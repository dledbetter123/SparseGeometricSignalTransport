"""Generate architecture_v16c.ipynb — Spectral-Native Architecture.

V16c: Stay entirely in spectral space. No irfft/rfft per block.
The FFN operates directly on (mag, phase, log_var) features.
Energy guarantees via CloudNorm + Parseval filter constraint.
Positional encoding via phase shift (built into constellation).

Removes: irfft/rfft round-trip, spatial positional encoding, position FFT.
Keeps: Wilson fiber, Parseval filter, cross-mode interaction, local conv, FFN, Gaussian clouds."""
import json
import os

cells = []

def md(source):
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source_list})

def code(source):
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": source_list,
                  "outputs": [], "execution_count": None})


# ═══════════════════════════════════════════════════════════════
md("""# V16c: Spectral-Native (No irfft Round-Trip)

## Why

The per-block irfft/rfft is a fixed basis change (DFT matrix multiply) applied per-position.
It provides no cross-position mixing. The FFN can learn whatever basis it needs from raw
spectral features. CloudNorm controls spectral energy. Parseval's theorem guarantees spatial
energy equivalence — whether or not we compute the spatial representation.

By staying in spectral space:
- Phase carries explicit positional encoding throughout (no separate pos_emb needed)
- No wasted compute on 2× FFT per block
- Simpler architecture, same energy guarantees

## Block Structure

```
V16cBlock:
  CloudNorm(constellation)
  ├── Wilson fiber: h[t] = z_t·h[t-1] + c_t     (causal, complex)
  ├── Parseval filter: W⊙h, |W|≤1, cross-mode    (per-position, spectral)
  ├── Local conv on spectral flat                  (causal, kernel=7)
  └── FFN on (mag, phase, log_var, filtered)       (per-position, nonlinear)
  → Updated constellation (mag, phase, log_var)
```

All causal. All spectral. No domain switching.""")

# ═══════════════════════════════════════════════════════════════
code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm.auto import tqdm
import math
import time
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

try:
    from datasets import load_dataset
    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\\n".join(ds["train"]["text"])
    val_text = "\\n".join(ds["validation"]["text"])
except ImportError:
    raise ImportError("pip install datasets")

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
except ImportError:
    from transformers import GPT2TokenizerFast
    enc = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = enc.vocab_size

def tokenize(text):
    if hasattr(enc, 'encode_ordinary'):
        return enc.encode_ordinary(text)
    if hasattr(enc, 'encode'):
        return enc.encode(text)
    return enc(text)['input_ids']

print("Tokenizing...")
train_ids = torch.tensor(tokenize(train_text), dtype=torch.long)
val_ids = torch.tensor(tokenize(val_text), dtype=torch.long)
print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens, Vocab: {vocab_size:,}")""")

# ═══════════════════════════════════════════════════════════════
code("""@dataclass
class V16cConfig:
    # Spectral structure
    n_subbundles: int = 8
    subbundle_dim: int = 32
    n_modes: int = 136
    fiber_dim: int = 256              # only used in decoder now

    # Wilson fiber
    wilson_hidden: int = 256

    # Parseval filter
    filter_hidden: int = 256

    # Local conv (on spectral flat = 3M dims: mag, phase, log_var)
    local_kernel: int = 7

    # FFN (on spectral features: input = 3M + M filtered = 4M, or just 3M)
    ffn_hidden: int = 1088            # ~4× the 272 (2M) working dim

    # Model
    n_blocks: int = 7
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    lr_hold_steps: int = 3000
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 20000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1

cfg = V16cConfig(vocab_size=vocab_size)
print(f"V16c: vocab={cfg.vocab_size:,} seq={cfg.seq_len} batch={cfg.batch_size}")
print(f"Modes: {cfg.n_modes}, Blocks: {cfg.n_blocks}")
print(f"LR: {cfg.learning_rate}, Steps: {cfg.max_steps}")

def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i+c.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
md("## Components")

code("""class Constellation:
    def __init__(self, mag, phase, log_var):
        self.mag = mag
        self.phase = phase
        self.log_var = log_var
    def to_complex(self):
        return self.mag * torch.exp(1j * self.phase)
    def to_flat(self):
        return torch.cat([self.mag, self.phase, self.log_var], dim=-1)
    def precision(self):
        return torch.exp(-self.log_var)


class CloudNorm(nn.Module):
    \"\"\"RMSNorm on magnitudes = spectral energy normalization.
    By Parseval: this controls spatial energy too, without computing irfft.\"\"\"
    def __init__(self, n_modes):
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(n_modes))
    def forward(self, c):
        mag_rms = (c.mag ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return Constellation(c.mag / mag_rms * self.mag_scale, c.phase, c.log_var)


class ConstellationEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.mag_emb = nn.Embedding(cfg.vocab_size, M)
        self.phase_emb = nn.Embedding(cfg.vocab_size, M)
        self.var_proj = nn.Linear(M, M, bias=True)
        nn.init.uniform_(self.phase_emb.weight, -math.pi, math.pi)
        nn.init.zeros_(self.var_proj.weight)
        nn.init.zeros_(self.var_proj.bias)
        freqs = torch.zeros(M)
        for k in range(cfg.n_subbundles):
            off = k * cfg.spectral_half_dim
            freqs[off:off+cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0))
        self.register_buffer('freqs', freqs)

    def forward(self, token_ids):
        B, T = token_ids.shape
        mag = self.mag_emb(token_ids)
        phase = self.phase_emb(token_ids)
        log_var = self.var_proj(mag)
        pos = torch.arange(T, device=token_ids.device).float()
        phase = phase + (pos.unsqueeze(-1) * self.freqs).unsqueeze(0)
        return Constellation(mag, phase, log_var)


class SpectralDecoder(nn.Module):
    \"\"\"Decode from spectral to logits. Only place irfft is used — at the output.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.fiber_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )
    def forward(self, constellation):
        precision = constellation.precision()
        weighted_mag = constellation.mag * torch.sigmoid(precision)
        spectral = weighted_mag * torch.exp(1j * constellation.phase)
        shd = self.cfg.spectral_half_dim
        subs = spectral.reshape(*spectral.shape[:-1], self.cfg.n_subbundles, shd)
        spatial = torch.fft.irfft(subs, n=self.cfg.subbundle_dim, dim=-1)
        spatial = spatial.reshape(*spectral.shape[:-1], self.cfg.fiber_dim)
        return self.head(self.norm(spatial))


def real_parallel_scan(alpha, x):
    N, T = alpha.shape
    a, b = alpha, x
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T: break
        b = torch.cat([b[:, :step],
                        a[:, step:] * b[:, :-step] + b[:, step:]], dim=1)
        a = torch.cat([a[:, :step],
                        a[:, step:] * a[:, :-step]], dim=1)
    return b


def complex_parallel_scan(a_re, a_im, b_re, b_im):
    N, T = a_re.shape
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T: break
        ar, ai = a_re[:, step:], a_im[:, step:]
        al, ail = a_re[:, :-step], a_im[:, :-step]
        bl, bil = b_re[:, :-step], b_im[:, :-step]
        ab_re = ar * bl - ai * bil
        ab_im = ar * bil + ai * bl
        aa_re = ar * al - ai * ail
        aa_im = ar * ail + ai * al
        b_re = torch.cat([b_re[:, :step], ab_re + b_re[:, step:]], dim=1)
        b_im = torch.cat([b_im[:, :step], ab_im + b_im[:, step:]], dim=1)
        a_re = torch.cat([a_re[:, :step], aa_re], dim=1)
        a_im = torch.cat([a_im[:, :step], aa_im], dim=1)
    return b_re, b_im


def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("Components loaded.")""")

# ═══════════════════════════════════════════════════════════════
md("## V16c Block: Everything in Spectral Space")

code("""class WilsonFiber(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.base_decay = nn.Parameter(torch.zeros(M))
        self.wilson_proj = nn.Sequential(
            nn.Linear(3 * M, cfg.wilson_hidden), nn.SiLU(),
            nn.Linear(cfg.wilson_hidden, 2 * M),
        )
        nn.init.zeros_(self.wilson_proj[-1].weight)
        nn.init.zeros_(self.wilson_proj[-1].bias)

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        flat = constellation.to_flat()
        wilson = self.wilson_proj(flat)
        decay = torch.sigmoid(self.base_decay + wilson[..., :M]).clamp(0.01, 0.99)
        theta = torch.tanh(wilson[..., M:]) * math.pi
        z_re = decay * torch.cos(theta)
        z_im = decay * torch.sin(theta)
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)
        z_re_f = z_re.permute(0,2,1).reshape(B*M, T)
        z_im_f = z_im.permute(0,2,1).reshape(B*M, T)
        c_re_f = c_re.permute(0,2,1).reshape(B*M, T)
        c_im_f = c_im.permute(0,2,1).reshape(B*M, T)
        h_re_f, h_im_f = complex_parallel_scan(z_re_f, z_im_f, c_re_f, c_im_f)
        h_re_f = F.pad(h_re_f[:, :-1], (1, 0))
        h_im_f = F.pad(h_im_f[:, :-1], (1, 0))
        h_re = h_re_f.reshape(B, M, T).permute(0, 2, 1)
        h_im = h_im_f.reshape(B, M, T).permute(0, 2, 1)
        # Variance accumulation
        input_var = torch.exp(constellation.log_var)
        decay_sq_f = (decay ** 2).permute(0,2,1).reshape(B*M, T)
        var_f = input_var.permute(0,2,1).reshape(B*M, T)
        h_var_f = real_parallel_scan(decay_sq_f, var_f)
        h_var_f = F.pad(h_var_f[:, :-1], (1, 0))
        h_var = h_var_f.reshape(B, M, T).permute(0, 2, 1)
        return h_re, h_im, h_var


class ParsevalFilter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        shd = cfg.spectral_half_dim
        nsub = cfg.n_subbundles
        self.filter_net = nn.Sequential(
            nn.Linear(4 * M, cfg.filter_hidden), nn.SiLU(),
            nn.Linear(cfg.filter_hidden, 2 * M),
        )
        nn.init.zeros_(self.filter_net[-1].weight)
        nn.init.zeros_(self.filter_net[-1].bias)
        self.cross_re = nn.Parameter(torch.eye(shd).unsqueeze(0).expand(nsub,-1,-1).clone())
        self.cross_im = nn.Parameter(torch.zeros(nsub, shd, shd))
        self.nsub = nsub
        self.shd = shd

    def forward(self, constellation, h_re, h_im, h_var):
        B, T, M = constellation.mag.shape
        inp = torch.cat([constellation.to_flat(), h_var], dim=-1)
        raw = self.filter_net(inp)
        w_mag = torch.sigmoid(raw[..., :M])
        w_phase = raw[..., M:]
        w_re = w_mag * torch.cos(w_phase)
        w_im = w_mag * torch.sin(w_phase)
        g_re = (w_re * h_re - w_im * h_im).reshape(B, T, self.nsub, self.shd)
        g_im = (w_re * h_im + w_im * h_re).reshape(B, T, self.nsub, self.shd)
        y_re = (torch.einsum('btsi,sio->btso', g_re, self.cross_re)
              - torch.einsum('btsi,sio->btso', g_im, self.cross_im))
        y_im = (torch.einsum('btsi,sio->btso', g_re, self.cross_im)
              + torch.einsum('btsi,sio->btso', g_im, self.cross_re))
        return y_re.reshape(B, T, M), y_im.reshape(B, T, M)


class SpectralLocalConv(nn.Module):
    \"\"\"Causal depthwise conv on spectral flat (mag, phase, log_var = 3M dims).\"\"\"
    def __init__(self, cfg):
        super().__init__()
        D = 3 * cfg.n_modes  # operate on full spectral flat
        k = cfg.local_kernel
        self.pad = k - 1
        self.conv = nn.Conv1d(D, D, kernel_size=k, groups=D, bias=True)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, flat):
        h = flat.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h).transpose(1, 2)
        return torch.sigmoid(self.gate) * h


class SpectralFFN(nn.Module):
    \"\"\"FFN operating directly on spectral features. No basis change needed.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        # Input: 2M (mag + phase from filtered) + M (messages magnitude) = 3M
        # Or simpler: operate on the constellation flat (3M) after updates
        self.norm = nn.LayerNorm(2 * M)
        self.net = nn.Sequential(
            nn.Linear(2 * M, cfg.ffn_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, 2 * M),
            nn.Dropout(cfg.dropout),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, mag, phase):
        x = torch.cat([mag, phase], dim=-1)
        delta = self.net(self.norm(x))
        M = mag.shape[-1]
        return delta[..., :M], delta[..., M:]


class V16cBlock(nn.Module):
    \"\"\"Spectral-native block. No irfft/rfft. All operations in spectral space.

    1. CloudNorm: spectral energy normalization
    2. Wilson fiber: causal complex EMA + variance accumulation
    3. Parseval filter: spectral gating + cross-mode interaction
    4. Combine: constellation + gated filtered context (in spectral space)
    5. Local conv: causal, on spectral features
    6. FFN: nonlinear processing on (mag, phase) directly
    7. Update constellation\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.norm = CloudNorm(cfg.n_modes)
        self.fiber = WilsonFiber(cfg)
        self.pfilter = ParsevalFilter(cfg)
        self.fiber_gate = nn.Parameter(torch.tensor(-2.0))
        self.local = SpectralLocalConv(cfg)
        self.ffn = SpectralFFN(cfg)

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        normed = self.norm(constellation)

        # --- Fiber + Parseval filter (all spectral) ---
        h_re, h_im, h_var = self.fiber(normed)
        y_re, y_im = self.pfilter(normed, h_re, h_im, h_var)

        # Combine filtered context with constellation in (re, im) form
        # Avoid mag/phase conversion (atan2 gradients unstable near zero)
        g = torch.sigmoid(self.fiber_gate)

        # Current constellation as (re, im)
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)

        # Add gated filtered context in cartesian
        updated_re = c_re + g * y_re
        updated_im = c_im + g * y_im

        # Convert back to (mag, phase)
        updated_mag = torch.sqrt(updated_re ** 2 + updated_im ** 2 + 1e-8)
        updated_phase = torch.atan2(updated_im, updated_re)

        # --- Local conv on spectral features ---
        flat = torch.cat([updated_mag, updated_phase, constellation.log_var], dim=-1)
        local_delta = self.local(flat)
        updated_mag = updated_mag + local_delta[..., :M]
        updated_phase = updated_phase + local_delta[..., M:2*M]
        # Wrap phase after local conv update
        updated_phase = torch.remainder(updated_phase + math.pi, 2 * math.pi) - math.pi
        updated_log_var = constellation.log_var + local_delta[..., 2*M:]

        # --- FFN on (mag, phase) ---
        d_mag, d_phase = self.ffn(updated_mag, updated_phase)
        final_mag = updated_mag + d_mag
        final_phase = updated_phase + d_phase

        # Wrap phase to [-π, π] and clamp magnitudes for numerical stability
        final_phase = torch.remainder(final_phase + math.pi, 2 * math.pi) - math.pi
        final_mag = final_mag.clamp(-50, 50)

        # Variance: shrink per block (confidence increases with depth)
        final_log_var = updated_log_var - 0.1
        final_log_var = final_log_var.clamp(min=-6, max=2)

        return Constellation(final_mag, final_phase, final_log_var)


class V16cModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V16cBlock(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = SpectralDecoder(cfg)

    def forward(self, token_ids):
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)
        logits = self.decoder(c)[:, :-1, :]
        return logits, {}


_b = V16cBlock(cfg)
print(f"V16cBlock: {count_params(_b):,} params")
print(f"  Fiber:   {count_params(_b.fiber):,}")
print(f"  Filter:  {count_params(_b.pfilter):,}")
print(f"  Local:   {count_params(_b.local):,}")
print(f"  FFN:     {count_params(_b.ffn):,}")""")

# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
    def __init__(self, vocab_size, n_embd=224, n_head=8, n_layer=12,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(n_embd),
                'attn_qkv': nn.Linear(n_embd, 3 * n_embd),
                'attn_proj': nn.Linear(n_embd, n_embd),
                'ln2': nn.LayerNorm(n_embd),
                'mlp_fc1': nn.Linear(n_embd, 4 * n_embd),
                'mlp_fc2': nn.Linear(4 * n_embd, n_embd),
            }))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        hd = self.n_embd // self.n_head
        for blk in self.blocks:
            h = blk['ln1'](x)
            qkv = blk['attn_qkv'](h).reshape(B, T, 3, self.n_head, hd)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            att = (q @ k.transpose(-2,-1)) * (hd**-0.5)
            att = att.masked_fill(self.causal_mask[:,:,:T,:T]==0, float('-inf'))
            y = (F.softmax(att, dim=-1) @ v).transpose(1,2).reshape(B, T, self.n_embd)
            x = x + blk['attn_proj'](y)
            x = x + blk['mlp_fc2'](F.gelu(blk['mlp_fc1'](blk['ln2'](x))))
        return self.lm_head(self.ln_f(x))[:, :-1, :], {}


models = {}
models['V16c'] = V16cModel(cfg).to(device)
models['GPT-224d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=224, n_head=8,
                              n_layer=12, block_size=cfg.seq_len).to(device)

print(f"\\n{'Model':<15} {'Total':>10}  {'Blocks':>10}  {'Emb+Dec':>10}")
print('=' * 50)
for name, m in models.items():
    total = count_params(m)
    blk = sum(count_params(b) for b in m.blocks)
    print(f"{name:<15} {total:>10,}  {blk:>10,}  {total-blk:>10,}")""")

# ═══════════════════════════════════════════════════════════════
md("## Training")

code("""@torch.no_grad()
def estimate_loss(model, c):
    model.eval()
    results = {}
    for name, sd in [('train', train_ids), ('val', val_ids)]:
        tot_ce, tot_ok, tot_n = 0., 0, 0
        for _ in range(c.eval_steps):
            b = get_batch(sd, c)
            logits, _ = model(b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
        n = c.eval_steps
        results[name] = {'ce': tot_ce/n, 'acc': tot_ok/tot_n}
    model.train()
    return results

def train_model(model, c, label='model'):
    opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.05)
    mr = c.min_lr / c.learning_rate
    he = c.warmup_steps + c.lr_hold_steps
    def lr_fn(s):
        if s < c.warmup_steps: return s / max(1, c.warmup_steps)
        if s < he: return 1.0
        p = (s - he) / max(1, c.max_steps - he)
        return max(mr, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    hist = {'step':[], 'train_ce':[], 'val_ce':[], 'train_acc':[], 'val_acc':[],
            'train_bpc':[], 'val_bpc':[], 'step_times':[], 'per_step_loss':[]}
    model.train()
    t0 = time.time()
    smooth_loss = None

    pbar = tqdm(range(c.max_steps + 1), desc=label, unit='step')
    for step in pbar:
        if step % c.eval_interval == 0:
            r = estimate_loss(model, c)
            tr, vl = r['train'], r['val']
            hist['step'].append(step)
            hist['train_ce'].append(tr['ce']); hist['val_ce'].append(vl['ce'])
            hist['train_acc'].append(tr['acc']); hist['val_acc'].append(vl['acc'])
            hist['train_bpc'].append(tr['ce']/math.log(2))
            hist['val_bpc'].append(vl['ce']/math.log(2))
            vl_ppl = math.exp(min(vl['ce'], 20))
            tqdm.write(f"  [{label}] {step:5d} | Val CE:{vl['ce']:.3f} "
                       f"BPC:{vl['ce']/math.log(2):.3f} PPL:{vl_ppl:.1f} Acc:{vl['acc']:.1%}")
        if step >= c.max_steps: break
        st = time.time()
        batch = get_batch(train_ids, c)
        opt.zero_grad()
        logits, _ = model(batch)
        tgt = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        elapsed = time.time() - st
        hist['step_times'].append(elapsed)
        hist['per_step_loss'].append(loss.item())
        if smooth_loss is None: smooth_loss = loss.item()
        else: smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        ppl = math.exp(min(smooth_loss, 20))
        pbar.set_postfix(loss=f"{smooth_loss:.3f}", ppl=f"{ppl:.1f}",
                         bpc=f"{smooth_loss/math.log(2):.2f}",
                         lr=f"{sched.get_last_lr()[0]:.1e}", ms=f"{elapsed*1000:.0f}")

    pbar.close()
    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000
    final_ppl = math.exp(min(hist['val_ce'][-1], 20))
    print(f"  {label} DONE: {el/60:.1f}min | BPC:{hist['val_bpc'][-1]:.3f} "
          f"PPL:{final_ppl:.1f} Acc:{hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms; hist['n_params'] = count_params(model)
    return hist""")

# ═══════════════════════════════════════════════════════════════
code("""all_hist = {}
for name, model in models.items():
    all_hist[name] = train_model(model, cfg, label=name)""")

# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""colors = {'V16c': 'tab:blue', 'GPT-224d': 'black'}
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V16c (Spectral-Native) vs GPT-224d on WikiText-103', fontsize=14, fontweight='bold')

ax = axes[0, 0]
for name, h in all_hist.items():
    ax.plot(h['step'], h['val_bpc'], '-o', color=colors[name], label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val BPC'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for name, h in all_hist.items():
    ppl = [math.exp(min(ce, 20)) for ce in h['val_ce']]
    ax.plot(h['step'], ppl, '-o', color=colors[name], label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Perplexity'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
for name, h in all_hist.items():
    ax.plot(h['step'], [a*100 for a in h['val_acc']], '-o', color=colors[name],
            label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Accuracy %'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
w = 100
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ax.plot(range(len(sm)), sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ppl_sm = [math.exp(min(x, 20)) for x in sm]
        ax.plot(range(len(ppl_sm)), ppl_sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Perplexity (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 2]; ax.axis('off')
rows = [[name, f"{h['n_params']:,}", f"{h['val_bpc'][-1]:.3f}",
         f"{math.exp(min(h['val_ce'][-1],20)):.1f}",
         f"{h['val_acc'][-1]:.1%}", f"{h['avg_step_ms']:.0f}"]
        for name, h in all_hist.items()]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','PPL','Acc','ms/step'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('v16c_results.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n' + '='*70)
for name, h in all_hist.items():
    ppl = math.exp(min(h['val_ce'][-1], 20))
    print(f"  {name:<15} BPC:{h['val_bpc'][-1]:.3f}  PPL:{ppl:.1f}  "
          f"Params:{h['n_params']:,}  {h['avg_step_ms']:.0f}ms/step")""")

# ═══════════════════════════════════════════════════════════════
code("""@torch.no_grad()
def gen(model, prompt_text, c, n=100, temp=0.8):
    model.eval()
    ids = torch.tensor(tokenize(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n):
        ctx = ids[:, -c.seq_len:]
        logits, _ = model(ctx)
        p = F.softmax(logits[:, -1, :] / temp, dim=-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return enc.decode(ids[0].tolist())

for prompt in ['The meaning of life is', 'In the beginning', 'Scientists discovered that']:
    print(f"\\nPrompt: {repr(prompt)}")
    for name, model in models.items():
        try:
            text = gen(model, prompt, cfg, n=50)
            print(f"  {name}: {text[len(prompt):len(prompt)+100]}")
        except Exception as e:
            print(f"  {name}: error - {e}")""")

# ═══════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells,
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v16c.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Created {outpath} with {len(cells)} cells")
import ast
errs = 0
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        try: ast.parse(''.join(c['source']))
        except SyntaxError as e:
            print(f"SYNTAX ERROR cell {i}: {e}"); errs += 1
if errs == 0:
    print(f"All {sum(1 for c in cells if c['cell_type']=='code')} code cells parse OK")
