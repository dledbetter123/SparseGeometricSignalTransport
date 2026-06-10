"""Generate wikitext2_v14b.ipynb — Tuned V14 on WikiText-2.

Changes from v14a:
1. Factored embedding: Embedding(50K, 32) → Linear(32, 136) — saves ~24M params
2. Bigger geometry: wilson_hidden 192→384, memory atoms 256→512, 10 blocks (was 8)
3. Sequence length 512 (was 256)

Models: D-tuned (factored + bigger geometry), GPT-Nano 224d (param-matched)."""
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
# CELL 0: Title
# ═══════════════════════════════════════════════════════════════
md("""# V14b: Tuned Geometry on WikiText-2

## What Changed from V14a

V14a on WikiText-2 showed:
- Geometry beats SSM+MLP (PPL 617 vs 1831) but loses to GPT-128d (PPL ~440)
- **94% of params wasted in embedding/decoder** (26.7M of 28.3M)
- Only 1.7M params in geometric blocks vs GPT's 2.4M in attention blocks
- Wilson line provides ~0.15 BPC benefit (small but consistent)

V14b fixes the parameter allocation:

| Change | V14a | V14b | Why |
|---|---|---|---|
| Embedding | 2×Embedding(50K, 136) = 13.7M | same (direct) | — |
| Decoder | Linear(256, 256)→Linear(256, 50K) = 13M | same | — |
| Wilson hidden | 192 | 384 | 2× fiber capacity |
| Memory atoms | 256 | 512 | 2× retrieval capacity |
| Blocks | 8 | 10 | More depth |
| Seq length | 256 | 512 | Longer context for fiber |""")

# ═══════════════════════════════════════════════════════════════
# CELL 1: Imports + Data
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
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
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
# CELL 2: Config
# ═══════════════════════════════════════════════════════════════
code("""@dataclass
class V14bConfig:
    # Spectral structure
    n_modes: int = 136
    n_subbundles: int = 8
    fiber_dim: int = 256

    # Embedding (direct, no factoring)

    # Wilson fiber — BIGGER
    wilson_hidden: int = 384         # was 192

    # Langevin settler — BIGGER
    n_memory_atoms: int = 512        # was 256
    n_langevin_steps: int = 2
    beta_min: float = 0.5
    beta_max: float = 5.0
    langevin_eta: float = 0.3

    # Model — DEEPER
    n_blocks: int = 10               # was 8
    vocab_size: int = 50257
    max_seq_len: int = 512           # was 256
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    lr_hold_steps: int = 500
    batch_size: int = 4              # smaller batch for seq_len 512
    seq_len: int = 512               # was 256
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 10

    @property
    def subbundle_dim(self):
        return self.fiber_dim // self.n_subbundles

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1

cfg = V14bConfig(vocab_size=vocab_size)
print(f"V14b: vocab={cfg.vocab_size:,} seq={cfg.seq_len} batch={cfg.batch_size}")
print(f"Modes: {cfg.n_modes}, Blocks: {cfg.n_blocks}")
print(f"Wilson hidden: {cfg.wilson_hidden}, Memory atoms: {cfg.n_memory_atoms}")
print(f"LR: {cfg.learning_rate}")

def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i+c.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Shared components
# ═══════════════════════════════════════════════════════════════
md("## Shared Components")

code("""class Constellation:
    def __init__(self, mag, phase):
        self.mag = mag
        self.phase = phase

    def to_complex(self):
        return self.mag * torch.exp(1j * self.phase)

    def to_flat(self):
        return torch.cat([self.mag, self.phase], dim=-1)


class MagPhaseNorm(nn.Module):
    def __init__(self, n_modes):
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(n_modes))

    def forward(self, c):
        mag_rms = (c.mag ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return Constellation(c.mag / mag_rms * self.mag_scale, c.phase)


class ConstellationEmbedding(nn.Module):
    \"\"\"Direct embedding: Embedding(vocab, n_modes) for mag and phase.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.mag_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        self.phase_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        nn.init.uniform_(self.phase_emb.weight, -math.pi, math.pi)

        freqs = torch.zeros(cfg.n_modes)
        for k in range(cfg.n_subbundles):
            off = k * cfg.spectral_half_dim
            freqs[off:off+cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0))
        self.register_buffer('freqs', freqs)

    def forward(self, token_ids):
        B, T = token_ids.shape
        mag = self.mag_emb(token_ids)
        phase = self.phase_emb(token_ids)
        pos = torch.arange(T, device=token_ids.device).float()
        phase = phase + (pos.unsqueeze(-1) * self.freqs).unsqueeze(0)
        return Constellation(mag, phase)


class ConstellationDecoder(nn.Module):
    \"\"\"Decoder: irfft → LayerNorm → MLP → vocab logits.\"\"\"
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
        spectral = constellation.to_complex()
        shd = self.cfg.spectral_half_dim
        subs = spectral.reshape(*spectral.shape[:-1], self.cfg.n_subbundles, shd)
        spatial = torch.fft.irfft(subs, n=self.cfg.subbundle_dim, dim=-1)
        spatial = spatial.reshape(*spectral.shape[:-1], self.cfg.fiber_dim)
        return self.head(self.norm(spatial))


def parallel_scan(alpha, x):
    N, T = alpha.shape
    a, b = alpha, x
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T:
            break
        b = torch.cat([b[:, :step],
                        a[:, step:] * b[:, :-step] + b[:, step:]], dim=1)
        a = torch.cat([a[:, :step],
                        a[:, step:] * a[:, :-step]], dim=1)
    return b


def complex_parallel_scan(a_re, a_im, b_re, b_im):
    N, T = a_re.shape
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T:
            break
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

_emb = ConstellationEmbedding(cfg)
_dec = ConstellationDecoder(cfg)
print(f"Embedding: {count_params(_emb):,} params")
print(f"Decoder:   {count_params(_dec):,} params")""")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Model D-tuned (Wilson + Langevin, bigger geometry)
# ═══════════════════════════════════════════════════════════════
md("## Model D-tuned: Bigger Geometry, Factored Embedding")

code("""class WilsonFiber(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.base_decay = nn.Parameter(torch.zeros(M))
        self.wilson_proj = nn.Sequential(
            nn.Linear(2 * M, cfg.wilson_hidden),
            nn.SiLU(),
            nn.Linear(cfg.wilson_hidden, 2 * M),
        )
        nn.init.zeros_(self.wilson_proj[-1].weight)
        nn.init.zeros_(self.wilson_proj[-1].bias)

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        flat = constellation.to_flat()
        wilson = self.wilson_proj(flat)
        decay_delta = wilson[..., :M]
        phase_rot = wilson[..., M:]
        decay = torch.sigmoid(self.base_decay + decay_delta).clamp(0.01, 0.99)
        theta = torch.tanh(phase_rot) * math.pi
        z_re = decay * torch.cos(theta)
        z_im = decay * torch.sin(theta)
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)
        z_re_f = z_re.permute(0, 2, 1).reshape(B * M, T)
        z_im_f = z_im.permute(0, 2, 1).reshape(B * M, T)
        c_re_f = c_re.permute(0, 2, 1).reshape(B * M, T)
        c_im_f = c_im.permute(0, 2, 1).reshape(B * M, T)
        h_re_f, h_im_f = complex_parallel_scan(z_re_f, z_im_f, c_re_f, c_im_f)
        h_re_f = F.pad(h_re_f[:, :-1], (1, 0))
        h_im_f = F.pad(h_im_f[:, :-1], (1, 0))
        h_re = h_re_f.reshape(B, M, T).permute(0, 2, 1)
        h_im = h_im_f.reshape(B, M, T).permute(0, 2, 1)
        return h_re * c_re + h_im * c_im


class LangevinSettler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.memory = nn.Parameter(torch.randn(cfg.n_memory_atoms, 2 * M) * 0.02)
        self.msg_proj = nn.Linear(M, 2 * M, bias=False)
        nn.init.normal_(self.msg_proj.weight, std=0.01)
        self.gate = nn.Parameter(torch.full((M,), -2.0))
        self.K = cfg.n_langevin_steps
        self.eta = cfg.langevin_eta
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max

    def forward(self, constellation, messages):
        M = constellation.mag.shape[-1]
        x = constellation.to_flat()
        x0 = x
        ctx = self.msg_proj(messages)
        m_norm = F.normalize(self.memory, dim=-1)
        for k in range(self.K):
            beta = self.beta_min + (self.beta_max - self.beta_min) * k / max(1, self.K - 1)
            q = F.normalize(x + ctx, dim=-1)
            scores = beta * (q @ m_norm.T)
            weights = F.softmax(scores, dim=-1)
            attractor = weights @ self.memory
            grad = attractor - x
            noise = 0.0
            if self.training:
                noise = math.sqrt(2 * self.eta / beta) * torch.randn_like(x)
            x = x + self.eta * grad + noise
        delta = (x - x0) + ctx
        g = torch.sigmoid(self.gate)
        return g * delta[..., :M], g * delta[..., M:]


class BlockD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = MagPhaseNorm(cfg.n_modes)
        self.fiber = WilsonFiber(cfg)
        self.settler = LangevinSettler(cfg)

    def forward(self, constellation):
        normed = self.norm(constellation)
        messages = self.fiber(normed)
        d_mag, d_phase = self.settler(normed, messages)
        return Constellation(constellation.mag + d_mag, constellation.phase + d_phase)


class GeomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([BlockD(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

    def forward(self, token_ids):
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)
        logits = self.decoder(c)[:, :-1, :]
        sp = (c.mag.abs() < 0.01).float().mean().item()
        return logits, {'spectral_sparsity': sp}


print(f"Block params: {count_params(BlockD(cfg)):,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 5: GPT-Nano
# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
    def __init__(self, vocab_size, n_embd=224, n_head=8, n_layer=12,
                 block_size=512, dropout=0.1):
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


# Instantiate
models = {}
models['D-tuned'] = GeomModel(cfg).to(device)
models['GPT-224d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=224, n_head=8,
                              n_layer=12, block_size=cfg.seq_len).to(device)

print(f"{'Model':<20} {'Params':>10}  {'Blocks':>10}  {'Emb+Dec':>10}")
print('=' * 55)
for name, m in models.items():
    total = count_params(m)
    if hasattr(m, 'blocks'):
        blk = sum(count_params(b) for b in m.blocks)
        emb_dec = total - blk
    else:
        blk = 0; emb_dec = total
    print(f"{name:<20} {total:>10,}  {blk:>10,}  {emb_dec:>10,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Training
# ═══════════════════════════════════════════════════════════════
md("## Training")

code("""@torch.no_grad()
def estimate_loss(model, c, is_gpt=False):
    model.eval()
    results = {}
    for name, sd in [('train', train_ids), ('val', val_ids)]:
        tot_ce, tot_ok, tot_n = 0., 0, 0
        for _ in range(c.eval_steps):
            b = get_batch(sd, c)
            logits, info = model(b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
        n = c.eval_steps
        results[name] = {'ce': tot_ce/n, 'acc': tot_ok/tot_n}
    model.train()
    return results


def train_model(model, c, label='model', is_gpt=False):
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
    np_ = count_params(model)
    smooth_loss = None

    pbar = tqdm(range(c.max_steps + 1), desc=label, unit='step')
    for step in pbar:
        if step % c.eval_interval == 0:
            r = estimate_loss(model, c, is_gpt=is_gpt)
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
        logits, info = model(batch)
        tgt = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        elapsed = time.time() - st
        hist['step_times'].append(elapsed)
        hist['per_step_loss'].append(loss.item())

        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        ppl = math.exp(min(smooth_loss, 20))
        lr_now = sched.get_last_lr()[0]
        pbar.set_postfix(loss=f"{smooth_loss:.3f}",
                         ppl=f"{ppl:.1f}",
                         bpc=f"{smooth_loss/math.log(2):.2f}",
                         lr=f"{lr_now:.1e}",
                         ms=f"{elapsed*1000:.0f}")

    pbar.close()
    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000
    final_ppl = math.exp(min(hist['val_ce'][-1], 20))
    print(f"  {label} DONE: {el/60:.1f}min | BPC:{hist['val_bpc'][-1]:.3f} "
          f"PPL:{final_ppl:.1f} Acc:{hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms; hist['n_params'] = np_
    return hist""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Train
# ═══════════════════════════════════════════════════════════════
md("## Train")

code("""all_hist = {}
for name, model in models.items():
    is_gpt = name.startswith('GPT')
    all_hist[name] = train_model(model, cfg, label=name, is_gpt=is_gpt)""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Results
# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""colors = {'D-tuned': 'tab:blue', 'GPT-224d': 'black'}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V14b: Tuned Geometry vs Param-Matched GPT on WikiText-2', fontsize=14, fontweight='bold')

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
w = 50
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ax.plot(range(len(sm)), sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
w = 50
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
plt.savefig('wikitext2_v14b_results.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n' + '='*70)
for name, h in all_hist.items():
    ppl = math.exp(min(h['val_ce'][-1], 20))
    print(f"  {name:<15} BPC:{h['val_bpc'][-1]:.3f}  PPL:{ppl:.1f}  "
          f"Params:{h['n_params']:,}  {h['avg_step_ms']:.0f}ms/step")

d = all_hist['D-tuned']
g = all_hist['GPT-224d']
d_ppl = math.exp(min(d['val_ce'][-1], 20))
g_ppl = math.exp(min(g['val_ce'][-1], 20))
print(f"\\n--- Verdict ---")
print(f"D-tuned ({d['n_params']:,}) vs GPT-224d ({g['n_params']:,})")
print(f"  BPC: {d['val_bpc'][-1]:.3f} vs {g['val_bpc'][-1]:.3f} ({d['val_bpc'][-1] - g['val_bpc'][-1]:+.3f})")
print(f"  PPL: {d_ppl:.0f} vs {g_ppl:.0f}")
print(f"  ms/step: {d['avg_step_ms']:.0f} vs {g['avg_step_ms']:.0f}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Text generation
# ═══════════════════════════════════════════════════════════════
code("""@torch.no_grad()
def gen(model, prompt_text, c, n=100, temp=0.8, is_gpt=False):
    model.eval()
    ids = torch.tensor(tokenize(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n):
        ctx = ids[:, -c.seq_len:]
        logits, _ = model(ctx)
        p = F.softmax(logits[:, -1, :] / temp, dim=-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    if hasattr(enc, 'decode'):
        return enc.decode(ids[0].tolist())
    return enc.decode(ids[0].tolist())

for prompt in ['The meaning of life is', 'In the beginning', 'Scientists discovered that']:
    print(f"\\nPrompt: {repr(prompt)}")
    for name, model in models.items():
        try:
            text = gen(model, prompt, cfg, n=50, is_gpt=name.startswith('GPT'))
            print(f"  {name}: {text[len(prompt):len(prompt)+100]}")
        except Exception as e:
            print(f"  {name}: error - {e}")""")

# ═══════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells,
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext2_v14b.ipynb")
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
