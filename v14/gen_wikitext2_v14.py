"""Generate wikitext2_v14.ipynb — V14 variants on WikiText-2 with BPE tokenization.

Tests the geometric machinery on a real dataset:
- 33K vocab (vs 65 chars) — spectral sparsity should activate
- 2M+ tokens — proper data scale for ~2M param models
- BPE tokenization — semantic tokens, not characters

Models: D (best from ablation: no sparsity), B (efficiency winner), E (SSM+MLP bar), GPT-Nano."""
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
md("""# V14 on WikiText-2: Real Vocabulary, Real Data

## Why WikiText-2

The Tiny Shakespeare ablation (vocab 65) showed:
- **Geometry earns its keep** (A >> E by 0.265 BPC)
- **Sparsity hurts** at small vocab (D > A by 0.112 BPC)
- **Wilson line is marginal** on character-level (B ≈ A)
- **Hopfield memory bank is the key contributor**

WikiText-2 tests what Tiny Shakespeare cannot:
- **Vocab 33,278** — 136 spectral modes must represent 33K tokens. Sparsity may activate.
- **2M+ tokens** — proper data scale
- **BPE tokenization** — semantic content per token
- **Longer effective context** — each BPE token ≈ 4 characters

## Models Tested

| Model | Description | Why |
|---|---|---|
| D: No Sparsity | Wilson fiber + Langevin, no threshold | Best from Shakespeare ablation |
| B: No Wilson | Fixed decay + Langevin | Efficiency winner (half params) |
| E: SSM+MLP | Real EMA + MLP (param-matched) | The bar to clear |
| GPT-Nano 128d | 12-layer attention, 128-dim (15M) | Small baseline |
| GPT-Nano 224d | 12-layer attention, 224-dim (28M) | Param-matched baseline |""")

# ═══════════════════════════════════════════════════════════════
# CELL 1: Imports + WikiText-2 Data
# ═══════════════════════════════════════════════════════════════
code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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

# --- WikiText-2 via HuggingFace datasets ---
try:
    from datasets import load_dataset
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\\n".join(ds["train"]["text"])
    val_text = "\\n".join(ds["validation"]["text"])
    test_text = "\\n".join(ds["test"]["text"])
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")
except ImportError:
    print("pip install datasets  — required for WikiText-2")
    raise

# --- BPE tokenizer ---
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50257
    print(f"Using tiktoken GPT-2 BPE: vocab {vocab_size:,}")
except ImportError:
    try:
        from transformers import GPT2TokenizerFast
        enc = GPT2TokenizerFast.from_pretrained("gpt2")
        vocab_size = enc.vocab_size  # 50257
        print(f"Using HF GPT2 tokenizer: vocab {vocab_size:,}")
    except ImportError:
        print("pip install tiktoken  OR  pip install transformers  — need a BPE tokenizer")
        raise

# Tokenize
def tokenize(text):
    if hasattr(enc, 'encode'):
        if hasattr(enc, 'encode_ordinary'):
            return enc.encode_ordinary(text)
        return enc.encode(text)
    return enc(text)['input_ids']

print("Tokenizing...")
train_ids = torch.tensor(tokenize(train_text), dtype=torch.long)
val_ids = torch.tensor(tokenize(val_text), dtype=torch.long)
print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens")
print(f"Vocab: {vocab_size:,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 2: Config
# ═══════════════════════════════════════════════════════════════
code("""@dataclass
class WT2Config:
    # Spectral structure
    n_modes: int = 136
    n_subbundles: int = 8
    fiber_dim: int = 256

    # Wilson fiber
    wilson_hidden: int = 192

    # Langevin settler
    n_memory_atoms: int = 256
    n_langevin_steps: int = 2
    beta_min: float = 0.5
    beta_max: float = 5.0
    langevin_eta: float = 0.3

    # SSM+MLP ablation
    ssm_mlp_hidden: int = 311

    # Model
    n_blocks: int = 8
    vocab_size: int = 50257          # GPT-2 BPE vocab
    max_seq_len: int = 256           # shorter seqlen for speed with larger vocab
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    lr_hold_steps: int = 500
    batch_size: int = 8              # smaller batch (larger vocab = more memory)
    seq_len: int = 256
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 10

    @property
    def subbundle_dim(self):
        return self.fiber_dim // self.n_subbundles

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1

cfg = WT2Config(vocab_size=vocab_size)
print(f"Config: vocab={cfg.vocab_size:,} seq={cfg.seq_len} batch={cfg.batch_size}")
print(f"Modes: {cfg.n_modes}, Blocks: {cfg.n_blocks}, Steps: {cfg.max_steps}")

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

print("Shared components loaded.")""")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Model D — No Sparsity (Best from Shakespeare)
# ═══════════════════════════════════════════════════════════════
md("## Model D: Wilson Fiber + Langevin, No Sparsity (Best from ablation)")

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

print(f"Block D params: {count_params(BlockD(cfg)):,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Model B — No Wilson (Efficiency Winner)
# ═══════════════════════════════════════════════════════════════
md("## Model B: Fixed Decay + Langevin (Efficiency Winner)")

code("""class FixedDecayFiber(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decay = nn.Parameter(torch.zeros(cfg.n_modes))

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)
        alpha = torch.sigmoid(self.decay).clamp(0.01, 0.99)
        alpha_flat = alpha.unsqueeze(0).expand(B, -1).reshape(B * M, 1).expand(-1, T)
        re_flat = c_re.permute(0, 2, 1).reshape(B * M, T)
        im_flat = c_im.permute(0, 2, 1).reshape(B * M, T)
        h_flat = parallel_scan(
            torch.cat([alpha_flat, alpha_flat], dim=0),
            torch.cat([re_flat, im_flat], dim=0))
        h_flat = F.pad(h_flat[:, :-1], (1, 0))
        h_re = h_flat[:B*M].reshape(B, M, T).permute(0, 2, 1)
        h_im = h_flat[B*M:].reshape(B, M, T).permute(0, 2, 1)
        return h_re * c_re + h_im * c_im


class BlockB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = MagPhaseNorm(cfg.n_modes)
        self.fiber = FixedDecayFiber(cfg)
        self.settler = LangevinSettler(cfg)

    def forward(self, constellation):
        normed = self.norm(constellation)
        messages = self.fiber(normed)
        d_mag, d_phase = self.settler(normed, messages)
        return Constellation(constellation.mag + d_mag, constellation.phase + d_phase)

print(f"Block B params: {count_params(BlockB(cfg)):,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Model E — SSM+MLP (The Bar)
# ═══════════════════════════════════════════════════════════════
md("## Model E: SSM+MLP (The Bar to Clear)")

code("""class BlockE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.norm = MagPhaseNorm(M)
        self.decay = nn.Parameter(torch.zeros(M))
        self.mlp = nn.Sequential(
            nn.Linear(3 * M, cfg.ssm_mlp_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ssm_mlp_hidden, 2 * M),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, constellation):
        normed = self.norm(constellation)
        B, T, M = normed.mag.shape
        c_re = normed.mag * torch.cos(normed.phase)
        c_im = normed.mag * torch.sin(normed.phase)
        alpha = torch.sigmoid(self.decay).clamp(0.01, 0.99)
        alpha_flat = alpha.unsqueeze(0).expand(B, -1).reshape(B * M, 1).expand(-1, T)
        re_flat = c_re.permute(0, 2, 1).reshape(B * M, T)
        im_flat = c_im.permute(0, 2, 1).reshape(B * M, T)
        h_flat = parallel_scan(
            torch.cat([alpha_flat, alpha_flat], dim=0),
            torch.cat([re_flat, im_flat], dim=0))
        h_flat = F.pad(h_flat[:, :-1], (1, 0))
        h_re = h_flat[:B*M].reshape(B, M, T).permute(0, 2, 1)
        h_im = h_flat[B*M:].reshape(B, M, T).permute(0, 2, 1)
        messages = h_re * c_re + h_im * c_im
        combined = torch.cat([normed.mag, normed.phase, messages], dim=-1)
        delta = self.mlp(combined)
        g = torch.sigmoid(self.gate)
        d_mag = g * delta[..., :M]
        d_phase = g * delta[..., M:]
        return Constellation(constellation.mag + d_mag, constellation.phase + d_phase)

print(f"Block E params: {count_params(BlockE(cfg)):,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: GPT-Nano + Model Factory
# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=12,
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


class GeomModel(nn.Module):
    def __init__(self, cfg, block_class):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([block_class(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

    def forward(self, token_ids):
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)
        logits = self.decoder(c)[:, :-1, :]
        sp = (c.mag.abs() < 0.01).float().mean().item()
        return logits, {'spectral_sparsity': sp}


# Instantiate
models = {}
for name, bcls in [('D: No Sparsity', BlockD), ('B: No Wilson', BlockB),
                    ('E: SSM+MLP', BlockE)]:
    models[name] = GeomModel(cfg, bcls).to(device)

# GPT-Nano at two scales: original (15M) and param-matched (28M)
models['GPT-Nano 128d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=128, n_head=4,
                                   n_layer=12, block_size=cfg.seq_len).to(device)
models['GPT-Nano 224d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=224, n_head=8,
                                   n_layer=12, block_size=cfg.seq_len).to(device)

print(f"{'Model':<20} {'Params':>10}")
print('=' * 32)
for name, m in models.items():
    print(f"{name:<20} {count_params(m):>10,}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Training
# ═══════════════════════════════════════════════════════════════
md("## Training")

code("""from tqdm.auto import tqdm

@torch.no_grad()
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
            tqdm.write(f"  [{label}] {step:5d} | Val CE:{vl['ce']:.3f} "
                       f"BPC:{vl['ce']/math.log(2):.3f} Acc:{vl['acc']:.1%}")

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

        # Smoothed loss for tqdm bar
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        lr_now = sched.get_last_lr()[0]
        ppl = math.exp(min(smooth_loss, 20))  # clamp to avoid overflow
        pbar.set_postfix(loss=f"{smooth_loss:.3f}",
                         ppl=f"{ppl:.1f}",
                         bpc=f"{smooth_loss/math.log(2):.2f}",
                         lr=f"{lr_now:.1e}",
                         ms=f"{elapsed*1000:.0f}")

    pbar.close()
    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000
    final_ppl = math.exp(hist['val_ce'][-1])
    print(f"  {label} DONE: {el/60:.1f}min | BPC:{hist['val_bpc'][-1]:.3f} "
          f"PPL:{final_ppl:.1f} Acc:{hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms; hist['n_params'] = np_
    return hist""")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Train all models
# ═══════════════════════════════════════════════════════════════
md("## Train All Models")

code("""all_hist = {}
for name, model in models.items():
    is_gpt = name.startswith('GPT')
    all_hist[name] = train_model(model, cfg, label=name, is_gpt=is_gpt)""")

# ═══════════════════════════════════════════════════════════════
# CELL 10: Results
# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""colors = {
    'D: No Sparsity': 'tab:purple',
    'B: No Wilson': 'tab:orange',
    'E: SSM+MLP': 'tab:red',
    'GPT-Nano 128d': 'tab:gray',
    'GPT-Nano 224d': 'black',
}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V14 Variants on WikiText-2 (BPE, vocab 50K)', fontsize=14, fontweight='bold')

# Val BPC
ax = axes[0, 0]
for name, h in all_hist.items():
    ax.plot(h['step'], h['val_bpc'], '-o', color=colors[name], label=name, markersize=2)
ax.set_xlabel('Step'); ax.set_title('Val BPC'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Val Perplexity
ax = axes[0, 1]
for name, h in all_hist.items():
    ppl = [math.exp(ce) for ce in h['val_ce']]
    ax.plot(h['step'], ppl, '-o', color=colors[name], label=name, markersize=2)
ax.set_xlabel('Step'); ax.set_title('Val Perplexity (lower = better)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Val Accuracy
ax = axes[0, 2]
for name, h in all_hist.items():
    ax.plot(h['step'], [a*100 for a in h['val_acc']], '-o', color=colors[name],
            label=name, markersize=2)
ax.set_xlabel('Step'); ax.set_title('Val Accuracy %'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Smoothed loss
ax = axes[1, 0]
w = 50
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ax.plot(range(len(sm)), sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Smoothed perplexity
ax = axes[1, 1]
w = 50
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ppl_sm = [math.exp(min(x, 20)) for x in sm]  # clamp to avoid overflow
        ax.plot(range(len(ppl_sm)), ppl_sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Perplexity (smooth {w})'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Table
ax = axes[1, 2]; ax.axis('off')
rows = [[name, f"{h['n_params']:,}", f"{h['val_bpc'][-1]:.3f}",
         f"{math.exp(h['val_ce'][-1]):.1f}",
         f"{h['val_acc'][-1]:.1%}", f"{h['avg_step_ms']:.0f}"]
        for name, h in all_hist.items()]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','PPL','Acc','ms/step'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.2, 1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('wikitext2_results.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n' + '='*70)
print('WIKITEXT-2 RESULTS')
print('='*70)
for name, h in all_hist.items():
    print(f"  {name:<20} BPC:{h['val_bpc'][-1]:.3f}  Acc:{h['val_acc'][-1]:.1%}  "
          f"Params:{h['n_params']:,}  {h['avg_step_ms']:.0f}ms/step")

d_bpc = all_hist['D: No Sparsity']['val_bpc'][-1]
e_bpc = all_hist['E: SSM+MLP']['val_bpc'][-1]
b_bpc = all_hist['B: No Wilson']['val_bpc'][-1]
g_small = all_hist['GPT-Nano 128d']['val_bpc'][-1]
g_match = all_hist['GPT-Nano 224d']['val_bpc'][-1]
tol = 0.05

print(f"\\n--- Verdict ---")
print(f"D vs E: {d_bpc - e_bpc:+.3f} BPC ({'geometry helps' if d_bpc < e_bpc - tol else 'comparable' if abs(d_bpc-e_bpc) <= tol else 'MLP wins'})")
print(f"D vs B: {d_bpc - b_bpc:+.3f} BPC ({'Wilson helps' if d_bpc < b_bpc - tol else 'Wilson marginal'})")
print(f"D vs GPT-128d: {d_bpc - g_small:+.3f} BPC (GPT at 15M)")
print(f"D vs GPT-224d: {d_bpc - g_match:+.3f} BPC (GPT param-matched ~28M)")
print(f"  {'BEATS param-matched GPT' if d_bpc < g_match - tol else 'Comparable to GPT' if abs(d_bpc - g_match) <= tol else 'GPT still wins'}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 11: Text generation
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
# CELL 12: Summary
# ═══════════════════════════════════════════════════════════════
md("""## Summary

This notebook tests V14 variants on WikiText-2 with GPT-2 BPE tokenization (vocab ~50K).

Key questions:
1. Does geometry still beat SSM+MLP with a real vocabulary? (D vs E)
2. Does the Wilson line matter with longer effective contexts? (D vs B)
3. How close can the geometric models get to attention? (D vs GPT-Nano)

The Shakespeare ablation showed the Hopfield memory bank is the key contributor.
WikiText-2 tests whether this holds with semantic tokens and real language.""")

# ═══════════════════════════════════════════════════════════════
# CELL 13: WikiText-2 Findings
# ═══════════════════════════════════════════════════════════════
md("""## Findings (WikiText-2, BPE vocab 50K, 5000 steps, 2026-03-31)

### Results

| Model | BPC | PPL | Acc | Params | ms/step |
|---|---|---|---|---|---|
| **GPT-Nano 128d** | **~8.79** | **~440** | **~19%** | 15.3M | 68 |
| D: No Sparsity | 9.268 | 616.5 | 17.1% | 28.3M | 190 |
| B: No Wilson | 9.424 | 686.7 | 16.2% | 27.5M | 127 |
| E: SSM+MLP | 10.838 | 1830.7 | 4.9% | 28.4M | 79 |
| GPT-Nano 224d | (still training) | — | — | 29.8M | — |

### Key Findings

**1. Geometry CRUSHES SSM+MLP on real data.**
D (BPC 9.27, PPL 617) vs E (BPC 10.84, PPL 1831) — a 1.57 BPC gap. E is essentially
not learning (4.9% accuracy, barely above random for 50K vocab). The SSM+MLP that
matched geometric models on Shakespeare completely fails on WikiText-2. The constellation
representation + Hopfield memory bank provides something the MLP cannot.

**2. GPT-Nano 128d beats all geometric models with HALF the parameters.**
GPT-128d (15.3M params, BPC ~8.79) outperforms D (28.3M params, BPC 9.27). Attention
is still the stronger mechanism for this task/scale. However, GPT has nearly 2× the
parameters in its processing blocks (2.4M vs 1.7M), and the geometric models waste
most params in the oversized embedding (2× 50K×136 for mag+phase).

**3. Wilson line provides marginal benefit on real data too.**
D (9.27) vs B (9.42) = 0.16 BPC gap. Small but consistent with Shakespeare. The
Wilson line's content-dependent phase rotation helps slightly but isn't transformative.

**4. The embedding bottleneck is real.**
With vocab 50K, each geometric model spends ~26.7M params on embedding+decoder
(50K×136×2 for mag/phase + 256→50K decoder). Only ~1.7M params are in the actual
geometric blocks. GPT-Nano spends ~12.9M on embedding+head, leaving 2.4M for
attention blocks. The geometric models are parameter-starved where it matters.

### Diagnosis: Why E Fails Completely

The SSM+MLP (Model E) drops from competitive on Shakespeare (BPC 2.77, 44% acc) to
catastrophic on WikiText-2 (BPC 10.84, 5% acc). The MLP receives (mag, phase, messages)
= 408-dimensional input and must learn to process 50K tokens' worth of spectral patterns
through a single hidden layer of 311 units. This is a severe bottleneck for a large
vocabulary. The Hopfield memory bank (in D and B) provides content-addressable retrieval
over 256 prototype patterns, which scales much better with vocab size.

### Implications for Hybrid Architecture

1. **The Hopfield memory bank is the winning mechanism** — confirmed on both Shakespeare
   and WikiText-2. It provides something neither MLP nor fixed SSM can match.

2. **The embedding needs to be more efficient.** Two full 50K×136 embedding tables is
   wasteful. Consider: shared embedding with phase offset, or factored embedding
   (50K×32 → Linear → 136), or use standard embedding + project to spectral.

3. **Processing blocks need more capacity.** Only 1.7M of 28.3M params are in the
   geometric blocks. The fiber+settler deserve more of the parameter budget.

4. **Wilson line is consistent but small.** ~0.15 BPC on both tasks. Worth keeping
   but not the primary mechanism.

### Next Ablation (Recommended)

Drop B (No Wilson) and E (SSM+MLP) — their stories are clear. Instead test:

| Model | Description | Tests |
|---|---|---|
| D | Wilson + Langevin (current best) | Baseline |
| D + MLP | Wilson + Langevin + small MLP | Does per-token nonlinearity help? |
| D factored | Factored embedding (50K×32→136) | Does efficient embedding help? |
| GPT-Nano 224d | Param-matched attention | Fair attention comparison |""")

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

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext2_v14.ipynb")
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
