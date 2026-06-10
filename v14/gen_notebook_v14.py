"""Generate architecture_v14.ipynb — Completing the Theory.

V14: Implements the three missing load-bearing mechanisms from the theory:
1. Wilson line: content-dependent complex recurrence (gauge transport)
2. Langevin settling: iterative Hopfield energy descent (content-addressable retrieval)
3. Proximal sparsity: soft-thresholding enforces spectral parsimony

Also fixes: separate mag/phase normalization, per-mode gates, no deep supervision."""
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
md("""# V14: Completing the Theory

## What Changed from V13

V13 had the right geometry (constellations, Parseval inner products) but wrong dynamics.
The three mechanisms that give the theory its expressive power were missing:

| Mechanism | Theory | V13 | V14 |
|---|---|---|---|
| **Gauge transport** | Content-dependent phase rotation (Wilson line) | Fixed scalar EMA | Content-dependent complex recurrence |
| **Langevin settling** | Iterative Hopfield energy descent | One-shot MLP | K-step Hopfield attractor descent |
| **Proximal sparsity** | Soft-thresholding for spectral parsimony | None | Magnitude thresholding after settling |

Additional fixes:
- **MagPhaseNorm**: RMSNorm on magnitudes only; phases left untouched (respects S¹ topology)
- **Per-mode gates**: each mode has its own residual gate (was single scalar)
- **No deep supervision**: loss only at final block (was blocks 2,4,6,8)
- **Lower LR**: 1e-3 (was 3e-3, caused instability in V13)

## Architecture

```
ConstellationEmbedding: token → (mag, phase) across 136 modes

V14Block (×8, pre-norm residual):
  normed = MagPhaseNorm(constellation)          # RMSNorm on mag only

  # === Gauge Transport (Wilson Line) ===
  decay_t, theta_t = WilsonProj(normed)         # content-dependent per-mode
  z_t = decay_t · exp(i·theta_t)                # complex recurrence coeff
  h[t] = z_t · h[t-1] + deposit(normed)         # complex parallel scan
  messages = Re(h · conj(c))                     # Parseval inner product

  # === Langevin Settling (Hopfield Energy Descent) ===
  ctx = msg_proj(messages)                       # context biases energy landscape
  for k in 1..K:
    query = normalize(x + ctx)                   # cosine routing
    attractor = softmax(β·query @ M^T) @ M       # Hopfield gradient
    x = x + η·(attractor - x) + noise            # Langevin step
  x_mag = proximal_threshold(x_mag)              # spectral sparsity

  delta = per_mode_gate · (x - x₀)              # gated residual
  return constellation + delta

ConstellationDecoder: mag · exp(iφ) → irfft → LayerNorm → MLP → logits
```""")

# ═══════════════════════════════════════════════════════════════
# CELL 1: Imports + Data
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

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.getcwd(), "tiny_shakespeare.txt")
if not os.path.exists(DATA_PATH):
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH) as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"Tiny Shakespeare: {len(data):,} chars, vocab {vocab_size}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 2: Config
# ═══════════════════════════════════════════════════════════════
code("""@dataclass
class V14Config:
    # Spectral structure
    n_modes: int = 136               # 8 subbundles × 17 frequencies
    n_subbundles: int = 8
    fiber_dim: int = 256             # spatial dim for decoder (8 × 32)

    # Wilson fiber
    wilson_hidden: int = 192         # bottleneck dim for content-dependent gates

    # Langevin settler
    n_memory_atoms: int = 256        # Hopfield memory bank size per block
    n_langevin_steps: int = 2        # settling steps per block
    beta_min: float = 0.5            # initial inverse temperature
    beta_max: float = 5.0            # final inverse temperature
    langevin_eta: float = 0.3        # Langevin step size
    sparsity_threshold: float = 0.05 # soft-thresholding lambda

    # Model
    n_blocks: int = 8
    vocab_size: int = 65
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3      # reduced from V13's 3e-3
    min_lr: float = 1e-4
    warmup_steps: int = 750
    lr_hold_steps: int = 1250        # shorter hold (V13: 3250)
    batch_size: int = 16
    seq_len: int = 512
    max_steps: int = 10000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def subbundle_dim(self):
        return self.fiber_dim // self.n_subbundles  # 32

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1          # 17

config = V14Config(vocab_size=vocab_size)
print(f"Modes: {config.n_modes} ({config.n_subbundles} subbundles × {config.spectral_half_dim} freqs)")
print(f"Blocks: {config.n_blocks}, Wilson hidden: {config.wilson_hidden}")
print(f"Memory atoms: {config.n_memory_atoms}, Langevin steps: {config.n_langevin_steps}")
print(f"Sparsity threshold: {config.sparsity_threshold}")
print(f"Seq: {config.seq_len}, Batch: {config.batch_size}, LR: {config.learning_rate}")

def get_batch(split_data, cfg):
    ix = torch.randint(0, len(split_data) - cfg.seq_len - 1, (cfg.batch_size,))
    return torch.stack([split_data[i:i+cfg.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Constellation representation + MagPhaseNorm
# ═══════════════════════════════════════════════════════════════
md("## Constellation Representation")

code("""class Constellation:
    \"\"\"A point on the spectral manifold: signed magnitudes and phases.
    Negative magnitude = π phase shift: -|m|·e^(iφ) = |m|·e^(i(φ+π)).\"\"\"

    def __init__(self, mag, phase):
        self.mag = mag      # (B, T, M) signed amplitude
        self.phase = phase   # (B, T, M) phase angle

    @property
    def shape(self):
        return self.mag.shape

    def to_complex(self):
        return self.mag * torch.exp(1j * self.phase)

    def to_flat(self):
        return torch.cat([self.mag, self.phase], dim=-1)  # (B, T, 2M)


class MagPhaseNorm(nn.Module):
    \"\"\"Separate normalization respecting geometry.
    RMSNorm on magnitudes only. Phases live on S¹ — left untouched.\"\"\"
    def __init__(self, n_modes):
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(n_modes))

    def forward(self, c):
        mag_rms = (c.mag ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        normed_mag = c.mag / mag_rms * self.mag_scale
        return Constellation(normed_mag, c.phase)


# Verify
_m = torch.randn(2, 4, 136)
_p = torch.randn(2, 4, 136)
_c = Constellation(_m, _p)
print(f"Constellation: mag {_c.mag.shape}, phase {_c.phase.shape}")
print(f"Complex: {_c.to_complex().shape}")
_norm = MagPhaseNorm(136)
_cn = _norm(_c)
print(f"Normed mag RMS: {(_cn.mag**2).mean(-1).sqrt().mean():.4f}")
print(f"Phase unchanged: {(_cn.phase == _c.phase).all()}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Embedding
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationEmbedding(nn.Module):
    \"\"\"Token → spectral constellation. Positional encoding via phase shift.\"\"\"

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
        return Constellation(mag, phase)""")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Complex Parallel Scan + Wilson Fiber
# ═══════════════════════════════════════════════════════════════
md("""## Wilson Fiber: Content-Dependent Gauge Transport

The core change from V13. The recurrence coefficient is now **complex and content-dependent**:

```
z_t = decay(c_t) · exp(i · θ(c_t))    — Wilson line coefficient
h[t] = z_t · h[t-1] + deposit(c_t)     — complex parallel scan
messages = Re(h · conj(c))              — Parseval inner product
```

The decay modulates how quickly past information fades (heat kernel).
The phase rotation θ accumulates contextual curvature (Wilson line / holonomy).
Both are **functions of the current token** — content-dependent transport.

This is what V3's GRU had (content-dependent gates) and V13 lost (fixed scalar decay).
The complex parallel scan preserves O(n log n) efficiency.""")

code("""def complex_parallel_scan(a_re, a_im, b_re, b_im):
    \"\"\"Complex parallel associative scan: h[t] = a[t]*h[t-1] + b[t].
    a = a_re + i·a_im (complex multiplier), b = b_re + i·b_im (complex addend).
    All inputs: (N, T). Returns (h_re, h_im): (N, T).\"\"\"
    N, T = a_re.shape
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T:
            break

        # Slices: "right" = [step:], "left" = [:-step]
        ar, ai = a_re[:, step:], a_im[:, step:]
        al, ail = a_re[:, :-step], a_im[:, :-step]
        bl, bil = b_re[:, :-step], b_im[:, :-step]

        # Complex multiply: a_right * b_left (for b update)
        ab_re = ar * bl - ai * bil
        ab_im = ar * bil + ai * bl

        # Complex multiply: a_right * a_left (for a update)
        aa_re = ar * al - ai * ail
        aa_im = ar * ail + ai * al

        # Update b: new_right = a_right * b_left + b_right
        b_re = torch.cat([b_re[:, :step], ab_re + b_re[:, step:]], dim=1)
        b_im = torch.cat([b_im[:, :step], ab_im + b_im[:, step:]], dim=1)

        # Update a: new_right = a_right * a_left
        a_re = torch.cat([a_re[:, :step], aa_re], dim=1)
        a_im = torch.cat([a_im[:, :step], aa_im], dim=1)

    return b_re, b_im


class WilsonFiber(nn.Module):
    \"\"\"Content-dependent complex EMA + Parseval inner product read.

    Learned parameters:
    - base_decay: per-mode baseline decay (like V13)
    - wilson_proj: constellation → (decay_delta, phase_rotation) per mode
      This is the gauge connection A: content-dependent transport coefficients.
    \"\"\"

    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes

        # Base decay rate (V13-compatible initialization)
        self.base_decay = nn.Parameter(torch.zeros(M))

        # Content-dependent Wilson line: (mag, phase) → (decay_delta, phase_rot)
        self.wilson_proj = nn.Sequential(
            nn.Linear(2 * M, cfg.wilson_hidden),
            nn.SiLU(),
            nn.Linear(cfg.wilson_hidden, 2 * M),
        )
        # Zero-init last layer → initially: decay_delta=0, phase_rot=0
        # This makes V14 start with V13's behavior (fixed real decay, no rotation)
        nn.init.zeros_(self.wilson_proj[-1].weight)
        nn.init.zeros_(self.wilson_proj[-1].bias)

    def forward(self, constellation):
        B, T, M = constellation.mag.shape

        # Content-dependent transport coefficients
        flat = constellation.to_flat()                   # (B, T, 2M)
        wilson = self.wilson_proj(flat)                  # (B, T, 2M)
        decay_delta = wilson[..., :M]                    # content adjustment to decay
        phase_rot = wilson[..., M:]                      # Wilson line phase rotation

        # Complex recurrence coefficient: z = decay · exp(i·θ)
        decay = torch.sigmoid(self.base_decay + decay_delta).clamp(0.01, 0.99)
        theta = torch.tanh(phase_rot) * math.pi          # bounded to [-π, π]
        z_re = decay * torch.cos(theta)                   # (B, T, M)
        z_im = decay * torch.sin(theta)                   # (B, T, M)

        # Complex deposit: the spectral coefficient itself
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)

        # Flatten for scan: (B*M, T)
        z_re_f = z_re.permute(0, 2, 1).reshape(B * M, T)
        z_im_f = z_im.permute(0, 2, 1).reshape(B * M, T)
        c_re_f = c_re.permute(0, 2, 1).reshape(B * M, T)
        c_im_f = c_im.permute(0, 2, 1).reshape(B * M, T)

        # Complex parallel scan
        h_re_f, h_im_f = complex_parallel_scan(z_re_f, z_im_f, c_re_f, c_im_f)

        # Causal shift: position t reads state BEFORE its own deposit
        h_re_f = F.pad(h_re_f[:, :-1], (1, 0))
        h_im_f = F.pad(h_im_f[:, :-1], (1, 0))

        # Reshape: (B*M, T) → (B, T, M)
        h_re = h_re_f.reshape(B, M, T).permute(0, 2, 1)
        h_im = h_im_f.reshape(B, M, T).permute(0, 2, 1)

        # Parseval inner product: Re(h · conj(c)) = h_re·c_re + h_im·c_im
        messages = h_re * c_re + h_im * c_im

        return messages, {'decay': decay, 'theta': theta}


# Verify
_cfg = V14Config()
_fib = WilsonFiber(_cfg)
_c = Constellation(torch.randn(2, 8, 136), torch.randn(2, 8, 136))
_msg, _info = _fib(_c)
print(f"WilsonFiber: input {_c.mag.shape} → messages {_msg.shape}")
print(f"  Fiber params: {sum(p.numel() for p in _fib.parameters()):,}")
print(f"  msg[0] all zero: {(_msg[:, 0, :].abs().max() < 1e-6).item()}")
print(f"  Decay mean: {_info['decay'].mean():.3f}, Phase rot mean abs: {_info['theta'].abs().mean():.4f}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Langevin Settler
# ═══════════════════════════════════════════════════════════════
md("""## Langevin Settler: Hopfield Energy Descent

Replaces V13's one-shot MLP with iterative dynamics on an energy landscape.

Each step:
1. **Hopfield routing**: cosine similarity between (context-biased) query and memory atoms
2. **Attractor pull**: softmax-weighted average of memory atoms = energy gradient
3. **Langevin step**: gradient descent + noise (annealed temperature)
4. **Proximal threshold**: soft-thresholding on magnitudes (final step only)

The memory bank provides **content-addressable retrieval** — the Hopfield energy
gradient is mathematically identical to softmax attention over memory atoms.
The messages from the Wilson fiber **bias the query**, modulating which attractors
are relevant given the current context.

With K=2 steps, this is a minimal but genuine implementation of the forward-reverse
loop: field reconstruction (fiber) → energy descent (Langevin) → sparsification (proximal).""")

code("""class LangevinSettler(nn.Module):
    \"\"\"Iterative Hopfield energy descent with proximal sparsity.

    Replaces V13's ConstellationUpdate MLP. The settling loop IS the update.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes

        # Hopfield memory bank: prototype constellation patterns
        self.memory = nn.Parameter(torch.randn(cfg.n_memory_atoms, 2 * M) * 0.02)

        # Project fiber messages into energy bias (contextual modulation)
        # Small init (not zero!) — zero blocks ALL gradients to the Wilson fiber
        self.msg_proj = nn.Linear(M, 2 * M, bias=False)
        nn.init.normal_(self.msg_proj.weight, std=0.01)

        # Per-mode residual gate (replaces V13's scalar gate)
        self.gate = nn.Parameter(torch.full((M,), -2.0))

        # Hyperparameters
        self.K = cfg.n_langevin_steps
        self.eta = cfg.langevin_eta
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max
        self.threshold = cfg.sparsity_threshold

    def forward(self, constellation, messages):
        M = constellation.mag.shape[-1]
        x = constellation.to_flat()       # (B, T, 2M)
        x0 = x                             # save for residual

        # Context bias from fiber messages
        ctx = self.msg_proj(messages)      # (B, T, 2M)

        # Pre-compute normalized memory for cosine routing
        m_norm = F.normalize(self.memory, dim=-1)  # (n_atoms, 2M)

        for k in range(self.K):
            beta = self.beta_min + (self.beta_max - self.beta_min) * k / max(1, self.K - 1)

            # Hopfield energy gradient: softmax attention over memory bank
            q = F.normalize(x + ctx, dim=-1)              # cosine query
            scores = beta * (q @ m_norm.T)                 # (B, T, n_atoms)
            weights = F.softmax(scores, dim=-1)            # routing weights
            attractor = weights @ self.memory              # (B, T, 2M) raw attractor

            # Langevin step: gradient + noise
            grad = attractor - x
            if self.training:
                noise = math.sqrt(2 * self.eta / beta) * torch.randn_like(x)
            else:
                noise = 0.0
            x = x + self.eta * grad + noise

            # Proximal sparsity on magnitudes (last step only — v12.2 lesson)
            if k == self.K - 1:
                mag = x[..., :M]
                mag = torch.sign(mag) * F.relu(mag.abs() - self.threshold)
                phase = x[..., M:]
                # Wrap phase to [-π, π] (respect S¹ topology)
                phase = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
                x = torch.cat([mag, phase], dim=-1)

        # Gated residual: per-mode gate
        # ctx enters BOTH the Langevin query AND the output delta directly.
        # This gives the fiber a gradient path that bypasses the Langevin chain
        # (which attenuates gradients through softmax * tiny-memory at init).
        delta = (x - x0) + ctx
        g = torch.sigmoid(self.gate)               # (M,) per-mode
        d_mag = g * delta[..., :M]
        d_phase = g * delta[..., M:]

        return d_mag, d_phase, {'weights': weights}


# Verify
_cfg = V14Config()
_settler = LangevinSettler(_cfg)
_c = Constellation(torch.randn(2, 8, 136), torch.randn(2, 8, 136))
_msg = torch.randn(2, 8, 136)
_dm, _dp, _si = _settler(_c, _msg)
print(f"LangevinSettler: msg {_msg.shape} → delta_mag {_dm.shape}, delta_phase {_dp.shape}")
print(f"  Settler params: {sum(p.numel() for p in _settler.parameters()):,}")
print(f"  Memory bank: {_cfg.n_memory_atoms} atoms × {2*_cfg.n_modes} dims")
print(f"  Gate mean: {torch.sigmoid(_settler.gate).mean():.3f}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: V14Block
# ═══════════════════════════════════════════════════════════════
code("""class V14Block(nn.Module):
    \"\"\"Pre-norm residual block implementing the full forward-reverse loop.

    1. MagPhaseNorm (separate mag/phase normalization)
    2. WilsonFiber (content-dependent complex EMA → Parseval messages)
    3. LangevinSettler (Hopfield energy descent → proximal sparsity)
    4. Gated residual add\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.norm = MagPhaseNorm(cfg.n_modes)
        self.fiber = WilsonFiber(cfg)
        self.settler = LangevinSettler(cfg)

    def forward(self, constellation):
        normed = self.norm(constellation)
        messages, fiber_info = self.fiber(normed)
        d_mag, d_phase, settler_info = self.settler(normed, messages)
        return Constellation(
            constellation.mag + d_mag,
            constellation.phase + d_phase), fiber_info, settler_info""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Decoder + Model
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationDecoder(nn.Module):
    \"\"\"Constellation → logits via irfft (same as V13).\"\"\"

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


class V14Model(nn.Module):
    \"\"\"No deep supervision. Loss only at final block.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V14Block(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

    def forward(self, token_ids):
        constellation = self.embedding(token_ids)

        for block in self.blocks:
            constellation, _, _ = block(constellation)

        logits = self.decoder(constellation)[:, :-1, :]
        sp = (constellation.mag.abs() < 0.01).float().mean().item()
        mag_mean = constellation.mag.abs().mean()

        return logits, {
            'spectral_sparsity': sp,
            'mag_mean': mag_mean,
        }""")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Baseline + Instantiate
# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
    def __init__(self, vocab_size=65, n_embd=128, n_head=4, n_layer=12,
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


model = V14Model(config).to(device)
gpt_model = GPTNano(vocab_size=vocab_size, block_size=config.seq_len).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"V14:         {count_params(model):>10,} params")
print(f"GPT-Nano:    {count_params(gpt_model):>10,} params")

n_e = sum(p.numel() for p in model.embedding.parameters())
n_nrm = sum(sum(p.numel() for p in b.norm.parameters()) for b in model.blocks)
n_fib = sum(sum(p.numel() for p in b.fiber.parameters()) for b in model.blocks)
n_set = sum(sum(p.numel() for p in b.settler.parameters()) for b in model.blocks)
n_dec = sum(p.numel() for p in model.decoder.parameters())
tot = count_params(model)
print(f"\\nV14 Breakdown:")
print(f"  Embedding:   {n_e:>8,} ({100*n_e/tot:.1f}%)")
print(f"  Norms:       {n_nrm:>8,} ({100*n_nrm/tot:.1f}%)")
print(f"  Fibers:      {n_fib:>8,} ({100*n_fib/tot:.1f}%) ← Wilson line + base decay")
print(f"  Settlers:    {n_set:>8,} ({100*n_set/tot:.1f}%) ← Hopfield memory + msg proj + gates")
print(f"  Decoder:     {n_dec:>8,} ({100*n_dec/tot:.1f}%)")""")

# ═══════════════════════════════════════════════════════════════
# CELL 10: Training
# ═══════════════════════════════════════════════════════════════
md("## Training")

code("""@torch.no_grad()
def estimate_loss(model, cfg, is_gpt=False):
    model.eval()
    results = {}
    for name, sd in [('train', train_data), ('val', val_data)]:
        tot_ce, tot_ok, tot_n, tot_sp = 0., 0, 0, 0.
        for _ in range(cfg.eval_steps):
            b = get_batch(sd, cfg)
            logits, info = model(b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
            if not is_gpt:
                tot_sp += info.get('spectral_sparsity', 0.0)
        n = cfg.eval_steps
        results[name] = {
            'ce': tot_ce/n, 'acc': tot_ok/tot_n,
            'sparsity': tot_sp/n if not is_gpt else 0.0}
    model.train()
    return results


def train_model(model, cfg, label='V14', is_gpt=False):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.05)
    mr = getattr(cfg, 'min_lr', 0) / cfg.learning_rate
    he = cfg.warmup_steps + getattr(cfg, 'lr_hold_steps', 0)

    def lr_fn(s):
        if s < cfg.warmup_steps: return s / max(1, cfg.warmup_steps)
        if s < he: return 1.0
        p = (s - he) / max(1, cfg.max_steps - he)
        return max(mr, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    hist = {'step':[], 'train_ce':[], 'val_ce':[], 'train_acc':[], 'val_acc':[],
            'train_bpc':[], 'val_bpc':[], 'sparsity':[], 'lr':[],
            'step_times':[], 'per_step_loss':[]}

    model.train()
    t0 = time.time()
    np_ = count_params(model)
    print(f"\\nTraining {label}: {np_:,} params")
    print(f"Steps: {cfg.max_steps}, Batch: {cfg.batch_size}, Seq: {cfg.seq_len}")
    print('=' * 70)

    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0:
            r = estimate_loss(model, cfg, is_gpt=is_gpt)
            tr, vl = r['train'], r['val']
            hist['step'].append(step)
            hist['train_ce'].append(tr['ce']); hist['val_ce'].append(vl['ce'])
            hist['train_acc'].append(tr['acc']); hist['val_acc'].append(vl['acc'])
            hist['train_bpc'].append(tr['ce']/math.log(2))
            hist['val_bpc'].append(vl['ce']/math.log(2))
            hist['sparsity'].append(vl['sparsity'])
            hist['lr'].append(sched.get_last_lr()[0])
            sp = f" | Sp: {vl['sparsity']:.1%}" if not is_gpt else ''
            print(f"[{label}] Step {step:5d} | Train CE: {tr['ce']:.3f} | "
                  f"Val CE: {vl['ce']:.3f} | Val BPC: {vl['ce']/math.log(2):.3f} | "
                  f"Acc: {vl['acc']:.1%}{sp}")

        if step >= cfg.max_steps: break
        st = time.time()
        batch = get_batch(train_data, cfg)
        opt.zero_grad()
        logits, info = model(batch)
        tgt = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist['step_times'].append(time.time() - st)
        hist['per_step_loss'].append(loss.item())
        if step % 500 == 0 and step > 0:
            print(f"    avg step time: {np.mean(hist['step_times'][-500:])*1000:.0f}ms")

    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000
    print(f"\\n{label} DONE in {el/60:.1f}min | BPC: {hist['val_bpc'][-1]:.3f} | "
          f"Acc: {hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms; hist['n_params'] = np_
    return hist""")

# ═══════════════════════════════════════════════════════════════
# CELL 11-12: Train
# ═══════════════════════════════════════════════════════════════
code("v14_hist = train_model(model, config, label='V14')")
md("## Baseline\nGPT-Nano (12-layer attention). Same schedule.")
code("gpt_hist = train_model(gpt_model, config, label='GPT', is_gpt=True)")

# ═══════════════════════════════════════════════════════════════
# CELL 13: Results + Plots
# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('V14 (Wilson + Langevin + Proximal) vs GPT-Nano',
             fontsize=16, fontweight='bold')
hs = [(v14_hist,'V14','b','o'), (gpt_hist,'GPT','r','s')]

for ax, key, title in [(axes[0,0],'val_ce','Val CE'), (axes[0,1],'val_bpc','Val BPC'),
                        (axes[0,2],'val_acc','Val Accuracy')]:
    for h,l,c,m in hs:
        y = [v*100 for v in h[key]] if 'acc' in key else h[key]
        ax.plot(h['step'], y, f'{c}-{m}', label=l, markersize=3)
    ax.set_xlabel('Step'); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]
w = 100
for h,l,c,m in hs:
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ax.plot(range(len(sm)), sm, f'{c}-', label=l, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.plot(v14_hist['step'], [s*100 for s in v14_hist['sparsity']], 'b-o', markersize=3)
ax.set_title('Effective Sparsity (% modes < 0.01)'); ax.grid(True, alpha=0.3)
ax.set_xlabel('Step'); ax.set_ylabel('%')

ax = axes[1,2]; ax.axis('off')
rows = [[l, f'{h["n_params"]:,}', f'{h["val_bpc"][-1]:.3f}', f'{h["val_acc"][-1]:.1%}', f'{h["avg_step_ms"]:.0f}']
        for h,l,_,_ in hs]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','Acc','ms/step'], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2,1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('v14_results.png', dpi=150, bbox_inches='tight'); plt.show()

print('\\n' + '='*70 + '\\nRESULTS\\n' + '='*70)
v, g = v14_hist['val_bpc'][-1], gpt_hist['val_bpc'][-1]
print(f"V14:      BPC {v:.3f} | Acc {v14_hist['val_acc'][-1]:.1%} | {v14_hist['avg_step_ms']:.0f}ms")
print(f"GPT-Nano: BPC {g:.3f} | Acc {gpt_hist['val_acc'][-1]:.1%} | {gpt_hist['avg_step_ms']:.0f}ms")""")

# ═══════════════════════════════════════════════════════════════
# CELL 14: Diagnostics
# ═══════════════════════════════════════════════════════════════
md("""## Diagnostics: Geometric Mechanism Analysis

The key question: **do the geometric mechanisms contribute measurably?**

We check:
1. **Wilson line**: Are phase rotations non-trivial? (θ ≠ 0 means content-dependent transport)
2. **Langevin settling**: How much does the state move during settling? Is memory utilization diverse?
3. **Proximal sparsity**: Are modes actually being pruned? Is the magnitude distribution bimodal?
4. **Per-mode gates**: Which modes have opened vs. still closed?""")

code("""@torch.no_grad()
def diagnostics(model, cfg):
    model.eval()
    batch = get_batch(val_data, cfg)
    c = model.embedding(batch)

    quiet = (c.mag.abs() < 0.01).float().mean().item()
    print(f"Embedding: {cfg.n_modes} modes, {quiet:.1%} effectively silent")
    print(f"  |mag|: mean={c.mag.abs().mean():.4f} max={c.mag.abs().max():.4f}")

    print("\\n--- Per-block dynamics ---")
    for i, block in enumerate(model.blocks):
        mag_before = c.mag.clone()

        # Run block
        normed = block.norm(c)
        messages, fiber_info = block.fiber(normed)
        d_mag, d_phase, settler_info = block.settler(normed, messages)
        c = Constellation(c.mag + d_mag, c.phase + d_phase)

        # Wilson line stats
        decay = fiber_info['decay']
        theta = fiber_info['theta']
        theta_abs = theta.abs().mean().item()

        # Langevin stats: memory utilization (softmax entropy)
        weights = settler_info['weights']  # (B, T, n_atoms)
        entropy = -(weights * (weights + 1e-10).log()).sum(-1).mean().item()
        max_entropy = math.log(cfg.n_memory_atoms)

        # Sparsity and magnitude change
        mag_delta = (c.mag - mag_before).abs().mean().item()
        now_quiet = (c.mag.abs() < 0.01).float().mean().item()
        gate = torch.sigmoid(block.settler.gate).mean().item()

        print(f"  Block {i+1}: |Δmag|={mag_delta:.4f} silent={now_quiet:.1%} "
              f"gate={gate:.3f}")
        print(f"    Wilson: decay={decay.mean():.3f} [{decay.min():.3f},{decay.max():.3f}] "
              f"|θ|={theta_abs:.4f}")
        print(f"    Langevin: entropy={entropy:.2f}/{max_entropy:.2f} "
              f"({entropy/max_entropy:.0%} of max)")
        print(f"    Messages: mean={messages.mean():.4f} std={messages.std():.4f}")

    print("\\n--- Per-mode gate values (sigmoid) ---")
    all_gates = []
    for i, block in enumerate(model.blocks):
        g = torch.sigmoid(block.settler.gate)
        all_gates.append(g)
        if i in [0, 3, 7]:  # show first, middle, last
            print(f"  Block {i+1}: mean={g.mean():.3f} min={g.min():.3f} max={g.max():.3f}")

    print("\\n--- Magnitude distribution (final) ---")
    mag = c.mag
    per_mode = mag.abs().mean(dim=(0, 1))
    print(f"  Per-mode |mag|: mean={per_mode.mean():.4f} std={per_mode.std():.4f}")
    print(f"  Top 5 modes: {[f'{v:.4f}' for v in per_mode.topk(5).values.tolist()]}")
    print(f"  Bottom 5 modes: {[f'{v:.4f}' for v in per_mode.topk(5, largest=False).values.tolist()]}")

    # Check if magnitude distribution is bimodal (sign of working sparsity)
    below_thresh = (mag.abs() < cfg.sparsity_threshold).float().mean().item()
    above_2x = (mag.abs() > 2 * cfg.sparsity_threshold).float().mean().item()
    print(f"  Below threshold ({cfg.sparsity_threshold}): {below_thresh:.1%}")
    print(f"  Above 2× threshold: {above_2x:.1%}")
    bimodal = below_thresh > 0.1 and above_2x > 0.3
    print(f"  Bimodal (sparsity working): {'YES' if bimodal else 'NO'}")

    print("\\n--- Memory bank diversity ---")
    for i in [0, 7]:
        mem = model.blocks[i].settler.memory
        mem_n = F.normalize(mem, dim=-1)
        cos_sim = (mem_n @ mem_n.T)
        # Zero out diagonal
        mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        avg_sim = cos_sim[mask].mean().item()
        print(f"  Block {i+1}: avg pairwise cosine sim = {avg_sim:.4f} "
              f"(0 = diverse, 1 = collapsed)")

    # Spectral inner product between adjacent tokens
    c_complex = c.mag * torch.exp(1j * c.phase)
    ip = (c_complex[:, :-1, :].conj() * c_complex[:, 1:, :]).real.sum(dim=-1)
    print(f"\\n--- Adjacent spectral inner product (Parseval metric) ---")
    print(f"  mean={ip.mean():.4f} std={ip.std():.4f}")

    # Gradient norms (run a quick backward to check)
    model.train()
    batch2 = get_batch(val_data, cfg)
    logits, info = model(batch2)
    tgt = batch2[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
    loss.backward()

    fiber_grad = 0.
    settler_grad = 0.
    for block in model.blocks:
        for p in block.fiber.parameters():
            if p.grad is not None:
                fiber_grad += p.grad.norm().item()
        for p in block.settler.parameters():
            if p.grad is not None:
                settler_grad += p.grad.norm().item()

    print(f"\\n--- Gradient norms ---")
    print(f"  Fiber total: {fiber_grad:.4f}")
    print(f"  Settler total: {settler_grad:.4f}")
    ratio = fiber_grad / max(settler_grad, 1e-8)
    print(f"  Fiber/Settler ratio: {ratio:.4f} (want > 0.1)")

    model.zero_grad()
    model.eval()

diagnostics(model, config)""")

# ═══════════════════════════════════════════════════════════════
# CELL 15: Text generation
# ═══════════════════════════════════════════════════════════════
code("""@torch.no_grad()
def gen(model, prompt, cfg, n=200, temp=0.8, is_gpt=False):
    model.eval()
    ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n):
        ctx = ids[:, -cfg.seq_len:]
        logits, _ = model(ctx)
        p = F.softmax(logits[:, -1, :] / temp, dim=-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return ''.join(itos[i.item()] for i in ids[0])

for p in ['ROMEO:\\n', 'To be or not to ', 'The king ']:
    print(f"\\nPrompt: {repr(p)}")
    print(f"  V14: {gen(model, p, config)[len(p):len(p)+120]}")
    print(f"  GPT: {gen(gpt_model, p, config, is_gpt=True)[len(p):len(p)+120]}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 16: Summary
# ═══════════════════════════════════════════════════════════════
md("""## Summary: V14 — Completing the Theory

V14 implements the three load-bearing mechanisms the theory prescribes:

### 1. Wilson Line (Gauge Transport)
Content-dependent complex recurrence: `z_t = decay(c_t) · exp(i·θ(c_t))`.
The phase rotation θ accumulates contextual curvature — it IS the gauge connection.
Initialized to V13's behavior (real decay, no rotation) and learns to use phase.

### 2. Langevin Settling (Hopfield Energy Descent)
K=2 iterative steps on a Hopfield energy landscape. Each step:
softmax attention over a learned memory bank → gradient step → noise.
The fiber messages bias the query, modulating which attractors are contextually relevant.
This replaces V13's one-shot MLP with content-addressable iterative refinement.

### 3. Proximal Sparsity (Soft-Thresholding)
Magnitudes below threshold are zeroed after the final Langevin step.
Enforces the "few dots on Fourier space" constraint that makes constellations sparse.

### What's gone from V13
- The ConstellationUpdate MLP (95% of V13's params) — replaced by Langevin settling
- Deep supervision at intermediate blocks — loss only at final block
- Joint mag/phase RMSNorm — separate MagPhaseNorm respects S¹ topology
- Single scalar gate — per-mode gates allow differential update rates
- Fixed scalar EMA — content-dependent complex recurrence

### The bar to clear
V12.2 ablation showed SSM+MLP alone matched the full model.
If V14's geometric mechanisms don't measurably outperform SSM+MLP,
the geometry is decorative regardless of how elegant the math is.

Check the diagnostics cell:
- Wilson line θ should be non-trivial (phase rotations ≠ 0)
- Langevin entropy should be < max (using specific memories, not uniform)
- Magnitude distribution should be bimodal (sparsity is working)
- Fiber/settler gradient ratio should be > 0.1 (both contributing)""")

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

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v14.ipynb")
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
