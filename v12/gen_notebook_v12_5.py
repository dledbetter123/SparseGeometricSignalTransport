"""Generate architecture_v12_5.ipynb — Sparse Constellation Architecture.

The simplest possible system where sparse mode overlaps do the work.

Tokens are a few dots in Fourier space. Shared dots = connections.
The network learns: which dots define tokens, what happens at overlaps,
when to move dots. Everything else falls out of the sparse structure."""
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
md("""# V12.5: Sparse Constellation Architecture

## Core Idea

A token is a pattern of magnitudes and phases across 136 Fourier modes
(8 subbundles × 17 frequencies). Sparsity is native — most magnitudes are near zero.
The loud modes define the token. Shared loud modes between tokens = connections.

## How It Works

**Magnitudes are the sparsity.** No binary mask, no top-k. A mode with near-zero
magnitude is effectively silent. The embedding learns which modes to activate
(loud) for each token. L1 pressure keeps constellations sparse.

**Connections are free.** If tokens A and B both have loud magnitude at mode m,
they're connected through m. No attention, no routing — shared resonance IS the connection.

**Each mode is a channel.** Mode m carries a running causal state (EMA) of what
tokens have deposited at frequency m. Loud tokens deposit more, quiet ones less.

**The update rule is learned.** Each token reads messages from all modes (weighted
by magnitude), and a learned function maps (constellation, messages) → updated
constellation. Magnitudes grow or shrink — topology restructures continuously.

## What's Removed (vs V12.0-V12.4)

- SSM, transport kernels, SpectralInteraction matrices, MLP as primary computation
- Binary masks / top-k sparsification (magnitudes handle it natively)
- Complex arithmetic (mag + phase stored as real vectors)

## What's Kept

- Sparse spectral representation (magnitude-native sparsity)
- rfft/irfft only at embed/decode boundaries
- No pairwise token attention
- Deep supervision

## Changes (v12.5.2) — Geometric Fiber

Previous runs plateaued at BPC ~2.84 with train/val identical (underfitting).
The fibers had no geometric structure — learned linear maps over separated
(mag, phase) reals, ignoring the complex spectral geometry the README demands.

**Root cause**: The architecture specifies metric (Parseval inner product),
connection (phase-relative transport), and curvature (phase variation across
modes). The fiber implemented none of these — just independent real EMAs.

**Fix**: Replace the learned deposit/read/gate/mixing machinery with the
actual complex geometry:

1. **Deposit = complex coefficient**: `mag · exp(iφ)` per mode. Not learned —
   the spectral coefficient IS the deposit. This is what the token is.
2. **State = complex EMA**: `h[t] = α·h[t-1] + c[t]` with real decay α.
   Re and Im decouple, so the existing parallel scan works unchanged.
3. **Read = Parseval inner product**: `Re(h · conj(c_reader))` — how aligned
   is the accumulated past spectrum with the current token's spectral pattern.
   This IS the metric (by Parseval). The phase rotation IS the connection.

Only learned fiber parameter: per-mode decay rate (136 scalars per block).
All other structure comes from the geometry, not from learned weights.""")

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
class V12_5Config:
    # Spectral structure
    n_modes: int = 136               # total modes (8 subbundles × 17 modes each)
    n_subbundles: int = 8
    fiber_dim: int = 256             # spatial dim for decoder (8 × 32)

    # Model
    update_hidden: int = 384         # hidden dim of constellation update MLP
    # mode_state_dim removed — fiber uses complex Re/Im natively (2 channels from geometry)
    n_blocks: int = 8
    vocab_size: int = 65
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 750
    lr_hold_steps: int = 3250
    sparsity_lambda: float = 0.0    # L1 on magnitudes — off for now, sparsity emerges naturally
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

config = V12_5Config(vocab_size=vocab_size)
print(f"Modes: {config.n_modes} total ({config.n_subbundles} subbundles × {config.spectral_half_dim} freqs)")
print(f"Sparsity pressure: L1 lambda = {config.sparsity_lambda}")
print(f"Blocks: {config.n_blocks}, Seq: {config.seq_len}, Batch: {config.batch_size}")

def get_batch(split_data, cfg):
    ix = torch.randint(0, len(split_data) - cfg.seq_len - 1, (cfg.batch_size,))
    return torch.stack([split_data[i:i+cfg.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Architecture header
# ═══════════════════════════════════════════════════════════════
md("""## Architecture

A constellation is: magnitudes (136) and phases (136). That's it.

Magnitudes encode sparsity natively — loud modes are active, quiet modes are silent.
No binary mask. Mode fibers carry complex spectral coefficients causally —
deposits are `mag·exp(iφ)`, reads are Parseval inner products `Re(h·conj(c))`.
The phase rotation at read IS parallel transport. The update MLP reshapes
constellations using these geometrically grounded messages.""")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Constellation representation
# ═══════════════════════════════════════════════════════════════
code("""class Constellation:
    \"\"\"A spectral pattern. Signed magnitudes and phases.
    Sparsity = modes near zero (from either side). Negative magnitude
    is just a π phase shift: -|m|·e^(iφ) = |m|·e^(i(φ+π)).
    Signed magnitudes enable clean additive residuals.\"\"\"

    def __init__(self, mag, phase):
        # mag:   (B, T, n_modes) — signed amplitude (can be negative)
        # phase: (B, T, n_modes) — phase at each mode
        self.mag = mag
        self.phase = phase

    @property
    def shape(self):
        return self.mag.shape

    def to_complex(self):
        \"\"\"Convert to complex for irfft at decoder.\"\"\"
        return self.mag * torch.exp(1j * self.phase)

    def to_flat(self):
        \"\"\"Flatten for MLP input: [mag, phase].\"\"\"
        return torch.cat([self.mag, self.phase], dim=-1)  # (B, T, 2*M)


class ConstellationNorm(nn.Module):
    \"\"\"RMSNorm on the full constellation (mag + phase as 2M-dim vector).
    Preserves relative structure, controls absolute scale.\"\"\"
    def __init__(self, n_modes):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2 * n_modes))

    def forward(self, c):
        x = c.to_flat()  # (B, T, 2M)
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        x = x / rms * self.scale
        M = c.mag.shape[-1]
        return Constellation(x[..., :M], x[..., M:])


# Verify
_m = torch.randn(2, 4, 136)  # signed magnitudes
_p = torch.randn(2, 4, 136)
_c = Constellation(_m, _p)
print(f"Constellation: mag {_c.mag.shape} (signed), phase {_c.phase.shape}")
print(f"Complex: {_c.to_complex().shape}, dtype={_c.to_complex().dtype}")
_norm = ConstellationNorm(136)
_cn = _norm(_c)
print(f"After norm: mag rms={(_cn.mag**2).mean():.3f}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Embedding
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationEmbedding(nn.Module):
    \"\"\"Token → sparse constellation. Each token learns its spectral pattern.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Each token gets magnitudes and phases
        self.mag_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        self.phase_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        nn.init.uniform_(self.phase_emb.weight, -math.pi, math.pi)

        # Positional encoding via phase shift
        freqs = torch.zeros(cfg.n_modes)
        for k in range(cfg.n_subbundles):
            off = k * cfg.spectral_half_dim
            freqs[off:off+cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0))
        self.register_buffer('freqs', freqs)

    def forward(self, token_ids):
        B, T = token_ids.shape
        mag = self.mag_emb(token_ids)  # signed magnitudes — sparsity is |mag| near 0
        phase = self.phase_emb(token_ids)
        # Add positional phase shift
        pos = torch.arange(T, device=token_ids.device).float()
        phase = phase + (pos.unsqueeze(-1) * self.freqs).unsqueeze(0)
        return Constellation(mag, phase)""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Mode Fiber
# ═══════════════════════════════════════════════════════════════
code("""def parallel_scan(alpha, x):
    \"\"\"Parallel associative scan for h[t] = alpha[t]*h[t-1] + x[t].
    alpha: (N, T), x: (N, T). Returns h: (N, T).
    Only multiply+add — no division, numerically stable in both directions.\"\"\"
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


class ModeFiber(nn.Module):
    \"\"\"Geometric mode fiber: complex spectral accumulation with phase-relative read.

    Deposit = the token's complex spectral coefficient: mag · exp(i·phase).
    State  = complex EMA via parallel scan (Re/Im decouple with real decay).
    Read   = Parseval inner product: Re(h · conj(c_reader)).

    This gives, directly from the representation geometry:
    - Metric: sum of reads = spectral inner product = spatial inner product (Parseval)
    - Connection: phase rotation at read = parallel transport on S¹ per mode
    - Curvature: variation of phase alignment across modes and positions

    Only learned parameter: per-mode decay rate (M scalars).\"\"\"

    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes  # 136
        self.decay = nn.Parameter(torch.zeros(M))  # sigmoid → 0.5 at init

    def forward(self, constellation):
        \"\"\"Returns messages: (B, T, M) — Parseval inner product per mode.\"\"\"
        B, T, M = constellation.mag.shape

        # Complex spectral coefficient: deposit IS the representation
        c_re = constellation.mag * torch.cos(constellation.phase)  # (B, T, M)
        c_im = constellation.mag * torch.sin(constellation.phase)  # (B, T, M)

        # Per-mode decay (real-valued, so Re and Im scans are independent)
        alpha = torch.sigmoid(self.decay).clamp(0.01, 0.99)  # (M,)
        alpha_flat = alpha.unsqueeze(0).expand(B, -1).reshape(B * M, 1).expand(-1, T)

        # Stack Re and Im for a single scan call: (2*B*M, T)
        re_flat = c_re.permute(0, 2, 1).reshape(B * M, T)
        im_flat = c_im.permute(0, 2, 1).reshape(B * M, T)
        dep_flat = torch.cat([re_flat, im_flat], dim=0)
        alpha_flat2 = torch.cat([alpha_flat, alpha_flat], dim=0)

        h_flat = parallel_scan(alpha_flat2, dep_flat)  # (2*B*M, T)

        # Causal shift: position t reads state BEFORE its own deposit
        h_flat = F.pad(h_flat[:, :-1], (1, 0))

        # Split back to Re and Im: (B*M, T) each
        h_re, h_im = h_flat[:B*M], h_flat[B*M:]

        # Reshape: (B, M, T) → (B, T, M)
        h_re = h_re.reshape(B, M, T).permute(0, 2, 1)
        h_im = h_im.reshape(B, M, T).permute(0, 2, 1)

        # Read: Re(h · conj(c)) = h_re·c_re + h_im·c_im
        # This IS the Parseval inner product per mode
        messages = h_re * c_re + h_im * c_im  # (B, T, M)

        return messages""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Constellation Update
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationUpdate(nn.Module):
    \"\"\"Learned update rule: how mode messages reshape the constellation.

    Takes: NORMALIZED constellation (mag + phase) + mode messages
    Produces: delta (mag, phase) to add as residual to the UNNORMALIZED constellation.

    Pre-norm residual pattern: x = x + f(Norm(x)). Same recipe that
    makes every working deep architecture stable.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes           # 136
        hd = cfg.update_hidden    # 384

        # Input: normalized (mag, phase, messages) = 3*M = 408
        # Output: delta (mag, phase) = 2*M = 272
        self.update_net = nn.Sequential(
            nn.Linear(3 * M, hd),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hd, 2 * M),
        )

        # Learnable gate (starts small so initial blocks are near-identity)
        self.gate = nn.Parameter(torch.tensor(-2.0))

        # Initialize last linear layer to near-zero so residual starts as identity
        nn.init.zeros_(self.update_net[-1].weight)
        nn.init.zeros_(self.update_net[-1].bias)

    def forward(self, normed_constellation, messages):
        \"\"\"
        normed_constellation: pre-normalized Constellation
        messages: (B, T, n_modes) — scalar message per mode
        Returns: delta_mag, delta_phase (to be added as residual by the block)
        \"\"\"
        M = normed_constellation.mag.shape[-1]

        combined = torch.cat([
            normed_constellation.mag, normed_constellation.phase, messages], dim=-1)

        delta = self.update_net(combined)  # (B, T, 2*M)

        g = torch.sigmoid(self.gate)
        return g * delta[..., :M], g * delta[..., M:]""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: V12_5Block
# ═══════════════════════════════════════════════════════════════
code("""class V12_5Block(nn.Module):
    \"\"\"Pre-norm residual block on the constellation manifold.

    x = x + f(Norm(x))

    1. Normalize constellation (RMSNorm on [mag, phase])
    2. Mode fibers read normalized constellation → causal messages
    3. Update MLP on (normalized mag, phase, messages) → delta
    4. Add delta to ORIGINAL (unnormalized) constellation\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.norm = ConstellationNorm(cfg.n_modes)
        self.fiber = ModeFiber(cfg)
        self.update = ConstellationUpdate(cfg)

    def forward(self, constellation):
        # Pre-norm
        normed = self.norm(constellation)

        # Fiber operates on normalized constellation
        messages = self.fiber(normed)

        # Update produces deltas from normalized constellation + messages
        d_mag, d_phase = self.update(normed, messages)

        # Clean additive residual on the ORIGINAL constellation
        return Constellation(
            constellation.mag + d_mag,
            constellation.phase + d_phase)""")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Decoder + Model
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationDecoder(nn.Module):
    \"\"\"Constellation → logits. The only irfft.\"\"\"

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
        # Convert sparse constellation to complex, then irfft to spatial
        spectral = constellation.to_complex()  # (B, T, 136) complex
        shd = self.cfg.spectral_half_dim
        subs = spectral.reshape(*spectral.shape[:-1], self.cfg.n_subbundles, shd)
        spatial = torch.fft.irfft(subs, n=self.cfg.subbundle_dim, dim=-1)
        spatial = spatial.reshape(*spectral.shape[:-1], self.cfg.fiber_dim)
        return self.head(self.norm(spatial))


class V12_5Model(nn.Module):
    \"\"\"Sparse constellation language model.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V12_5Block(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

        # Deep supervision weights: blocks 2, 4, 6
        weights = torch.zeros(cfg.n_blocks)
        for i in range(cfg.n_blocks):
            if (i + 1) % 2 == 0:
                weights[i] = (i + 1) / cfg.n_blocks
        weights[-1] = 1.0
        self.register_buffer('block_loss_weights', weights)

    def forward(self, token_ids):
        cfg = self.cfg
        constellation = self.embedding(token_ids)

        intermediate_logits = []
        for i, block in enumerate(self.blocks):
            constellation = block(constellation)
            if self.block_loss_weights[i] > 0:
                logits = self.decoder(constellation)[:, :-1, :]
                intermediate_logits.append((logits, self.block_loss_weights[i]))

        # Effective sparsity: fraction of modes with |magnitude| < 0.01
        sp = (constellation.mag.abs() < 0.01).float().mean().item()
        # Mean |magnitude| (for diagnostics / optional L1)
        mag_mean = constellation.mag.abs().mean()

        return intermediate_logits[-1][0], {
            'spectral_sparsity': sp,
            'mag_mean': mag_mean,
            'intermediate_logits': intermediate_logits,
        }""")

# ═══════════════════════════════════════════════════════════════
# CELL 10: Baselines + instantiate
# ═══════════════════════════════════════════════════════════════
code("""# -- GPT-Nano ---------------------------------------------------------------

class GPTNano(nn.Module):
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


# -- Instantiate ------------------------------------------------------------
model = V12_5Model(config).to(device)
gpt_model = GPTNano(vocab_size=vocab_size, block_size=config.seq_len).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"V12.5:       {count_params(model):>10,} params")
print(f"GPT-Nano:    {count_params(gpt_model):>10,} params")

# Breakdown
n_e = sum(p.numel() for p in model.embedding.parameters())
n_fib = sum(sum(p.numel() for p in b.fiber.parameters()) for b in model.blocks)
n_upd = sum(sum(p.numel() for p in b.update.parameters()) for b in model.blocks)
n_dec = sum(p.numel() for p in model.decoder.parameters())
tot = count_params(model)
print(f"\\nV12.5 Breakdown:")
print(f"  Embedding:   {n_e:>8,} ({100*n_e/tot:.1f}%)")
print(f"  ModeFibers:  {n_fib:>8,} ({100*n_fib/tot:.1f}%)")
print(f"  Updates:     {n_upd:>8,} ({100*n_upd/tot:.1f}%)")
print(f"  Decoder:     {n_dec:>8,} ({100*n_dec/tot:.1f}%)")""")

# ═══════════════════════════════════════════════════════════════
# CELL 11: Training header
# ═══════════════════════════════════════════════════════════════
md("## Training")

# ═══════════════════════════════════════════════════════════════
# CELL 12: Training functions
# ═══════════════════════════════════════════════════════════════
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


def train_model(model, cfg, label='V12.5', is_gpt=False):
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

        if is_gpt:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
        else:
            ce, tw = 0., 0.
            for bl, w in info['intermediate_logits']:
                ce += w * F.cross_entropy(bl.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
                tw += w
            loss = ce / tw + cfg.sparsity_lambda * info['mag_mean']

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
# CELL 13-14: Train
# ═══════════════════════════════════════════════════════════════
code("v12_hist = train_model(model, config, label='V12.5')")
md("## Baseline\nGPT-Nano (12-layer attention). Same schedule.")
code("gpt_hist = train_model(gpt_model, config, label='GPT', is_gpt=True)")

# ═══════════════════════════════════════════════════════════════
# CELL 15: Results
# ═══════════════════════════════════════════════════════════════
md("## Results")

# ═══════════════════════════════════════════════════════════════
# CELL 16: Plots
# ═══════════════════════════════════════════════════════════════
code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('V12.5 (Sparse Constellation) vs GPT-Nano — seq_len=512',
             fontsize=16, fontweight='bold')
hs = [(v12_hist,'V12.5','b','o'), (gpt_hist,'GPT','r','s')]

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
ax.plot(v12_hist['step'], [s*100 for s in v12_hist['sparsity']], 'b-o', markersize=3)
ax.set_title('Effective Sparsity (% modes < 0.01)'); ax.grid(True, alpha=0.3)

ax = axes[1,2]; ax.axis('off')
rows = [[l, f'{h["n_params"]:,}', f'{h["val_bpc"][-1]:.3f}', f'{h["val_acc"][-1]:.1%}', f'{h["avg_step_ms"]:.0f}']
        for h,l,_,_ in hs]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','Acc','ms/step'], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2,1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('v12_5_results.png', dpi=150, bbox_inches='tight'); plt.show()

print('\\n' + '='*70 + '\\nRESULTS\\n' + '='*70)
v, g = v12_hist['val_bpc'][-1], gpt_hist['val_bpc'][-1]
print(f"V12.5:    BPC {v:.3f} | Acc {v12_hist['val_acc'][-1]:.1%} | {v12_hist['avg_step_ms']:.0f}ms")
print(f"GPT-Nano: BPC {g:.3f} | Acc {gpt_hist['val_acc'][-1]:.1%} | {gpt_hist['avg_step_ms']:.0f}ms")""")

# ═══════════════════════════════════════════════════════════════
# CELL 17: Diagnostics
# ═══════════════════════════════════════════════════════════════
md("## Diagnostics: Constellation Dynamics")

code("""@torch.no_grad()
def diagnostics(model, cfg):
    model.eval()
    batch = get_batch(val_data, cfg)
    c = model.embedding(batch)

    # Effective sparsity: how many modes are quiet?
    quiet = (c.mag.abs() < 0.01).float().mean().item()
    print(f"Embedding: {cfg.n_modes} modes, {quiet:.1%} effectively silent")
    print(f"  |mag|: mean={c.mag.abs().mean():.4f} max={c.mag.abs().max():.4f}")

    print("\\n--- Per-block magnitude dynamics ---")
    for i, block in enumerate(model.blocks):
        mag_before = c.mag.clone()
        c = block(c)

        # How much did magnitudes change?
        mag_delta = (c.mag - mag_before).abs().mean().item()
        # Modes that crossed the 0.01 threshold (activated or deactivated)
        was_quiet = (mag_before.abs() < 0.01)
        now_quiet = (c.mag.abs() < 0.01)
        activated = (was_quiet & ~now_quiet).float().mean().item()
        deactivated = (~was_quiet & now_quiet).float().mean().item()

        gate = torch.sigmoid(block.update.gate).item()
        quiet = now_quiet.float().mean().item()

        print(f"  Block {i+1}: |Δmag|={mag_delta:.4f} activated={activated:.1%} "
              f"deactivated={deactivated:.1%} silent={quiet:.1%} gate={gate:.3f}")

    print("\\n--- Mode fiber gating (learned base decay + content modulation) ---")
    for i, block in enumerate(model.blocks):
        base_gate = torch.sigmoid(block.fiber.gate_b)
        gate_w_norm = block.fiber.gate_w.norm()
        print(f"  Block {i+1}: base decay mean={base_gate.mean():.3f} "
              f"range=[{base_gate.min():.3f}, {base_gate.max():.3f}] "
              f"content_w_norm={gate_w_norm:.4f}")

    print("\\n--- Cross-mode mixing (off-diagonal = learned interactions) ---")
    for i, block in enumerate(model.blocks):
        w = block.fiber.mode_mix.weight
        off_diag = w - torch.diag(torch.diag(w))
        print(f"  Block {i+1}: off-diag norm={off_diag.norm():.4f} "
              f"diag mean={torch.diag(w).mean():.4f}")

    # Magnitude distribution across modes
    print("\\n--- Magnitude distribution (final) ---")
    mag = c.mag  # (B, T, M)
    per_mode = mag.abs().mean(dim=(0, 1))  # (M,) avg |magnitude| per mode
    print(f"  Per-mode |mag|: mean={per_mode.mean():.4f} std={per_mode.std():.4f}")
    print(f"  Top 5 modes: {per_mode.topk(5).values.tolist()}")
    print(f"  Bottom 5 modes: {per_mode.topk(5, largest=False).values.tolist()}")

    # Shared resonance between adjacent tokens (continuous version)
    # dot product of magnitude vectors = how much two tokens resonate together
    m1, m2 = mag[:, :-1, :], mag[:, 1:, :]
    resonance = (m1 * m2).sum(dim=-1)  # (B, T-1)
    print(f"  Adjacent resonance: mean={resonance.mean():.2f} std={resonance.std():.2f}")
    model.train()

diagnostics(model, config)""")

# ═══════════════════════════════════════════════════════════════
# CELL 18: Text gen
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
    print(f"  V12.5: {gen(model, p, config)[len(p):len(p)+120]}")
    print(f"  GPT:   {gen(gpt_model, p, config, is_gpt=True)[len(p):len(p)+120]}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 19: Summary
# ═══════════════════════════════════════════════════════════════
md("""## Summary: V12.5 Sparse Constellation

**The simplest version of the thesis.**

Tokens are magnitudes and phases across 136 Fourier modes.
Sparsity is native — loud modes are active, quiet modes are silent.
Shared loud modes between tokens = connections. No binary masks.

**Block flow:**
```
constellation (136 modes, magnitude = activation level)
  → ModeFiber: content-gated recurrence + cross-mode mixing
  → ConstellationUpdate: (mag, phase, messages) → evolved (mag, phase)
  → magnitudes grow/shrink → topology restructures continuously
```

**What's gone:**
- No SSM, no transport kernels, no attention, no binary masks, no top-k
- No complex arithmetic (mag + phase as real vectors)

**Key diagnostics:**
- `activated`/`deactivated`: modes crossing the silence threshold per block
- `|Δmag|`: how much the constellation reshapes
- `adjacent resonance`: magnitude dot product between neighboring tokens""")

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

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v12_5.ipynb")
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
