"""Generate architecture_v13.ipynb — Geometric Spectral Transport.

V13: The mode fiber IS the geometry. Deposits are complex spectral coefficients,
reads are Parseval inner products, phase rotation is parallel transport.
No learned deposit/read/gate — the structure comes from the math."""
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
md("""# V13: Geometric Spectral Transport

## What Changed from V12.5

V12.5 plateaued at BPC ~2.84 / 42% accuracy with train ≈ val (underfitting).
The mode fibers — the only cross-token mechanism — used learned linear maps
over separated (mag, phase) reals. No complex structure, no geometric meaning.
The README's core constructions (Parseval metric, phase-relative transport,
spectral inner products) were absent from the actual computation.

V13 replaces the learned fiber machinery with the native complex geometry:

| | V12.5 | V13 |
|---|---|---|
| **Deposit** | `Linear(mag, phase) → R^sd` | `mag · exp(iφ)` (the spectral coeff itself) |
| **State** | Real EMA, sd dims | Complex EMA (Re/Im decouple with real decay) |
| **Read** | `dot(state, read_w) → R` | `Re(h · conj(c))` = Parseval inner product |
| **Fiber params** | ~22K/block (gate, deposit, read, mix) | 136/block (just decay) |
| **Geometric content** | None | Metric + connection + curvature handle |

The Parseval inner product `Re(h · conj(c))` simultaneously gives:
- **Metric**: magnitude overlap between accumulated past and current token
- **Connection**: phase rotation = parallel transport on S¹ per mode
- **Curvature**: how phase alignment varies across modes

## Architecture

```
ConstellationEmbedding: token → (mag, phase) across 136 modes

V13Block (×N, pre-norm residual):
  normed = ConstellationNorm(constellation)
  c = normed.mag · exp(i · normed.phase)           # complex spectral coeff
  h = complex_EMA(c, learned_decay)                 # causal accumulation
  messages = Re(h · conj(c))                        # Parseval inner product
  d_mag, d_phase = UpdateMLP(mag, phase, messages)  # learned nonlinear update
  return constellation + (d_mag, d_phase)            # clean residual

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
class V13Config:
    # Spectral structure
    n_modes: int = 136               # 8 subbundles × 17 frequencies
    n_subbundles: int = 8
    fiber_dim: int = 256             # spatial dim for decoder (8 × 32)

    # Model
    update_hidden: int = 384         # hidden dim of constellation update MLP
    n_blocks: int = 8
    vocab_size: int = 65
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 750
    lr_hold_steps: int = 3250
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

config = V13Config(vocab_size=vocab_size)
print(f"Modes: {config.n_modes} ({config.n_subbundles} subbundles × {config.spectral_half_dim} freqs)")
print(f"Blocks: {config.n_blocks}, Hidden: {config.update_hidden}")
print(f"Seq: {config.seq_len}, Batch: {config.batch_size}")

def get_batch(split_data, cfg):
    ix = torch.randint(0, len(split_data) - cfg.seq_len - 1, (cfg.batch_size,))
    return torch.stack([split_data[i:i+cfg.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Constellation representation
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


class ConstellationNorm(nn.Module):
    \"\"\"RMSNorm on [mag, phase]. Controls scale without breaking relative structure.\"\"\"
    def __init__(self, n_modes):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2 * n_modes))

    def forward(self, c):
        x = c.to_flat()
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        x = x / rms * self.scale
        M = c.mag.shape[-1]
        return Constellation(x[..., :M], x[..., M:])


# Verify
_m = torch.randn(2, 4, 136)
_p = torch.randn(2, 4, 136)
_c = Constellation(_m, _p)
print(f"Constellation: mag {_c.mag.shape}, phase {_c.phase.shape}")
print(f"Complex: {_c.to_complex().shape}")""")

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
# CELL 5: Geometric Mode Fiber
# ═══════════════════════════════════════════════════════════════
md("""## Geometric Mode Fiber

The core construction. No learned deposit, read, gate, or mixing weights.
The geometry does the work:

- **Deposit** = `mag · exp(iφ)` — the complex spectral coefficient itself
- **State** = complex EMA via parallel scan (Re/Im decouple with real decay)
- **Read** = `Re(h · conj(c))` — the Parseval inner product

`Re(h · conj(c)) = h_re · c_re + h_im · c_im` measures how aligned the
accumulated past spectrum is with the current token's spectral pattern.
By Parseval's theorem, summing across modes gives the spatial inner product.
The phase rotation at read IS parallel transport on S¹ per mode.""")

code("""def parallel_scan(alpha, x):
    \"\"\"Parallel associative scan: h[t] = alpha[t]*h[t-1] + x[t].
    alpha, x: (N, T). Returns h: (N, T).
    Only multiply+add — numerically stable in both directions.\"\"\"
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
    \"\"\"Geometric mode fiber: complex EMA + Parseval inner product read.
    Only learned parameter: per-mode decay rate (M scalars).\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.decay = nn.Parameter(torch.zeros(cfg.n_modes))

    def forward(self, constellation):
        B, T, M = constellation.mag.shape

        # Complex spectral coefficient: the deposit IS the representation
        c_re = constellation.mag * torch.cos(constellation.phase)
        c_im = constellation.mag * torch.sin(constellation.phase)

        # Per-mode decay (real → Re/Im scans are independent)
        alpha = torch.sigmoid(self.decay).clamp(0.01, 0.99)
        alpha_flat = alpha.unsqueeze(0).expand(B, -1).reshape(B * M, 1).expand(-1, T)

        # Single scan over stacked Re/Im: (2·B·M, T)
        re_flat = c_re.permute(0, 2, 1).reshape(B * M, T)
        im_flat = c_im.permute(0, 2, 1).reshape(B * M, T)
        h_flat = parallel_scan(
            torch.cat([alpha_flat, alpha_flat], dim=0),
            torch.cat([re_flat, im_flat], dim=0))

        # Causal shift: position t reads state BEFORE its own deposit
        h_flat = F.pad(h_flat[:, :-1], (1, 0))

        # Split Re/Im, reshape: (B, M, T) → (B, T, M)
        h_re = h_flat[:B*M].reshape(B, M, T).permute(0, 2, 1)
        h_im = h_flat[B*M:].reshape(B, M, T).permute(0, 2, 1)

        # Parseval inner product: Re(h · conj(c)) = h_re·c_re + h_im·c_im
        messages = h_re * c_re + h_im * c_im

        return messages


# Verify shapes and that scan produces causal output
_cfg = V13Config()
_fib = ModeFiber(_cfg)
_c = Constellation(torch.randn(2, 8, 136), torch.randn(2, 8, 136))
_msg = _fib(_c)
print(f"ModeFiber: input {_c.mag.shape} → messages {_msg.shape}")
print(f"  Fiber params: {sum(p.numel() for p in _fib.parameters()):,}")
# Position 0 should have zero message (no past to read from)
print(f"  msg[0] all zero: {(_msg[:, 0, :].abs().max() < 1e-6).item()}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Constellation Update
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationUpdate(nn.Module):
    \"\"\"Learned update: (mag, phase, geometric_messages) → delta constellation.

    The messages now carry actual geometric content (Parseval inner products),
    so the MLP has meaningful inputs to work with. Pre-norm residual pattern.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        hd = cfg.update_hidden

        # Input: (mag, phase, messages) = 3M. Output: delta (mag, phase) = 2M
        self.net = nn.Sequential(
            nn.Linear(3 * M, hd),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hd, 2 * M),
        )

        # Gate starts small → blocks begin as near-identity
        self.gate = nn.Parameter(torch.tensor(-2.0))

        # Zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normed_constellation, messages):
        M = normed_constellation.mag.shape[-1]
        combined = torch.cat([
            normed_constellation.mag, normed_constellation.phase, messages], dim=-1)
        delta = self.net(combined)
        g = torch.sigmoid(self.gate)
        return g * delta[..., :M], g * delta[..., M:]""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: V13Block
# ═══════════════════════════════════════════════════════════════
code("""class V13Block(nn.Module):
    \"\"\"Pre-norm residual block: x = x + f(Norm(x)).

    1. Normalize constellation
    2. Geometric fiber: complex EMA → Parseval inner product messages
    3. Update MLP: (mag, phase, messages) → delta
    4. Residual add to original constellation\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.norm = ConstellationNorm(cfg.n_modes)
        self.fiber = ModeFiber(cfg)
        self.update = ConstellationUpdate(cfg)

    def forward(self, constellation):
        normed = self.norm(constellation)
        messages = self.fiber(normed)
        d_mag, d_phase = self.update(normed, messages)
        return Constellation(
            constellation.mag + d_mag,
            constellation.phase + d_phase)""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Decoder + Model
# ═══════════════════════════════════════════════════════════════
code("""class ConstellationDecoder(nn.Module):
    \"\"\"Constellation → logits via irfft.\"\"\"

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


class V13Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V13Block(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

        # Deep supervision at even blocks
        weights = torch.zeros(cfg.n_blocks)
        for i in range(cfg.n_blocks):
            if (i + 1) % 2 == 0:
                weights[i] = (i + 1) / cfg.n_blocks
        weights[-1] = 1.0
        self.register_buffer('block_loss_weights', weights)

    def forward(self, token_ids):
        constellation = self.embedding(token_ids)

        intermediate_logits = []
        for i, block in enumerate(self.blocks):
            constellation = block(constellation)
            if self.block_loss_weights[i] > 0:
                logits = self.decoder(constellation)[:, :-1, :]
                intermediate_logits.append((logits, self.block_loss_weights[i]))

        sp = (constellation.mag.abs() < 0.01).float().mean().item()
        mag_mean = constellation.mag.abs().mean()

        return intermediate_logits[-1][0], {
            'spectral_sparsity': sp,
            'mag_mean': mag_mean,
            'intermediate_logits': intermediate_logits,
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


model = V13Model(config).to(device)
gpt_model = GPTNano(vocab_size=vocab_size, block_size=config.seq_len).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"V13:         {count_params(model):>10,} params")
print(f"GPT-Nano:    {count_params(gpt_model):>10,} params")

n_e = sum(p.numel() for p in model.embedding.parameters())
n_nrm = sum(sum(p.numel() for p in b.norm.parameters()) for b in model.blocks)
n_fib = sum(sum(p.numel() for p in b.fiber.parameters()) for b in model.blocks)
n_upd = sum(sum(p.numel() for p in b.update.parameters()) for b in model.blocks)
n_dec = sum(p.numel() for p in model.decoder.parameters())
tot = count_params(model)
print(f"\\nV13 Breakdown:")
print(f"  Embedding:   {n_e:>8,} ({100*n_e/tot:.1f}%)")
print(f"  Norms:       {n_nrm:>8,} ({100*n_nrm/tot:.1f}%)")
print(f"  Fibers:      {n_fib:>8,} ({100*n_fib/tot:.1f}%) ← 136 decay params × {config.n_blocks} blocks")
print(f"  Updates:     {n_upd:>8,} ({100*n_upd/tot:.1f}%)")
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


def train_model(model, cfg, label='V13', is_gpt=False):
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
            loss = ce / tw

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
code("v13_hist = train_model(model, config, label='V13')")
md("## Baseline\nGPT-Nano (12-layer attention). Same schedule.")
code("gpt_hist = train_model(gpt_model, config, label='GPT', is_gpt=True)")

# ═══════════════════════════════════════════════════════════════
# CELL 13: Results + Plots
# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('V13 (Geometric Spectral Transport) vs GPT-Nano',
             fontsize=16, fontweight='bold')
hs = [(v13_hist,'V13','b','o'), (gpt_hist,'GPT','r','s')]

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
ax.plot(v13_hist['step'], [s*100 for s in v13_hist['sparsity']], 'b-o', markersize=3)
ax.set_title('Effective Sparsity (% modes < 0.01)'); ax.grid(True, alpha=0.3)

ax = axes[1,2]; ax.axis('off')
rows = [[l, f'{h["n_params"]:,}', f'{h["val_bpc"][-1]:.3f}', f'{h["val_acc"][-1]:.1%}', f'{h["avg_step_ms"]:.0f}']
        for h,l,_,_ in hs]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','Acc','ms/step'], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2,1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('v13_results.png', dpi=150, bbox_inches='tight'); plt.show()

print('\\n' + '='*70 + '\\nRESULTS\\n' + '='*70)
v, g = v13_hist['val_bpc'][-1], gpt_hist['val_bpc'][-1]
print(f"V13:      BPC {v:.3f} | Acc {v13_hist['val_acc'][-1]:.1%} | {v13_hist['avg_step_ms']:.0f}ms")
print(f"GPT-Nano: BPC {g:.3f} | Acc {gpt_hist['val_acc'][-1]:.1%} | {gpt_hist['avg_step_ms']:.0f}ms")""")

# ═══════════════════════════════════════════════════════════════
# CELL 14: Diagnostics
# ═══════════════════════════════════════════════════════════════
md("## Diagnostics: Geometric Fiber Dynamics")

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
        normed = block.norm(c)

        # Geometric fiber messages (Parseval inner products)
        msgs = block.fiber(normed)
        msg_pos = (msgs > 0).float().mean().item()

        c = block(c)

        mag_delta = (c.mag - mag_before).abs().mean().item()
        now_quiet = (c.mag.abs() < 0.01).float().mean().item()
        gate = torch.sigmoid(block.update.gate).item()

        print(f"  Block {i+1}: |Δmag|={mag_delta:.4f} silent={now_quiet:.1%} "
              f"gate={gate:.3f} | msg: mean={msgs.mean():.4f} "
              f"std={msgs.std():.4f} pos={msg_pos:.1%}")

    print("\\n--- Fiber decay rates ---")
    for i, block in enumerate(model.blocks):
        decay = torch.sigmoid(block.fiber.decay)
        print(f"  Block {i+1}: mean={decay.mean():.3f} "
              f"min={decay.min():.3f} max={decay.max():.3f}")

    print("\\n--- Magnitude distribution (final) ---")
    mag = c.mag
    per_mode = mag.abs().mean(dim=(0, 1))
    print(f"  Per-mode |mag|: mean={per_mode.mean():.4f} std={per_mode.std():.4f}")
    print(f"  Top 5 modes: {per_mode.topk(5).values.tolist()}")
    print(f"  Bottom 5 modes: {per_mode.topk(5, largest=False).values.tolist()}")

    # Spectral inner product between adjacent tokens (the geometric metric)
    c_complex = c.mag * torch.exp(1j * c.phase)
    ip = (c_complex[:, :-1, :].conj() * c_complex[:, 1:, :]).real.sum(dim=-1)
    print(f"\\n--- Adjacent spectral inner product (Parseval metric) ---")
    print(f"  mean={ip.mean():.4f} std={ip.std():.4f}")
    model.train()

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
    print(f"  V13: {gen(model, p, config)[len(p):len(p)+120]}")
    print(f"  GPT: {gen(gpt_model, p, config, is_gpt=True)[len(p):len(p)+120]}")""")

# ═══════════════════════════════════════════════════════════════
# CELL 16: Summary
# ═══════════════════════════════════════════════════════════════
md("""## Summary: V13 Geometric Spectral Transport

**The fiber IS the geometry.**

V12.5 tried to learn cross-token communication from scratch with linear maps.
V13 uses the complex spectral structure that was already in the representation:

- **Deposit**: `mag · exp(iφ)` — the complex coefficient, not a learned projection
- **State**: complex EMA — parallel scan on Re/Im independently
- **Read**: `Re(h · conj(c))` — the Parseval inner product = spectral metric
- **Update**: MLP on (mag, phase, geometric_messages) — the only learned part

The fiber has 136 params/block (just decay rates). Everything else comes from the
math: Parseval gives the metric, phase rotation gives the connection, variation
across modes gives a handle on curvature.

**What's gone from V12.5:**
- Learned deposit weights, read weights, gates, cross-mode mixing
- `mode_state_dim` config (now fixed at 2 = Re/Im from geometry)
- Content-dependent gating (the geometry handles selectivity via phase alignment)""")

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

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v13.ipynb")
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
