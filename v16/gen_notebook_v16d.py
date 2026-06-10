"""Generate architecture_v16d.ipynb — Matrix-Valued Fiber (Linear Attention).

V16d: Upgrade the Wilson fiber from rank-1 (one complex scalar per mode) to
matrix-valued (d×d matrix per subbundle). This is linear attention in spectral space.

Base: V16 architecture (irfft round-trip, Parseval filter, FFN). Proven stable at PPL 275.
Change: Replace scalar fiber with matrix fiber for richer state capacity.
Added: Gaussian clouds from V16b.

The matrix fiber stores key-value outer products instead of scalar deposits.
Each subbundle carries a 17×17 complex state matrix. Query against it to read.
O(n) and causal, but with d² state capacity instead of d."""
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
md("""# V16d: Matrix-Valued Fiber (Linear Attention in Spectral Space)

## Why

The scalar fiber h[t] = z·h[t-1] + c stores one complex number per mode = 136 values.
This is rank-1: each mode carries a single decayed summary. It can't store "token X at
position Y had value Z" — only "the accumulated blob at this mode has magnitude M and phase P."

The matrix fiber S[t] = Z·S[t-1] + k·v^T stores a d×d matrix per subbundle = 17×17 = 289
values per subbundle × 8 = 2,312 values. Each subbundle can store multiple key-value
associations. Query with q to retrieve: output = q @ S.

This is linear attention: the state S accumulates outer products (like attention's KV cache)
and queries read from it (like attention's Q@K^T@V). But it's O(n) because S is fixed-size,
not growing with sequence length.

## Architecture

```
V16dBlock:
  CloudNorm(constellation)
  ├── MatrixFiber: S[t] = Z_t · S[t-1] + k_t · v_t^T   (causal, matrix-valued)
  │   Query: output[t] = q_t @ S[t]                       (read from matrix state)
  ├── Parseval filter on fiber output                      (spectral gating)
  ├── irfft → spatial                                      (stable basis change)
  ├── Local conv                                           (causal local patterns)
  └── FFN                                                  (per-token nonlinear)
  rfft → back to spectral
```""")

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
class V16dConfig:
    # Spectral structure
    n_subbundles: int = 8
    subbundle_dim: int = 32
    n_modes: int = 136
    fiber_dim: int = 256

    # Matrix fiber
    fiber_key_dim: int = 17           # = spectral_half_dim, key/value dim per subbundle
    fiber_hidden: int = 256           # MLP for computing q, k, v, decay

    # Parseval filter
    filter_hidden: int = 256

    # Local conv
    local_kernel: int = 7

    # FFN
    ffn_mult: int = 4

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
    max_steps: int = 10000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1

cfg = V16dConfig(vocab_size=vocab_size)
print(f"V16d: vocab={cfg.vocab_size:,} seq={cfg.seq_len} batch={cfg.batch_size}")
print(f"Subbundles: {cfg.n_subbundles}, Modes: {cfg.n_modes}")
print(f"Matrix fiber: {cfg.n_subbundles} × ({cfg.fiber_key_dim}×{cfg.fiber_key_dim}) state")
print(f"Blocks: {cfg.n_blocks}, LR: {cfg.learning_rate}, Steps: {cfg.max_steps}")

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


class SpatialDecoder(nn.Module):
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


def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("Components loaded.")""")

# ═══════════════════════════════════════════════════════════════
md("""## Matrix-Valued Fiber

The scalar fiber stores one complex number per mode: h[t] = z·h[t-1] + c.
The matrix fiber stores a d×d matrix per subbundle: S[t] = Z·S[t-1] + k·v^T.

For each subbundle (17 modes):
- Project constellation to q, k, v (each 17-dim)
- Decay the state matrix: S = gamma · S
- Deposit: S += k · v^T (outer product adds a new key-value pair)
- Read: output = q @ S (query retrieves from the accumulated state)

This is linear attention: softmax(QK^T)V ≈ Q(K^TV) = Q @ S.
But causal and O(n) because S is updated incrementally.""")

code("""class MatrixFiber(nn.Module):
    \"\"\"Matrix-valued state space model per subbundle.
    State: S ∈ R^(d×d) per subbundle, where d = spectral_half_dim = 17.
    Deposit: S[t] = gamma_t * S[t-1] + k_t ⊗ v_t
    Read: output[t] = q_t @ S[t-1]  (causal: read before deposit)
    \"\"\"
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        d = cfg.fiber_key_dim        # 17
        nsub = cfg.n_subbundles      # 8

        # Project constellation → q, k, v per subbundle + decay
        # Input: 3M (mag, phase, log_var). Output: per sub: q(d) + k(d) + v(d) + decay(1)
        self.proj = nn.Linear(3 * M, nsub * (3 * d + 1))

        # Output projection: nsub * d → M (messages in spectral space)
        self.out_proj = nn.Linear(nsub * d, M)  # 512 → 136

        self.nsub = nsub
        self.d = d

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        d = self.d
        nsub = self.nsub

        # Project to q, k, v, decay per subbundle
        flat = constellation.to_flat()  # (B, T, 3M)
        proj = self.proj(flat)          # (B, T, nsub * (3d + 1))
        proj = proj.reshape(B, T, nsub, 3 * d + 1)

        q = proj[..., :d]                          # (B, T, nsub, d)
        k = proj[..., d:2*d]                        # (B, T, nsub, d)
        v = proj[..., 2*d:3*d]                      # (B, T, nsub, d)
        gamma_logit = proj[..., 3*d:]               # (B, T, nsub, 1)
        gamma = torch.sigmoid(gamma_logit).squeeze(-1)  # (B, T, nsub) in (0, 1)

        # Normalize q and k for stability (like in linear attention)
        q = F.elu(q) + 1  # positive queries
        k = F.elu(k) + 1  # positive keys

        # Two-part linear attention:
        # Part 1: Within-chunk causal attention (batched, no loop over positions)
        # Part 2: Cross-chunk state carry (loop over chunks only)
        C = min(64, T)
        n_chunks = (T + C - 1) // C

        # Pad to even chunks
        pad_len = n_chunks * C - T
        if pad_len > 0:
            q = F.pad(q, (0,0,0,0,0,pad_len))
            k = F.pad(k, (0,0,0,0,0,pad_len))
            v = F.pad(v, (0,0,0,0,0,pad_len))
            gamma = F.pad(gamma, (0,0,0,pad_len), value=0.5)

        # Reshape into chunks: (B, n_chunks, C, nsub, d)
        q_ch = q.reshape(B, n_chunks, C, nsub, d)
        k_ch = k.reshape(B, n_chunks, C, nsub, d)
        v_ch = v.reshape(B, n_chunks, C, nsub, d)
        g_ch = gamma.reshape(B, n_chunks, C, nsub)

        # Part 1: Within-chunk causal linear attention (batched over chunks)
        # For each chunk, compute causal Q @ cumsum(K^T V) with decay
        # Simplified: compute q_i @ (sum_{j<=i} k_j v_j^T) within each chunk
        # This is causal linear attention within the chunk

        # Compute KV outer products for all positions in all chunks
        # kv: (B, n_chunks, C, nsub, d, d)
        kv = torch.einsum('bncsf,bncse->bncsfe', k_ch, v_ch)

        # Causal cumsum of KV within each chunk (with decay)
        # Sequential over C positions within chunk, but C is small (64)
        chunk_S = torch.zeros(B, n_chunks, nsub, d, d, device=flat.device)
        intra_outputs = torch.zeros(B, n_chunks, C, nsub, d, device=flat.device)

        for t_local in range(C):
            # Read before deposit
            intra_outputs[:, :, t_local] = torch.einsum(
                'bnsd,bnsde->bnse', q_ch[:, :, t_local], chunk_S)
            # Deposit with decay
            g = g_ch[:, :, t_local].unsqueeze(-1).unsqueeze(-1)
            chunk_S = g * chunk_S + kv[:, :, t_local]

        # chunk_S now holds the final state of each chunk: (B, n_chunks, nsub, d, d)

        # Part 2: Cross-chunk state carry (sequential over n_chunks only)
        # Each chunk's output gets additional contribution from all previous chunks' states
        carry_S = torch.zeros(B, nsub, d, d, device=flat.device)
        cross_outputs = torch.zeros_like(intra_outputs)

        for ci in range(n_chunks):
            # Every position in this chunk queries the carried state
            # q_ch[:, ci]: (B, C, nsub, d), carry_S: (B, nsub, d, d)
            cross_out = torch.einsum('bcsd,bsde->bcse', q_ch[:, ci], carry_S)
            cross_outputs[:, ci] = cross_out

            # Decay the carry by this chunk's cumulative decay, then add chunk's final state
            # Approximate chunk decay as product of all gammas in chunk
            chunk_decay = g_ch[:, ci].prod(dim=1)  # (B, nsub) — product over C positions
            cd = chunk_decay.unsqueeze(-1).unsqueeze(-1)  # (B, nsub, 1, 1)
            carry_S = cd * carry_S + chunk_S[:, ci]

        # Total output = intra-chunk + cross-chunk
        output = (intra_outputs + cross_outputs).reshape(B, n_chunks * C, nsub, d)
        output = output[:, :T]  # trim padding
        # Flatten subbundles and project to spectral mode space
        output_flat = output.reshape(B, T, nsub * d)  # (B, T, nsub*d = 136)
        messages = self.out_proj(output_flat)           # (B, T, M)

        return messages


class ParsevalFilter(nn.Module):
    \"\"\"Spectral gating with Parseval constraint on the fiber output.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        shd = cfg.spectral_half_dim
        nsub = cfg.n_subbundles
        # Input: constellation flat (3M) + messages (M) = 4M
        self.filter_net = nn.Sequential(
            nn.Linear(4 * M, cfg.filter_hidden), nn.SiLU(),
            nn.Linear(cfg.filter_hidden, M),
        )
        nn.init.zeros_(self.filter_net[-1].weight)
        nn.init.zeros_(self.filter_net[-1].bias)
        self.cross_re = nn.Parameter(torch.eye(shd).unsqueeze(0).expand(nsub,-1,-1).clone())
        self.cross_im = nn.Parameter(torch.zeros(nsub, shd, shd))
        self.nsub = nsub
        self.shd = shd

    def forward(self, constellation, messages):
        B, T, M = constellation.mag.shape
        inp = torch.cat([constellation.to_flat(), messages], dim=-1)
        raw = self.filter_net(inp)
        # Parseval constraint: gate magnitudes in (0, 1)
        gate = torch.sigmoid(raw)  # (B, T, M)
        # Gate the messages
        gated_re = gate * messages
        # Cross-mode interaction within subbundles
        gated = gated_re.reshape(B, T, self.nsub, self.shd)
        y = torch.einsum('btsi,sio->btso', gated, self.cross_re)
        return y.reshape(B, T, M)


class LocalRefinement(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.fiber_dim
        k = cfg.local_kernel
        self.pad = k - 1
        self.conv = nn.Conv1d(D, D, kernel_size=k, groups=D, bias=True)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    def forward(self, spatial):
        h = spatial.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h).transpose(1, 2)
        return torch.sigmoid(self.gate) * h


class SpatialFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.fiber_dim
        self.norm = nn.LayerNorm(D)
        self.net = nn.Sequential(
            nn.Linear(D, D * cfg.ffn_mult),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D * cfg.ffn_mult, D),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, spatial):
        return spatial + self.net(self.norm(spatial))


class V16dBlock(nn.Module):
    \"\"\"Matrix fiber + Parseval filter + irfft + local conv + FFN.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm = CloudNorm(cfg.n_modes)
        self.fiber = MatrixFiber(cfg)
        self.pfilter = ParsevalFilter(cfg)
        self.fiber_gate = nn.Parameter(torch.tensor(-2.0))
        self.local = LocalRefinement(cfg)
        self.ffn = SpatialFFN(cfg)

    def forward(self, constellation):
        B, T, M = constellation.mag.shape
        normed = self.norm(constellation)
        shd = self.cfg.spectral_half_dim
        nsub = self.cfg.n_subbundles
        sdim = self.cfg.subbundle_dim

        # --- Matrix fiber → messages ---
        messages = self.fiber(normed)

        # --- Parseval filter on messages ---
        filtered = self.pfilter(normed, messages)

        # --- irfft to spatial (stable basis change) ---
        # Current constellation
        c_complex = normed.to_complex()
        c_subs = c_complex.reshape(B, T, nsub, shd)
        current_spatial = torch.fft.irfft(c_subs, n=sdim, dim=-1).reshape(B, T, self.cfg.fiber_dim)

        # Filtered messages as spectral → spatial
        # Messages are real-valued (from the q@S read). Convert to spatial via
        # treating as magnitude-only spectral coefficients in each subbundle
        f_subs = filtered.reshape(B, T, nsub, shd).to(torch.cfloat)
        filtered_spatial = torch.fft.irfft(f_subs, n=sdim, dim=-1).reshape(B, T, self.cfg.fiber_dim)

        # Combine
        spatial = current_spatial + torch.sigmoid(self.fiber_gate) * filtered_spatial

        # --- Local conv ---
        spatial = spatial + self.local(spatial)

        # --- FFN ---
        spatial = self.ffn(spatial)

        # --- Back to spectral ---
        spatial_subs = spatial.reshape(B, T, nsub, sdim)
        new_complex = torch.fft.rfft(spatial_subs, dim=-1).reshape(B, T, M)
        new_mag = new_complex.abs()
        new_phase = new_complex.angle()

        # Variance: shrink per block
        new_log_var = constellation.log_var - 0.1
        new_log_var = new_log_var.clamp(min=-6, max=2)

        return Constellation(new_mag, new_phase, new_log_var)


class V16dModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V16dBlock(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = SpatialDecoder(cfg)
    def forward(self, token_ids):
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)
        logits = self.decoder(c)[:, :-1, :]
        return logits, {}


_b = V16dBlock(cfg)
print(f"V16dBlock: {count_params(_b):,} params")
print(f"  MatrixFiber: {count_params(_b.fiber):,}")
print(f"  Filter:      {count_params(_b.pfilter):,}")
print(f"  Local:       {count_params(_b.local):,}")
print(f"  FFN:         {count_params(_b.ffn):,}")""")

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
models['V16d'] = V16dModel(cfg).to(device)
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

code("""colors = {'V16d': 'tab:blue', 'GPT-224d': 'black'}
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V16d (Matrix Fiber) vs GPT-224d on WikiText-103', fontsize=14, fontweight='bold')
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
    ax.plot(h['step'], [a*100 for a in h['val_acc']], '-o', color=colors[name], label=name, markersize=3)
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
plt.savefig('v16d_results.png', dpi=150, bbox_inches='tight')
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
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v16d.ipynb")
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
