"""
V12.1: Streamlined Spectral Sparsity Architecture for Character-Level LM

Changes from V12:
- Langevin 5→2 steps
- Simplified memory: dot-product Hopfield routing (no MLP routers)
- SpatialMLP per block (nonlinear channel mixing)
- 4→6 blocks (more depth)
- context_dim 256→128 (param reallocation)
- spectral_sparsity 10 modes per subbundle

Spectral fixes (post-ablation):
- rfft/irfft: unique half-spectrum only, eliminates conjugate redundancy
  (10 selected = 10 independent DoF, not ~5 from paired full-spectrum)
- Spectral proximal only after final Langevin step (settler can explore freely)
- Signed diffusion in transport kernel (tanh, can amplify or dampen modes)

Non-negotiables preserved:
1. Sparse in spectral, dense only transiently in spatial
2. Field reconstruction IS the irfft
3. Langevin starts from the reconstructed (irfft) field
4. Context warps the spectral metric (D, A context-dependent)
5. Spectral proximal enforced (after settling completes)
6. Subbundles are independent spectral channels
7. No pairwise token attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class V12_1Config:
    # Fiber geometry
    fiber_dim: int = 256
    n_subbundles: int = 8
    spectral_sparsity: int = 10      # s_k: active spectral modes per subbundle (was 8)

    # Vocabulary
    vocab_size: int = 65
    max_seq_len: int = 128

    # Architecture
    n_blocks: int = 6                # was 4
    context_dim: int = 128           # was 256
    atoms_per_subbundle: int = 16    # was 32
    mlp_ratio: float = 1.5           # MLP hidden = fiber_dim * mlp_ratio = 384

    # Langevin settling
    langevin_steps: int = 2          # was 5
    langevin_lr: float = 0.03
    beta_init: float = 4.0           # was 1.0
    beta_final: float = 12.0

    # Training
    learning_rate: float = 7e-3      # restored from V12 (was 5e-3)
    min_lr: float = 1e-4             # cosine floor (was 0)
    warmup_steps: int = 750
    lr_hold_steps: int = 3250        # hold at peak LR until step 4000 (40% of training)
    dropout: float = 0.1
    batch_size: int = 32
    seq_len: int = 128
    max_steps: int = 10000           # was 5000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def subbundle_dim(self):
        return self.fiber_dim // self.n_subbundles

    @property
    def total_active_modes(self):
        return self.n_subbundles * self.spectral_sparsity

    @property
    def mlp_hidden(self):
        return int(self.fiber_dim * self.mlp_ratio)

    @property
    def spectral_half_dim(self):
        """Unique spectral modes per subbundle (DC through Nyquist).
        For subbundle_dim=32: 17 unique modes."""
        return self.subbundle_dim // 2 + 1

    @property
    def spectral_dim(self):
        """Total unique spectral dimension across all subbundles.
        For 8 subbundles × 17 modes = 136."""
        return self.n_subbundles * self.spectral_half_dim


# ── Shared utilities ─────────────────────────────────────────────────

def spectral_sparsify(x_complex, cfg):
    """Per-subbundle top-s_k sparsification in half-spectrum (rfft domain).
    Each selected mode is an independent spectral degree of freedom —
    no conjugate redundancy. Uses pairwise comparison (MPS-friendly)."""
    shape = x_complex.shape
    x_subs = x_complex.reshape(*shape[:-1], cfg.n_subbundles, cfg.spectral_half_dim)
    with torch.no_grad():
        mags = x_subs.abs()
        gt_count = (mags.unsqueeze(-1) < mags.unsqueeze(-2)).sum(dim=-1)
        mask = (gt_count < cfg.spectral_sparsity).float()
    return (x_subs * mask).reshape(shape)


def spectral_to_spatial(x_spectral, cfg):
    """irfft per subbundle: unique half-spectrum → dense spatial (real).
    No .real needed — irfft returns real by construction."""
    shape = x_spectral.shape[:-1]
    subs = x_spectral.reshape(*shape, cfg.n_subbundles, cfg.spectral_half_dim)
    spatial = torch.fft.irfft(subs, n=cfg.subbundle_dim, dim=-1)
    return spatial.reshape(*shape, cfg.fiber_dim)


def spatial_to_spectral(x_spatial, cfg):
    """rfft per subbundle: dense spatial → unique half-spectrum (complex).
    Only stores modes 0..N/2 — each mode is independent."""
    shape = x_spatial.shape[:-1]
    subs = x_spatial.reshape(*shape, cfg.n_subbundles, cfg.subbundle_dim)
    spectral = torch.fft.rfft(subs, dim=-1)
    return spectral.reshape(*shape, cfg.spectral_dim)


def spectral_proximal(x_spatial, cfg):
    """V12 proximal operator: FFT → top-s_k per subbundle → IFFT.
    Enforces spectral sparsity while operating on spatial state."""
    x_spec = spatial_to_spectral(x_spatial, cfg)
    x_sparse = spectral_sparsify(x_spec, cfg)
    return spectral_to_spatial(x_sparse, cfg)


def parallel_associative_scan(A, B):
    """O(log T) Hillis-Steele parallel scan: q_t = A_t * q_{t-1} + B_t."""
    _, T, _ = A.shape
    a, b = A, B
    for d in range(int(math.ceil(math.log2(T)))):
        step = 2 ** d
        if step >= T:
            break
        b = torch.cat([b[:, :step, :],
                        a[:, step:, :] * b[:, :-step, :] + b[:, step:, :]], dim=1)
        a = torch.cat([a[:, :step, :],
                        a[:, step:, :] * a[:, :-step, :]], dim=1)
    return b


# ── Spectral Token Embedding ────────────────────────────────────────

class SpectralTokenEmbedding(nn.Module):
    """Tokens as sparse spectral configurations (half-spectrum).
    Magnitude embedding + learned phase + positional phase → complex spectral → sparsify.
    Uses rfft frequencies: each mode is an independent spectral DoF."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mag_embedding = nn.Embedding(cfg.vocab_size, cfg.spectral_dim)
        self.phase_embedding = nn.Embedding(cfg.vocab_size, cfg.spectral_dim)
        nn.init.uniform_(self.phase_embedding.weight, -math.pi, math.pi)

        freqs = torch.zeros(cfg.spectral_dim)
        for k in range(cfg.n_subbundles):
            offset = k * cfg.spectral_half_dim
            freqs[offset:offset + cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0)
            )
        self.register_buffer("freqs", freqs)

    def forward(self, token_ids):
        B, T = token_ids.shape
        mag = self.mag_embedding(token_ids)
        phase_offset = self.phase_embedding(token_ids)
        positions = torch.arange(T, device=token_ids.device).float()
        pos_phase = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)
        total_phase = phase_offset + pos_phase.unsqueeze(0)
        spectral = mag * torch.exp(1j * total_phase)
        return spectral_sparsify(spectral, self.cfg)


# ── Context Accumulator (SSM) ───────────────────────────────────────

class ContextAccumulator(nn.Module):
    """Content-dependent SSM with parallel scan.
    q_t = A(x_t) * q_{t-1} + B(x_t) * psi(x_t)
    Modified: fiber_dim → context_dim (128, was 256)."""

    def __init__(self, cfg):
        super().__init__()
        self.A_proj = nn.Linear(cfg.fiber_dim, cfg.context_dim)
        self.B_proj = nn.Linear(cfg.fiber_dim, cfg.context_dim)
        self.psi_proj = nn.Linear(cfg.fiber_dim, cfg.context_dim)

    def forward(self, x_spatial):
        A = torch.sigmoid(self.A_proj(x_spatial))
        B = torch.sigmoid(self.B_proj(x_spatial))
        psi = self.psi_proj(x_spatial)
        return parallel_associative_scan(A, B * psi)


# ── Spectral Transport ──────────────────────────────────────────────

class SpectralTransport(nn.Module):
    """Context-dependent spectral transport kernel (half-spectrum).
    X_tilde(w) = X(w) * exp(-D(ctx)*w^2 - i*w*A(ctx))
    Signed diffusion: tanh allows both damping (D>0) and amplification (D<0).
    Projects context_dim → spectral_dim (136, not fiber_dim)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.D_proj = nn.Linear(cfg.context_dim, cfg.spectral_dim)
        self.A_proj = nn.Linear(cfg.context_dim, cfg.spectral_dim)

    def forward(self, spectral_x, q_t):
        cfg = self.cfg
        # Signed diffusion: range [-2, 2], allows amplification of modes
        diffusion = torch.tanh(self.D_proj(q_t)) * 1.0
        gauge = self.A_proj(q_t)

        freqs = torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0, device=spectral_x.device)
        freqs = freqs.repeat(cfg.n_subbundles)
        w2 = freqs ** 2
        w = freqs

        kernel = torch.exp(-diffusion * w2 - 1j * w * gauge)
        return spectral_x * kernel


# ── Simplified Memory Bank ───────────────────────────────────────────

class SimplifiedMemoryBank(nn.Module):
    """Memory atoms with direct dot-product Hopfield routing.
    No MLP routers — sim = x_k @ M_k.T, weights = softmax(beta * sim).
    Atoms stored as half-spectrum (real, imag), irfft'd to spatial for comparison.
    Ramsauer 2021: this IS modern Hopfield retrieval."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        shd = cfg.spectral_half_dim
        K = cfg.n_subbundles
        A = cfg.atoms_per_subbundle

        self.dict_real = nn.ParameterList([
            nn.Parameter(torch.randn(A, shd) * 0.02) for _ in range(K)
        ])
        self.dict_imag = nn.ParameterList([
            nn.Parameter(torch.randn(A, shd) * 0.02) for _ in range(K)
        ])

    def get_spatial_atoms(self):
        """Pre-compute normalized spatial atoms from spectral parameters.
        irfft: half-spectrum → real spatial (no .real needed)."""
        cfg = self.cfg
        all_real = torch.stack(list(self.dict_real))   # (K, A, shd)
        all_imag = torch.stack(list(self.dict_imag))   # (K, A, shd)
        all_spatial = F.normalize(
            torch.fft.irfft(torch.complex(all_real, all_imag),
                            n=cfg.subbundle_dim, dim=-1),
            dim=-1
        )  # (K, A, sd)
        return all_spatial


# ── Spectral Hopfield Settler ───────────────────────────────────────

class SpectralHopfieldSettler(nn.Module):
    """2-step Langevin settling with spectral proximal after final step.

    Non-negotiable #3: starts from irfft field (dense_field).
    Non-negotiable #5: spectral proximal enforced (after settling completes).

    Proximal moved to end: settler explores spatial domain freely during
    Langevin steps, then projects to sparse spectral manifold once.
    This prevents proximal from fighting the Hopfield gradient at every step."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_inh = nn.Parameter(torch.ones(cfg.fiber_dim) * 0.01)

    def forward(self, dense_field, spatial_atoms):
        """dense_field: (B, T, D) real dense spatial.
        spatial_atoms: (K, A, sd) normalized spatial atom patterns."""
        cfg = self.cfg
        B, T, D = dense_field.shape
        sd = cfg.subbundle_dim
        K = cfg.n_subbundles
        BT = B * T

        x = dense_field  # Non-negotiable #3: init from irfft field
        betas = torch.linspace(cfg.beta_init, cfg.beta_final, cfg.langevin_steps,
                               device=dense_field.device)

        # spatial_atoms: (K, A, sd) → expand for batched einsum
        M_all = spatial_atoms.unsqueeze(0).expand(BT, -1, -1, -1)  # (BT, K, A, sd)

        for step in range(cfg.langevin_steps):
            beta = betas[step].item()

            # Vectorized Hopfield gradient across all K subbundles
            x_subs = x.reshape(BT, K, sd)
            sim = torch.einsum('bks,bkas->bka', x_subs, M_all)  # (BT, K, A)
            w = F.softmax(beta * sim, dim=-1)
            grad_E = -torch.einsum('bka,bkas->bks', w, M_all)   # (BT, K, sd)

            # Lateral inhibition
            inhib = self.W_inh * x

            # Langevin step
            x = x - cfg.langevin_lr * (grad_E.reshape(B, T, D) + inhib)

            # Annealing noise (eval only)
            if not self.training:
                x = x + math.sqrt(2.0 * cfg.langevin_lr / beta) * torch.randn_like(x)

        # Spectral proximal after final step (not at every step)
        x = spectral_proximal(x, cfg)

        return x


# ── Spatial MLP ──────────────────────────────────────────────────────

class SpatialMLP(nn.Module):
    """Nonlinear channel mixing in spatial domain.
    The missing component from V12: GPT has attention + MLP, V12 had no MLP.
    Operates on transiently dense spatial representation (non-negotiable #1)."""

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.fiber_dim, cfg.mlp_hidden)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.fiber_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fc1(x))))


# ── V12.1 Block ──────────────────────────────────────────────────────

class V12_1Block(nn.Module):
    """One forward-reverse spectral diffusion cycle + spatial MLP.

    Flow:
    spectral → IFFT → SSM context → spectral transport →
    IFFT (field reconstruction) → 2-step Hopfield settling →
    SpatialMLP → gated residual → FFT + re-sparsify → spectral out"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_acc = ContextAccumulator(cfg)
        self.transport = SpectralTransport(cfg)
        self.memory = SimplifiedMemoryBank(cfg)
        self.settler = SpectralHopfieldSettler(cfg)
        self.mlp = SpatialMLP(cfg)
        self.norm1 = nn.LayerNorm(cfg.fiber_dim)
        self.norm2 = nn.LayerNorm(cfg.fiber_dim)
        self.res_gate = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, spectral_x):
        cfg = self.cfg

        # 1. Spatial projection for context accumulation
        x_spatial_in = spectral_to_spatial(spectral_x, cfg)

        # 2. Context accumulator (SSM on spatial)
        q_t = self.context_acc(x_spatial_in)

        # 3. Spectral transport (context-dependent kernel)
        transported = self.transport(spectral_x, q_t)

        # 4. Field reconstruction: IFFT → dense spatial (non-negotiable #2)
        dense_field = spectral_to_spatial(transported, cfg)
        dense_field = self.norm1(dense_field)

        # 5. Pre-compute spatial atoms, then 2-step Hopfield settling
        spatial_atoms = self.memory.get_spatial_atoms()
        settled = self.settler(dense_field, spatial_atoms)

        # 6. Spatial MLP (NEW: nonlinear channel mixing)
        mlp_out = self.mlp(self.norm2(settled))

        # 7. Gated residual in spatial domain
        gate = torch.sigmoid(self.res_gate)
        x_spatial_out = x_spatial_in + gate * self.dropout(mlp_out)

        # 8. Back to spectral and enforce sparsity (non-negotiable #1)
        spectral_out = spatial_to_spectral(x_spatial_out, cfg)
        spectral_out = spectral_sparsify(spectral_out, cfg)

        return spectral_out


# ── Full V12.1 Model ────────────────────────────────────────────────

class V12_1Model(nn.Module):
    """V12.1: Streamlined Spectral Sparsity CLM.

    Selective deep supervision at blocks 2, 4, 6 (every other)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = SpectralTokenEmbedding(cfg)
        self.blocks = nn.ModuleList([V12_1Block(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.fiber_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )
        # Deep supervision only at even-indexed blocks (1, 3, 5 = blocks 2, 4, 6)
        weights = torch.zeros(cfg.n_blocks)
        for i in range(cfg.n_blocks):
            if (i + 1) % 2 == 0:  # blocks 2, 4, 6
                weights[i] = (i + 1) / cfg.n_blocks
        weights[-1] = 1.0  # final block always gets weight 1.0
        self.register_buffer("block_loss_weights", weights)

    def _decode_spatial(self, spectral_x):
        spatial = spectral_to_spatial(spectral_x, self.cfg)
        return self.decoder(self.final_norm(spatial))

    def forward(self, token_ids):
        B, T = token_ids.shape
        cfg = self.cfg

        # 1. Spectral embedding
        spectral_x = self.embedding(token_ids)

        # 2. Process through blocks with selective deep supervision
        intermediate_logits = []
        for i, block in enumerate(self.blocks):
            spectral_x = block(spectral_x)
            if self.block_loss_weights[i] > 0:
                logits = self._decode_spatial(spectral_x)[:, :-1, :]
                intermediate_logits.append((logits, self.block_loss_weights[i]))

        # 3. Sparsity diagnostic
        spatial_final = spectral_to_spatial(spectral_x, cfg)
        spec_check = spatial_to_spectral(spatial_final, cfg)
        spec_sparsity = (spec_check.abs() < 1e-6).float().mean().item()

        final_logits = intermediate_logits[-1][0]
        info = {
            "spectral_sparsity": spec_sparsity,
            "intermediate_logits": intermediate_logits,  # list of (logits, weight)
        }
        return final_logits, info


# ── GPT-Nano Baseline ───────────────────────────────────────────────

class GPTNano(nn.Module):
    """Minimal GPT for baseline comparison.
    Architecture: Embedding + [CausalAttention + MLP] x n_layer + LMHead"""

    def __init__(self, vocab_size=65, n_embd=128, n_head=4, n_layer=12,
                 block_size=128, dropout=0.1):
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

        causal_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, block_size, block_size))

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        head_dim = self.n_embd // self.n_head
        for block in self.blocks:
            h = block['ln1'](x)
            qkv = block['attn_qkv'](h).reshape(B, T, 3, self.n_head, head_dim)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + block['attn_proj'](y)
            h = block['ln2'](x)
            x = x + block['mlp_fc2'](F.gelu(block['mlp_fc1'](h)))

        logits = self.lm_head(self.ln_f(x))
        return logits[:, :-1, :], {}
