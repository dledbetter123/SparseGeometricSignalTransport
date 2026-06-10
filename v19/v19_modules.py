"""V19 core modules.

V19 = V18's load-bearing subset + the one unexplored combination from
SYNTHESIS.md (U(K) unitary transport + delta rule) + a geometric (spectral)
context accumulator replacing the SSM + learned per-block sparse frequency
bands + CurvBias on a single attention head per block.

This file is a self-contained PyTorch module. It is intentionally dependency-free
beyond torch so that test_v19.py can import it directly and gen_notebook_v19.py
can either import it or inline-copy its source into the notebook.

Organization:
  1. Config dataclass
  2. Utility kernels:
       - rms_norm, matrix_parallel_scan, unitary_delta_parallel_scan
       - make_skew_symmetric, fast_orthogonal
  3. Components:
       - PrecisionEmbedding           (V18 survivor)
       - VarianceUpdate               (V17 survivor)
       - LearnedBandMask              (NEW, sparse-FFT frequency allocation)
       - GeometricContextAccum        (NEW, replaces SSM)
       - UnitaryDeltaFiber            (NEW, main innovation: U(K) + delta rule)
       - ParsevalSpectralFilter       (V16 survivor, simplified)
       - CurvBiasAttention            (thesis primary contribution, single head)
       - FFN                          (non-negotiable V5-V18 survivor)
       - V19Block                     (combines everything with a learned gate)
       - V19Model                     (stack of blocks + precision embedding + head)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V19Config:
    # --- model dimensions ---
    d_model: int = 256
    n_blocks: int = 8
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # --- precision routing (V17/V18 survivor) ---
    # Same knob names as V18 for drop-in comparability.

    # --- unitary + delta rule fiber (main innovation) ---
    fiber_heads: int = 16         # parallel matrix fibers per block
    fiber_K: int = 32             # SO(K) size; state per head is (K x K).
                                  # K=32 gives state 16*32*32 = 16,384 per block,
                                  # 16x V18's 1024. This closes most of the
                                  # V18-audit "128x state gap" to attention.
    fiber_hidden_mult: int = 2    # width of the skew-symmetric projection MLP

    # --- geometric context accumulator ---
    # The running spectral summary runs an independent content-dependent
    # complex decay per *active* frequency mode.
    ctx_n_bands: int = 32         # total candidate frequency bands per block
    ctx_active_bands: int = 16    # soft target for active bands (via L1 reg on mask)

    # --- Parseval spectral filter ---
    parseval_hidden: int = 128

    # --- CurvBias attention (multi-head per block) ---
    # Default: 8 heads × 32 dim = 256 total, matching GPT-Nano's attention.
    # The original V19 used a single 64-dim head, which left the attention
    # path 4× under-provisioned vs GPT-Nano and caused V19 to plateau ~1 nat
    # higher than GPT-Nano at 3.5K steps on WikiText-103. The topology_arch
    # CurvBias paper itself used multi-head attention (n_heads=8, head_dim=32).
    curvbias_heads: int = 8
    curvbias_dim: int = 32        # per-head dimension

    # --- FFN ---
    ffn_mult: int = 4

    # --- regularization ---
    band_mask_l1: float = 1e-3    # L1 on LearnedBandMask to keep it sparse

    # --- optimizer (used by the training script; not by the modules) ---
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    lr_hold_steps: int = 3000
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 20000
    eval_interval: int = 500
    eval_steps: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utility kernels
# ─────────────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm used across modules. Last-dim normalization."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def matrix_parallel_scan(A_diag: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """S[t] = a[t] * S[t-1] + B[t], scalar decay * matrix deposit.

    Out-of-place Hillis–Steele sweep. V16e/V17/V18 compatible.

    A_diag: (N, T) scalar decays.
    B_mat : (N, T, d, d) deposits.
    Returns: (N, T, d, d) accumulated states (S[0] = B[0], S[t] as above).
    """
    N, T, d, _ = B_mat.shape
    a = A_diag
    b = B_mat
    steps = int(math.ceil(math.log2(max(T, 2))))
    for s in range(steps):
        step = 2 ** s
        if step >= T:
            break
        a_r = a[:, step:].unsqueeze(-1).unsqueeze(-1)
        b = torch.cat([b[:, :step], a_r * b[:, :-step] + b[:, step:]], dim=1)
        a = torch.cat([a[:, :step], a[:, step:] * a[:, :-step]], dim=1)
    return b


def scalar_parallel_scan(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Plain scalar prefix scan: out[t] = A[t] * out[t-1] + B[t], out[-1] = 0.

    Out-of-place Hillis–Steele sweep on (N, T) tensors.

    A: (N, T) scalar decays.
    B: (N, T) scalar deposits.
    Returns: (N, T) accumulated sequence.
    """
    assert A.shape == B.shape, "A and B must share shape"
    N, T = B.shape
    a = A
    b = B
    steps = int(math.ceil(math.log2(max(T, 2))))
    for s in range(steps):
        step = 2 ** s
        if step >= T:
            break
        a_r = a[:, step:]
        b = torch.cat([b[:, :step], a_r * b[:, :-step] + b[:, step:]], dim=1)
        a = torch.cat([a[:, :step], a[:, step:] * a[:, :-step]], dim=1)
    return b


def unitary_delta_parallel_scan(
    U_all: torch.Tensor, B_all: torch.Tensor
) -> torch.Tensor:
    """S[t] = U[t] @ S[t-1] + B[t] with S[-1] = 0.

    Out-of-place Hillis–Steele sweep. Semigroup element is the pair (U_eff,
    B_eff) representing:
        S_after = U_eff @ S_before + B_eff
    Combination (older, newer) -> combined:
        (U1, B1) · (U2, B2) = (U2 @ U1, U2 @ B1 + B2)
    Associative. O(T log T) matrix products.

    Matmul via torch.matmul (CUBLAS path) rather than torch.einsum: for 4D
    tensors both compile to the same GEMM but torch.matmul skips the einsum
    parser overhead, saving a measurable fraction when the scan is called 8
    times per forward at K=32.

    U_all: (N, T, K, K).
    B_all: (N, T, K, K).
    Returns: (N, T, K, K) with S[t] as above.
    """
    assert U_all.shape == B_all.shape, "U and B must share shape"
    N, T, K, _ = U_all.shape
    U = U_all
    B = B_all
    steps = int(math.ceil(math.log2(max(T, 2))))
    for s in range(steps):
        step = 2 ** s
        if step >= T:
            break
        U_left = U[:, :-step]      # older
        U_right = U[:, step:]      # newer
        B_left = B[:, :-step]
        B_right = B[:, step:]
        new_B_tail = torch.matmul(U_right, B_left) + B_right
        new_U_tail = torch.matmul(U_right, U_left)
        B = torch.cat([B[:, :step], new_B_tail], dim=1)
        U = torch.cat([U[:, :step], new_U_tail], dim=1)
    return B


def make_skew_symmetric(params: torch.Tensor, K: int) -> torch.Tensor:
    """Map a (..., K*(K-1)/2) parameter tensor to (..., K, K) skew-symmetric.

    Vectorized via advanced indexing on triu_indices. The original version
    used a Python double loop over K*(K-1)/2 pairs, which was unacceptable
    at K=32 (~500 Python calls per forward pass per block). This version
    does the whole fill in two tensor assignments.

    torch.triu_indices is a supported torch.compile op, so this stays inside
    the compiled graph on CUDA.
    """
    shape = params.shape[:-1]
    A = torch.zeros(*shape, K, K, device=params.device, dtype=params.dtype)
    triu = torch.triu_indices(K, K, offset=1, device=params.device)
    i_idx, j_idx = triu[0], triu[1]
    A[..., i_idx, j_idx] = params
    A[..., j_idx, i_idx] = -params
    return A


def fast_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """Four-term Taylor approximation of exp(A) for small skew-symmetric A.

    Accurate to ~1e-8 when ||A|| < 0.1. MPS-safe (no complex ops). We rely on
    caller scaling `A` so the spectral radius stays small; see
    UnitaryDeltaFiber.forward for the 0.01 scaling.
    """
    K = A.shape[-1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    A2 = A @ A
    return I + A + A2 / 2 + (A2 @ A) / 6


# ─────────────────────────────────────────────────────────────────────────────
# 3a. PrecisionEmbedding (V18 survivor)
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionEmbedding(nn.Module):
    """Dense token + positional embedding with learned positional precision.

    V18 verbatim — kept as a load-bearing component. Position is encoded both
    as an additive vector (pos_emb) and as a precision template (pos_prec).
    """

    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, D)
        self.pos_prec = nn.Embedding(cfg.max_seq_len, D)
        self.prec_mix = nn.Linear(2 * D, D)
        nn.init.zeros_(self.pos_prec.weight)
        nn.init.zeros_(self.prec_mix.weight)
        nn.init.zeros_(self.prec_mix.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        tok = self.tok_emb(token_ids)
        x = tok + self.pos_emb(pos)
        x = self.drop(x)
        pp = self.pos_prec(pos).unsqueeze(0).expand(B, -1, -1)
        log_var = self.prec_mix(torch.cat([tok, pp], dim=-1))
        return x, log_var


# ─────────────────────────────────────────────────────────────────────────────
# 3b. VarianceUpdate (V17 survivor)
# ─────────────────────────────────────────────────────────────────────────────

class VarianceUpdate(nn.Module):
    """Learned content-dependent log_var evolution. V17 verbatim."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh())
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, log_var: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([context, log_var], dim=-1))
        return (log_var + 0.1 * delta).clamp(min=-6, max=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3c. LearnedBandMask  (NEW, thesis §8.3.3)
# ─────────────────────────────────────────────────────────────────────────────

class LearnedBandMask(nn.Module):
    """Differentiable sparse mask over candidate frequency bands.

    Each block owns one LearnedBandMask. We initialize with a deterministic
    wavelet-like schedule (block ℓ prefers bands 2^(ℓ-1)..2^ℓ) and let training
    move the mask around. An L1 penalty (applied externally via `cfg.band_mask_l1`)
    keeps the mask sparse.

    Output is a per-band scalar in (0, 1) via sigmoid of the learned logits.
    """

    def __init__(self, cfg: V19Config, block_idx: int):
        super().__init__()
        self.n_bands = cfg.ctx_n_bands
        self.block_idx = block_idx

        # Initialize with a soft wavelet-like preference:
        # block 0 favors the lowest bands, block L-1 favors the highest.
        logits = torch.full((cfg.ctx_n_bands,), -2.0)
        lo = int(math.floor(cfg.ctx_n_bands * block_idx / max(1, cfg.n_blocks)))
        hi = int(math.ceil(cfg.ctx_n_bands * (block_idx + 1) / max(1, cfg.n_blocks)))
        lo = max(0, min(cfg.ctx_n_bands - 1, lo))
        hi = max(lo + 1, min(cfg.ctx_n_bands, hi))
        logits[lo:hi] = 2.0  # sigmoid(2) ≈ 0.88 on preferred bands
        self.logits = nn.Parameter(logits)

    def forward(self) -> torch.Tensor:
        """Returns (n_bands,) soft mask in (0, 1)."""
        return torch.sigmoid(self.logits)

    def l1_penalty(self) -> torch.Tensor:
        """L1 of the soft mask — use in training loss for sparsification."""
        return self.forward().sum()


# ─────────────────────────────────────────────────────────────────────────────
# 3d. GeometricContextAccum  (NEW, replaces SSM; thesis §7.3.4)
# ─────────────────────────────────────────────────────────────────────────────

class GeometricContextAccum(nn.Module):
    """Spectral running summary.

    Instead of an SSM (the thesis §7.3.4 flagged as the least geometrically
    motivated component), we maintain a per-frequency-band complex running
    summary with content-dependent decay. Only the active bands (as selected
    by LearnedBandMask) participate in each block's summary.

    Concretely, for each candidate band k we learn a real frequency ω_k and
    run the recurrence:
        h_k[t] = γ_k(x_t) · e^{-i ω_k} · h_k[t-1] + α_k(x_t) · x_t^{(k)}

    where:
      - γ_k(x_t) ∈ (0, 1) is a content-dependent decay (per-band, per-step)
      - α_k(x_t) is a content-dependent deposit
      - x_t^{(k)} is a learned projection of x_t onto band k (complex-valued,
        implemented as a 2-channel real pair to avoid MPS complex ops)

    The output at step t is a real vector of size d_model obtained by
    projecting the concatenated band summaries back through a linear layer.
    """

    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        S = cfg.ctx_n_bands
        self.n_bands = S
        # Per-band learnable frequency (initialized with RoPE-like 1/10000^(2k/S))
        omegas = 1.0 / (10000.0 ** (torch.arange(S, dtype=torch.float32) / max(1, S)))
        self.omega = nn.Parameter(omegas * 2 * math.pi / max(1, cfg.seq_len))
        # Content projections: deposit (real, imag) and decay per band
        self.dep_proj = nn.Linear(D, 2 * S)   # real+imag deposit per band
        self.dec_proj = nn.Linear(D, S)       # scalar decay per band (sigmoid)
        # Output projection from concatenated (real, imag) band summary (2S) to D
        self.out_proj = nn.Linear(2 * S, D)
        # Initialize deposit near-zero so band summary starts calm
        nn.init.zeros_(self.dep_proj.weight)
        nn.init.zeros_(self.dep_proj.bias)
        nn.init.zeros_(self.dec_proj.weight)
        nn.init.constant_(self.dec_proj.bias, 2.0)  # sigmoid(2) ≈ 0.88 (slow decay)

    def forward(self, x: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        band_mask: (S,) soft mask from LearnedBandMask
        returns: (B, T, D) running summary at each time step

        Vectorized implementation. The original sequential Python loop over T
        was killing H100 throughput because the inner ops were tiny and the
        kernel-launch overhead dominated. We use the pre-rotate trick:

            Original: h[t] = γ[t] · R · h[t-1] + d[t]   (R is fixed 2x2 rotation)
            Substitute h'[t] = R^{-t} · h[t]:
                h'[t] = γ[t] · h'[t-1] + R^{-t} · d[t]
                       └── scalar scan ─┘   └── precomputable deposits ─┘

        This reduces to a per-band scalar prefix scan with content-dependent
        decay, which we run via scalar_parallel_scan in O(log T) parallel
        steps. The final h[t] = R^t · h'[t] is a pure pointwise post-rotate.
        """
        B, T, D = x.shape
        S = self.n_bands

        # Content-dependent deposits and decays
        dep = self.dep_proj(x).reshape(B, T, S, 2)          # (B, T, S, 2) re+im
        dep = dep * band_mask.view(1, 1, S, 1)               # sparse-FFT gating
        gam = torch.sigmoid(self.dec_proj(x))                # (B, T, S) in (0, 1)

        # Precompute ω·t grid and its cos/sin ONCE per forward (not per-t)
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
        omega_t = t_idx.view(T, 1) * self.omega.view(1, S)   # (T, S)
        cos_ot = torch.cos(omega_t)                          # (T, S)
        sin_ot = torch.sin(omega_t)                          # (T, S)

        # Pre-rotate deposits: d̃[t] = R^{-t} · d[t]
        # Rotation by -ω·t: (a + b i) · e^{-i ω t} = a cos + b sin  +  i (b cos - a sin)
        d_re = dep[..., 0]                                   # (B, T, S)
        d_im = dep[..., 1]
        dtil_re = d_re * cos_ot + d_im * sin_ot              # (B, T, S)
        dtil_im = d_im * cos_ot - d_re * sin_ot              # (B, T, S)

        # Parallel scalar scan per (batch, band): z'[t] = γ[t]·z'[t-1] + d̃[t]
        gam_flat = gam.permute(0, 2, 1).reshape(B * S, T).contiguous()
        dtil_re_flat = dtil_re.permute(0, 2, 1).reshape(B * S, T).contiguous()
        dtil_im_flat = dtil_im.permute(0, 2, 1).reshape(B * S, T).contiguous()
        zp_re = scalar_parallel_scan(gam_flat, dtil_re_flat)  # (B*S, T)
        zp_im = scalar_parallel_scan(gam_flat, dtil_im_flat)  # (B*S, T)
        zp_re = zp_re.reshape(B, S, T).permute(0, 2, 1)       # (B, T, S)
        zp_im = zp_im.reshape(B, S, T).permute(0, 2, 1)       # (B, T, S)

        # Post-rotate: h[t] = R^t · z'[t]
        # (a + b i) · e^{i ω t} = a cos - b sin  +  i (a sin + b cos)
        h_re = zp_re * cos_ot - zp_im * sin_ot                # (B, T, S)
        h_im = zp_re * sin_ot + zp_im * cos_ot                # (B, T, S)

        summary = torch.cat([h_re, h_im], dim=-1)             # (B, T, 2S)
        return self.out_proj(summary)


# ─────────────────────────────────────────────────────────────────────────────
# 3e. UnitaryDeltaFiber  (NEW, the main V19 innovation)
# ─────────────────────────────────────────────────────────────────────────────

class UnitaryDeltaFiber(nn.Module):
    """SO(K) unitary transport + delta-rule write + associative read.

    This is the unexplored combination identified in topology/SYNTHESIS.md:
      S[t] = U[t] @ S[t-1] + k[t] (v[t] - (U[t] S[t-1])^T k[t])^T

    We parameterize U[t] as fast_orthogonal(skew(content(x_t))) — a small
    skew-symmetric content projection then Taylor-approximated matrix
    exponential. This is MPS-safe (no complex ops, real SO(K)) and consistent
    with the HolonomicRotary pattern already in topology_arch.

    For the scan we use unitary_delta_parallel_scan which runs in O(T log T)
    matrix products.

    State dimensions:
        H heads × K × K   = cfg.fiber_heads × cfg.fiber_K × cfg.fiber_K
        total per block   = H K^2 values
        V19 default       = 16 × 32 × 32 = 16 384 values per block
                            (16× the V16e/V18 budget of 1024)
    """

    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        H = cfg.fiber_heads
        K = cfg.fiber_K
        ns = K * (K - 1) // 2
        hidden = cfg.fiber_hidden_mult * D

        # Content -> skew-symmetric params for U[t]
        self.skew_proj = nn.Sequential(
            nn.Linear(D, hidden),
            nn.SiLU(),
            nn.Linear(hidden, H * ns),
        )
        nn.init.zeros_(self.skew_proj[-1].weight)
        nn.init.zeros_(self.skew_proj[-1].bias)

        # Content -> k, v for the delta rule
        # k is (H, K), v is (H, K), so together 2 H K per step.
        self.kv_proj = nn.Linear(D, 2 * H * K)

        # Output projection from concatenated fiber reads (H*K) to D
        self.out_proj = nn.Linear(H * K, D)

        # Gate from fiber output to the block residual. Start at sigmoid(-2) ≈ 0.12
        # so the fiber doesn't dominate at init but is learnable.
        self.gate = nn.Parameter(torch.tensor(-2.0))

        # Initialize out_proj small so the fiber doesn't inject noise at init.
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.H = H
        self.K = K
        self.ns = ns
        self.D = D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D) fiber output
        """
        B, T, D = x.shape
        H, K, ns = self.H, self.K, self.ns

        # 1. Content -> skew params -> orthogonal U[t]
        skew_params = self.skew_proj(x).reshape(B, T, H, ns)
        # Scale for Taylor accuracy (4-term exp is ~1e-8 accurate for ||A|| < 0.1)
        skew_params = skew_params * 0.01
        skew_flat = skew_params.reshape(B * T * H, ns)
        A = make_skew_symmetric(skew_flat, K)
        U_step = fast_orthogonal(A).reshape(B, T, H, K, K)

        # 2. Content -> k, v for delta rule
        kv = self.kv_proj(x).reshape(B, T, H, 2, K)
        k = kv[:, :, :, 0, :]  # (B, T, H, K)
        v = kv[:, :, :, 1, :]  # (B, T, H, K)
        # Normalize k to unit length so that (S^T k) magnitudes stay bounded.
        k = F.normalize(k, dim=-1)

        # 3. Build the deposit B[t] for the scan. The delta-rule recurrence is:
        #     S[t] = U[t] @ S[t-1] + k[t] (v[t] - (U[t] S[t-1])^T k[t])^T
        # That "(U[t] S[t-1])^T k[t]" term depends on the past state, which the
        # parallel scan does not see at t. We use the design-note mitigation:
        # apply the delta in the unitary-conjugate basis so that the correction
        # depends only on local content:
        #     S'[t] = S[t-1] + k[t] (v[t] - (k[t])^T S[t-1])^T   (no U in the correction)
        #     S[t]  = U[t] @ S'[t]
        # Expanded:
        #     S[t] = U[t] @ S[t-1] + U[t] @ (k[t] (v[t] - k[t]^T S[t-1])^T)
        # Grouping:
        #     S[t] = (U[t] - U[t] @ k[t] k[t]^T) @ S[t-1] + U[t] @ k[t] v[t]^T
        #
        # Since the delta correction also depends on S[t-1] via k[t]^T S[t-1],
        # the parallel scan must handle a transition matrix of the form
        #     (U[t] - U[t] @ k[t] k[t]^T)
        # which is (U[t] @ (I - k[t] k[t]^T)) — still a well-defined K×K matrix
        # that we can feed to unitary_delta_parallel_scan as the "U" argument.
        # The deposit is U[t] @ (k[t] v[t]^T).
        kkT = torch.einsum("bthi,bthj->bthij", k, k)                            # (B,T,H,K,K)
        I = torch.eye(K, device=x.device, dtype=x.dtype).reshape(1, 1, 1, K, K)
        eff_trans = torch.einsum("bthij,bthjk->bthik", U_step, I - kkT)         # (B,T,H,K,K)
        kvT = torch.einsum("bthi,bthj->bthij", k, v)                            # (B,T,H,K,K)
        deposit = torch.einsum("bthij,bthjk->bthik", U_step, kvT)              # (B,T,H,K,K)

        # 4. Parallel scan over T for each (B, H)
        # Reshape to (N, T, K, K) where N = B*H
        eff_trans_flat = eff_trans.permute(0, 2, 1, 3, 4).reshape(B * H, T, K, K)
        deposit_flat = deposit.permute(0, 2, 1, 3, 4).reshape(B * H, T, K, K)
        S_all = unitary_delta_parallel_scan(eff_trans_flat, deposit_flat)
        S_all = S_all.reshape(B, H, T, K, K)

        # 5. Causal shift: the query at time t must read S[t-1] (state before
        # writing the current token). Pad S_all with a zero state at t=0.
        zero_state = torch.zeros(B, H, 1, K, K, device=x.device, dtype=x.dtype)
        S_prev = torch.cat([zero_state, S_all[:, :, :-1]], dim=2)  # (B, H, T, K, K)

        # 6. Associative read: output[t] = q[t] @ S[t-1].
        # Use a separate linear head as query (reusing k would couple read and write).
        # To keep the parameter count bounded we reuse the kv_proj's k as the query
        # after a tiny learned rotation — but that couples read and write. Cleaner:
        # promote k to a second role by projecting through a small linear. Here we
        # simply reuse k as the query, which is a documented simplification (the
        # delta rule makes k content-addressable, so using k for both read and write
        # is the classical DeltaNet choice).
        out = torch.einsum("bthi,bhtij->bthj", k, S_prev)  # (B, T, H, K)
        out = out.reshape(B, T, H * K)
        return torch.sigmoid(self.gate) * self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3f. ParsevalSpectralFilter  (V16 survivor, simplified)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelGate(nn.Module):
    """Content-dependent per-channel gate on the fiber output.

    Replaces V19's original ParsevalSpectralFilter, which did an rFFT along
    the POSITION axis and was therefore non-causal: perturbing the last
    token bled into every earlier position through the inverse FFT. The
    original V19Block causality test only barely passed (err ~ 1e-4) because
    at init the fiber output is small; during training the leakage became
    significant and the model could not learn.

    V16's Parseval filter was safe because its FFT was along the CHANNEL
    axis of a spectral constellation, not the position axis of a spatial
    signal. V19 has no spectral constellation, so there is nothing natural
    to FFT. The right replacement is a simple per-channel gate:

        gate[b, t, d] = sigmoid(W · ctx[b, t]) ∈ (0, 1)
        y_out[b, t, d] = gate[b, t, d] · y[b, t, d]

    This is position-wise (causal by construction), content-dependent
    (ctx drives the gate), and energy-bounded (|gate| ≤ 1). It retains the
    "Parseval spirit" of V16 (bounded-energy filter) without introducing
    the position-axis FFT that would violate causality.
    """

    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        self.filter_proj = nn.Linear(D, D)
        nn.init.zeros_(self.filter_proj.weight)
        nn.init.zeros_(self.filter_proj.bias)

    def forward(self, y: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        y:   (B, T, D) fiber output
        ctx: (B, T, D) context summary (drives the gate)
        returns: (B, T, D) gated output with 0 <= gate <= 1 per channel per step
        """
        gate = torch.sigmoid(self.filter_proj(ctx))      # (B, T, D)
        return gate * y


# Backwards-compatible alias for code that still imports the old name.
ParsevalSpectralFilter = ChannelGate


# ─────────────────────────────────────────────────────────────────────────────
# 3g. CurvBiasAttention  (thesis primary contribution, single head)
# ─────────────────────────────────────────────────────────────────────────────

class CurvBiasAttention(nn.Module):
    """Multi-head attention per V19 block with CurvBias position encoding.

    Formula (from topology_arch/gen_notebook_v5_ablation.py, verbatim logic,
    applied per-head):
      1. Per-head base RoPE frequencies: θ_base[t, k] = t · ω_k
      2. Per-head content-dependent delta: δ[t, h, k] = tanh(x_t · W_θ) · (π/T)
      3. Per-head accumulated θ[t, h] = base + cumsum(δ)
      4. Per-head curvature bias: B[h, i, j] = α[h] · ‖θ[i, h] − θ[j, h]‖
      5. Attention per head: softmax(qk/√d_head + B[h]) v  with causal mask
      6. Concatenate heads → output projection → gate → block residual

    Default cfg: n_heads=8, head_dim=32, total 256 dim (matches GPT-Nano's
    attention capacity). Earlier V19 used a single 64-dim head which was
    4× under-provisioned vs GPT-Nano and caused training to plateau ~1 nat
    above GPT-Nano on WikiText-103. This multi-head version restores parity
    and lets the CurvBias geometric signal actually help the attention.
    """

    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        H = cfg.curvbias_heads
        hd = cfg.curvbias_dim
        n_rot = hd // 2
        total = H * hd
        self.q_proj = nn.Linear(D, total)
        self.k_proj = nn.Linear(D, total)
        self.v_proj = nn.Linear(D, total)
        self.o_proj = nn.Linear(total, D)
        # Per-head, per-rotation-plane content-dependent delta.
        self.theta_proj = nn.Linear(D, H * n_rot)
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.theta_proj.bias)
        # Per-head learnable curvature-bias strength α[h]
        self.curv_alpha = nn.Parameter(torch.zeros(1, H, 1, 1))
        # Base RoPE frequencies (shared across heads, like GPT-Nano)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, n_rot, dtype=torch.float32) / max(1, n_rot)))
        self.register_buffer("base_freqs", freqs)
        self.H = H
        self.hd = hd
        self.n_rot = n_rot
        # Gate back into the residual stream. Start open (sigmoid(0) = 0.5) so
        # the attention path carries meaningful signal from step 0. The fiber
        # path, by contrast, starts gated down (sigmoid(-2) ≈ 0.12) so V19
        # looks transformer-ish at initialization and the fiber contribution
        # has to earn its weight during training.
        self.gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
        """x: (..., hd); cos/sin: (..., n_rot). Returns (..., hd)."""
        d2 = x.shape[-1] // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        H, hd, n_rot = self.H, self.hd, self.n_rot

        # Project and reshape to (B, H, T, hd)
        q = self.q_proj(x).reshape(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, hd).transpose(1, 2)

        # Per-head content-dependent angles: (B, T, H, n_rot) → (B, H, T, n_rot)
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        base_theta = positions.unsqueeze(-1) * self.base_freqs  # (T, n_rot)
        base_theta = base_theta.view(1, 1, T, n_rot).expand(B, H, T, n_rot)
        delta = torch.tanh(self.theta_proj(x)) * (math.pi / max(1, T))   # (B, T, H*n_rot)
        delta = delta.reshape(B, T, H, n_rot).transpose(1, 2)            # (B, H, T, n_rot)
        theta = base_theta + torch.cumsum(delta, dim=2)                  # (B, H, T, n_rot)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        q = self._apply_rotary(q, cos_t, sin_t)
        k = self._apply_rotary(k, cos_t, sin_t)

        # Scores: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)

        # CurvBias term per head: ‖θ[i, h] - θ[j, h]‖ via cdist
        # theta is (B, H, T, n_rot); flatten to (B*H, T, n_rot) for cdist.
        theta_flat = theta.reshape(B * H, T, n_rot)
        curv = torch.cdist(theta_flat, theta_flat).reshape(B, H, T, T)
        scores = scores + self.curv_alpha * curv

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        # Aggregate values: (B, H, T, hd)
        out = torch.matmul(attn, v)
        # Concatenate heads: (B, T, H*hd)
        out = out.transpose(1, 2).reshape(B, T, H * hd)
        return torch.sigmoid(self.gate) * self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3h. FFN (non-negotiable V5-V18 survivor)
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: V19Config):
        super().__init__()
        D = cfg.d_model
        hidden = D * cfg.ffn_mult
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, D),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3i. V19Block  (puts everything together with a learned mix gate)
# ─────────────────────────────────────────────────────────────────────────────

class V19Block(nn.Module):
    """One V19 block.

    Data flow:
        x, log_var
        -> RMSNorm
        -> precision gating          (V17/V18)
        -> GeometricContextAccum     (replaces SSM)
        -> UnitaryDeltaFiber         (main innovation)
        -> ParsevalSpectralFilter    (V16 survivor)
        -> CurvBiasAttention         (thesis primary contribution, 1 head)
        -> learned gate between fiber-path and attention-path outputs
        -> residual into x
        -> RMSNorm -> FFN -> residual into x
        -> VarianceUpdate(log_var, context)
    """

    def __init__(self, cfg: V19Config, block_idx: int, is_last: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_last = is_last
        self.norm1 = RMSNorm(cfg.d_model)
        self.band_mask = LearnedBandMask(cfg, block_idx)
        self.ctx = GeometricContextAccum(cfg)
        self.fiber = UnitaryDeltaFiber(cfg)
        self.pfilter = ChannelGate(cfg)
        self.attn = CurvBiasAttention(cfg)
        # Mix gate initialized at sigmoid(2) ≈ 0.88 so the attention path
        # dominates initially. This makes V19 behave like a standard
        # transformer at step 0 (which we know can learn on WikiText-103) and
        # lets the fiber + geometric context paths prove their value by
        # shifting the gate during training.
        self.mix_gate = nn.Parameter(torch.tensor(2.0))
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)
        # The last block's log_var update would be unused (no downstream block
        # consumes it), so we skip the VarianceUpdate module on the final block
        # to avoid stranded parameters.
        self.var_update = None if is_last else VarianceUpdate(cfg.d_model)

    def forward(
        self, x: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Normalization
        h = self.norm1(x)

        # 2. Precision gating (V17/V18)
        precision = torch.exp(-log_var)
        prec_gate = torch.sigmoid(precision - 1.0)
        h_eff = h * prec_gate

        # 3. Geometric context accumulator (replaces SSM)
        band_mask = self.band_mask()
        ctx = self.ctx(h_eff, band_mask)              # (B, T, D)

        # 4. Unitary + delta-rule fiber
        fiber_out = self.fiber(h_eff)                  # (B, T, D)

        # 5. Parseval spectral filter on the fiber output
        fiber_out = self.pfilter(fiber_out, ctx)

        # 6. Single-head attention with CurvBias
        attn_out = self.attn(h_eff)                    # (B, T, D)

        # 7. Learned mix of fiber vs attention paths
        g = torch.sigmoid(self.mix_gate)
        mix = g * attn_out + (1.0 - g) * fiber_out + ctx

        # 8. Residual
        x = x + mix

        # 9. FFN (non-negotiable)
        x = x + self.ffn(self.norm2(x))

        # 10. Variance evolution (V17/V18). Skipped on the last block.
        if self.var_update is not None:
            log_var = self.var_update(log_var, mix)

        return x, log_var

    def band_mask_l1(self) -> torch.Tensor:
        return self.band_mask.l1_penalty()


# ─────────────────────────────────────────────────────────────────────────────
# 3j. V19Model  (stack of blocks, head, loss with band-mask regularization)
# ─────────────────────────────────────────────────────────────────────────────

class V19Model(nn.Module):
    def __init__(self, cfg: V19Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = PrecisionEmbedding(cfg)
        self.blocks = nn.ModuleList([
            V19Block(cfg, block_idx=i, is_last=(i == cfg.n_blocks - 1))
            for i in range(cfg.n_blocks)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Return (logits, aux).

        logits: (B, T-1, V) next-token logits over positions 0..T-2.
        aux:    dict with "band_mask_l1" for the training loss.
        """
        x, log_var = self.embedding(token_ids)
        for block in self.blocks:
            x, log_var = block(x, log_var)
        x = self.norm_f(x)
        logits = self.head(x)[:, :-1, :]
        # Aggregate band-mask L1 across blocks for regularization
        band_l1 = sum(block.band_mask_l1() for block in self.blocks)
        return logits, {"band_mask_l1": band_l1}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
