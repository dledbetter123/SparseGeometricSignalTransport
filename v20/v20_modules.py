"""V20 core modules — the spectral return.

V20 is the return to the V12.1 spectral-constellation line with the
non-abelian transport and speed infrastructure that came out of the V19
experiment grafted on.

This file is self-contained: every module V20 needs lives here, including
the utility kernels (RMSNorm, make_skew_symmetric, fast_orthogonal,
unitary_delta_parallel_scan) that were previously imported from
`../v19/v19_modules.py` via a sys.path hack. V20 is the main architectural
line now, so it should not depend on V19's directory layout or V19's config
attribute names.

Layout:
  1. Config dataclass
  2. Utility kernels:
       - RMSNorm
       - make_skew_symmetric, fast_orthogonal
       - unitary_delta_parallel_scan
       - count_params
  3. Constellation container (mag, phase)
  4. CloudNorm
  5. ConstellationEmbedding
  6. SpectralTransportKernel  (V12.1's exp(-D_k(q) ω² - i A_k(q) ω))
  7. LearnedBandMask           (differentiable sparse frequency mask per block)
  8. SparseFFT / SparseIFFT    (Level 0: full rFFT/irFFT + band mask)
  9. PerSubbundleUnitaryDeltaFiber
 10. SpatialMLP
 11. PerSubbundleMLP           (strict orthogonal FFN variant)
 12. ProximalTopK
 13. V20Block
 14. V20Model
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V20Config:
    # --- constellation shape ---
    n_subbundles: int = 8           # orthogonal feature channels
    subbundle_dim: int = 32         # spatial size per subbundle (power of 2)
    # derived: spectral_half_dim = subbundle_dim // 2 + 1 = 17 at default
    # derived: n_modes = n_subbundles * spectral_half_dim = 136
    # derived: fiber_dim = n_subbundles * subbundle_dim = 256

    # --- sparse-FFT proximal top-k ---
    active_modes_per_sub: int = 8   # keep this many modes per subbundle
    band_mask_l1: float = 1e-3      # L1 on LearnedBandMask

    # --- spectral transport kernel (V12.1 D_k(q) + A_k(q)) ---
    transport_hidden: int = 128

    # --- per-subbundle unitary delta fiber ---
    # Each subbundle is one fiber; fiber state is (K, K) per subbundle,
    # where K = spectral_half_dim by default (= 17), so state = 8 * 17 * 17
    # = 2,312 values per block. Much smaller than V19's 16,384 but the
    # design thesis is that this is the natural state capacity for the
    # number of active modes per subbundle, not a tuning knob.
    fiber_hidden_mult: int = 2

    # --- SpatialMLP ---
    ffn_mult: int = 4               # keeps SpatialMLP as dominant compute

    # --- stack ---
    n_blocks: int = 6
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # --- training (used by the notebook, not the modules) ---
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    lr_hold_steps: int = 3000
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 20000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def spectral_half_dim(self) -> int:
        return self.subbundle_dim // 2 + 1

    @property
    def n_modes(self) -> int:
        return self.n_subbundles * self.spectral_half_dim

    @property
    def fiber_dim(self) -> int:
        return self.n_subbundles * self.subbundle_dim

    @property
    def fiber_K(self) -> int:
        # Fiber state dim per subbundle. We use the spatial subbundle size
        # (not the spectral half-dim) so the fiber operates on the post-IFFT
        # real signal per subbundle.
        return self.subbundle_dim


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utility kernels
# ─────────────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm along the last dimension."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def make_skew_symmetric(params: torch.Tensor, K: int) -> torch.Tensor:
    """Map a (..., K*(K-1)/2) parameter tensor to (..., K, K) skew-symmetric.

    Vectorized via advanced indexing on torch.triu_indices — no Python loop
    over the K*(K-1)/2 upper-triangle pairs.
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

    Accurate to ~1e-8 when ||A|| < 0.1. MPS-safe (no complex ops). Caller
    must scale A so spectral radius stays small; PerSubbundleUnitaryDeltaFiber
    multiplies skew_params by 0.01 before this call.
    """
    K = A.shape[-1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    A2 = A @ A
    return I + A + A2 / 2 + (A2 @ A) / 6


def unitary_delta_parallel_scan(
    U_all: torch.Tensor, B_all: torch.Tensor
) -> torch.Tensor:
    """S[t] = U[t] @ S[t-1] + B[t] with S[-1] = 0.

    Hillis–Steele sweep over the semigroup
        (U1, B1) · (U2, B2) = (U2 @ U1, U2 @ B1 + B2)
    which is associative. O(T log T) matrix products.

    U_all: (N, T, K, K)
    B_all: (N, T, K, K)
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
        U_left = U[:, :-step]
        U_right = U[:, step:]
        B_left = B[:, :-step]
        B_right = B[:, step:]
        new_B_tail = torch.matmul(U_right, B_left) + B_right
        new_U_tail = torch.matmul(U_right, U_left)
        B = torch.cat([B[:, :step], new_B_tail], dim=1)
        U = torch.cat([U[:, :step], new_U_tail], dim=1)
    return B


def count_params(model: nn.Module) -> int:
    """Total number of parameters in a module."""
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constellation container
# ─────────────────────────────────────────────────────────────────────────────

class Constellation(NamedTuple):
    """A sparse spectral constellation: (mag, phase) per mode.

    Stored as two parallel real tensors of shape (B, T, n_modes) so the
    whole pipeline is compile-friendly (no Python object construction in
    the hot path). Use .to_complex() to get the (B, T, n_modes) complex
    tensor when you need it for rFFT operations.

    Earlier versions carried a third `log_var` field for V17-style
    Gaussian-cloud precision routing, but the precision mechanism was
    never actually wired into any downstream computation in V20 — it was
    computed, passed through every module unchanged, and contributed
    nothing to the loss. Removed on 2026-04-10 to stop dead code from
    misleading future readers.
    """
    mag: torch.Tensor
    phase: torch.Tensor

    def to_complex(self) -> torch.Tensor:
        return self.mag * torch.exp(1j * self.phase)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CloudNorm  (V17 verbatim)
# ─────────────────────────────────────────────────────────────────────────────

class CloudNorm(nn.Module):
    """RMS-normalize the constellation magnitudes and apply a learned
    per-mode rescaling. Phase passes through unchanged.
    """

    def __init__(self, n_modes: int):
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(n_modes))

    def forward(self, c: Constellation) -> Constellation:
        rms = (c.mag ** 2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        new_mag = (c.mag / rms) * self.mag_scale
        return Constellation(new_mag, c.phase)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ConstellationEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class ConstellationEmbedding(nn.Module):
    """Token → sparse spectral constellation.

    Each vocab entry gets a learned (magnitude, phase) pair per mode. The
    position-dependent phase offset is added via the RoPE-style frequency
    schedule of V17 (cheap baseline that gives a smooth position prior;
    V20's main positional mechanism is the non-abelian fiber transport).
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        M = cfg.n_modes
        self.cfg = cfg

        self.mag_emb = nn.Embedding(cfg.vocab_size, M)
        self.phase_emb = nn.Embedding(cfg.vocab_size, M)
        nn.init.uniform_(self.phase_emb.weight, -math.pi, math.pi)

        # Supplementary phase shift per position via RoPE-style frequencies
        # (V17 uses this as a cheap positional baseline.)
        freqs = torch.zeros(M)
        for s in range(cfg.n_subbundles):
            off = s * cfg.spectral_half_dim
            freqs[off:off + cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0)
            )
        self.register_buffer("pos_freqs", freqs)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> Constellation:
        B, T = token_ids.shape
        mag = self.mag_emb(token_ids)                       # (B, T, M)
        phase = self.phase_emb(token_ids)                    # (B, T, M)

        pos = torch.arange(T, device=token_ids.device, dtype=phase.dtype)
        phase = phase + (pos.unsqueeze(-1) * self.pos_freqs).unsqueeze(0)  # (B, T, M)

        mag = self.drop(mag)
        return Constellation(mag, phase)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SpectralTransportKernel  (V12.1's D_k(q) + A_k(q), with Parseval bound)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralTransportKernel(nn.Module):
    """Apply exp(-D_k(q) ω_k² - i A_k(q) ω_k) to the constellation.

    This is V12.1's content-dependent spectral transport kernel. Two
    learned MLPs produce:

      - D_k(q) ≥ 0 via softplus  — per-mode damping rate
      - A_k(q) ∈ ℝ               — per-mode gauge phase rotation

    The context q is the current constellation itself (flattened to
    concat(mag, phase)). This is deliberately per-token rather
    than running-context; we can swap in a cumulative summary later if
    V20's ablations show the per-token version is insufficient.

    Parseval bound: since D ≥ 0, |damping| ≤ 1 per mode, so the magnitude
    of every mode can only shrink. This is the |W| ≤ 1 constraint V16 and
    V19 both rely on for numerical stability.
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.cfg = cfg
        M = cfg.n_modes
        hid = cfg.transport_hidden

        self.q_proj = nn.Sequential(
            nn.Linear(2 * M, hid),
            nn.SiLU(),
        )
        self.D_head = nn.Linear(hid, M)
        self.A_head = nn.Linear(hid, M)
        # Initialize so the kernel starts as NEAR-IDENTITY.
        #
        # Subtle but critical: softplus(0) = ln(2) ≈ 0.693, NOT zero.
        # If D_head.bias is zero-initialized, then D = softplus(0) ≈ 0.693
        # at step 0, and the damping becomes exp(-0.693 * ω_k²) which crushes
        # high-frequency modes before any learning happens. For the default
        # subbundle_dim=32 this means:
        #     Nyquist (ω² = π² ≈ 9.87):  damping ≈ 0.00108 per block
        #     After 6 blocks:            ≈ 10⁻¹⁸ (dead)
        #     ω² ≈ 2.5:                  damping ≈ 0.18, after 6 ≈ 3e-5 (dead)
        # The model then has to overcome its own init to recover high-freq
        # content, but softplus is bounded below by 0 so D can never be
        # driven negative. This is the same low-pass pathology that V5-V9's
        # "45% wall" exhibited, reintroduced as an init bug.
        #
        # Fix: shift D_head.bias to -6 so softplus(-6) ≈ 0.00248 ≈ 0 at init.
        # Damping at Nyquist is then ≈ 0.976 per block, ≈ 0.86 after 6 blocks —
        # all modes survive initialization and the model can learn to INCREASE
        # D if damping is useful.
        nn.init.zeros_(self.D_head.weight)
        nn.init.constant_(self.D_head.bias, -6.0)
        # A_head stays zero-init: gradient flow through A_head does NOT
        # depend on its current value (d(loss)/d(A_head) =
        # d(loss)/d(phase_shift) · q_hid · ω, non-zero at init because
        # downstream is sensitive to phase_shift). There's also no
        # symmetry trap because A_head's output is per-position-per-mode.
        # Keeping A_head at zero avoids injecting noise at init.
        nn.init.zeros_(self.A_head.weight)
        nn.init.zeros_(self.A_head.bias)

        # Per-mode base frequencies ω_k = 2π k / subbundle_dim, flattened
        # across subbundles.
        omegas = torch.zeros(M)
        for s in range(cfg.n_subbundles):
            off = s * cfg.spectral_half_dim
            omegas[off:off + cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0)
            )
        self.register_buffer("omega", omegas)          # (M,)
        self.register_buffer("omega_sq", omegas ** 2)  # (M,)

    def forward(self, c: Constellation) -> Constellation:
        B, T, M = c.mag.shape
        q_in = torch.cat([c.mag, c.phase], dim=-1)      # (B, T, 2M)
        q_hid = self.q_proj(q_in)
        D = F.softplus(self.D_head(q_hid))              # (B, T, M), ≥ 0
        A = self.A_head(q_hid)                           # (B, T, M)

        omega = self.omega.view(1, 1, M)
        omega_sq = self.omega_sq.view(1, 1, M)

        damping = torch.exp(-D * omega_sq)              # (B, T, M), ≤ 1
        phase_shift = A * omega                          # (B, T, M)

        new_mag = c.mag * damping
        new_phase = c.phase + phase_shift
        return Constellation(new_mag, new_phase)


# ─────────────────────────────────────────────────────────────────────────────
# 7. LearnedBandMask — per-block differentiable sparse frequency mask
# ─────────────────────────────────────────────────────────────────────────────

class LearnedBandMask(nn.Module):
    """Differentiable sparse mask over the n_modes frequency bands.

    One instance per V20Block. Initialized with a soft wavelet-like
    schedule: block $\\ell$ prefers the frequency band range
    $[\\ell / L, (\\ell + 1) / L)$ of the total mode count, so shallow
    blocks favor low frequencies and deep blocks favor high frequencies.
    An L1 penalty on the soft mask (applied via `cfg.band_mask_l1` in the
    training loss) pushes the mask toward sparsity.

    Output is a per-mode scalar in (0, 1) via sigmoid of the learned logits.
    Reads `cfg.n_modes` and `cfg.n_blocks` directly — no adapter shim.
    """

    def __init__(self, cfg: "V20Config", block_idx: int):
        super().__init__()
        self.n_modes = cfg.n_modes
        self.block_idx = block_idx

        # Wavelet-like init: the preferred window for this block is at
        # sigmoid(2) ≈ 0.88, everything else at sigmoid(-2) ≈ 0.12.
        logits = torch.full((cfg.n_modes,), -2.0)
        lo = int(math.floor(cfg.n_modes * block_idx / max(1, cfg.n_blocks)))
        hi = int(math.ceil(cfg.n_modes * (block_idx + 1) / max(1, cfg.n_blocks)))
        lo = max(0, min(cfg.n_modes - 1, lo))
        hi = max(lo + 1, min(cfg.n_modes, hi))
        logits[lo:hi] = 2.0
        self.logits = nn.Parameter(logits)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def l1_penalty(self) -> torch.Tensor:
        return self.forward().sum()


# ─────────────────────────────────────────────────────────────────────────────
# 8. SparseFFT / SparseIFFT  (Level 0: full rFFT/irFFT + LearnedBandMask)
# ─────────────────────────────────────────────────────────────────────────────

class SparseIFFT(nn.Module):
    """Constellation → spatial signal per subbundle.

    Level 0 implementation: run the full rFFT → inverse transform on every
    subbundle, then multiply by a band mask. This is correctness-equivalent
    to a true sparse inverse FFT; the speed-up from computing only the
    active bands is a follow-up (Level 1 / 2 in V20_DESIGN.md §IV).
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, c: Constellation, band_mask: torch.Tensor) -> torch.Tensor:
        """
        c: Constellation with mag, phase of shape (B, T, n_modes).
        band_mask: (n_modes,) soft mask in (0, 1) from LearnedBandMask.
        Returns: (B, T, fiber_dim) spatial signal.
        """
        B, T, M = c.mag.shape
        shd = self.cfg.spectral_half_dim
        nsub = self.cfg.n_subbundles
        sdim = self.cfg.subbundle_dim

        # Apply the band mask to the magnitudes (zeros suppress inactive bands)
        masked_mag = c.mag * band_mask.view(1, 1, M)
        # Build the complex spectral representation per subbundle
        complex_spec = (masked_mag * torch.exp(1j * c.phase)).reshape(B, T, nsub, shd)
        # Inverse FFT back to spatial
        spatial = torch.fft.irfft(complex_spec, n=sdim, dim=-1)   # (B, T, n_sub, sdim)
        return spatial.reshape(B, T, nsub * sdim)


class SparseFFT(nn.Module):
    """Spatial signal → constellation.

    Level 0: full rFFT on each subbundle, apply band mask, return
    (mag, phase).
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        spatial: torch.Tensor,
        band_mask: torch.Tensor,
    ) -> Constellation:
        """
        spatial: (B, T, fiber_dim)
        band_mask: (n_modes,)
        Returns: new Constellation with fresh mag, phase.
        """
        B, T, D = spatial.shape
        sdim = self.cfg.subbundle_dim
        nsub = self.cfg.n_subbundles
        shd = self.cfg.spectral_half_dim
        M = self.cfg.n_modes

        spatial_s = spatial.reshape(B, T, nsub, sdim)
        complex_spec = torch.fft.rfft(spatial_s, dim=-1)           # (B, T, n_sub, shd)
        complex_flat = complex_spec.reshape(B, T, M)

        new_mag = complex_flat.abs() * band_mask.view(1, 1, M)
        new_phase = complex_flat.angle()
        return Constellation(new_mag, new_phase)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PerSubbundleUnitaryDeltaFiber
# ─────────────────────────────────────────────────────────────────────────────

class PerSubbundleUnitaryDeltaFiber(nn.Module):
    """One SO(K) unitary-delta fiber per subbundle.

    Each subbundle s has:
      - Its own q, k, v projections: x_s → (q_s, k_s, v_s) each of size K
      - Its own content-dependent skew-symmetric projection: x_s → skew_s
      - Its own fiber state S_s ∈ ℝ^(K × K)
      - Its own output projection: read_s → out_s of size K

    where K = cfg.fiber_K = cfg.subbundle_dim (= 32 at default).

    Projections across subbundles are independent: we use a weight tensor
    of shape (n_sub, K_in, K_out) and an einsum 'btsi,sio->btso' so the
    subbundles don't share parameters.

    The scan is batched with the (B × n_sub) dimension as the outer scan
    dim, so all subbundles run through unitary_delta_parallel_scan in one
    call — compile-friendly.

    Operates on the SPATIAL subbundle signal (post-IFFT), not on the
    spectral constellation directly. This keeps the fiber transport in
    real arithmetic (no complex ops) and matches V12.1's block order:
    spectral transport first, then spatial associative memory, then the
    FFN.
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.n_sub = cfg.n_subbundles
        self.K = cfg.fiber_K            # subbundle_dim (spatial) at default
        self.ns = self.K * (self.K - 1) // 2

        # Per-subbundle projections. Init scale 0.1 (not 0.02) to avoid a
        # closed-gate collapse: the fiber read is q @ S where S ≈ k v^T, so
        # raw output is order (q·k·v·out_std) ≈ scale^4. At 0.02 that's
        # ~1e-7 per dim, times a closed gate sigmoid(-2)≈0.12, vs a spatial
        # residual of order 1 — the fiber cannot earn its own gradient
        # and never activates. 0.1 gives ~1e-4 per dim with gate = 0.5.
        def _pw(scale: float = 0.1) -> nn.Parameter:
            return nn.Parameter(torch.randn(self.n_sub, self.K, self.K) * scale)

        self.q_w = _pw()
        self.k_w = _pw()
        self.v_w = _pw()
        self.out_w = _pw()

        # Zero-init: U_step = I at init, fiber starts as plain DeltaNet.
        self.skew_w = nn.Parameter(torch.zeros(self.n_sub, self.K, self.ns))

        # sigmoid(0) = 0.5, the max-gradient point; lets the gate move
        # freely. sigmoid(-2) ≈ 0.12 would trap it (see _pw above).
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """
        spatial: (B, T, fiber_dim = n_sub * K)
        Returns: (B, T, fiber_dim) fiber read output.
        """
        B, T, D = spatial.shape
        n_sub, K, ns = self.n_sub, self.K, self.ns

        # (B, T, n_sub, K)
        x = spatial.reshape(B, T, n_sub, K)

        # Independent per-subbundle projections via batched einsum.
        # 'btsi,sio->btso' treats s (subbundle) as a batched group.
        q = torch.einsum("btsi,sio->btso", x, self.q_w)
        k = torch.einsum("btsi,sio->btso", x, self.k_w)
        v = torch.einsum("btsi,sio->btso", x, self.v_w)
        skew_params = torch.einsum("btsi,sio->btso", x, self.skew_w)
        # Scale down skew_params so Taylor-4 exp of skew stays accurate
        # (||A|| < 0.1 keeps fast_orthogonal within ~1e-8).
        skew_params = skew_params * 0.01

        # Normalize k so (k · S^T k) stays bounded in the delta rule
        k = F.normalize(k, dim=-1)

        # Build U_step per (B, T, n_sub) via skew -> fast_orthogonal
        skew_flat = skew_params.reshape(B * T * n_sub, ns)
        A_skew = make_skew_symmetric(skew_flat, K)
        U_step = fast_orthogonal(A_skew).reshape(B, T, n_sub, K, K)

        # Delta-rule effective transition and deposit in the unitary-conjugate
        # basis (see v19_modules.py UnitaryDeltaFiber for the derivation):
        #   eff_trans = U (I - k kᵀ)
        #   deposit   = U (k vᵀ)
        kkT = torch.einsum("btsi,btsj->btsij", k, k)
        I = torch.eye(K, device=x.device, dtype=x.dtype).view(1, 1, 1, K, K)
        eff_trans = torch.matmul(U_step, I - kkT)                 # (B, T, n_sub, K, K)
        kvT = torch.einsum("btsi,btsj->btsij", k, v)
        deposit = torch.matmul(U_step, kvT)                       # (B, T, n_sub, K, K)

        # Flatten (B, n_sub) → one outer scan dim
        N = B * n_sub
        eff_flat = eff_trans.permute(0, 2, 1, 3, 4).reshape(N, T, K, K)
        dep_flat = deposit.permute(0, 2, 1, 3, 4).reshape(N, T, K, K)
        S_all = unitary_delta_parallel_scan(eff_flat, dep_flat)
        S_all = S_all.reshape(B, n_sub, T, K, K)

        # Causal shift: read at position t must use state before writing t
        zero_state = torch.zeros(B, n_sub, 1, K, K, device=x.device, dtype=x.dtype)
        S_prev = torch.cat([zero_state, S_all[:, :, :-1]], dim=2)  # (B, n_sub, T, K, K)

        # Associative read: output = q @ S_prev
        q_perm = q.permute(0, 2, 1, 3)                              # (B, n_sub, T, K)
        out = torch.einsum("bsti,bstij->bstj", q_perm, S_prev)      # (B, n_sub, T, K)

        # Independent per-subbundle output projection
        out = out.permute(0, 2, 1, 3)                                # (B, T, n_sub, K)
        out = torch.einsum("btsi,sio->btso", out, self.out_w)
        out = out.reshape(B, T, n_sub * K)

        return torch.sigmoid(self.gate) * out


# ─────────────────────────────────────────────────────────────────────────────
# 10. SpatialMLP  (dominant nonlinearity)  +  11. PerSubbundleMLP (A10)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialMLP(nn.Module):
    """Per-token FFN in the spatial domain — the reaction term.

    V12.1 made SpatialMLP ~50% of its block parameter budget. V20 follows
    the same convention: standard 4× FFN with SiLU, applied position-wise
    on the spatial (post-IFFT) signal.

    This variant mixes across subbundles (the single Linear layer takes
    the full fiber_dim as input). V12.1 did the same. For strict orthogonal
    subbundle decomposition, swap for `PerSubbundleMLP` (A10 in the
    ablation matrix).
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        D = cfg.fiber_dim
        self.norm = nn.LayerNorm(D)
        hidden = D * cfg.ffn_mult
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, D),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PerSubbundleMLP(nn.Module):
    """Strictly orthogonal per-subbundle FFN — the A10 ablation variant.

    Replaces `SpatialMLP`'s single `fiber_dim -> hidden -> fiber_dim`
    Linear with `n_subbundles` independent MLPs, one per subbundle,
    applied via batched einsums with weight tensors of shape
    `(n_sub, K, hidden)` and `(n_sub, hidden, K)`. No information flows
    across subbundles at the FFN step, preserving the orthogonal
    subbundle decomposition end-to-end.

    Also uses a per-subbundle RMSNorm (`rms = √(⟨x²⟩ over K)`) instead of
    a full fiber_dim LayerNorm, because LayerNorm over the full fiber_dim
    would compute mean/variance across all subbundles and so leak
    information between them.

    Parameter count vs `SpatialMLP`:
        SpatialMLP      : 2 * D * D * ffn_mult + D * ffn_mult + D
                        = 2 * 256 * 1024 + 1024 + 256 = 525,824 at defaults
        PerSubbundleMLP : n_sub * (2 * K * K * ffn_mult + K * ffn_mult + K)
                        = 8 * (2 * 32 * 128 + 128 + 32) = 8 * 8352 = 66,816
                        ≈ 8× smaller at the same ffn_mult.

    This leaves ~460 k parameters per block on the table. If V20's design
    budget wants to keep SpatialMLP as the dominant nonlinearity, the
    per-subbundle variant should use a larger per-subbundle hidden dim
    (ffn_mult=16 gets close to parity) or accept the smaller budget as a
    structural tax for orthogonality.
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.n_sub = cfg.n_subbundles
        self.K = cfg.subbundle_dim
        self.hidden = self.K * cfg.ffn_mult

        # Per-subbundle RMS scale
        self.norm_scale = nn.Parameter(torch.ones(self.n_sub, self.K))

        # Per-subbundle MLP weights: each subbundle has its own (K, hidden)
        # and (hidden, K) projection. Initialized Kaiming-like per subbundle.
        self.w1 = nn.Parameter(
            torch.randn(self.n_sub, self.K, self.hidden) * (2.0 / self.K) ** 0.5
        )
        self.b1 = nn.Parameter(torch.zeros(self.n_sub, self.hidden))
        self.w2 = nn.Parameter(
            torch.randn(self.n_sub, self.hidden, self.K) * (2.0 / self.hidden) ** 0.5
        )
        self.b2 = nn.Parameter(torch.zeros(self.n_sub, self.K))

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, fiber_dim = n_sub * K)
        Returns: (B, T, fiber_dim) with a residual added per subbundle.
        """
        B, T, D = x.shape
        K = self.K
        n_sub = self.n_sub
        hidden = self.hidden

        # Per-subbundle reshape
        xs = x.reshape(B, T, n_sub, K)

        # Per-subbundle RMSNorm (only over K within each subbundle)
        rms = xs.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        normed = (xs / rms) * self.norm_scale.view(1, 1, n_sub, K)

        # First Linear per subbundle: (B,T,s,K) @ (s,K,hidden) -> (B,T,s,hidden)
        h = torch.einsum("btsi,sij->btsj", normed, self.w1)
        h = h + self.b1.view(1, 1, n_sub, hidden)
        h = F.silu(h)
        h = self.drop(h)

        # Second Linear per subbundle: (B,T,s,hidden) @ (s,hidden,K) -> (B,T,s,K)
        out = torch.einsum("btsj,sjk->btsk", h, self.w2)
        out = out + self.b2.view(1, 1, n_sub, K)
        out = self.drop(out)

        # Residual in subbundle-local coordinates, then flatten back
        return (xs + out).reshape(B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 12. ProximalTopK  (per-subbundle top-k by magnitude, hard mask)
# ─────────────────────────────────────────────────────────────────────────────

class ProximalTopK(nn.Module):
    """Keep only the top-k modes per subbundle (by magnitude).

    Hard-masking implementation, matching V12.1. Dropped modes get zero
    gradient and are re-evaluated at every forward pass, so a mode that
    was dropped at step t can come back at step t+1 if its magnitude
    crosses the threshold through weight updates on upstream parameters.

    Phase of a zero-magnitude mode is undefined, so we zero it alongside
    the magnitude.
    """

    def __init__(self, cfg: V20Config):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.active_modes_per_sub

    def forward(self, c: Constellation) -> Constellation:
        B, T, M = c.mag.shape
        nsub = self.cfg.n_subbundles
        shd = self.cfg.spectral_half_dim
        k = min(self.k, shd)

        mag_s = c.mag.reshape(B, T, nsub, shd)
        phase_s = c.phase.reshape(B, T, nsub, shd)

        # Top-k indices by magnitude along the mode axis
        _, topk_idx = torch.topk(mag_s.detach(), k=k, dim=-1)
        mask = torch.zeros_like(mag_s)
        mask.scatter_(-1, topk_idx, 1.0)
        mask = mask.detach()

        new_mag = (mag_s * mask).reshape(B, T, M)
        new_phase = (phase_s * mask).reshape(B, T, M)
        return Constellation(new_mag, new_phase)


# ─────────────────────────────────────────────────────────────────────────────
# 13. V20Block
# ─────────────────────────────────────────────────────────────────────────────

class V20Block(nn.Module):
    """One V20 block — the full spectral return.

    Flow:
      constellation
        → CloudNorm
        → SpectralTransportKernel      (V12.1 D_k(q), A_k(q))
        → SparseIFFT                    (to spatial, gated by LearnedBandMask)
        → +PerSubbundleUnitaryDeltaFiber (associative memory on spatial)
        → +SpatialMLP                   (reaction)
        → SparseFFT                     (back to spectral, gated by band mask)
      new constellation

    Design note: V20 *used* to apply a hard `ProximalTopK` at the end of
    every block to enforce per-subbundle top-$k$ sparsity. This was
    removed because (a) it cut gradients exactly to zero on the dropped
    modes, combined with the softplus-init damping to systematically kill
    the high-frequency subspace, and (b) it wasted compute on
    `torch.topk` + scatter per block at every forward. Sparsity is now
    enforced *softly* through two mechanisms only: the `LearnedBandMask`
    (which multiplies the spectrum by a sigmoid-bounded per-mode gate and
    carries an L1 penalty via `band_mask_l1`), and the natural damping
    from the `SpectralTransportKernel`'s `D_k(q)` head.

    `ProximalTopK` is still available in `v20_modules.py` as a module
    (the tests still exercise it) so it can be re-added as an ablation
    condition if needed.
    """

    def __init__(self, cfg: V20Config, block_idx: int):
        super().__init__()
        self.cfg = cfg
        self.cloud_norm = CloudNorm(cfg.n_modes)
        self.band_mask = LearnedBandMask(cfg, block_idx=block_idx)
        self.transport = SpectralTransportKernel(cfg)
        self.ifft = SparseIFFT(cfg)
        self.fiber = PerSubbundleUnitaryDeltaFiber(cfg)
        self.spatial_mlp = SpatialMLP(cfg)
        self.fft = SparseFFT(cfg)

    def forward(self, c: Constellation) -> Constellation:
        c = self.cloud_norm(c)
        band = self.band_mask()                           # (n_modes,)
        c = self.transport(c)
        spatial = self.ifft(c, band)                      # (B, T, fiber_dim)
        spatial = spatial + self.fiber(spatial)
        spatial = self.spatial_mlp(spatial)
        c = self.fft(spatial, band)
        return c

    def band_mask_l1(self) -> torch.Tensor:
        return self.band_mask.l1_penalty()


# ─────────────────────────────────────────────────────────────────────────────
# 14. V20Model
# ─────────────────────────────────────────────────────────────────────────────

class V20Model(nn.Module):
    def __init__(self, cfg: V20Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList(
            [V20Block(cfg, block_idx=i) for i in range(cfg.n_blocks)]
        )
        # Decode from the final constellation's spatial reconstruction.
        self.ifft_final = SparseIFFT(cfg)
        self.norm_f = nn.LayerNorm(cfg.fiber_dim)
        self.head = nn.Linear(cfg.fiber_dim, cfg.vocab_size)

        # A unit band mask for the final decoder — we want all modes
        # contributing to the prediction logits, not just the active ones.
        self.register_buffer("unit_band", torch.ones(cfg.n_modes))

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Returns (logits, aux):
          logits: (B, T-1, vocab) next-token logits for positions 0..T-2
          aux:    dict with band_mask_l1 for the training loss
        """
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)

        # Decode to spatial and project to logits
        spatial = self.ifft_final(c, self.unit_band)
        spatial = self.norm_f(spatial)
        logits = self.head(spatial)[:, :-1, :]

        band_l1 = sum(block.band_mask_l1() for block in self.blocks)
        return logits, {"band_mask_l1": band_l1}
