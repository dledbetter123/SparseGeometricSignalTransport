"""Unit tests for V19 modules.

Run with: python test_v19.py
or:       pytest test_v19.py -v

Tests cover:
  - shape correctness for every module
  - causality of the UnitaryDeltaFiber and CurvBiasAttention (future tokens
    must not affect past outputs)
  - correctness of unitary_delta_parallel_scan against a sequential reference
  - correctness of matrix_parallel_scan against a sequential reference
  - orthogonality of fast_orthogonal output for small skew inputs
  - LearnedBandMask initialization places the active band on the right block
  - V19Block end-to-end shape + causality
  - V19Model full forward pass and loss computability

Written to be runnable without pytest — each test is a function and the
bottom of the file runs them all if invoked directly.
"""
from __future__ import annotations

import math
import sys

import torch

from v19_modules import (
    V19Config,
    V19Block,
    V19Model,
    PrecisionEmbedding,
    VarianceUpdate,
    LearnedBandMask,
    GeometricContextAccum,
    UnitaryDeltaFiber,
    ChannelGate,
    ParsevalSpectralFilter,  # backwards-compat alias for ChannelGate
    CurvBiasAttention,
    FFN,
    matrix_parallel_scan,
    scalar_parallel_scan,
    unitary_delta_parallel_scan,
    make_skew_symmetric,
    fast_orthogonal,
    count_params,
)


def _tiny_cfg(**overrides) -> V19Config:
    base = dict(
        d_model=32,
        n_blocks=2,
        vocab_size=257,
        max_seq_len=16,
        dropout=0.0,
        fiber_heads=2,
        fiber_K=4,
        fiber_hidden_mult=1,
        ctx_n_bands=8,
        ctx_active_bands=4,
        parseval_hidden=16,
        curvbias_dim=16,
        ffn_mult=2,
        seq_len=16,
    )
    base.update(overrides)
    return V19Config(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Utility kernels
# ─────────────────────────────────────────────────────────────────────────────

def test_matrix_parallel_scan_matches_sequential() -> None:
    torch.manual_seed(0)
    N, T, d = 3, 8, 4
    A = torch.sigmoid(torch.randn(N, T))        # decays in (0, 1)
    B = torch.randn(N, T, d, d)
    # Sequential reference
    S_ref = torch.zeros(N, T, d, d)
    s = torch.zeros(N, d, d)
    for t in range(T):
        s = A[:, t].view(N, 1, 1) * s + B[:, t]
        S_ref[:, t] = s
    # Parallel
    S = matrix_parallel_scan(A.clone(), B.clone())
    err = (S - S_ref).abs().max().item()
    assert err < 1e-5, f"matrix_parallel_scan mismatch: {err}"


def test_scalar_parallel_scan_matches_sequential() -> None:
    torch.manual_seed(10)
    N, T = 4, 12
    A = torch.sigmoid(torch.randn(N, T))
    B = torch.randn(N, T)
    s_ref = torch.zeros(N, T)
    acc = torch.zeros(N)
    for t in range(T):
        acc = A[:, t] * acc + B[:, t]
        s_ref[:, t] = acc
    s = scalar_parallel_scan(A.clone(), B.clone())
    err = (s - s_ref).abs().max().item()
    assert err < 1e-5, f"scalar_parallel_scan mismatch: {err}"


def test_unitary_delta_parallel_scan_matches_sequential() -> None:
    torch.manual_seed(1)
    N, T, K = 3, 8, 4
    # Use small random U (not strictly orthogonal; the scan algebra only
    # depends on associativity, so this is a fine correctness test)
    U = torch.randn(N, T, K, K) * 0.3
    B = torch.randn(N, T, K, K)
    # Sequential reference: S[t] = U[t] @ S[t-1] + B[t], S[-1] = 0
    S_ref = torch.zeros(N, T, K, K)
    s = torch.zeros(N, K, K)
    for t in range(T):
        s = U[:, t] @ s + B[:, t]
        S_ref[:, t] = s
    # Parallel
    S = unitary_delta_parallel_scan(U.clone(), B.clone())
    err = (S - S_ref).abs().max().item()
    assert err < 1e-4, f"unitary_delta_parallel_scan mismatch: {err}"


def test_fast_orthogonal_preserves_norm_small_skew() -> None:
    torch.manual_seed(2)
    K = 6
    ns = K * (K - 1) // 2
    params = torch.randn(4, ns) * 0.05  # small to stay in Taylor accuracy range
    A = make_skew_symmetric(params, K)
    U = fast_orthogonal(A)  # (4, K, K)
    # Check ||U v|| ≈ ||v|| for a random unit vector
    v = torch.randn(4, K)
    v = v / v.norm(dim=-1, keepdim=True)
    Uv = torch.einsum("nij,nj->ni", U, v)
    norms = Uv.norm(dim=-1)
    # 4-term Taylor is not exactly orthogonal but for ||A|| ≈ 0.12 the error
    # should be < 5e-3 per component.
    err = (norms - 1.0).abs().max().item()
    assert err < 5e-3, f"fast_orthogonal norm drift: {err}"


def test_make_skew_symmetric_is_skew() -> None:
    torch.manual_seed(3)
    K = 5
    ns = K * (K - 1) // 2
    params = torch.randn(2, ns)
    A = make_skew_symmetric(params, K)
    # A + A^T should be zero
    err = (A + A.transpose(-1, -2)).abs().max().item()
    assert err < 1e-7, f"make_skew_symmetric failed: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# Component shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_precision_embedding_shape() -> None:
    cfg = _tiny_cfg()
    emb = PrecisionEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    x, log_var = emb(ids)
    assert x.shape == (2, cfg.seq_len, cfg.d_model)
    assert log_var.shape == (2, cfg.seq_len, cfg.d_model)


def test_variance_update_shape_and_clamp() -> None:
    vu = VarianceUpdate(32)
    lv = torch.randn(2, 8, 32) * 10  # out of clamp range
    ctx = torch.randn(2, 8, 32)
    out = vu(lv, ctx)
    assert out.shape == lv.shape
    assert out.min().item() >= -6.0001
    assert out.max().item() <= 2.0001


def test_learned_band_mask_init() -> None:
    cfg = _tiny_cfg(n_blocks=4, ctx_n_bands=16)
    # Each block should have its active window in a different portion of the
    # frequency spectrum (soft wavelet schedule).
    masks = []
    for i in range(cfg.n_blocks):
        m = LearnedBandMask(cfg, block_idx=i)
        masks.append(m().detach())
    peaks = [int(torch.argmax(m).item()) for m in masks]
    # Peaks should be monotonically non-decreasing across blocks
    for a, b in zip(peaks, peaks[1:]):
        assert a <= b, f"Band mask peaks not monotonic: {peaks}"


def test_geometric_context_accum_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    ctx = GeometricContextAccum(cfg)
    band = torch.ones(cfg.ctx_n_bands)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    y = ctx(x, band)
    assert y.shape == x.shape
    # Causality: changing x at position T-1 must not change y at positions < T-1.
    x_mod = x.clone()
    x_mod[:, -1] = torch.randn(2, cfg.d_model)
    y_mod = ctx(x_mod, band)
    err = (y_mod[:, :-1] - y[:, :-1]).abs().max().item()
    assert err < 1e-6, f"GeometricContextAccum not causal: max err {err}"


def test_unitary_delta_fiber_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    fiber = UnitaryDeltaFiber(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    y = fiber(x)
    assert y.shape == x.shape
    # Causality: modifying the last token must not affect outputs at < T-1
    x_mod = x.clone()
    x_mod[:, -1] = torch.randn(2, cfg.d_model)
    y_mod = fiber(x_mod)
    err = (y_mod[:, :-1] - y[:, :-1]).abs().max().item()
    assert err < 1e-5, f"UnitaryDeltaFiber not causal: max err {err}"


def test_channel_gate_shape_and_bound() -> None:
    cfg = _tiny_cfg()
    gate = ChannelGate(cfg)
    y = torch.randn(2, cfg.seq_len, cfg.d_model)
    ctx = torch.randn(2, cfg.seq_len, cfg.d_model)
    out = gate(y, ctx)
    assert out.shape == y.shape
    # Energy bound: gate is sigmoid(...) ∈ (0, 1) applied per channel per step,
    # so |out[b, t, d]| ≤ |y[b, t, d]| pointwise, hence ||out|| ≤ ||y||.
    e_in = y.pow(2).sum(dim=(-1, -2))
    e_out = out.pow(2).sum(dim=(-1, -2))
    ratio = (e_out / e_in.clamp(min=1e-8)).max().item()
    assert ratio <= 1.001, f"ChannelGate bound violated: ratio={ratio}"


def test_channel_gate_is_causal() -> None:
    """ChannelGate must be per-position, so perturbing x[:, -1] must not
    affect any output position <T-1."""
    cfg = _tiny_cfg()
    gate = ChannelGate(cfg)
    # Use a non-zero filter_proj so the test exercises the gate
    with torch.no_grad():
        gate.filter_proj.weight.copy_(torch.randn_like(gate.filter_proj.weight) * 0.1)
    y = torch.randn(2, cfg.seq_len, cfg.d_model)
    ctx = torch.randn(2, cfg.seq_len, cfg.d_model)
    out1 = gate(y, ctx)
    y_mod = y.clone()
    y_mod[:, -1] = torch.randn(2, cfg.d_model)
    ctx_mod = ctx.clone()
    ctx_mod[:, -1] = torch.randn(2, cfg.d_model)
    out2 = gate(y_mod, ctx_mod)
    err = (out2[:, :-1] - out1[:, :-1]).abs().max().item()
    assert err < 1e-6, f"ChannelGate not causal: max err {err}"


def test_curvbias_attention_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    attn = CurvBiasAttention(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    y = attn(x)
    assert y.shape == x.shape
    # Causality: modifying the last token must not affect outputs at < T-1
    x_mod = x.clone()
    x_mod[:, -1] = torch.randn(2, cfg.d_model)
    y_mod = attn(x_mod)
    err = (y_mod[:, :-1] - y[:, :-1]).abs().max().item()
    assert err < 1e-5, f"CurvBiasAttention not causal: max err {err}"


def test_ffn_shape() -> None:
    cfg = _tiny_cfg()
    ffn = FFN(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    y = ffn(x)
    assert y.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# Block and model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_v19_block_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    block = V19Block(cfg, block_idx=0)
    # Force non-zero weights in every sub-module so the test exercises real
    # signal, not just the zero-init state.
    with torch.no_grad():
        for p in block.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn_like(p) * 0.02)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    log_var = torch.zeros(2, cfg.seq_len, cfg.d_model)
    y, lv = block(x, log_var)
    assert y.shape == x.shape
    assert lv.shape == log_var.shape
    # Causality check: perturb last token
    x_mod = x.clone()
    x_mod[:, -1] = torch.randn(2, cfg.d_model)
    y_mod, _ = block(x_mod, log_var)
    # Stricter threshold now that the non-causal Parseval filter is gone.
    err = (y_mod[:, :-1] - y[:, :-1]).abs().max().item()
    assert err < 1e-4, f"V19Block not causal: max err {err}"


def test_v19_model_forward_and_loss() -> None:
    cfg = _tiny_cfg()
    model = V19Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, aux = model(ids)
    assert logits.shape == (2, cfg.seq_len - 1, cfg.vocab_size)
    assert "band_mask_l1" in aux
    assert aux["band_mask_l1"].dim() == 0  # scalar
    # Compute a real loss and backprop to verify the computation graph
    tgt = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
    )
    total = loss + cfg.band_mask_l1 * aux["band_mask_l1"]
    total.backward()
    # Every parameter should have a non-None grad (or be unused, but V19Block
    # uses every parameter in its forward).
    missing = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"Params without grad: {missing[:5]}"


def test_v19_model_param_count_reasonable() -> None:
    cfg = _tiny_cfg()
    model = V19Model(cfg)
    n = count_params(model)
    # Rough bounds: tiny config should be well under a million parameters.
    assert 1_000 < n < 2_000_000, f"Unreasonable param count: {n}"


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_matrix_parallel_scan_matches_sequential,
    test_scalar_parallel_scan_matches_sequential,
    test_unitary_delta_parallel_scan_matches_sequential,
    test_fast_orthogonal_preserves_norm_small_skew,
    test_make_skew_symmetric_is_skew,
    test_precision_embedding_shape,
    test_variance_update_shape_and_clamp,
    test_learned_band_mask_init,
    test_geometric_context_accum_shape_and_causality,
    test_unitary_delta_fiber_shape_and_causality,
    test_channel_gate_shape_and_bound,
    test_channel_gate_is_causal,
    test_curvbias_attention_shape_and_causality,
    test_ffn_shape,
    test_v19_block_shape_and_causality,
    test_v19_model_forward_and_loss,
    test_v19_model_param_count_reasonable,
]


def main() -> int:
    torch.manual_seed(42)
    failed = 0
    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    n = len(ALL_TESTS)
    print(f"\n{n - failed}/{n} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
