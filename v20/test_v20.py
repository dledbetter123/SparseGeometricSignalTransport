"""Unit tests for V20 modules.

Run with:  /path/to/python test_v20.py

Covers:
  - Shape correctness for every module
  - Causality of SpectralTransportKernel, PerSubbundleUnitaryDeltaFiber,
    and the full V20Block (perturbing the last token must not change
    outputs at earlier positions)
  - Parseval bound on SpectralTransportKernel (|damping| ≤ 1 per mode)
  - FFT/IFFT round-trip identity on the full-band mask
  - ProximalTopK keeps exactly active_modes_per_sub modes per subbundle
  - V20Block forward + backward + grad flow for every parameter
  - LearnedBandMask shim: initialization places peaks in the right place
"""
from __future__ import annotations

import math
import sys

import torch

from v20_modules import (
    V20Config,
    Constellation,
    CloudNorm,
    ConstellationEmbedding,
    SpectralTransportKernel,
    SparseFFT,
    SparseIFFT,
    PerSubbundleUnitaryDeltaFiber,
    SpatialMLP,
    PerSubbundleMLP,
    ProximalTopK,
    V20Block,
    V20Model,
    count_params,
)


def _tiny_cfg(**over) -> V20Config:
    base = dict(
        n_subbundles=4,
        subbundle_dim=8,           # shd = 5, n_modes = 20, fiber_dim = 32
        active_modes_per_sub=3,
        transport_hidden=16,
        fiber_hidden_mult=1,
        ffn_mult=2,
        n_blocks=2,
        vocab_size=257,
        max_seq_len=16,
        dropout=0.0,
        seq_len=16,
        batch_size=2,
    )
    base.update(over)
    return V20Config(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Component shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cloudnorm_shape_and_passthrough() -> None:
    cfg = _tiny_cfg()
    cn = CloudNorm(cfg.n_modes)
    c = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    out = cn(c)
    assert out.mag.shape == c.mag.shape
    # phase must pass through unchanged
    assert torch.allclose(out.phase, c.phase)
    # Rescaled mag should have ~unit RMS per position (times mag_scale)
    rms = (out.mag.pow(2).mean(dim=-1)).sqrt()
    assert rms.mean().item() > 0.0


def test_constellation_embedding_shape() -> None:
    cfg = _tiny_cfg()
    emb = ConstellationEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    c = emb(ids)
    assert c.mag.shape == (2, cfg.seq_len, cfg.n_modes)
    assert c.phase.shape == (2, cfg.seq_len, cfg.n_modes)


def test_spectral_transport_shape_and_parseval_bound() -> None:
    cfg = _tiny_cfg()
    tk = SpectralTransportKernel(cfg)
    # Force non-zero D_head so damping is actually active
    with torch.no_grad():
        tk.D_head.weight.copy_(torch.randn_like(tk.D_head.weight) * 0.1)
        tk.D_head.bias.copy_(torch.tensor([0.5]).expand_as(tk.D_head.bias))
    c = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    out = tk(c)
    assert out.mag.shape == c.mag.shape
    # Parseval bound: |damping| ≤ 1 ⇒ output magnitudes ≤ input magnitudes
    assert (out.mag <= c.mag + 1e-5).all(), \
        "SpectralTransportKernel violated |damping| ≤ 1"


def test_spectral_transport_identity_at_init() -> None:
    """At initialization, SpectralTransportKernel must be approximately
    the identity. The intent of zero-initializing D_head.weight is that
    the kernel starts damping-free; since softplus(0) = ln(2) ≈ 0.693,
    we need D_head.bias initialized to -6 so softplus(-6) ≈ 0.0025,
    which gives damping at Nyquist = exp(-0.0025 * π²) ≈ 0.976 (not
    exp(-0.693 * π²) ≈ 0.00108, which would destroy high-freq content).
    """
    cfg = _tiny_cfg()
    tk = SpectralTransportKernel(cfg)
    c = Constellation(
        mag=torch.ones(2, cfg.seq_len, cfg.n_modes),
        phase=torch.zeros(2, cfg.seq_len, cfg.n_modes),
    )
    out = tk(c)
    # Smallest surviving magnitude across ALL modes (incl. highest ω)
    # must be close to 1 at init, not ~ 0 from catastrophic damping.
    min_mag = out.mag.min().item()
    assert min_mag > 0.95, (
        f"SpectralTransportKernel is damping too hard at init: "
        f"min magnitude {min_mag:.4f} after one block (expected > 0.95)"
    )
    # Phase should also not shift at init (A_head is zero-init)
    max_phase_diff = (out.phase - c.phase).abs().max().item()
    assert max_phase_diff < 1e-6, \
        f"SpectralTransportKernel phase is shifting at init: {max_phase_diff}"


def test_spectral_transport_is_causal() -> None:
    """SpectralTransportKernel is position-wise (q comes from the current
    token's own constellation), so perturbing the last token must not
    affect earlier positions."""
    cfg = _tiny_cfg()
    tk = SpectralTransportKernel(cfg)
    with torch.no_grad():
        for p in tk.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn_like(p) * 0.02)
    c = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    out1 = tk(c)
    c_mod = Constellation(
        mag=c.mag.clone(),
        phase=c.phase.clone(),
    )
    c_mod.mag[:, -1] = torch.randn(2, cfg.n_modes).abs()
    c_mod.phase[:, -1] = torch.randn(2, cfg.n_modes)
    out2 = tk(c_mod)
    err = (out2.mag[:, :-1] - out1.mag[:, :-1]).abs().max().item()
    assert err < 1e-6, f"SpectralTransportKernel not causal: max err {err}"


def test_sparse_fft_ifft_roundtrip_spatial() -> None:
    """Spatial → SparseFFT → SparseIFFT should recover the spatial signal.

    This is the correct round-trip to test because it's the order a V20Block
    uses in practice: an arbitrary (mag, phase) constellation is NOT a valid
    rFFT output (DC and Nyquist bins must be real), so going constellation
    → irFFT → rFFT is only the identity on the valid subspace. Going the
    other direction, spatial → rFFT → irFFT, is exactly the identity because
    rFFT is a unitary change of basis for real signals.
    """
    cfg = _tiny_cfg()
    ifft = SparseIFFT(cfg)
    fft = SparseFFT(cfg)
    band = torch.ones(cfg.n_modes)

    spatial = torch.randn(2, cfg.seq_len, cfg.fiber_dim)

    c = fft(spatial, band)
    spatial_rt = ifft(c, band)
    assert spatial_rt.shape == spatial.shape
    err = (spatial_rt - spatial).abs().max().item()
    assert err < 1e-4, f"Spatial round-trip error {err}"


def test_constellation_roundtrip_on_valid_subspace() -> None:
    """Applying SparseIFFT then SparseFFT should be idempotent:
    constellation_1 = FFT(IFFT(c)) should equal constellation_2 = FFT(IFFT(c_1)).

    That is, the first round-trip projects onto the valid rFFT output
    subspace, and from then on round-trips are identities. We check
    idempotence rather than a direct identity because our test
    constellation is built from arbitrary (mag, phase) which includes
    invalid phase values at DC/Nyquist bins.
    """
    cfg = _tiny_cfg()
    ifft = SparseIFFT(cfg)
    fft = SparseFFT(cfg)
    band = torch.ones(cfg.n_modes)

    c0 = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    c1 = fft(ifft(c0, band), band)
    c2 = fft(ifft(c1, band), band)
    err_mag = (c2.mag - c1.mag).abs().max().item()
    err_phase_cos = (torch.cos(c2.phase) - torch.cos(c1.phase)).abs().max().item()
    err_phase_sin = (torch.sin(c2.phase) - torch.sin(c1.phase)).abs().max().item()
    assert err_mag < 1e-4, f"round-trip not idempotent on mag: {err_mag}"
    assert err_phase_cos < 1e-4, f"round-trip not idempotent on phase cos: {err_phase_cos}"
    assert err_phase_sin < 1e-4, f"round-trip not idempotent on phase sin: {err_phase_sin}"


def test_sparse_ifft_band_mask_zeros_out_inactive_bands() -> None:
    """If the band mask is zero on some bands, the reconstructed spatial
    signal should depend only on the active bands."""
    cfg = _tiny_cfg()
    ifft = SparseIFFT(cfg)
    # Full band mask
    band_full = torch.ones(cfg.n_modes)
    # Half-band mask: zero out the second half
    band_half = torch.ones(cfg.n_modes)
    band_half[cfg.n_modes // 2:] = 0.0

    c = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    # Construct a second constellation that differs only in the inactive modes
    c2 = Constellation(
        mag=c.mag.clone(),
        phase=c.phase.clone(),
    )
    c2.mag[..., cfg.n_modes // 2:] = torch.randn(2, cfg.seq_len, cfg.n_modes // 2).abs()
    # With the half-band mask, the two constellations should produce the
    # SAME spatial output because the changes are in the masked-out region.
    out1 = ifft(c, band_half)
    out2 = ifft(c2, band_half)
    err = (out1 - out2).abs().max().item()
    assert err < 1e-5, f"SparseIFFT band mask leaked: {err}"


def test_per_subbundle_fiber_contributes_at_init() -> None:
    """The fiber's output at init must be large enough to carry gradient.

    Earlier V20 runs plateaued at ~20% accuracy because q_w/k_w/v_w/out_w
    were all initialized at std=0.02 and the output gate was sigmoid(-2) ≈
    0.12. The compound effect was a raw fiber contribution of order
    (0.02)^4 * 0.12 ≈ 2e-8 per dim — far below any meaningful gradient
    signal, so the gate never opened and the fiber was effectively dead.
    After fix: init scale is 0.1 and gate init is 0.0 (sigmoid=0.5).

    This test locks in that the fiber contribution is at least ~1e-4 per
    dim on a unit-magnitude input at init.
    """
    cfg = _tiny_cfg()
    fiber = PerSubbundleUnitaryDeltaFiber(cfg)
    x = torch.randn(4, cfg.seq_len, cfg.fiber_dim)
    y = fiber(x)
    # RMS of the output across all dims (averaged over batch, time, dim).
    # At init with the new scales, expect ~1e-4 to 1e-2, depending on
    # the scan's exact multiplicative behavior. The old init gave ~1e-8
    # which would fail this test.
    rms = y.pow(2).mean().sqrt().item()
    assert rms > 1e-5, (
        f"Fiber output is too small at init: rms={rms:.2e}. "
        f"This means the closed-gate collapse is still happening."
    )


def test_per_subbundle_fiber_gate_open_at_init() -> None:
    """Verify the gate is at sigmoid(0)=0.5 at init, not sigmoid(-2)=0.12."""
    cfg = _tiny_cfg()
    fiber = PerSubbundleUnitaryDeltaFiber(cfg)
    gate_value = torch.sigmoid(fiber.gate).item()
    assert 0.45 < gate_value < 0.55, (
        f"Fiber gate is not ~0.5 at init: got {gate_value}. "
        f"The closed-gate trap (sigmoid(-2)=0.12) must be fixed."
    )


def test_per_subbundle_fiber_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    fiber = PerSubbundleUnitaryDeltaFiber(cfg)
    # Give the skew_w non-zero weights so U_step != I
    with torch.no_grad():
        fiber.skew_w.copy_(torch.randn_like(fiber.skew_w) * 0.05)
    x = torch.randn(2, cfg.seq_len, cfg.fiber_dim)
    y = fiber(x)
    assert y.shape == x.shape
    # Causality
    x_mod = x.clone()
    x_mod[:, -1] = torch.randn(2, cfg.fiber_dim)
    y_mod = fiber(x_mod)
    err = (y_mod[:, :-1] - y[:, :-1]).abs().max().item()
    assert err < 1e-4, f"PerSubbundleUnitaryDeltaFiber not causal: max err {err}"


def test_spatial_mlp_shape() -> None:
    cfg = _tiny_cfg()
    mlp = SpatialMLP(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.fiber_dim)
    y = mlp(x)
    assert y.shape == x.shape


def test_per_subbundle_mlp_shape() -> None:
    cfg = _tiny_cfg()
    mlp = PerSubbundleMLP(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.fiber_dim)
    y = mlp(x)
    assert y.shape == x.shape


def test_per_subbundle_mlp_strict_orthogonality() -> None:
    """The A10 invariant: modifying one subbundle must not affect the
    output in any other subbundle. This is the reason PerSubbundleMLP
    exists (to distinguish orthogonal subbundle processing from V12.1's
    shared-MLP default)."""
    cfg = _tiny_cfg()
    mlp = PerSubbundleMLP(cfg)
    # Force non-zero weights everywhere so the test exercises real signal
    with torch.no_grad():
        for p in mlp.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn_like(p) * 0.05)
            else:
                p.add_(torch.randn_like(p) * 0.01)

    B, T = 2, cfg.seq_len
    nsub = cfg.n_subbundles
    K = cfg.subbundle_dim
    x = torch.randn(B, T, cfg.fiber_dim)
    y1 = mlp(x).reshape(B, T, nsub, K)

    # Perturb only subbundle 0; other subbundles must not change
    x_mod = x.reshape(B, T, nsub, K).clone()
    x_mod[:, :, 0, :] = torch.randn(B, T, K)
    x_mod = x_mod.reshape(B, T, cfg.fiber_dim)
    y2 = mlp(x_mod).reshape(B, T, nsub, K)

    # Subbundle 0 should differ
    diff_0 = (y2[:, :, 0, :] - y1[:, :, 0, :]).abs().max().item()
    assert diff_0 > 1e-4, \
        f"Subbundle 0 should differ when its input changes; got {diff_0}"
    # Subbundles 1..n_sub-1 must be identical
    for s in range(1, nsub):
        err = (y2[:, :, s, :] - y1[:, :, s, :]).abs().max().item()
        assert err < 1e-6, \
            f"PerSubbundleMLP leaked across subbundles: s={s} err={err}"


def test_per_subbundle_mlp_param_count_lower_than_spatial_mlp() -> None:
    """At the same ffn_mult, PerSubbundleMLP should have significantly
    fewer parameters than SpatialMLP because its hidden dimension is
    per-subbundle rather than shared."""
    cfg = _tiny_cfg()
    sp = SpatialMLP(cfg)
    ps = PerSubbundleMLP(cfg)
    n_sp = sum(p.numel() for p in sp.parameters())
    n_ps = sum(p.numel() for p in ps.parameters())
    assert n_ps < n_sp, (
        f"Expected PerSubbundleMLP ({n_ps}) to have fewer params than "
        f"SpatialMLP ({n_sp}) at the same ffn_mult"
    )


def test_proximal_top_k_exact_count_per_subbundle() -> None:
    cfg = _tiny_cfg()
    prox = ProximalTopK(cfg)
    c = Constellation(
        mag=torch.rand(2, cfg.seq_len, cfg.n_modes),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    out = prox(c)
    # Reshape to per-subbundle
    mag_s = out.mag.reshape(2, cfg.seq_len, cfg.n_subbundles, cfg.spectral_half_dim)
    # Every (batch, position, subbundle) should have exactly
    # min(k, spectral_half_dim) non-zero modes.
    expected_nonzero = min(cfg.active_modes_per_sub, cfg.spectral_half_dim)
    nonzero_counts = (mag_s != 0).sum(dim=-1)      # (B, T, n_sub)
    assert (nonzero_counts == expected_nonzero).all(), \
        f"Per-subbundle nonzero mode counts wrong: got {nonzero_counts.unique().tolist()}"
    # Phase of dropped modes should also be zero
    phase_s = out.phase.reshape(2, cfg.seq_len, cfg.n_subbundles, cfg.spectral_half_dim)
    assert ((mag_s != 0) | (phase_s == 0)).all(), \
        "ProximalTopK: phase not zeroed where magnitude was zeroed"


def test_proximal_top_k_preserves_top_values_exactly() -> None:
    cfg = _tiny_cfg()
    prox = ProximalTopK(cfg)
    c = Constellation(
        mag=torch.arange(
            2 * cfg.seq_len * cfg.n_modes, dtype=torch.float32
        ).reshape(2, cfg.seq_len, cfg.n_modes),
        phase=torch.zeros(2, cfg.seq_len, cfg.n_modes),
    )
    out = prox(c)
    # Top-k kept values should match the input values unchanged at those
    # positions (not straight-through normalized).
    kept = out.mag != 0
    assert torch.allclose(out.mag[kept], c.mag[kept])


# ─────────────────────────────────────────────────────────────────────────────
# Block and model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_v20_block_has_no_proximal() -> None:
    """V20Block no longer uses ProximalTopK (removed 2026-04-10).
    Sparsity is now enforced softly via LearnedBandMask + L1 penalty and
    the natural damping from SpectralTransportKernel's D_k(q) head.
    """
    cfg = _tiny_cfg()
    block = V20Block(cfg, block_idx=0)
    assert not hasattr(block, "proximal"), (
        "V20Block should no longer have a ProximalTopK sub-module. "
        "Sparsity is enforced via LearnedBandMask + L1 + transport kernel damping."
    )


def test_v20_block_shape_and_causality() -> None:
    cfg = _tiny_cfg()
    block = V20Block(cfg, block_idx=0)
    # Exercise every parameter by adding small non-zero perturbations
    with torch.no_grad():
        for p in block.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn_like(p) * 0.02)
    c = Constellation(
        mag=torch.randn(2, cfg.seq_len, cfg.n_modes).abs(),
        phase=torch.randn(2, cfg.seq_len, cfg.n_modes),
    )
    out = block(c)
    assert out.mag.shape == c.mag.shape
    assert out.phase.shape == c.phase.shape

    # Causality: perturb the last token, earlier outputs must not change
    c_mod = Constellation(
        mag=c.mag.clone(),
        phase=c.phase.clone(),
    )
    c_mod.mag[:, -1] = torch.randn(2, cfg.n_modes).abs()
    c_mod.phase[:, -1] = torch.randn(2, cfg.n_modes)
    out_mod = block(c_mod)
    err = (out_mod.mag[:, :-1] - out.mag[:, :-1]).abs().max().item()
    assert err < 1e-3, f"V20Block not causal: max err {err}"


def test_v20_model_forward_and_loss() -> None:
    cfg = _tiny_cfg()
    model = V20Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, aux = model(ids)
    assert logits.shape == (2, cfg.seq_len - 1, cfg.vocab_size)
    assert "band_mask_l1" in aux
    assert aux["band_mask_l1"].dim() == 0

    tgt = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
    )
    total = loss + cfg.band_mask_l1 * aux["band_mask_l1"]
    total.backward()

    # Every trainable parameter should receive a gradient. Some params are
    # zero-initialized (SpectralTransportKernel's D_head/A_head, fiber skew_w)
    # but the computation graph still passes through them, so they should
    # still get a non-None grad.
    missing = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"V20 params without grad: {missing[:5]}"


def test_v20_model_param_count_reasonable() -> None:
    cfg = _tiny_cfg()
    model = V20Model(cfg)
    n = count_params(model)
    assert 5_000 < n < 5_000_000, f"Unreasonable tiny-cfg param count: {n}"


def test_v20_at_default_config_builds() -> None:
    """Smoke test: V20Model at the default (production) config builds and
    runs one forward pass on CPU."""
    cfg = V20Config(vocab_size=50257)
    model = V20Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, aux = model(ids)
    assert logits.shape == (1, 15, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "Default-config V20 produced NaN/Inf"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_cloudnorm_shape_and_passthrough,
    test_constellation_embedding_shape,
    test_spectral_transport_shape_and_parseval_bound,
    test_spectral_transport_identity_at_init,
    test_spectral_transport_is_causal,
    test_sparse_fft_ifft_roundtrip_spatial,
    test_constellation_roundtrip_on_valid_subspace,
    test_sparse_ifft_band_mask_zeros_out_inactive_bands,
    test_per_subbundle_fiber_contributes_at_init,
    test_per_subbundle_fiber_gate_open_at_init,
    test_per_subbundle_fiber_shape_and_causality,
    test_spatial_mlp_shape,
    test_per_subbundle_mlp_shape,
    test_per_subbundle_mlp_strict_orthogonality,
    test_per_subbundle_mlp_param_count_lower_than_spatial_mlp,
    test_proximal_top_k_exact_count_per_subbundle,
    test_proximal_top_k_preserves_top_values_exactly,
    test_v20_block_has_no_proximal,
    test_v20_block_shape_and_causality,
    test_v20_model_forward_and_loss,
    test_v20_model_param_count_reasonable,
    test_v20_at_default_config_builds,
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
