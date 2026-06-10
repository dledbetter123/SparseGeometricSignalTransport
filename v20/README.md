# V20 — The Spectral Return

V20 returns the architectural line to the V12.1 spectral-constellation
direction with V19's non-abelian transport and speed infrastructure
grafted on. `CurvBias` is relegated to a diagnostic baseline; the main
contribution is back to *the spectral architecture itself*.

See `V20_DESIGN.md` for the full retrospective of V5–V19, the explicit
reasons each V12.1 piece was dropped in V19 and why each is now back,
and the V12.2-style ablation protocol that V20 is designed to answer.

## Files

| File | Purpose |
|------|---------|
| `V20_DESIGN.md` | Full design document (9 parts + 2 appendices) |
| `v20_modules.py` | All core PyTorch modules, imports utility kernels from `../v19/v19_modules.py` |
| `test_v20.py` | 15 unit tests: shape, causality, Parseval bound, FFT round-trip, top-k correctness, grad flow |
| `README.md` | this file |

**Not yet written** (Part IX of V20_DESIGN.md):
- `benchmark_v20.py`
- `gen_notebook_v20.py`
- `architecture_v20.ipynb`

## Quick start

```bash
# All unit tests (requires torch; miniconda python on this machine)
/Users/davidledbetter/miniconda3/bin/python test_v20.py
```

Expected output:

```
  PASS  test_cloudnorm_shape_and_passthrough
  PASS  test_constellation_embedding_shape
  PASS  test_spectral_transport_shape_and_parseval_bound
  PASS  test_spectral_transport_is_causal
  PASS  test_sparse_fft_ifft_roundtrip_spatial
  PASS  test_constellation_roundtrip_on_valid_subspace
  PASS  test_sparse_ifft_band_mask_zeros_out_inactive_bands
  PASS  test_per_subbundle_fiber_shape_and_causality
  PASS  test_spatial_mlp_shape
  PASS  test_proximal_top_k_exact_count_per_subbundle
  PASS  test_proximal_top_k_preserves_top_values_exactly
  PASS  test_v20_block_shape_and_causality
  PASS  test_v20_model_forward_and_loss
  PASS  test_v20_model_param_count_reasonable
  PASS  test_v20_at_default_config_builds

15/15 tests passed
```

## The architectural vision

V20 encodes the framing that came out of our design conversation:

> Each token is a sparse spectral constellation — a learned optimal
> fingerprint distributed across $K$ orthogonal subbundles. Tokens
> interact through shared mode activations during content-dependent
> non-abelian transport along the sequence. The accumulated per-subbundle
> fiber state is the dynamic associative memory. Word order is
> structurally encoded because the transport group is non-abelian.
> There is no stored Hopfield bank: the fiber state *is* the bank,
> built per-sequence, read by the current token's constellation,
> exactly analogous to attention's KV cache in spectral space.

The concrete components that realize this vision:

| Component | Role |
|-----------|------|
| `ConstellationEmbedding` | Token → learned (mag, phase, log_var) per mode; V17's constellation embedding verbatim |
| `CloudNorm` | RMS-normalize magnitudes, learnable per-mode rescaling |
| `SpectralTransportKernel` | V12.1's content-dependent $\exp(-D_k(q)\omega_k^2 - iA_k(q)\omega_k)$; Parseval bound enforced by $D \geq 0$ via softplus |
| `SparseIFFT` / `SparseFFT` | Level 0 (full rFFT + band mask) spectral-to-spatial round trip |
| `PerSubbundleUnitaryDeltaFiber` | One $SO(K)$ unitary-delta fiber per subbundle with independent q/k/v/skew projections; dynamic associative memory |
| `SpatialMLP` | Dominant nonlinearity; 4× FFN on the spatial reconstruction |
| `ProximalTopK` | Hard top-$k$ per subbundle — the invariant that keeps the representation *actually* sparse in frequency space |
| `LearnedBandMask` | Per-block differentiable frequency-band sparse mask (imported from V19 via a config shim) |
| `V20Block` | Full block: CloudNorm → SpectralTransportKernel → SparseIFFT → +Fiber → SpatialMLP → SparseFFT → ProximalTopK |
| `V20Model` | Stack of `V20Block`s with `ConstellationEmbedding` in front and a `SparseIFFT` + linear decoder at the end |

## Default config

```python
V20Config(
    n_subbundles=8,             # orthogonal feature channels
    subbundle_dim=32,           # spatial size per subbundle (rFFT-friendly)
    # derived:
    #   spectral_half_dim = 17
    #   n_modes = 8 * 17 = 136
    #   fiber_dim = 8 * 32 = 256
    active_modes_per_sub=8,     # proximal top-k keeps 8 / 17 modes per subbundle
    transport_hidden=128,
    ffn_mult=4,
    n_blocks=6,
)
```

At this default: **~31 M total parameters**, **8,192 state values per
block** (`n_subbundles × fiber_K² = 8 × 32 × 32`), state is ~4× V18's 1024
but still ~16× smaller than attention's KV cache at T=256.

Per-block parameter breakdown:

| Component | Params | % of block |
|-----------|-------:|-----------:|
| `CloudNorm` | 136 | 0.0% |
| `LearnedBandMask` | 136 | 0.0% |
| `SpectralTransportKernel` | 87 440 | 18.8% |
| `PerSubbundleUnitaryDeltaFiber` | 159 745 | 34.4% |
| `SpatialMLP` | 526 080 | 113.3% of block\* |
| (others) | 0 | 0.0% |

\* Per-block total is 773 537 ≈ 4.64 M across 6 blocks; `SpatialMLP` is
the dominant nonlinearity by design, as in V12.1.

## What V20 is and isn't

**V20 is:**
- a spectral architecture: tokens live as sparse complex constellations
- non-abelian: per-subbundle $SO(K)$ transport, not V12.1's abelian $U(1)^M$
- structurally sparse in Fourier space: `ProximalTopK` every block
- compatible with V19's speed infrastructure: vectorized `make_skew_symmetric`,
  `unitary_delta_parallel_scan`, `torch.compile`, `bf16` autocast
- directly comparable to V12.1 via the ablation matrix in `V20_DESIGN.md` §V

**V20 is not:**
- a Hopfield memory bank of stored sequence prototypes (the fiber state
  is the only associative memory)
- a dense-vector model (V18/V19 are dense; V20 is spectral throughout)
- a CurvBias variant (`CurvBiasAttention` stays in V19 for the baseline)
- yet running at production scale (MVP is 31 M params; scaling is a
  follow-up)

## Changelog

**2026-04-10 — Expressivity fixes after first H100 run**

- **`SpectralTransportKernel.D_head.bias` now initialized to −6** (was 0).
  `softplus(0) = ln(2) ≈ 0.693`, which was producing near-identity *damping*
  at init (`exp(-0.693 · ω²)`) and crushing high-frequency modes before any
  learning happened. At `subbundle_dim = 32`, mode `ω = π` was being damped
  to `0.00108` per block and to `≈10⁻¹⁸` after 6 blocks — the representation
  was effectively low-pass filtered at step 0, with no way for the model to
  recover because softplus has a floor at 0. Fixed by biasing so that
  `softplus(−6) ≈ 0.0025 ≈ 0` at init; damping at Nyquist is now `≈ 0.976`
  per block, `≈ 0.86` after 6 blocks. Locked in by
  `test_spectral_transport_identity_at_init`.

- **`ProximalTopK` removed from `V20Block`.** The hard top-$k$ mask was
  cutting gradients exactly to zero for dropped modes, which combined with
  the softplus-init bug above to systematically kill the high-frequency
  subspace. It also wasted compute on `torch.topk` + `scatter` per block.
  Sparsity is now enforced softly via `LearnedBandMask` (+ L1 penalty) and
  the natural damping from `SpectralTransportKernel`. The `ProximalTopK`
  class is still in `v20_modules.py` for possible ablation use. Locked in
  by `test_v20_block_has_no_proximal`.

## Known caveats and follow-ups

1. **Sparse FFT is at Level 0.** `SparseIFFT` and `SparseFFT` both compute
   the full rFFT on every subbundle and then multiply by the band mask.
   This is correctness-equivalent to a true sparse FFT but has the same
   compute cost as a dense FFT. If benchmarking shows FFT cost
   dominates, Level 1 (batched per-subbundle sparse FFT) is the next
   step; Level 2 (true $O(s \log s)$ kernel) is the long-tail follow-up.
   See `V20_DESIGN.md` §IV.

2. **No `VarianceUpdate` yet.** `log_var` passes through the blocks
   unchanged. V17's learned variance evolution can be added back once
   the core architecture is validated; for the MVP we want the minimum
   number of moving parts.

3. **Transport context is per-token, not running.** `SpectralTransportKernel`
   computes $D_k(q)$ and $A_k(q)$ from the current token's constellation,
   not from an accumulated running context. If ablations show this is
   insufficient, the follow-up is a cheap EMA of the fiber read over
   time, used as $q$ for the next block's transport kernel. V12.1 used
   an explicit SSM for this, but V14 showed the SSM itself was not
   load-bearing.

4. **`ProximalTopK` is hard masking.** Dropped modes get zero gradient.
   Modes can still come back through upstream weight updates because
   the top-k is recomputed every forward pass, but this is not the only
   reasonable choice — `α-entmax` or `sparsemax` would give a
   soft-differentiable version. V12.1 used hard masking, so V20 starts
   there.

5. **The `LearnedBandMask` shim.** V20Block instantiates V19's
   `LearnedBandMask` via a tiny `_LearnedBandMaskShim` so we don't
   fork the module or edit `v19_modules.py`. The shim exposes
   `ctx_n_bands = cfg.n_modes` and `n_blocks = cfg.n_blocks` because
   those are the only two attributes `LearnedBandMask.__init__` reads.

6. **Content-dependent diffusion rate $D_k(q)$ stability.** $D$ goes
   through `softplus` so it's non-negative; `exp(-D ω²) ∈ (0, 1]` so the
   Parseval bound holds. If training is unstable, a max-clamp on
   $D_k(q)$ is the mitigation.

## Next steps (from `V20_DESIGN.md` §IX)

1. ✅ Write `v20_modules.py` — done
2. ✅ Write `test_v20.py` — done (15 tests passing)
3. ✅ Run unit tests — done
4. Write `benchmark_v20.py` — measure FFT cost fraction, decide Level 0
   vs Level 1
5. Write `gen_notebook_v20.py` — the full A0–A9 ablation matrix from
   `V20_DESIGN.md` §V
6. Syntax-check the notebook
7. Run A2 (SSM+MLP) and A8 (V20 full) on H100 as the first comparison;
   if A8 wins, run the rest of the matrix; if it loses, implement V20.1
   with a small learned prototype bank and re-run A8 vs A2 only

## Acceptance criteria

From `V20_DESIGN.md` §VIII, V20 is a success if any of:

1. **A8 beats A1** — V20 beats GPT + CurvBias at matched wall-clock
2. **A8 beats A2 by ≥ 10%** AND **A5 does not win** — validates that
   non-abelian $SO(K)$ transport is load-bearing
3. **A8 beats A3 by ≥ 10%** — validates that the dynamic fiber state is
   the load-bearing associative memory (replicates V14 at scale without
   a literal bank)
4. **A9 beats A8** — V20 + CurvBias beats V20 alone, justifying a hybrid

Any one is publishable.
