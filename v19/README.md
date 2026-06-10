# V19 — Unitary Delta-Rule Fiber + Geometric Context + CurvBias

V19 is the first architecture in the SGST line that implements every thesis
§8.3 future-work direction plus the one unexplored combination identified in
`topology/SYNTHESIS.md` (`U(K)` unitary transport + delta rule).

See `V19_DESIGN.md` for the full retrospective of V5–V18, the motivating
ablation evidence, and the component-by-component justification.

## Files

| File | Purpose |
|------|---------|
| `V19_DESIGN.md` | Full retrospective + design rationale + experimental protocol |
| `v19_modules.py` | All PyTorch modules. Importable and testable in isolation |
| `test_v19.py` | 15 unit tests: shapes, causality, Parseval bound, grad flow |
| `gen_notebook_v19.py` | Training notebook generator (follows V18 pattern) |
| `architecture_v19.ipynb` | Generated training notebook (WikiText-103) |
| `README.md` | This file |

## Quick start

```bash
# 1. Run the unit tests (uses conda python with torch installed)
/Users/davidledbetter/miniconda3/bin/python test_v19.py

# 2. Regenerate the training notebook
/Users/davidledbetter/miniconda3/bin/python gen_notebook_v19.py

# 3. Open architecture_v19.ipynb in VSCode / Jupyter and run it.
#    The notebook loads v19_modules.py via %run, so any edits you make to
#    the modules are picked up on the next notebook cell execution.
```

## What V19 keeps from V5–V18 (load-bearing survivors)

1. **Content-dependent state transitions** — V3/V12/V14 lesson.
2. **Per-token FFN** — non-negotiable; every ablation without it underperformed.
3. **irfft round-trip** — V16c went spectral-native and hit NaN at step 5500.
4. **Matrix fiber associative memory** — V16d/e over scalar EMA.
5. **Complex / holonomic state structure** — implemented via real SO(K) for
   MPS safety.
6. **Sufficient state capacity** — V19 increases to 16,384 values per block,
   16× the V16e/V18 budget of 1,024.

## What V19 drops (decorative; see V19_DESIGN.md §1.3)

Spatial sparsity, Hopfield memory bank, iterative Langevin settling, hard
top-k thresholding, phase-only position encoding, local causal conv,
unguarded position-axis FFT, spectral-native (no irfft) path, fixed decay
schedules, the SSM context accumulator (thesis §7.3.4 flagged it).

## What V19 adds

| Module | Role |
|--------|------|
| `LearnedBandMask` | Differentiable sparse mask over candidate frequency bands per block (thesis §8.3.3 "learned sparsity patterns") |
| `GeometricContextAccum` | Per-band complex running summary with content-dependent decay; replaces the SSM context accumulator |
| `UnitaryDeltaFiber` | SO(K) unitary transport (`fast_orthogonal` of skew-symmetric content projection) + delta-rule write + associative read. The unexplored SYNTHESIS.md combination |
| `ParsevalSpectralFilter` | Energy-bounded `|W| ≤ 1` spectral gate on the fiber output (V16 survivor) |
| `CurvBiasAttention` | Single attention head per block with the thesis's primary contribution: content-dependent curvature bias derived from cumulative theta distances |
| Learned mix gate | One scalar per block; decides how much of the fiber path vs the attention path to use. Lets the data choose which mechanism is load-bearing at each depth |

## Thesis §8.3 coverage

| Thesis direction | V19 realization |
|------------------|-----------------|
| §8.3.1 CurvBias at production scale | `CurvBiasAttention` is structurally drop-in for any transformer's attention layer |
| §8.3.2 Geometric enhancement of production models | Same |
| §8.3.3 Sparse FFT | `GeometricContextAccum` operates only on active bands |
| §8.3.3 Learned sparsity patterns | `LearnedBandMask` per block, L1-regularized |
| §8.3.3 Multi-scale spectral hierarchy | Each block's band mask is initialized to a wavelet-like schedule across depth and learned further |
| §8.3.3 100M+ scaling study | `V19Config` default is 69M params; scale by knob adjustment |
| §8.3.4 Formal proofs | Modules are named, isolated, and unit-testable for the four thesis theorems |

## Config knobs (V19Config)

```python
V19Config(
    d_model=256,             # token embedding dim
    n_blocks=8,              # number of V19Block layers
    fiber_heads=16,          # independent U(K) + delta fibers per block
    fiber_K=32,              # SO(K) size; state per head = K*K
    ctx_n_bands=32,          # candidate frequency bands for the context accumulator
    curvbias_dim=64,         # dimension of the single attention head per block
    ffn_mult=4,              # FFN hidden = ffn_mult * d_model
    band_mask_l1=1e-3,       # L1 penalty on learned band masks
    # ...plus standard training hyperparameters
)
```

At the default config:
- Total parameters: **~69M**
- State per block: 16 × 32 × 32 = **16,384 values** (16× V16e/V18)
- Block count: 8

## Running the tests

```
  PASS  test_matrix_parallel_scan_matches_sequential
  PASS  test_unitary_delta_parallel_scan_matches_sequential
  PASS  test_fast_orthogonal_preserves_norm_small_skew
  PASS  test_make_skew_symmetric_is_skew
  PASS  test_precision_embedding_shape
  PASS  test_variance_update_shape_and_clamp
  PASS  test_learned_band_mask_init
  PASS  test_geometric_context_accum_shape_and_causality
  PASS  test_unitary_delta_fiber_shape_and_causality
  PASS  test_parseval_filter_shape_and_bound
  PASS  test_curvbias_attention_shape_and_causality
  PASS  test_ffn_shape
  PASS  test_v19_block_shape_and_causality
  PASS  test_v19_model_forward_and_loss
  PASS  test_v19_model_param_count_reasonable

15/15 tests passed
```

Causality tests verify that perturbing the last token does not affect
outputs at earlier positions for: `GeometricContextAccum`,
`UnitaryDeltaFiber`, `CurvBiasAttention`, and the full `V19Block`.

The Parseval filter test verifies `||out|| ≤ ||in||` (energy-bounded gating).

The unitary delta parallel scan is checked against a sequential reference
implementation to confirm the Hillis–Steele sweep composes the
`(U, B) · (U', B') = (U' @ U, U' @ B + B')` semigroup correctly.

## Speed notes (post-2026-04-09 speed fixes)

The first V19 notebook run on H100 clocked 2.85 s/step with loss stuck at 11
(worse than random uniform). Two separate bugs:

1. **Non-causal `ParsevalSpectralFilter`.** The original filter ran `rfft`
   along the position axis of the fiber output and `irfft` back. This leaked
   information from the last token into every earlier position, which the
   model learned to exploit, then collapsed. Replaced with `ChannelGate`,
   a position-wise content-dependent gate with `gate ∈ (0, 1)` per
   (position, channel) that is causal by construction and still enforces
   the `|W| ≤ 1` energy bound.

2. **Many small ops + fp32 on H100.** The `UnitaryDeltaFiber` scan at
   `K = 32` runs log₂(T) = 8 matmul/concat levels per block, plus a
   Python `make_skew_symmetric` double loop (~500 Python calls per forward
   per block), plus a sequential Python loop in `GeometricContextAccum`.
   All of these produced tiny kernels that kept H100 at 10–30% utilization.

Fixes now in:
- `make_skew_symmetric` vectorized with `torch.triu_indices` (2 tensor
  assignments instead of ~500 Python iterations per call).
- `GeometricContextAccum` vectorized via the pre-rotate trick
  (`h'[t] = R^{-t} · h[t]` reduces the complex-decay recurrence to a plain
  scalar scan, which runs in `O(log T)` parallel sweeps via
  `scalar_parallel_scan`).
- `unitary_delta_parallel_scan` uses `torch.matmul` instead of `torch.einsum`
  for the 4D GEMM (same FLOPs, much lower parser overhead).
- The training notebook enables bf16 autocast (`torch.autocast(dtype=bfloat16)`)
  on CUDA — H100 tensor cores are ~13× faster in bf16 than fp32 for the
  matmul-heavy fiber scan.
- The training notebook enables TF32 for the fp32 matmuls that stay outside
  autocast.
- The training notebook wraps both V19 and GPT-Nano in `torch.compile(mode="reduce-overhead")`
  which fuses the fiber scan's many small ops into a single graph, which is
  the biggest lever for kernel-launch overhead.

`fiber_K` is kept at 32 per the thesis state-capacity goal: the fiber state
per block is `16 * 32 * 32 = 16,384` values, 16× V18's 1024.

Use `benchmark_v19.py` on H100 to measure the new numbers:

```bash
/path/to/python benchmark_v19.py                     # fp32, no compile
/path/to/python benchmark_v19.py --amp               # bf16 autocast
/path/to/python benchmark_v19.py --amp --compile     # bf16 + torch.compile
```

This prints a component breakdown and the V19 / GPT-Nano ratio. On H100
with `--amp --compile` the ratio should drop from the initial 20× down
to somewhere around 2–4×.

## Known caveats

1. **MPS complex ops**: V19 deliberately avoids `torch.complex*` types. The
   `GeometricContextAccum` represents complex numbers as a pair of real
   channels, and the fiber uses real `SO(K)` via `fast_orthogonal`. This
   follows `feedback_no_complex_mps.md` to avoid MPS hangs seen in earlier
   versions.

2. **`GeometricContextAccum` is sequential over T**: The complex-decay
   recurrence has a per-band rotation that `matrix_parallel_scan` would need
   to be extended to handle. For the current scale (T = 256, S ≤ 32) the
   inner loop is a negligible fraction of step time compared to the fiber
   matmuls. If T grows, the parallel scan extension is a straightforward
   followup.

3. **`make_skew_symmetric` uses a Python double loop**: At K = 32 this is
   ~496 Python iterations per call, executed once per forward. Fine at
   current scale; replace with a pre-computed index tensor if profiling
   shows it on the hot path.

4. **Band-mask L1 tuning**: The default `band_mask_l1 = 1e-3` is a starting
   point. If the masks saturate at 1.0 everywhere (losing the sparse-FFT
   benefit), increase the penalty. If they collapse to all-zero, decrease.
   The band-mask-inspection cell in `architecture_v19.ipynb` plots the
   masks across blocks for diagnosis.

## Next steps (per V19_DESIGN.md §2.8)

1. Ablation-first construction: train the component-wise ablations before
   the full V19 run. The ablation matrix is in `V19_DESIGN.md §2.8`.
2. Main WikiText-103 run (77M config): compare V19, V18, GPT-Nano,
   GPT+CurvBias at 40K steps.
3. Scaling study: if the 77M run is within 15% of GPT+CurvBias, scale to
   350M then 1B on The Pile.
4. Unit-test audit for the four §8.3.4 formal theorems against V19's
   actual implementations.

## Success criteria

Any one of the following is publishable; all four together justify a V20:

1. WikiText-103 PPL gap to GPT-Nano+CurvBias ≤ 15% at 77M params, matched
   wall-clock time.
2. V19 fiber path beats V18 by ≥ 10% PPL (validating the U(K) + delta-rule
   combination as a load-bearing mechanism).
3. Learned per-block `band_ids` empirically specialize for low/mid/high
   frequency across block depth, with ≥ 60% per-block spectral sparsity.
4. CurvBias alone (extracted from V19, applied to a vanilla transformer at
   1B scale on The Pile) continues to outperform RoPE, replicating the
   thesis §6.7 result at production scale.
