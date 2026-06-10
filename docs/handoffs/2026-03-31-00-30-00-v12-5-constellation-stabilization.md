# Handoff: V12.5 Constellation Stabilization & Scale-Up (2026-03-31)

## Session Arc

Started with V12.5 blocked at 13s/step due to parallel scan over 69K streams.
Ended with a stable, pre-norm residual architecture training monotonically at
BPC 2.784 (43.2% accuracy) on the small model, and a rebalanced larger model
(2.2M params) ready to train.

## What Was Done

### 1. Speed Fix: 13s → 93ms/step

**Problem:** ModeFiber's parallel scan ran over B×M×sd = 16×136×32 = 69,632
independent streams, each doing log2(512)=9 iterations.

**Fix progression:**
- First tried cumsum EMA trick (Option C from previous handoff) → 1s/step
- Realized `sparsify()` top-k was calling MPS's slow topk kernel 7x/forward → 855ms
- David pointed out: "sparsity is in the constellation encoding natively" — removed
  binary mask entirely. Magnitudes ARE the sparsity. No topk needed → 138ms/step
- Per-mode deposit weights (M×2 instead of shared Linear(2,1)) → 93ms/step

**Key insight:** The binary mask + topk was never needed. Magnitude encodes activation
level natively. Loud modes are active, quiet modes are silent.

### 2. NaN Debugging (3 rounds)

**Round 1 (step 2000):** `.abs()` on magnitudes caused sign-flip oscillation.
Fixed with `softplus`. Phase grew unbounded. Fixed with `torch.remainder` wrapping.

**Round 2 (step 1500):** Cumsum EMA trick had unstable BACKWARD pass.
`scaled = x / w.clamp(1e-20)` → gradient `dx/dw = -x/w²` with w=1e-20 gives 1e40.
Fixed by going back to parallel scan (only multiply+add, no division).
With mode_state_dim=1, scan runs over B×M = 2,176 streams at 3.4ms.

**Round 3 (step 1500):** No normalization anywhere in the constellation path.
Magnitudes accumulated unchecked across 6 blocks.

### 3. Architecture Stabilization: Pre-Norm Residual

**The breakthrough fix.** Applied standard pre-norm transformer pattern:

```
constellation = constellation + f(RMSNorm(constellation))
```

Specific changes:
- **Signed magnitudes:** Removed softplus/abs from residual path entirely.
  Negative magnitude = π phase shift: -|m|·e^(iφ) = |m|·e^(i(φ+π)).
  Clean additive residual with no nonlinearity.
- **ConstellationNorm (RMSNorm):** Applied at START of each block (pre-norm).
  Normalizes concatenated [mag, phase] by RMS, learned scale.
- **Zero-init last MLP layer:** Blocks start as identity, gradually learn.
- **Update returns deltas, not new constellation:** Block adds deltas to
  ORIGINAL (unnormalized) constellation. Clean gradient highway.
- **No phase wrapping:** exp(iφ) is periodic, wrapping breaks gradients.

**Result:** Monotonically decreasing loss, healthy train/val gap, no NaN.

### 4. L1 Sparsity Penalty

Tried L1 on magnitudes (lambda=0.01) to encourage sparsity. It fought the CE
loss — magnitudes oscillated between "L1 pushes down" and "CE pushes back up."
Both losses increased. **Removed L1 entirely.** Sparsity should emerge naturally.
Currently ~1% effective sparsity (modes with |mag| < 0.01).

### 5. Scale-Up

**Small model (stable, trained to 6500 steps):**
- update_hidden=256, n_blocks=6, mode_state_dim=1
- 1,152,423 params, 75ms/step
- BPC 2.784, Acc 43.2% at step 6500 (still improving)

**First scale-up (trained to 3000 steps):**
- update_hidden=512, n_blocks=8, mode_state_dim=1
- 2,897,689 params, 107ms/step
- BPC 2.837, Acc 42.2% at step 3000 (crossed 42% earlier)

**Final rebalanced model (NOT YET TRAINED):**
- update_hidden=384, n_blocks=8, mode_state_dim=4
- 2,214,489 params, ~952ms/step
- Shifted capacity from MLP into mode fibers (0.2% → 0.8% of params)
- Each mode carries 4-dim state vector, per-mode deposit (M,2,4), per-mode read (M,4)
- Parallel scan over B×M×sd = 8,704 streams

## Current Architecture

```
ConstellationEmbedding:
  mag_emb(token) → signed magnitudes (136)
  phase_emb(token) + positional phase shift → phases (136)
  → Constellation(mag, phase)

V12_5Block (×8, pre-norm residual):
  normed = ConstellationNorm(constellation)        # RMSNorm on [mag, phase]
  messages = ModeFiber(normed)                      # parallel scan EMA, sd=4
  d_mag, d_phase = ConstellationUpdate(normed, messages)  # MLP + gate
  return Constellation(mag + d_mag, phase + d_phase)     # clean residual

ConstellationDecoder:
  mag * exp(i*phase) → irfft → LayerNorm → MLP → logits

Deep supervision at blocks 2, 4, 6, 8 (weights 0.25, 0.5, 0.75, 1.0)
```

## Param Breakdown (rebalanced model)

```
Embedding:     17,680  (0.8%)
Norms:          2,176  (0.1%)
ModeFibers:    17,408  (0.8%)  ← up from 0.2%, each mode has 4-dim state
Updates:    2,094,216  (94.6%) ← MLP 384 hidden (down from 512)
Decoder:       83,009  (3.7%)
Total:      2,214,489
```

## Speed Issue with Rebalanced Model

The rebalanced model runs at **952ms/step** — much slower than the 107ms of the
512-hidden model with scalar fibers. The 4-dim mode state means the parallel scan
runs over 8,704 streams (vs 2,176). The scan itself was 3.4ms for 2,176 streams,
so ~14ms for 8,704 — that's not the bottleneck.

The slowdown is likely from:
- `torch.einsum('btmi,mid->btmd', ...)` for per-mode deposits (136 modes × 4 state dims)
- The scan's 9 iterations with 4x more data
- Backward through all of the above

**This needs profiling before training.** 952ms/step for a 2.2M model is too slow.
The 2.9M model with scalar fibers was only 107ms/step.

Possible fixes:
- Replace einsum with batched matmul or manual reshape+bmm
- Reduce mode_state_dim to 2 (half the streams, still richer than 1)
- Profile to find the actual bottleneck

## What Remains To Do

### Immediate: Profile and fix speed of rebalanced model
952ms/step is too slow. Need to identify whether it's the einsum, the scan, or
backward that's slow, and optimize.

### After speed fix: Train rebalanced model
Run for 10K steps on Tiny Shakespeare, compare against:
- Small V12.5 (1.15M, BPC ~2.6-2.7 projected at 10K)
- GPT-Nano (2.4M, BPC ~1.5 at 10K)

### Architecture questions to explore
1. **Sparsity:** Currently ~1%. Will it increase with longer training?
   If not, should we add a gentle L1 (lambda=0.001)?
2. **Mode fiber expressivity:** Does sd=4 actually help vs sd=1?
   Compare learning curves at matched param count.
3. **Update MLP dominance:** 94.6% of params in the pointwise MLP.
   Is the model just learning a per-position transformation that ignores
   the mode fiber messages? Check gradient norms on fiber vs update params.
4. **Deep supervision overhead:** 4 decoder calls in forward + backward.
   Consider reducing to 2 (blocks 4, 8) to save compute.

## File Inventory

```
v12/
  gen_notebook_v12_5.py    # Current generator — rebalanced model (384 hidden, sd=4, 8 blocks)
  architecture_v12_5.ipynb  # Generated notebook, ready to train
  gen_notebook_v12_3.py    # V12.3 spectral-native (archived)
  gen_notebook_v12_4.py    # V12.4 geometric transport (archived)
  v12_1.py                 # V12.2 base (archived)
```

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Remove binary mask / topk | Magnitudes encode sparsity natively |
| Signed magnitudes | Enables clean additive residual (no softplus in path) |
| Pre-norm residual | Standard recipe for stable deep architectures |
| Parallel scan over cumsum | Cumsum has unstable backward (division by tiny w) |
| Zero-init last MLP layer | Blocks start as identity, learn gradually |
| Remove L1 sparsity | Fought CE loss, caused oscillation |
| Per-mode deposit weights | Each mode learns own (mag,phase) → deposit mapping |
| RMSNorm not LayerNorm | Preserves magnitude signs and relative structure |
