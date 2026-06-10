# SSM Ablation Study: Is the SSM Doing All the Heavy Lifting?

## Pre-Fix Results (2026-03-30)

4-way ablation, 3000 steps each, Tiny Shakespeare (65 vocab, 128 seq_len):

| Model | Params | Val BPC | Val Acc | ms/step | Delta |
|-------|--------|---------|---------|---------|-------|
| A: Full V12.1 | 2,345,031 | 2.302 | 53.3% | 317 | baseline |
| B: Zero SSM Context | 2,345,031 | 3.589 | 27.0% | 279 | +1.287 |
| C: SSM+MLP Only | 2,112,327 | 2.267 | 53.8% | 68 | -0.035 |
| D: No Sparsity | 2,345,031 | 2.275 | 53.4% | 279 | -0.027 |

**Verdict:** The SSM context is critical (B collapses), but the spectral machinery
(FFT/IFFT, transport, Hopfield memory, sparsification) adds zero measurable value.
The SSM+MLP alone matches the full architecture while being 4.7x faster.

## Root Cause Diagnosis

Investigation revealed three bugs/design flaws that prevented the spectral
machinery from contributing:

### Bug 1: Conjugate Symmetry Halves Effective Spectral Capacity

**The problem:** After block 1, all spatial signals are real. `fft(real)` produces
conjugate-symmetric spectra where `X[k] = conj(X[N-k])`, meaning `|X[k]| = |X[N-k]|`.
The top-k sparsification selects modes by magnitude, so conjugate pairs are always
selected together. With `spectral_sparsity=10` and `subbundle_dim=32`:

- 10 modes selected, but ~4-5 are conjugate mirrors
- Effective independent degrees of freedom: ~5-6 per subbundle
- The model thought it had 10 spectral knobs, it actually had ~5

The spectral representation was so impoverished that the SSM+MLP could match it
trivially with its 256-dim dense spatial representation.

**The fix:** Use `rfft/irfft` instead of `fft/ifft`. The `rfft` of a length-32
real signal gives 17 unique complex modes (DC through Nyquist). Selecting 10 from
17 gives 10 truly independent modes. 2x the effective information capacity.

- `spectral_to_spatial`: `irfft(subs, n=subbundle_dim)` instead of `ifft(subs).real`
- `spatial_to_spectral`: `rfft(subs)` instead of `fft(subs)`
- `spectral_sparsify`: operates on `spectral_half_dim=17` instead of `subbundle_dim=32`
- Spectral dimension: 136 (8 x 17) instead of 256 (8 x 32)

### Bug 2: Spectral Proximal at Every Langevin Step Fights the Settler

**The problem:** At each of the 2 Langevin steps, the settler:
1. Computes Hopfield gradient (pushes state toward memory attractor in spatial domain)
2. Takes a step
3. Immediately applies spectral proximal: `FFT -> top-k -> IFFT`

Step 3 projects the state back to a 10-mode spectral subspace, destroying most of
the spatial-domain progress made in step 1-2. With only 2 steps, the settler
essentially does nothing useful — each step's contribution is immediately projected away.

This is analogous to taking a gradient step and then projecting onto a constraint
surface that's nearly orthogonal to the gradient direction.

**The fix:** Apply spectral proximal only after the final Langevin step, not at every
step. The settler can freely explore the full spatial domain during settling, then
project to the spectral manifold once at the end. The Hopfield gradient can actually
do useful work.

### Bug 3: Transport Kernel Only Dampens, Never Amplifies

**The problem:** The transport kernel is `exp(-D(ctx) * w^2 - i * w * A(ctx))` where
`D = softplus(projection)`, which forces `D >= 0`. This means:

- `|kernel| = exp(-D * w^2) <= 1` for all modes
- Transport can only attenuate spectral modes, never amplify them
- The kernel is a low-pass filter with context-dependent bandwidth
- Combined with subsequent sparsification, the transport just selects which modes
  survive, but in a roundabout way that adds 4.7x compute

For the transport to actively create useful spectral structure, it needs the ability
to amplify some modes while damping others.

**The fix:** Use `tanh`-bounded diffusion instead of `softplus`:
```python
diffusion = tanh(D_proj(q_t)) * 2.0  # range [-2, 2]
```
- Positive D: damping (low-pass filtering)
- Negative D: amplification (high-pass boosting)
- Bounded for stability (max gain ~exp(2 * 0.25) = e^0.5 at Nyquist)

## Post-Fix Architecture

After fixes, V12.1 has 2,120,631 params (down from 2,345,031 due to smaller spectral
dimension). The spectral and spatial dimensions are now cleanly separated:

| Domain | Dimension | Description |
|--------|-----------|-------------|
| Spectral | 136 (8 x 17) | Complex, unique half-spectrum per subbundle |
| Spatial | 256 (8 x 32) | Real, dense spatial per subbundle |
| Context | 128 | Real, SSM output |

The rfft/irfft pair is information-lossless for real signals — a length-32 real signal
has exactly 17 complex degrees of freedom (2 real-valued: DC and Nyquist; 15 complex),
totaling 32 real numbers. No information is created or destroyed.

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| FFT/IFFT | `fft/ifft` (256 complex modes, conjugate-redundant) | `rfft/irfft` (136 unique modes) |
| Sparsification budget | 10/32 nominal, ~5 effective | 10/17, all 10 independent |
| Transport diffusion | `softplus` (dampen only) | `tanh * 2` (dampen or amplify) |
| Settler proximal | Every Langevin step | After final step only |
| Embedding dim | 65 x 256 x 2 (mag+phase) | 65 x 136 x 2 (mag+phase) |
| Transport projections | 128 -> 256 | 128 -> 136 |
| Memory atoms | 16 x 32 per subbundle | 16 x 17 per subbundle |
| Total params | 2,345,031 | 2,120,631 |

## How to Rerun the Ablation

```bash
python v12/ablation_ssm.py
```

This runs all 4 ablation variants for 3000 steps each. Expected runtime: ~40 minutes
on MPS (Mac), less on CUDA.

## Interpreting Results

After the fix, the critical comparison is still **A vs C**:

- If `A (Full V12.1) >> C (SSM+MLP Only)` in BPC: **spectral machinery now contributes**.
  The fixes unlocked the spectral domain's potential.
- If `A ~= C`: the spectral machinery is still redundant at this scale. This doesn't
  invalidate the thesis (the theoretical framework is a contribution regardless), but
  the empirical argument needs to emphasize scaling expectations.

**B vs A** should remain large (SSM context is critical — this was already validated).

**D vs A** tests whether sparsity matters once the spectral machinery is working.
With rfft (every mode is independent), removing sparsity removes the rank-control
argument. A gap here supports the thesis directly.
