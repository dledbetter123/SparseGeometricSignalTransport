# V17: Precision-Routed Gaussian Constellations

## Core Idea

Tokens are Gaussian clouds in spectral space. The PRECISION (1/variance) of each cloud
determines routing: high-precision modes form strong, selective connections between tokens.
Low-precision modes are effectively silent. Position is encoded through WHICH MODES ARE
TIGHT, not through phase rotation.

The connectivity between tokens emerges from overlapping high-precision modes — like neurons
in distributed brain regions that fire together (Wernicke's + Broca's for language). Not
all modes activate for every token. The activation pattern IS the routing mechanism.

## What Changed from V16e

| | V16e | V17 |
|---|---|---|
| Position encoding | Phase shift (fixed sinusoidal) | Learned precision pattern = positional signature |
| Fiber deposit | All modes deposit equally | Precision-weighted: confident modes deposit strongly |
| Fiber query | All modes query equally | Precision-weighted: confident modes query precisely |
| Variance evolution | Fixed -0.1 per block | Learned, content+position dependent |
| Token routing | Broadcast to all modes | Sparse, selective, through overlapping tight cones |
| Connectivity | Implicit in shared state | Explicit through Gaussian overlap structure |

## Architecture

```
Embedding:
  mag = Embedding(vocab, M)           # what this token is
  phase = Embedding(vocab, M)         # spectral phase
  log_var = VarNet(mag, position)     # which modes are tight = positional + content routing

V17Block:
  CloudNorm(constellation)

  # Precision gates routing into the matrix fiber
  precision = exp(-log_var)           # per mode confidence
  effective_mag = mag * sigmoid(precision)  # only confident modes visible

  # Matrix fiber with precision-weighted q, k, v
  q, k, v = project(effective_mag, phase, precision)
  S[t] = gamma[t] * S[t-1] + k[t] v[t]^T    # parallel matrix scan
  messages = q[t] @ S[t-1]                    # precision-weighted retrieval

  # Parseval filter on messages
  filtered = ParsevalFilter(constellation, messages)

  # irfft to spatial (stable basis change)
  spatial = irfft(filtered) + irfft(constellation)

  # Local conv + FFN
  spatial = LocalConv(spatial) + FFN(spatial)

  # rfft back to spectral
  new_constellation = rfft(spatial)

  # Variance evolves through learned transform (not fixed -0.1)
  new_log_var = VarUpdate(log_var, messages, precision)
```

## Why This Should Help

1. **Sparse routing without hard thresholding.** V14 proved hard thresholding hurts.
   Gaussian precision gives soft, differentiable sparsity. A mode with variance=10 is
   effectively zero. A mode with variance=0.001 is a precise point. The transition is
   smooth and learnable.

2. **Position as connectivity pattern.** Instead of "position 5 has phase shift X on all
   modes" (rigid), position 5 has "modes 3,7,15 are tight" (learnable). Nearby positions
   share tight modes (communicate). Distant positions don't (independent). The model
   learns the connectivity topology.

3. **Content-dependent routing.** Different tokens at the same position can have different
   precision patterns. "The" at position 5 might tighten syntactic modes. "Paris" at
   position 5 might tighten entity modes. The routing depends on both WHAT and WHERE.

4. **Selective fiber utilization.** The matrix fiber's limited state (1024 values) is
   used efficiently — only the high-precision modes actually deposit. Low-precision modes
   don't waste state capacity on noise.
