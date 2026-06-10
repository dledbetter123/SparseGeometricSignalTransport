# V15: Parseval Spectral Attention

## The Idea

Replace the Langevin settler with a Parseval spectral filter. The fiber accumulates
causal context into h[t] (complex state per mode). The spectral filter then
redistributes energy across modes — amplifying relevant frequencies, suppressing
irrelevant ones — subject to an energy constraint.

## The V14 Lesson

V14 taught us:
- The Wilson fiber (content-dependent complex EMA) works — beats SSM+MLP
- The Langevin settler helps vs plain SSM but isn't competitive with attention
- Per-token nonlinearity (FFN) is essential
- The Hopfield memory bank is the key V14 contributor, not the Wilson line
- Attention (GPT-224d, PPL 211) crushes fiber+Langevin+FFN (PPL 646) on WikiText-2

The gap is in **retrieval expressivity**: attention computes exact pairwise similarity
in one shot. The fiber blurs through exponential decay. The Langevin does softmax over
a fixed memory bank. Neither can match attention's precision.

## What Parseval Attention Offers

Standard attention: "how much does Token A relate to Token B?" → pairwise, O(n²)

Parseval attention: "how do I redistribute spectral energy to minimize the spatial
energy function?" → global, O(n) per operation

The mechanism:
1. Tokens live in Fourier space as constellations (mag, phase) across M modes
2. The fiber accumulates causal context per mode: h[t] = z_t · h[t-1] + c_t
3. A learned complex filter W reshapes the spectral energy: y = W ⊙ h
4. Parseval constraint ||W||∞ ≤ 1 ensures spatial energy is controlled
5. Multiplication in Fourier domain = convolution in spatial domain = global mixing

Key properties:
- **O(n) global mixing**: one elementwise multiply = full spatial receptive field
- **Energy preservation**: Parseval constraint prevents vanishing/exploding signals
- **Scale separation**: low modes = global semantics, high modes = local syntax
- **No sequential bottleneck**: unlike the fiber scan, the filter is instant

## Architecture

```
V15Block (pre-norm residual, two sub-layers):

  # Sub-layer 1: Causal context gathering
  normed = MagPhaseNorm(constellation)
  messages = WilsonFiber(normed)                    # causal complex EMA → Parseval read

  # Parseval spectral filter on messages
  # Content-dependent filter: current token determines which modes to amplify
  W = filter_net(normed.to_flat())                  # → complex weights per mode
  W = W / W.abs().clamp(min=1).detach()             # Parseval constraint: |W| ≤ 1
  filtered = W * messages_complex                    # Hadamard product in spectral domain
  # This is O(M) per token — no scan, no softmax over memory bank

  combined = cat(normed.to_flat(), filtered_flat)    # constellation + filtered context
  d1 = gate1 · project(combined)

  constellation = constellation + d1

  # Sub-layer 2: Per-token FFN
  normed2 = MagPhaseNorm(constellation)
  d2 = FFN(normed2.to_flat())
  constellation = constellation + gate2 · d2
```

## Why This Should Work Better Than Langevin

The Langevin settler does: softmax(β · q @ M_norm^T) @ M
- Fixed memory bank of 256-512 atoms
- Softmax routing → weighted average of prototypes
- K=2 iterative steps
- Content-addressable but limited by memory bank size

The Parseval filter does: W(c_t) ⊙ h_complex(t)
- Content-dependent filter (W depends on current token)
- Direct spectral energy control (amplify/suppress per mode)
- Single pass, no iteration
- Energy-constrained (Parseval prevents instability)

The filter is simpler, faster, and more direct. Instead of "find the nearest memory
prototype and move toward it," it says "given what I'm looking at right now, which
spectral modes of the accumulated context are relevant, and how much should I amplify
each one?"

## The Causality Solution

The Hadamard product in Fourier space is non-causal (circular convolution in time).
But we're NOT filtering across time — we're filtering across MODES at a single
time position. The fiber already did the causal accumulation. The spectral filter
operates on the fiber state h[t], which is causal by construction. The filter
redistributes energy across the 136 modes of h[t], not across positions.

So: causal in time (fiber), global in spectrum (filter). No causality violation.

## Parameter Budget Per Block

- WilsonFiber: ~105K (wilson_proj + base_decay) — unchanged from V14
- ParsevalFilter: Linear(2M, hidden) → SiLU → Linear(hidden, 2M) ≈ similar to wilson_proj
  Plus projection from (2M + 2M) → 2M for combining filtered + constellation
  Estimate: ~120K
- FFN: Linear(2M, 4*2M) → SiLU → Linear(4*2M, 2M) ≈ 594K
- Norms + gates: ~1K
- Total per block: ~820K
- 8 blocks: ~6.6M
- Embedding + decoder: ~26.7M (direct, 50K vocab)
- Total: ~33M (similar to V14c)
