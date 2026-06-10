# v9: Geometric Attention — The Missing Piece

**Author:** David Ledbetter
**Date:** 03/14/2026

---

## The Progression So Far

### The Architecture

A causal language model built on fiber bundles, where tokens are sparse sections and computation happens through Langevin settling on Hopfield energy landscapes with causal cross-position mixing. No standard Transformer components — no softmax attention, no FFN, no causal mask. Causality is structural (Finsler geometry), not masked.

### Version History

| Version | Cross-Position Mechanism | Val Acc (Synthetic) | BPC (Shakespeare) | Key Insight |
|---|---|---|---|---|
| v1-v2 | None (position-only Wilson line) | 5-12% | — | No cross-position = no learning |
| v3 | Sequential ContextGate (GRU-style) | **35%** | — | Content-dependent selectivity is essential |
| v4 | Causal spectral convolution (FFT) | 18-20% | — | Parallel but content-independent = weak |
| v4.2 | Causal conv + content gate | 18.5% | — | Post-hoc gate can't fix content-independent kernel |
| v5 | QK attention-gauge + causal conv | 19.8% | — | Attention on wrong thing (per-fiber, not cross-position) |
| v6 | Sparse subspace routing (no gate) | ~20% | — | Sparsity-as-routing: theoretically elegant, undertrained |
| v7.0 | Sequence Langevin (blended forces) | 13% | — | Blending context into Hopfield query corrupts matching |
| v7.1 | Additive forces + ramping sparsity | 27.5% | — | Separate Hopfield settle from causal routing force |
| v7.2 | Remove gauge, gated residual, content mem | ~28% | — | Gauge transport fought causal mixing (different frames) |
| v7.3 | Deep supervision (every block gets signal) | **36.5%** | — | Gradient was dying through 21 chained Langevin steps |
| v8.0 | v7.3 on real text | 35% | 3.17 | First real text. Capacity-limited (train-val gap 0.02) |
| v8.1 | Scaled dense components | — | ~2.7 | Bigger context MLP. Helped but still plateaued at 35% |
| v8.2 | Scaled geometric components | **45%** | **2.65** | Per-subbundle dictionaries, deeper manifold, more Langevin steps |

### The Consistent Finding

Every version hits the same wall: **the causal convolution cannot selectively attend to specific past tokens based on content**. It applies the same exponential decay kernel regardless of what the tokens contain. The additive causal force (context_proj MLP) can process the mixed signal differently per-position, but the mixing itself is content-independent.

v3's sequential ContextGate worked because it was content-dependent (GRU-style gating decides per-feature how much context to keep). Every parallel variant we've tried provides content-independent mixing.

The Anthropic paper ("When Models Manipulate Manifolds") proved that attention heads ARE geometric operators — they compute rotations (twists) that align manifold coordinate frames. The QK interaction is a gauge connection. This is exactly what our architecture needs but doesn't have for cross-position routing.

### Why v5's Attention Attempt Failed

v5 applied QK attention to the **per-fiber gauge rotation** (Stage 1 — within-token transport). This made the transport content-dependent, but the **cross-position mixing** (Stage 2 — the causal convolution) remained content-independent. The QK parameters added 24K trainable parameters per block that overfitted without improving the thing that actually mattered: which past tokens influence which future tokens.

The attention was on the wrong thing.

---

## The v9 Hypothesis

### What's Missing

The architecture needs a mechanism where token t can selectively access specific past tokens based on content, not just proximity. Currently:

- Causal convolution: all past tokens contribute, weighted by exponential decay (same for all inputs)
- Context projection: processes the mixed signal, but the mixing already happened with wrong weights
- Hopfield gradient: settles toward local attractors, no cross-position selectivity

### What the Research Points To

**1. Attention IS Gauge Connection (NeurIPS 2025: "Gauge Fiber Bundle Geometry of Transformers")**
This paper formally proves that multi-head attention parameters have gauge symmetry with nonzero curvature, and attention behaves as a connection on a principal fiber bundle. The head-wise symmetry group creates gauge orbits as fibers. This is not a metaphor — it's a theorem. Our architecture was designed to exploit this exact structure, but we removed the mechanism that implements it.

**2. Mamba's Selective State Spaces**
Content-dependent state transitions via input-dependent A(x_t), B(x_t), C(x_t) matrices. Parallel scan algorithm achieves O(T) training. The state transition decides what to remember/forget based on token content — exactly the content-dependent selectivity we need.

**3. Hyena's Content-Gated Convolutions**
Implicitly parameterized long convolutions with multiplicative gating. The kernel shape depends on the input. Matches Transformer quality at sub-quadratic cost.

**4. SLAY: Spherical Linear Attention**
Constrains Q, K to the unit sphere (like our dictionary atoms). Angular alignment determines attention. O(L) complexity.

### The v9 Design: Sparse Geometric Attention Inside the Langevin Loop

The key realization: attention should operate **inside the Langevin settling loop**, not as a separate stage. At each settling step, every position should be able to selectively query the most relevant past positions. As states sharpen through annealing, the attention patterns sharpen too.

**Per-subbundle sparse attention:**
Each subbundle independently computes lightweight attention over past positions:

1. Project subbundle content into Q, K, V (small: subbundle_dim -> subbundle_dim)
2. Causal attention scores: Q_t @ K_{0..t-1}^T (per-subbundle, per-position)
3. Sparse top-k: only attend to the k most relevant positions (e.g., k=8)
4. Weighted sum of V values at selected positions
5. This provides a "selective context" signal alongside the causal convolution

Why this is different from v5:

| v5 (failed) | v9 (proposed) |
|---|---|
| Attention for per-fiber gauge rotation | Attention for **cross-position routing** |
| Applied once, before Langevin | Applied at **every Langevin step** |
| Full fiber_dim QK interaction | **Per-subbundle** QK (smaller, structured) |
| Content-dependent transport, content-independent mixing | Content-dependent **mixing** |
| No interaction with settling dynamics | Attention sharpens **as states settle** |

Why per-subbundle:
- Each subbundle independently decides which past tokens are relevant for its feature channel
- The "syntax" subbundle might attend to recent punctuation; the "semantics" subbundle might attend to the subject noun 20 tokens back
- Smaller QK matrices (32x32 per subbundle vs 256x256 full fiber) — less prone to overfitting
- Respects the orthogonal decomposition that the architecture was designed around

Complexity: O(T * k * K * d_sub) per Langevin step. With T=64, k=8, K=8, d_sub=32: 131K operations per step — much less than full attention's O(T^2 * D) = O(64^2 * 256) = 1M.

### The Three Forces Become Four

Currently the score function has three additive forces:
1. **Hopfield gradient** (settle toward nearest memory attractor)
2. **Causal context force** (what the causal convolution says about the local context mix)
3. **Lateral inhibition** (competitive dynamics between fiber dimensions)

v9 adds:
4. **Selective attention force** (what the sparse per-subbundle attention retrieves from specific past tokens)

The causal convolution provides a smooth, position-weighted context baseline. The sparse attention provides sharp, content-addressed retrieval of specific tokens. Together they give both local smoothing AND selective long-range access.

### Relationship to the Anthropic Paper

The Anthropic paper proved that LLM attention heads twist manifolds to align coordinate frames. In v9:

- Each subbundle's QK interaction computes the geometric alignment between the current token's features and each past token's features — this IS the manifold twist
- The top-k sparsification selects which past positions have the highest geometric alignment — the most relevant "twists"
- The V projection retrieves the aligned content from those positions
- The per-subbundle independence means different feature channels can twist toward different past positions — multi-scale geometric manipulation

This is attention as it was always meant to be used in this architecture: not as a probability distribution over positions (no softmax), but as a geometric alignment detector that identifies which manifold twists are most informative for denoising the current position.

### Open Questions

1. **Should we keep the causal convolution alongside the sparse attention?** The conv provides smooth local context; the attention provides selective retrieval. They may be complementary, or the attention may subsume the conv's role.

2. **How many attention positions (k)?** Too few = can't retrieve enough context. Too many = approaches quadratic cost and may overfit.

3. **Should k vary per subbundle?** Some channels might need very local attention (k=4), others might need broader reach (k=16).

4. **Should the attention use softmax or raw geometric alignment?** The Anthropic paper and our prior work suggest no softmax (these are rotations, not probabilities). But softmax stabilizes gradients.

5. **Per-block or shared Q/K/V projections?** Different blocks might need to attend to different aspects. But shared projections reduce parameters.

---

*Research notes, 03/14/2026. David Ledbetter.*
*Bridging the gap between geometric sparsity and selective cross-position routing.*
