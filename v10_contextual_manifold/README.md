# v10: The Contextual Manifold

**Author:** David Ledbetter
**Date:** 03/14/2026

---

## The Diagnosis

Nine architecture versions. Four different routing mechanisms. One consistent result: **~45% accuracy on Shakespeare, ~2.65 BPC, regardless of routing strategy.**

| Routing Mechanism | Result |
|---|---|
| Causal convolution (v4) | 45% |
| Sparse subspace routing (v6) | 45% |
| Sequence-level diffusion (v7) | 45% |
| Per-subbundle sparse attention (v9) | 45% |

The routing isn't the bottleneck. The **representations are**.

---

## The Root Cause: The Manifold Is Fake

The architecture was designed around a beautiful theoretical framework: tokens as sparse sections of a fiber bundle over a base manifold, with parallel transport along geodesics and Langevin settling into Hopfield attractors.

But the "manifold" in the implementation is `nn.Embedding(128, 128)` — **fixed positional vectors**. Position 5 always has the same manifold coordinate regardless of whether the text says "ROMEO" or "sword" or "the". There is no geometry. No curvature. No notion of contextual distance. The manifold is a lookup table with a fancy name.

This matters because:

1. **The memory bank routes on position, not context.** The router sees `[q, x]` where `q` is the same for position 5 in every sequence. The same dictionary atoms are available regardless of what came before. The attractor landscape is context-blind at the geometric level.

2. **The Hopfield gradient dominates cross-position signals.** The Hopfield gradient (magnitude ~1.0, unit-normalized atoms) overwhelms the routing forces (causal conv diff + attention diff, magnitude << 1.0, further scaled by per-step sigmoid). The settling is overwhelmingly local. Cross-position signals are whispers against the Hopfield gradient's shout.

3. **Representations don't accumulate context across positions.** In a Transformer, the residual stream at position t encodes ALL information from 0..t. In our architecture, each position independently settles to a memory atom, with cross-position influence entering as a small additive perturbation that barely shifts the attractor basin.

**The manifold should have always been contextual.** The coordinate `q` at position t should reflect what the model has seen so far — the accumulated context — not just the position index.

---

## What "Contextual Manifold" Means

### The Theory

In the mathematical framework (Architecture.md), the base manifold M represents the space of possible contexts. A point q on M encodes "where am I in meaning space?" Moving along the manifold traces the evolution of understanding as the model reads the sequence.

This was always the intent. But the implementation shortcut it to positional embeddings.

### What It Should Be

The manifold coordinate at position t should be a **running summary of the sequence history** — a compressed representation of tokens 0..t that determines:

1. **Which attractors are available** (memory bank routing)
2. **What the local geometry looks like** (how the fiber is shaped at this context)
3. **How to interpret the current token** (the same character means different things in different contexts)

Concretely: `q_t = f(x_0, x_1, ..., x_t)` — the manifold coordinate is a function of the accumulated context, not just the position.

### How This Changes the Architecture

**Before (v1-v9):**
```
q_t = positional_embedding[t]              # fixed, context-blind
M_q = memory_bank(q_t, x_t)               # same atoms for same position
hopfield_grad = settle(x_t, M_q)          # context enters only as perturbation
```

**After (v10):**
```
q_t = context_accumulator(x_0, ..., x_t)   # CONTEXTUAL, reflects history
M_q = memory_bank(q_t, x_t)               # different atoms for different histories
hopfield_grad = settle(x_t, M_q)          # settling is context-aware at the geometric level
```

The memory bank now selects completely different attractors depending on what came before. The Hopfield gradient — the dominant force — is itself context-dependent. The routing mechanisms (causal conv, attention) no longer need to fight against a context-blind settling process.

---

## Design Options for the Context Accumulator

The manifold coordinate `q_t` needs to be a compressed summary of positions 0..t. Several options:

### Option A: Causal Attention Pooling
Use lightweight causal attention to compute a context vector at each position. This is similar to what a Transformer does internally but produces manifold coordinates rather than token representations.

### Option B: Exponential Moving Average
`q_t = alpha * q_{t-1} + (1-alpha) * project(x_t)` — a running average of projected token states. Simple, fast, no extra parameters. But fixed decay rate limits what can be remembered.

### Option C: GRU/LSTM-style Gating
`q_t = GRU(q_{t-1}, x_t)` — a learned gating mechanism that decides what to remember and forget. This is essentially v3's ContextGate (which hit 35% on synthetic) but applied to the manifold coordinates rather than the token states. Sequential, but the manifold coordinates are lower-dimensional (128 vs 256), so the sequential cost is smaller.

### Option D: Parallel Scan (Mamba-style)
Content-dependent state transitions: `q_t = A(x_t) * q_{t-1} + B(x_t) * x_t`. Can be parallelized via the associative scan algorithm. Content-dependent, parallel, and geometrically principled.

### Option E: The Attention IS the Manifold
Rather than computing manifold coordinates separately, let the per-subbundle attention (v9) output serve double duty: it provides both the routing force AND the context for memory bank routing. The attention output at position t already encodes "what past tokens are relevant" — this is a contextual summary.

---

## What This Fixes

### The Hopfield Gradient Becomes Context-Aware
The dominant force in the settling process will now pull toward **different attractors depending on sequence history**. The character 'e' after "th" will settle into a different attractor than 'e' after "qu". No routing mechanism needs to fight the Hopfield gradient — the Hopfield gradient itself knows the context.

### The Memory Bank Becomes Truly Dynamic
Instead of "position 5 always sees atoms 12, 47, 93", the memory bank becomes "position 5 after 'ROMEO:' sees atoms 3, 28, 71, but position 5 after 'To be' sees atoms 14, 55, 89". The attractor landscape reshapes itself based on context.

### The Routing Forces Can Focus on Refinement
With the Hopfield gradient handling context-dependent settling, the causal conv and attention forces don't need to overcome a context-blind dominant force. They can focus on fine-grained adjustments — distinguishing between "the king is dead" and "the king is angry" when the Hopfield gradient has already settled into the right neighborhood.

---

## Relationship to Prior Findings

### v3's ContextGate Was Right (for the Wrong Reason)
v3 hit 35% on synthetic with a GRU-style sequential gate — the best result until v7.3's deep supervision matched it. The reason: the GRU accumulated context into the token state itself, making the representation at each position inherently contextual. Every subsequent version replaced the GRU with parallel mechanisms but never replaced the contextual accumulation.

### The Anthropic Paper Connection
Anthropic proved that attention heads twist manifolds to align coordinate frames. In v10, the manifold itself would be constructed from context. The attention heads (v9's per-subbundle attention) would then twist a **meaningful** manifold — one whose geometry reflects the actual sequence — rather than twisting a flat positional embedding with no structure.

### The Progression

```
v1-v5:  Fixed manifold + various routing → 12-35% (synthetic)
v6:     Fixed manifold + sparse subspace routing → ~20%
v7:     Fixed manifold + diffusion routing → 36.5%
v8:     Fixed manifold + geometric scaling → 45% (Shakespeare)
v9:     Fixed manifold + per-subbundle attention → 45% (same ceiling)
v10:    CONTEXTUAL manifold + ??? → ???
```

Every version improved routing on a fixed manifold. v10 changes the manifold.

---

## Open Questions

1. **Which context accumulator?** Parallel scan (Option D) is the most principled, but GRU (Option C) is simpler and v3 proved it works. Start simple?

2. **Should the manifold coordinates feed back into the Langevin settling?** If q_t is contextual, the memory bank already sees context through q. Do we still need the causal conv and attention forces, or does the contextual manifold subsume their role?

3. **Dimensionality of the contextual manifold?** Currently manifold_dim=128. A contextual summary might need more or fewer dimensions depending on how much history it encodes.

4. **What happens to the per-subbundle attention (v9)?** It might become unnecessary if the manifold itself carries context. Or it might become more powerful — attention on a contextual manifold is more meaningful than attention on positional embeddings.

---

*Research notes, 03/14/2026. David Ledbetter.*
*The manifold was always supposed to be contextual. Nine versions to find the real bottleneck.*
