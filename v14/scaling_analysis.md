# Scaling Analysis: SGST vs. Attention Models

## Complexity Comparison Table

| Component | Attention Transformer | V13 (as-is) | SGST (theory-complete) |
|---|---|---|---|
| Cross-token mixing | O(n^2 * d) | O(n * M) parallel scan | O(n * M) + O(K * n * s) Langevin |
| Per-token processing | O(n * d^2) | O(n * M * h) MLP | same |
| **Total per layer** | **O(n^2 * d + n * d^2)** | **O(n * M * h)** | **O(n * M * h + K * n * s)** |
| Inference memory | O(L * n * d) KV cache | O(L * M) fiber state | O(L * M + L * \|M\| * d) mem bank |
| Per-token generation | O(n * d) KV lookup | O(M) fiber update | O(M + K * \|M\| * d) |

Where:
- n = sequence length
- d = model dimension
- M = total spectral modes (136 in current config)
- s = active modes per token (s << M)
- K = Langevin steps per block
- h = MLP hidden dimension
- L = number of layers
- |M| = memory bank size (fixed, independent of n)

---

## Sequence Length Scaling

### The Headline: O(n) vs. O(n^2)

The fiber parallel scan is O(n) in sequence length. Self-attention is O(n^2). At concrete sequence lengths:

| Sequence Length | Attention (relative) | Fiber (relative) | Ratio |
|---|---|---|---|
| 512 | 1x | 1x | 1:1 |
| 4,096 | 64x | 8x | 8:1 |
| 32,768 | 4,096x | 64x | 64:1 |
| 1,048,576 | 4,194,304x | 2,048x | 2,048:1 |

For a 1M token context, the fiber is ~2000x cheaper than attention for the cross-token mixing operation.

**Caveat:** This advantage is shared with all SSM/linear attention models (Mamba, RWKV, RetNet, etc.). The SGST-specific advantage is the spectral structure, not the O(n) complexity.

### Inference Memory: Constant vs. Linear

The fiber state is a fixed-size complex vector: M complex numbers per layer. It does not grow with context.

Attention's KV cache grows as O(n * d) per layer.

Concrete comparison at d = 256, L = 8:

| Context Length | KV Cache Size | Fiber State Size | Ratio |
|---|---|---|---|
| 1K tokens | 4 MB | 8.7 KB | 460:1 |
| 32K tokens | 128 MB | 8.7 KB | 14,700:1 |
| 1M tokens | 4 GB | 8.7 KB | 460,000:1 |

This is a massive deployment advantage. A single GPU that can serve one 1M-context attention model could serve hundreds of thousands of SGST models at the same context length.

---

## Parameter Scaling (The Real Question)

Scaling laws care about: **how much does loss decrease per parameter added?**

### Attention: Parameters Scale Into Retrieval Expressivity

Each attention head learns an independent (Q, K, V) projection — a distinct "what to look for, where to look, what to retrieve" pattern. The parameter budget is:

```
Params per head: 3 * d * d_head = 3 * d * (d / n_heads)
Total attention params per layer: 3 * d^2 + d^2 (output) = 4 * d^2
```

Double the heads → double the diversity of retrieval patterns. The O(n^2) cost is also what makes it expressive: every token-pair interaction is parameterized through the QK product.

**Empirical scaling laws are proven.** Chinchilla scaling (Hoffmann et al. 2022) gives precise loss-vs-parameter curves for attention transformers. We know with confidence how much data and compute are needed for a given loss target.

### SGST Fiber: Parameters Scale Into Bandwidth

More modes = more independent channels. But each channel is still a fixed exponential decay. Doubling modes doubles the bandwidth of the leaky integrator, but doesn't add selectivity.

Current parameter distribution in V13:

| Component | Params | % of Total |
|---|---|---|
| Embeddings | 90,272 | 4.1% |
| Fiber (decay rates) | 1,088 | 0.05% |
| ConstellationUpdate MLPs | 2,094,216 | 95.3% |
| ConstellationNorms | 4,352 | 0.2% |
| Decoder | 7,680 | 0.3% |

The architecture is telling you where the scaling goes: **into the MLP, not the geometry.** The fiber is 0.05% of parameters. The geometric path is a thin wire connecting large MLP blocks.

### With Theory-Complete Implementation

If the Wilson line, Langevin settling, and Hopfield memory bank are implemented:

```
Wilson line params per layer:  O(M * d)  for content-dependent decay/phase functions
Memory bank params per layer:  O(|M| * d)  for memory atoms
Langevin params per layer:     O(d)  for step size, temperature schedule
```

This shifts the parameter balance: the geometric machinery acquires O(M * d + |M| * d) learnable parameters per layer, making it a genuine compute pathway rather than a thin wire.

---

## Width Scaling (Increasing Model Dimension d)

### Attention

d → 2d means:
- Attention: O(4n^2 * d) if heads scale with d (standard practice)
- FFN: O(4n * d^2) (quadratic in d)
- Total: dominated by O(n * d^2) for moderate sequence lengths

### SGST

d → 2d means M → 2M modes (M = d/2 for rfft of length d). The spectral decomposition into subbundles allows independent scaling of:
- K (number of subbundles) — like attention heads but without O(n^2)
- Modes per subbundle — spectral resolution within each subspace

This gives two axes of width scaling versus attention's one (heads). But whether the additional axis provides useful expressivity depends on whether the geometric machinery is active.

---

## Depth Scaling

Both architectures scale linearly in depth (layers). But the theoretical depth story differs:

### Attention

Each layer applies an independent attention pattern + FFN. Depth provides compositional expressivity — layer L can attend to features computed by layers 1..L-1. But there's no geometric constraint on what each layer does; expressivity comes from the combinatorial space of attention patterns.

### SGST (Theory)

If constellations evolve along geodesics through blocks, each layer advances the token along a curve in constellation space:

```
Block 1: constellation_0 -> constellation_1  (initial spectral identity)
Block 2: constellation_1 -> constellation_2  (contextual refinement)
...
Block L: constellation_{L-1} -> constellation_L  (deep geometric features)
```

Deeper networks trace longer geodesics. Each block contributes geometric content (curvature accumulation, mode activation/deactivation), not just another arbitrary nonlinear transformation. This could provide better depth efficiency — each layer has a structured role rather than a generic one.

**Current status:** V13's deep supervision at blocks 2, 4, 6, 8 actively prevents this progressive refinement by forcing the shared decoder to work at every depth.

---

## The Critical Scaling Question: Associative Recall

The empirical weakness of every O(n) model is **associative recall** — the "needle in a haystack" problem. Given a key at position t, retrieve the corresponding value from position t' << t.

### Why Attention Wins Here

```
score(t, t') = q_t^T * k_{t'} / sqrt(d)
```

Exact content-based lookup. Every past position is explicitly scored against the query. Cost: O(n * d). Selective: the softmax concentrates mass on the matching position(s).

### Why EMA Fails Here

```
h[t] = alpha * h[t-1] + x[t]
```

Information from position t' decays as alpha^(t - t'). For t - t' = 1000 and alpha = 0.99, the retention is 0.99^1000 ≈ 4.3 x 10^{-5}. The signal is gone. No amount of mode diversity fixes this — every mode has exponential decay.

### The Theory's Answer: Hopfield Memory Bank

The Modern Hopfield energy gradient IS softmax attention, but over a **fixed-size memory bank** rather than all past tokens:

```
output = softmax(beta * M^T * x) @ M    # M is |M| x d memory bank
```

- Cost: O(|M| * d) per token — independent of sequence length
- Selectivity: content-addressable, softmax concentrates on matching memories
- Capacity: Modern Hopfield networks store exponentially many patterns (2^{d/2})

If |M| = 256 memory atoms at d = 256:
- Per-token retrieval cost: 256 * 256 = 65,536 ops
- Full-attention equivalent at n = 32K: 32,768 * 256 = 8,388,608 ops
- Ratio: 128:1 advantage

The bet: a fixed-size memory bank with exponential capacity (Hopfield) can match attention's retrieval expressivity at O(1) cost in sequence length. This is the architectural claim that needs to be tested — **and it requires actually implementing the Hopfield memory bank.**

---

## Scaling Comparison Summary

| Scaling Axis | Attention | SGST (theory-complete) | SGST Advantage |
|---|---|---|---|
| Sequence length | O(n^2) | O(n) | Yes, shared with all SSMs |
| Inference memory | O(n) per layer | O(1) per layer | Yes, massive for deployment |
| Parameter efficiency | Proven Chinchilla curves | Unknown, untested | Unknown |
| Associative recall | Exact, O(n) | Via Hopfield bank, O(\|M\|) | Theoretically yes, unproven |
| Width (dimension) | 1 axis (heads) | 2 axes (subbundles x modes) | Possible, unverified |
| Depth | Generic composition | Geodesic evolution | Possible, currently blocked by deep supervision |
| Training FLOP efficiency | Well-understood | Potentially better (sparse gradients) | Theoretical, unmeasured |

### Where SGST Should Win At Scale

1. **Long contexts** — O(n) vs O(n^2), but only matters if the model can actually USE long context (requires working retrieval)
2. **Inference deployment** — constant memory per token is transformative for serving
3. **Spectral gradient locality** — gradients touch s modes, not d dimensions, potentially better FLOP efficiency
4. **Subbundle parallelism** — naturally shardable across devices

### Where Attention Still Wins

1. **Proven scaling laws** — we know Chinchilla curves; SGST has no empirical scaling data
2. **Retrieval expressivity** — exact content-addressable lookup over full history, no information loss
3. **Parameter → expressivity conversion** — adding heads directly adds retrieval diversity
4. **Ecosystem** — FlashAttention, quantization, distillation, all optimized for attention

### The Honest Assessment

The O(n) sequence scaling is real but not unique — Mamba has it too. The constant inference memory is genuinely novel and valuable. But the architecture's real scaling bet is that Hopfield memory + gauge transport can match attention's retrieval quality at O(1) cost in sequence length. **This bet is untested because neither mechanism is implemented in v13.** The current model is testing the scaffolding (constellation geometry, mode fibers, MLP updates), not the building (Wilson line transport, Langevin dynamics, Hopfield retrieval, proximal sparsity).

If the full theory works as the math predicts, the scaling regime is:

```
O(n) sequence, O(1) retrieval, O(s) gradient sparsity, exponential memory capacity
```

That would be a genuinely superior scaling regime to attention. But "if" is doing a lot of work in that sentence.
