# v10: The Contextual Manifold — A Comprehensive Mathematical Autopsy and Reconstruction

**Author:** David Ledbetter
**Date:** March 15, 2026
**Status:** Research exploration — open to structural departures from all prior versions

---

## Preface: What This Document Is

This is not a changelog. This is a forensic mathematical analysis of why nine iterations of a theoretically principled architecture converged to the same ceiling, and a detailed proposal for the structural renovation required to break through it. We draw on two external formalisms — Gurnee et al.'s empirical geometry of counting manifolds in Claude 3.5 Haiku (arXiv:2601.04480v1), and the anonymous ICLR submission on gauge fiber bundle geometry of Transformers (Paper 19168) — to diagnose what went wrong, what was lost from the original mathematical vision, and what a faithful implementation would actually look like.

The key claim: **the architecture's mathematical framework was correct from the start; every implementation betrayed it in the same place**.

---

## Part I: The Original Mathematical Vision

### 1.1 The Axioms (Architecture.md)

The architecture, called "Vega" internally, rests on five axioms that distinguish it from a standard Transformer:

**Axiom 1 — Fiber Bundle Topology.** Computation occurs over a base manifold $\mathcal{M}$ (possibly Finsler). At each contextual coordinate $q \in \mathcal{M}$, a local fiber $\mathcal{F}_q$ provides the representational space. The total space $E = \coprod_{q \in \mathcal{M}} \mathcal{F}_q$ is a fiber bundle. Tokens are not dense vectors in $\mathbb{R}^d$ but sparse sections of this bundle, decomposed into $K$ orthogonal subbundles:

$$\mathcal{F}_q = \bigoplus_{k=1}^{K} \mathcal{F}_q^{(k)}, \qquad x_q = (S_q, a_q)$$

**Axiom 2 — Spectral Gauge-Covariant Transport.** Because fibers at different points are distinct vector spaces, direct addition is invalid. Transport between fibers requires a gauge connection $A$. This is executed spectrally:

$$\tilde{X}_q = X_p \odot \exp\left(-D\omega^2 - i\omega \int_\gamma A\right)$$

The advective term $\exp(-i\omega \int_\gamma A)$ is the $U(1)$ holonomy (Wilson line). The diffusive term $\exp(-D\omega^2)$ is the heat kernel. Combined, they form a single complex exponential multiply in Fourier space: $O(d \log d)$.

**Axiom 3 — Dynamic Memory Bank.** At each coordinate $q$, a local attractor landscape $M_q$ is carved from a global overcomplete dictionary $D \in \mathbb{R}^{d \times N_{\text{global}}}$ via geometric gating:

$$g(q) = \mathrm{k\text{-}WTA}(W_{\text{route}}\, q), \qquad M_q = D \odot g(q)$$

The topographic continuity property guarantees that nearby manifold points share overlapping dictionaries, while distant points are orthogonal.

**Axiom 4 — Langevin-Hopfield Energy Descent.** The predicted token is recovered by running annealed Langevin dynamics on the Modern Hopfield energy landscape:

$$x_{t - \Delta t} = x_t - \eta \nabla_x E_q(x_t; M_q) + \sqrt{2\eta \beta_t^{-1}} \,\epsilon_t$$

where $E_q(x; M_q) = -\beta^{-1} \log \sum_j \exp(\beta\, x^\top m_j^{(q)})$. The gradient $\nabla_x E_q = -\sum_j \mathrm{softmax}(\beta\, x^\top m_j)\, m_j$ is mathematically identical to an attention operation — iterative, implicit attention refinement.

**Axiom 5 — Proximal Sparsity.** Soft-thresholding enforces biological lateral inhibition at each Langevin step:

$$x^{\text{sparse}} = \mathrm{sign}(x) \odot \max(|x| - \lambda\eta,\; 0)$$

### 1.2 The Causal Language Modeling Correspondence (CLMWithArch.md)

The architecture maps onto autoregressive language modeling with precise geometric correspondences:

| Standard Transformer | Vega Architecture | Mathematical Object |
|---|---|---|
| Token embedding in $\mathbb{R}^d$ | Sparse fiber section $x_q = (S, a)$ | Section of fiber bundle |
| Positional embedding | Manifold coordinate $q \in \mathcal{M}$ | Point on base manifold |
| Causal attention mask | Finsler metric asymmetry | Structural causality |
| KV cache | Wilson line $U_\gamma = \mathcal{P}\exp(i\int A)$ | Path-ordered exponential (holonomy) |
| Self-attention + FFN | Spectral advection-diffusion | Gauge-covariant parallel transport |
| Softmax + temperature | Langevin descent + proximal sparsity | Annealed energy-based sampling |

The Wilson line is the critical correspondence. In a standard Transformer, all past tokens are stored in the KV cache and queried via attention. In Vega, the history is encoded in the accumulated gauge phase — the holonomy of the connection along the sequence path:

$$U_\gamma = \mathcal{P}\exp\left(i \int_{p_1}^{p_t} A\right)$$

This holonomy is the mathematical incarnation of "accumulated context." The coordinate $q_t$ on the base manifold encodes not just *where* you are in the sequence, but *what you have seen* along the path to get there.

### 1.3 What the Vision Required

For the framework to function as designed, the manifold coordinate $q_t$ must satisfy:

1. **Context-dependence**: $q_t = f(x_0, x_1, \ldots, x_t)$, not $q_t = \text{Embedding}(t)$
2. **Path-dependence**: Different orderings of the same tokens produce different $q_t$ (holonomy depends on the path, not just the endpoint)
3. **Causal structure**: $q_t$ depends only on $x_0, \ldots, x_t$ (Finsler asymmetry enforces this)
4. **Smooth dependence on context**: Nearby contexts produce nearby manifold coordinates (topographic continuity of the memory bank requires this)

---

## Part II: The Mathematical Autopsy — Where Every Version Went Wrong

### 2.1 The Single Fatal Line

Across all nine versions, one line of code destroyed the framework:

```python
self.manifold_coords = nn.Embedding(max_seq_len, manifold_dim)
# ...
q = self.manifold_coords(positions)  # positions = torch.arange(T)
```

This makes $q_t = e_t$ — a fixed learned vector for position $t$, independent of the tokens. The "manifold" became a lookup table. Every downstream component that depended on $q_t$ for context-awareness was operating on a fiction.

### 2.2 The Cascade of Consequences

**Consequence 1: The Memory Bank Routes on Position, Not Meaning.**
The memory bank router receives $[q_t, x_t]$ (or just $q_t$ in early versions). When $q_t$ is positional, the router selects the same dictionary atoms for position 5 regardless of whether the text reads "ROMEO:" or "the end." Content-aware routing (v7.2+) partially compensated by including $x_t$, but $x_t$ is the *current* token only — it carries no accumulated context.

**Consequence 2: The Hopfield Gradient Dominates Context Signals.**
The Hopfield gradient $\|\nabla_x E_q\| \approx 1.0$ is parameterized by $M_q$, which is position-dependent but context-blind. The cross-position routing forces (causal convolution, attention) enter as additive perturbations with magnitude $\ll 1.0$, further attenuated by learned per-step scheduling ($\alpha \in [0.3, 0.7]$). The settling process is overwhelmingly local: the Hopfield gradient shouts; the routing whispers.

**Consequence 3: The Wilson Line Accumulates Nothing Meaningful.**
In v1-v5, the gauge connection $A_{t \to t+1}$ was computed from manifold coordinates alone: $A = f(q_t, q_{t+1})$. Since $q_t$ is positional, the Wilson line $\phi_{t+1} = \phi_t + A_{t \to t+1}$ accumulates *position-to-position* phase shifts — a fixed function of the sequence length, identical for every input. The Wilson line was supposed to encode the KV cache analog; instead it encoded the identity function.

**Consequence 4: No Geometric Structure to Twist.**
The Anthropic paper proved that attention heads twist manifolds to align coordinate frames. The gauge fiber bundle paper proved that attention induces an Ehresmann connection with nonzero curvature (Theorem 4.1). Both rely on there being a *non-trivial manifold* to twist. Applying attention (v9) to twist a flat positional grid is rotating a featureless plane — geometrically vacuous.

### 2.3 Version-by-Version Failure Modes

Each version attempted to compensate for the context-blind manifold via a different routing mechanism. All hit the same ceiling because the bottleneck was upstream of the routing.

| Version | What It Tried | Why It Failed | Accuracy |
|---|---|---|---|
| v1-v2 | Position-only Wilson line | No cross-position information at all | 5-12% |
| v3 | GRU-style ContextGate (sequential) | **Worked** — GRU implicitly contextualizes $x_t$ by accumulating history. But sequential, $O(T)$ | **35%** |
| v4 | Causal spectral convolution (FFT) | Content-independent kernel: same exponential decay for all inputs, regardless of what the tokens contain | 18-20% |
| v4.2 | Causal conv + content gate | Post-hoc gating scales the mixed signal, but the mixing already happened with wrong weights | 18.5% |
| v5 | QK attention-gauge + causal conv | Attention applied to per-fiber gauge rotation (Stage 1), not cross-position mixing (Stage 2). 24K extra parameters overfitted | 19.8% |
| v6 | Sparse subspace routing | Elegant theory: sparsity *is* the gate. But requires the Langevin settling to produce codes where "related tokens share active dims" — this presupposes the very context-dependence it's trying to achieve | ~20% |
| v7.0 | Sequence-level Langevin (blended) | Blended context into Hopfield query, corrupting the matching — memory atoms encode single tokens, not token+context mixtures | 13% |
| v7.1 | Additive forces + ramping sparsity | Separated Hopfield settle from causal force — correct decomposition, but causal force is still content-independent (same kernel) | 27.5% |
| v7.2 | Removed gauge + gated residual | Gauge transport rotated each position into a different frame, then causal conv mixed incompatible representations. Removing the gauge fixed one conflict but lost geometric transport entirely | ~28% |
| v7.3 | Deep supervision (all blocks) | The gradient was dying through 21 chained Langevin steps. Direct signal at each block: 36.5%. But the *representations themselves* haven't changed — each block independently settles to position-dependent attractors | **36.5%** |
| v8.0-v8.2 | Scaled to real text (Shakespeare) | Per-subbundle dictionaries, deeper manifold, more Langevin steps. 45% accuracy, BPC 2.65. But the train-val gap is small (0.14 nats), suggesting the model is *near capacity for the representational bottleneck*, not underfitting in the usual sense | **45%** |
| v9 | Per-subbundle sparse attention (4th force) | Content-dependent cross-position retrieval inside the Langevin loop. Added 98K attention parameters (2.6% of total). Result: **identical 45%** — the attention is fighting the same context-blind Hopfield gradient; adding a content-dependent force doesn't help when the dominant force is content-blind | **45%** |

### 2.4 The Smoking Gun: v3 Outperformed Everything Until v7.3

v3's GRU-style ContextGate hit 35% on synthetic data — a result that only v7.3's deep supervision matched (36.5%). The reason is now transparent: the GRU gate is a *context accumulator*:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t, \qquad z_t = \sigma(W_z [x_t, h_{t-1}])$$

This is exactly the Wilson line accumulation the architecture was designed around, but implemented at the token-state level rather than the manifold-coordinate level. The GRU *de facto* contextualizes the representation at each position by incorporating all previous tokens via gated accumulation. Every subsequent version removed this accumulation and never replaced it.

The insight: **v3 was the closest any version came to the intended mathematical framework**, despite being dismissed as "not parallel enough."

---

## Part III: External Mathematical Validation

### 3.1 The Anthropic Paper: When Models Manipulate Manifolds (Gurnee et al., 2026)

This paper provides direct empirical validation of four architectural principles, and devastating clarity on what was missing.

**Finding 1: Scalar quantities are represented on curved 1D manifolds in low-dimensional subspaces.**

Character counts are not stored as scalar numbers but as points on a helical curve embedded in a 6-dimensional subspace of the residual stream (95% variance captured). The curve has high curvature — it "ripples" — and this curvature is *optimal* for distinguishing nearby count values under finite dimensionality constraints. The ringing is not noise; it is a Gibbs-phenomenon-like artifact of truncating a Fourier representation of a narrow-peaked similarity function to $k$ dimensions. The discrete Fourier transform diagonalizes the circulant similarity matrix, and the low-rank approximation is equivalent to truncating small Fourier coefficients.

*Connection to Vega:* The manifold coordinate $q_t$ was supposed to live on exactly such a curved manifold. But $q_t = e_t$ (positional embedding) defines a collection of *isolated learned vectors* with no guaranteed geometric structure — no curvature, no helical topology, no Fourier structure. The 128-dimensional embedding space is massively over-parameterized for what amounts to a 1D positional index. The architecture was given a highway when it needed a winding mountain road.

**Finding 2: Discrete features tile manifolds as "place cells."**

The 10 character-count features tile the counting manifold like biological place cells — each activating for a specific range, with overlapping receptive fields that collectively provide continuous coverage. Subsequent features activate over increasingly large ranges (receptive field dilation), matching biological Weber-Fechner properties of numerical perception.

*Connection to Vega:* The memory bank atoms $\{m_j^{(q)}\}$ are the architectural analog of place cells. The $\mathrm{k\text{-}WTA}$ gating function $g(q)$ selects which atoms activate at each manifold point — exactly the mechanism for tiling a manifold with discrete feature receptive fields. But when $q$ is positional, the tiling is over sequence position, not over semantic content. The place cells tile the wrong space.

**Finding 3: QK matrices physically twist manifolds to align them.**

To detect the line boundary, the model must compare the character count manifold with the line width manifold. Specific attention heads have QK matrices that *rotate* one manifold to align it with the other at a target offset. When the manifolds geometrically align, the inner product spikes. Multiple heads implement different offsets, creating a stereoscopic distance-to-boundary system. The twist is a linear group action (rotation) on the embedded curve — possible precisely because the manifold is multidimensional (1D representations cannot support rotational actions, only scaling).

*Connection to Vega:* This is the gauge connection. The QK matrix *is* the Lie algebra element generating the gauge transformation. The Anthropic paper proves empirically what the gauge fiber bundle paper proves theoretically: attention computes geometric rotations on representation manifolds. Vega was designed to make this explicit via the spectral gauge operator $\exp(-i\omega \int_\gamma A)$. But when v7.2 removed the gauge transport (because it conflicted with the causal convolution), this geometric twisting was eliminated entirely. v9 reintroduced attention as an additive force, but not as the *geometric transformation* it mathematically is.

**Finding 4: The feature-geometry duality is fundamental.**

The paper establishes that computation in LLMs has dual descriptions: (a) discrete sparse features firing in circuits, and (b) continuous geometric transformations of manifolds. Neither description alone captures the full picture. There is a "complexity tax" when using only discrete features — the manifold perspective reduces it by providing compact geometric descriptions of operations that would require tracking hundreds of feature interactions.

*Connection to Vega:* The architecture was designed to make this duality explicit and structural:
- **Continuous geometry:** Spectral advection-diffusion (Fourier domain, smooth transport along the manifold)
- **Discrete features:** Langevin-Hopfield descent with proximal sparsity (spatial domain, collapse to sparse attractors)

The Fourier-Langevin split *is* the feature-geometry duality, architecturally encoded. This is the deepest vindication of the framework's design philosophy.

**Finding 5: Cooperative distributed construction of geometry.**

No single attention head creates the full character count manifold. Instead, 5 Layer-0 heads achieve $R^2 = 0.93$; adding 6 Layer-1 heads raises this to $R^2 = 0.97$. Individual head outputs are nearly 1-dimensional, but their *sum* forms the curved 6D manifold. Each head contributes a piece of the overall curvature — a distributed algorithm where the manifold is *cooperatively assembled*.

*Connection to Vega:* The architecture's multi-block structure (4 blocks in v8-v9) should similarly cooperate to construct the contextual manifold. With deep supervision, each block gets a direct training signal, but the blocks currently operate on the *same* static manifold coordinates. If $q_t$ were contextual, successive blocks could *refine* the manifold — each block contributing additional curvature, analogous to the distributed counting heads in Claude.

### 3.2 The Gauge Fiber Bundle Paper (Anonymous, ICLR 2025)

This paper provides the rigorous mathematical scaffolding that the architecture was reaching for intuitively.

**Theorem 2.3 (Principal Bundle Structure).** On the generic stratum $\Theta_0$ of parameter space, the head-wise gauge group $G_{\max} = ((GL(d_k))^h \times (GL(d_v))^h) \rtimes S_h$ acts freely and properly, making $\pi: \Theta_0 \to Q := \Theta_0 / G_{\max}$ a principal $G_{\max}$-bundle.

*Implication for Vega:* The architecture's parameter space has the exact fiber bundle structure postulated in Axiom 1. But Vega's fiber bundle is over the *representation space* (token features as sections), while this theorem is about the *parameter space*. Both structures coexist: the parameter-space bundle governs optimization geometry, while the representation-space bundle governs computation geometry.

**Theorem 3.2 (Natural Gradient as Horizontal Riesz Representative).** With the Fisher-Rao connection, the natural gradient is:

$$\tilde{\nabla}L = (G_{\theta|H_\theta})^\dagger P_{H_\theta}^\top \nabla L$$

It reduces to orthogonal projection onto $H_\theta$ only when $G_{\theta|H_\theta} = I$.

*Implication for Vega:* Standard gradient descent in Vega moves along *all* parameter directions, including gauge-redundant ones. This wastes optimization capacity on directions that don't change the realized function. A Fisher-aware optimizer that restricts updates to horizontal (function-changing) directions would be strictly more efficient. This is particularly relevant because Vega's Langevin dynamics introduce their own gauge redundancies (the energy landscape $E_q$ is invariant under certain rotations of the dictionary atoms).

**Theorem 4.1 (Attention Curvature).** Multi-head attention induces an Ehresmann connection on the representation bundle with generically nonzero curvature. Transporting around a small rectangle produces a nontrivial gauge displacement proportional to $\Omega(u,v)$.

The curvature arises from the commutator of horizontal lifts: $[X_u, X_v] \cdot Y = \sum_i (D_u D_v \alpha_i - D_v D_u \alpha_i) V_i$, driven by the non-commutativity of attention weight sensitivities.

*Implication for Vega:* Nonzero curvature means path-dependent transport — the result of transporting a representation depends on *which path* you take through the sequence. This is *exactly* context sensitivity. Different orderings of the same tokens produce different representations at the end. The architecture needs this curvature, but a flat positional manifold has zero curvature everywhere. The attention in v9 adds curvature through its QK interactions, but it's curvature *within* the Langevin settling process, not curvature *of the base manifold that governs attractor selection*.

**Proposition 4.2 (FFN Near-Verticality).** The FFN block produces gradients that are predominantly vertical (along gauge orbits) and nearly orthogonal to attention in the Fisher-Rao metric, with $\cos \angle_{g_\theta}(\nabla_{\text{FFN}}, \nabla_{\text{Att}}) \leq C \sqrt{d_{\text{head}}/d_{\text{model}}}$.

*Implication for Vega:* The FFN is largely fiber-preserving — it transforms representations within a fiber without moving between fibers. Vega's Langevin-Hopfield descent (which replaces the FFN) should exhibit similar fiber-preserving behavior. The Hopfield gradient pulls $x$ toward the nearest memory atom within the *same* fiber $\mathcal{F}_q$. The cross-position forces (causal convolution, attention) are the horizontal/connection components. This confirms the decomposition in v7.1: Hopfield gradient (vertical/settling) + causal force (horizontal/transport) is the correct split, not the blended approach of v7.0.

**Theorem 6.1 (Morse-Bott Structure).** When $L$ is gauge-invariant and $\theta$ is critical, the entire orbit $G_{\max} \cdot \theta$ lies in the critical set. The Hessian vanishes on $V_\theta$ and has a well-defined restriction to $H_\theta$. This explains why "seemingly distant" minima in $\Theta_0$ can sit in the same basin on the quotient $Q$ — small horizontal steps can move far in Euclidean parameter norm while staying close in function space.

*Implication for Vega:* The memory bank's dictionary atoms have permutation symmetry — $S_N$ acts on the $N$ atoms without changing the energy landscape. This creates flat directions (Morse-Bott degeneracy) in the loss landscape. The dictionary coherence regularizer partially addresses this by encouraging orthogonality, but doesn't eliminate the permutation redundancy. Gauge-aware optimization (Algorithm 1 from the paper) would project out these wasted gradient directions.

---

## Part IV: The Contextual Manifold — Mathematical Reconstruction

### 4.1 The Core Requirement

The manifold coordinate $q_t$ must be a causal function of the accumulated sequence:

$$q_t = \Phi(x_0, x_1, \ldots, x_t)$$

satisfying:

1. **Causality**: $q_t$ depends only on $x_{0:t}$, enforced structurally
2. **Content-dependence**: different tokens at position $t$ produce different $q_t$
3. **History-dependence**: $q_t$ encodes information from all of $x_{0:t}$, not just $x_t$
4. **Smoothness**: $\Phi$ is differentiable w.r.t. all its arguments (for gradient flow)
5. **Parallelizability**: computable in $O(T)$ or $O(T \log T)$ during training

The Wilson line formulation suggests:

$$q_t = q_{t-1} + A(x_t, q_{t-1})$$

where $A$ is the gauge connection. This is a content-dependent recurrence — the manifold coordinate at position $t$ is determined by the connection field evaluated at the current token and the current manifold position. The path-ordered exponential $\mathcal{P}\exp(i\int A)$ becomes the iterative application of this recurrence.

### 4.2 The Parallel Scan Implementation (Wilson Line as Structured State Space)

The connection between the Wilson line and structured state spaces is not a metaphor — it is a mathematical identity. The Wilson line in the Abelian ($U(1)$) case reduces to phase accumulation:

$$\phi_t = \phi_{t-1} + A(x_t, q_{t-1})$$

Generalizing to a vector-valued manifold coordinate with content-dependent transitions:

$$q_t = A(x_t) \odot q_{t-1} + B(x_t) \odot \psi(x_t)$$

where:
- $A(x_t) \in \mathbb{R}^{d_m}$ is the state transition (what to remember/forget, derived from token content)
- $B(x_t) \in \mathbb{R}^{d_m}$ is the input gate (how much of the new token to incorporate)
- $\psi(x_t) \in \mathbb{R}^{d_m}$ is a projection of the token into manifold coordinates

This is precisely the selective state space model (Mamba) recurrence. The associative scan algorithm computes $q_t$ for all $t$ simultaneously in $O(T)$ work and $O(\log T)$ depth, maintaining full parallelism during training.

**The mathematical identity:**

| Wilson Line (Gauge Theory) | SSM (Mamba) | Interpretation |
|---|---|---|
| $\phi_t = \phi_{t-1} + A_{t \to t+1}$ | $q_t = A(x_t) \odot q_{t-1} + B(x_t) \odot \psi(x_t)$ | Content-dependent state accumulation |
| Path-ordered exponential $\mathcal{P}\exp$ | Parallel associative scan | Efficient parallel computation of recurrence |
| Connection 1-form $A$ | State transition matrix $A(x_t)$ | Content-dependent dynamics |
| Holonomy | Hidden state $q_t$ after full sequence | Accumulated context |
| Gauge curvature $\Omega$ | Path-dependence of $q_t$ | Context sensitivity |

### 4.3 What Changes Downstream

Once $q_t$ is contextual, every downstream component transforms:

**Memory Bank Routing:**
```python
# BEFORE: same atoms for same position, regardless of content
q_t = positional_embedding[t]
M_q = memory_bank(q_t, x_t)  # x_t is current token only

# AFTER: different atoms for different histories
q_t = parallel_scan(A(x), B(x), psi(x))  # encodes x_0, ..., x_t
M_q = memory_bank(q_t, x_t)  # q_t carries the full history
```

The router now sees a compressed representation of the *entire* past sequence through $q_t$, plus the current token through $x_t$. This is functionally equivalent to what the KV cache provides in a Transformer, but compressed into the manifold coordinate rather than stored as raw vectors.

**Hopfield Gradient:**
The dominant settling force $-\nabla_x E_q(x; M_{q_t})$ is now parameterized by a context-aware $M_{q_t}$. The character 'e' after "th" settles toward different attractors than 'e' after "qu", because the manifold coordinate $q_t$ — and therefore the selected dictionary atoms — reflects the accumulated context. The Hopfield gradient *itself* becomes the primary carrier of context, rather than fighting against it.

**Cross-Position Forces:**
With the Hopfield gradient handling context-dependent settling, the causal convolution and attention forces (Forces 2 and 4 in v9) shift from "fighting the dominant force" to "providing complementary refinement." The causal conv offers smooth local context; the attention offers selective retrieval. Neither needs to overcome a context-blind settling process.

### 4.4 The Four-Force Score Function in the Contextual Regime

The Langevin score function becomes a hierarchy of aligned forces:

$$\text{score}(x_t, q_t, \text{step}) = \underbrace{-\nabla_x E_{q_t}(x_t; M_{q_t})}_{\text{Context-aware settling}} + \underbrace{\alpha_s \cdot f_{\text{conv}}(\mathbf{x})}_{\text{Local smoothing}} + \underbrace{\alpha_a \cdot f_{\text{attn}}(\mathbf{x})}_{\text{Selective retrieval}} + \underbrace{\gamma\, W_{\text{inh}}\, x_t}_{\text{Competition}}$$

In v1-v9, the first term dominated ($\|\nabla_x E_q\| \gg \|\text{other forces}\|$) and was context-blind. The additive routing forces were perturbations trying to redirect a stubborn attractor. In v10, the first term is context-aware *by construction*, and the additive forces can focus on fine-grained adjustments.

**The radical simplification hypothesis:** If the contextual manifold carries sufficient history, Forces 2 (causal conv) and 4 (attention) may become *entirely redundant*. The pure SDE returns:

$$dx = -\nabla_x E_{q_t}(x; M_{q_t})\, dt + \sqrt{2/\beta_t}\, dW$$

The context is not a force *applied* to the settling token. The context *is the landscape itself*.

### 4.5 Mathematical Relationship to the External Papers

**The Contextual Manifold and Theorem 4.1 (Attention Curvature):**

The gauge fiber bundle paper proves that attention induces a connection with nonzero curvature on the representation bundle. In the contextual manifold framework, this curvature has a precise interpretation: the curvature tensor $\Omega(u,v)$ at manifold point $q_t$ measures how much the transport of a representation depends on the *order* in which neighboring tokens are processed.

For a flat positional manifold, $\Omega = 0$ everywhere — there is no path dependence, because the manifold has no structure for paths to depend *on*. The contextual manifold, by encoding sequence history in $q_t$, provides the geometric substrate on which curvature can be nonzero. Different token orderings produce different $q_t$ trajectories, and therefore different memory banks $M_{q_t}$, producing genuinely path-dependent representations.

**The Contextual Manifold and Manifold Manipulation (Gurnee et al.):**

Gurnee et al. showed that Claude 3.5 Haiku builds curved 1D manifolds to represent character counts, and attention heads twist these manifolds for comparison. In Vega:

- The manifold $\mathcal{M}$ is now the trajectory $\{q_0, q_1, \ldots, q_T\}$ of the parallel scan — a curve in $\mathbb{R}^{d_m}$ whose shape is determined by the token sequence.
- The per-subbundle attention (v9's Force 4) acts as the manifold-twisting mechanism discovered by Gurnee et al. — each subbundle's QK interaction detects geometric alignment between positions.
- The memory bank atoms are the place cells that tile this contextual manifold.
- The Fourier-domain gauge transport $\exp(-i\omega A)$ implements the linear rotational action on the manifold that Gurnee et al. proved is the mechanism underlying boundary detection.

The architecture now natively builds, twists, and resolves manifolds — the operations that standard Transformers must learn implicitly.

---

## Part V: What Was Lost From microgpt.py — And Why It Matters

### 5.1 The Baseline

Karpathy's `microgpt.py` — a pure-Python, 200-line GPT — serves as the reference Transformer. Despite its extreme simplicity (1 layer, 16-dim embeddings, 4 heads, ~7K parameters), it demonstrates the *minimal viable architecture* for autoregressive language modeling.

### 5.2 What microgpt.py Does That Vega Doesn't

**Explicit KV Cache (Context Accumulation):**
```python
keys[li].append(k)
values[li].append(v)
```
Every previous token's key and value vectors are stored and accessible. Token $t$ can attend to *any* previous token $t'$ based on content similarity $q_t \cdot k_{t'}$. The KV cache is the *explicit* context accumulator that Vega's Wilson line was supposed to replace — but the Wilson line was implemented as a position-only phase accumulation that discards all token content.

**Content-Dependent Routing (Attention):**
The query-key interaction selects *which* past tokens are relevant, based on content. This is the content-dependent selectivity that v3's GRU achieved sequentially, and that every parallel version of Vega failed to match until v9 — which added it but still couldn't break through because the Hopfield gradient (the dominant force) remained context-blind.

**Residual Stream as Information Backbone:**
In microgpt.py, the residual connection means that every piece of information added at any layer is preserved and accessible to all subsequent layers. The residual stream at position $t$ after layer $L$ encodes *everything* the model knows about positions $0, \ldots, t$ — it is an implicit context accumulator.

Vega's Langevin settling destroys this property. Each block runs $L$ Langevin steps that collapse $x_t$ toward a memory attractor. The gated residual (v7.2+) mitigates this, but the settling process is fundamentally *lossy* — it projects the rich residual stream onto the nearest attractor in the memory bank, discarding information that doesn't align with any atom.

### 5.3 What Vega Has That microgpt.py Doesn't

The exchange is not entirely in microgpt.py's favor. Vega provides:

1. **Structured sparsity**: outputs have meaningful zero structure (biological lateral inhibition), unlike the dense vectors of a Transformer
2. **Iterative refinement**: Langevin settling performs implicit attention refinement over multiple steps, increasing precision with compute
3. **Energy-based sampling**: token prediction is grounded in a thermodynamic framework (Boltzmann distribution, simulated annealing), not an arbitrary softmax
4. **Geometric transport**: the spectral operator provides a principled mechanism for translating representations between different "coordinate frames" (fibers)
5. **Overcomplete dictionary**: the memory bank provides a structured, interpretable basis for representations, unlike the unstructured residual stream

The challenge is combining these advantages with the context-accumulation capability that microgpt.py gets for free from its KV cache + attention.

---

## Key Discovery: Token Construction as Sparse Fiber Bundle Manipulation

One of the most important insights to emerge from this line of research — and one that connects the architecture to both external papers at a fundamental level — is that **token construction in Vega is inherently a process of sparse manipulation on the fiber bundle, producing quantifiable activation patterns that tile the total space $E$**.

### The Observation

In a standard Transformer, a token embedding is a dense vector $e \in \mathbb{R}^d$ — every dimension carries a value, and the "identity" of the token is spread uniformly across the space. In Vega, a token embedding is a *sparse section* of the fiber bundle:

$$x_q = (S_q, a_q), \qquad S_q \subset \{1, \ldots, d\}, \quad |S_q| \ll d$$

Distributed across $K$ orthogonal subbundles, the token becomes $K$ independent sparse patterns:

$$x_q^{(k)} \in \mathcal{F}_q^{(k)}, \qquad \text{supp}(x_q^{(k)}) = S_q^{(k)}, \quad |S_q^{(k)}| = \text{top-}k_s$$

The *composite activation pattern* — which dimensions are active in which subbundles, with what magnitudes — is the token's identity on the fiber bundle. This pattern is not incidental; it is the fundamental representational primitive of the architecture.

### Why This Matters: Quantifiability

Dense vectors are opaque — every dimension is nonzero, and the "meaning" is distributed inscrutably across all dimensions. Sparse activation patterns are **quantifiable**:

1. **Combinatorial capacity**: The number of distinct activation patterns scales as $\prod_{k=1}^K \binom{d_k}{\text{top-}k_s}$, which is exponentially large in $K$. Eight subbundles with 32 dimensions and top-4 selection yield $\binom{32}{4}^8 \approx 10^{30}$ possible patterns — a vast discrete codebook emerging from continuous dynamics.

2. **Measurable similarity**: Two tokens' representational similarity can be decomposed into (a) support overlap (Jaccard index of active dimensions per subbundle), and (b) activation correlation (cosine similarity restricted to shared support). This provides a structured, interpretable similarity metric that dense cosine similarity lacks.

3. **Compositional structure**: Because subbundles are orthogonal, the activation pattern in subbundle $k$ is independent of subbundle $k'$. The composite pattern is a *conjunction* of independent aspects — each subbundle captures a factored component of the token's identity. This enables systematic compositional generalization: a token can share its subbundle-3 pattern with one token and its subbundle-7 pattern with another, without interference.

4. **Discrete codebook from continuous dynamics**: The Langevin settling process is continuous (gradient descent + noise on an energy landscape), but its output — after proximal sparsity — is a discrete, identifiable pattern. The continuous-to-discrete transition is not an arbitrary quantization; it is a *thermodynamic phase transition* governed by the annealing schedule $\beta_t$. At high temperature, the token explores the full fiber; at low temperature, it "crystallizes" into a specific activation pattern. The pattern is the equilibrium state of a physical process.

### Connection to Place Cells (Gurnee et al.)

This is precisely the structure Gurnee et al. discovered empirically in Claude 3.5 Haiku. Their "place cell" features:
- Activate for specific regions of a manifold (≡ specific patterns of active subbundle dimensions in Vega)
- Tile the manifold with overlapping receptive fields (≡ nearby manifold points share overlapping supports via topographic continuity of the routing function)
- Exhibit Weber-Fechner receptive field dilation (≡ the k-WTA threshold creates larger receptive fields for less discriminable regions)

The per-subbundle memory bank atoms $\{m_j^{(k)}\}$ are the *generators* of these place-cell patterns. Each atom, when selected by the k-WTA gate, carves out a receptive field on the fiber. The Langevin settling process drives the token's activation pattern toward the nearest pattern in the selected atom set — the token "snaps" to a place cell.

The discovery is that this is not merely an analogy: **the sparse activation patterns produced by Vega's settling process are the architectural implementation of biological place cells on the fiber bundle**. The fiber bundle provides the geometric substrate; the subbundle decomposition provides the factored structure; the k-WTA gating provides the tiling; and the Langevin dynamics provides the settling into discrete, identifiable patterns.

### Connection to Gauge Curvature (Paper 19168)

The sparse activation pattern also connects to the gauge fiber bundle paper's curvature results. Theorem 4.1 proves that attention induces a connection with nonzero curvature — transporting a representation around a closed loop produces a nontrivial gauge displacement. In the sparse pattern language:

- **Parallel transport** of a sparse pattern along the manifold (via the gauge connection $A$) *changes the pattern's support*. The spectral operator $\exp(-i\omega \int_\gamma A)$ rotates the phase of each frequency component, which in the spatial domain shifts which dimensions are above the sparsity threshold. Transport along different paths produces different support sets — this is curvature manifesting as *path-dependent activation patterns*.

- **Holonomy** is the net change in activation pattern after a closed loop on the manifold. Nonzero holonomy means the token "remembers" the path it took — not in its values, but in *which dimensions are active*. The sparsity structure itself is the geometric memory.

### Connection to Feature-Geometry Duality

The sparse activation pattern is the concrete meeting point of the two dual descriptions identified by Gurnee et al.:

| Perspective | Description | Mathematical Object |
|---|---|---|
| **Feature view** | Which atoms are active, with what weights | Sparse code $a_q$ over dictionary $M_q$ |
| **Geometry view** | Which point on the fiber bundle, in which subbundle | Section $x_q \in \Gamma(E)$ |
| **Pattern view** | The composite activation fingerprint across all subbundles | Quantifiable discrete structure |

The "pattern view" is the bridge. It is simultaneously a discrete combinatorial object (which dimensions are active — the feature view) and a geometric object (a point in the fiber over $q$ — the geometry view). The feature-geometry duality is not an abstract principle; it is *concretely realized* in the sparse activation pattern at each position.

This explains why both descriptions are necessary: the feature view reveals the combinatorial capacity and compositional structure; the geometry view reveals the transport, curvature, and path-dependence. Neither alone captures the full picture. The sparse fiber bundle construction is where they become one.

### Implications for v10: Context-Dependent Activation Patterns

With a contextual manifold coordinate $q_t = \Phi(x_{0:t})$, the sparse activation patterns become **context-dependent**:

- The same token 'e' produces *different* sparse patterns depending on its history: after "th", the pattern aligns with high-frequency English morpheme attractors; after "qu", it aligns with rare-completion attractors. The sparsity pattern itself encodes context.
- The *support structure* carries information beyond the nonzero values. *Which* dimensions are zero vs. nonzero — across all $K$ subbundles — is a binary fingerprint of the token-in-context. Two instances of 'e' in different contexts have different fingerprints, enabling downstream components to distinguish them.
- Nearby contexts produce nearby patterns (topographic continuity of $g(q_t)$), while distant contexts produce orthogonal patterns (disjoint supports from k-WTA selection in distant manifold regions). The place-cell tiling now operates on the *contextual manifold* rather than a flat positional grid — every token position has a unique, context-shaped receptive field landscape.

The quantifiable activation pattern becomes a **contextual fingerprint**: a discrete, measurable, compositional summary of "this token, in this context, on this fiber." This is what makes the architecture's representations fundamentally different from dense Transformer embeddings — not just sparser, but *structurally richer* in a way that supports analysis, probing, and principled comparison.

### Diagnostic Implication

This discovery motivates an additional diagnostic for v10:

**Test 9: Activation pattern divergence under context variation.** For the same token at the same position in two different contexts, measure (a) the Hamming distance between the binary support masks $S_q^{(k)}$ across subbundles, and (b) the earth mover's distance between the full activation patterns. In v1-v9, both should be near zero (same position → same manifold coordinate → same routing → same pattern). In v10, both should be nonzero and should increase monotonically with context divergence. Plotting pattern divergence against a context divergence metric (e.g., edit distance of the preceding token history) reveals whether the architecture achieves the place-cell tiling of the contextual manifold that the theory predicts.

**Test 10: Activation pattern entropy across subbundles.** Measure the entropy of the support distribution across subbundles: $H = -\sum_k p_k \log p_k$ where $p_k$ is the fraction of tokens whose subbundle-$k$ support matches a given canonical pattern. Low entropy means the subbundle has collapsed to a small number of stereotyped patterns (potential bottleneck); high entropy means the subbundle is utilizing its full combinatorial capacity. With contextual manifold coordinates, entropy should increase (more diverse patterns because context differentiates tokens that share position).

---

## Part VI: Novel Proposals — Beyond the Status Quo

The following proposals go beyond restoring the Wilson line as a parallel scan. They address structural limitations that would remain even with a contextual manifold.

### 6.1 Non-Abelian Gauge Group: From $U(1)$ to $SU(n)$

All implementations to date use the Abelian gauge group $U(1)$: the Wilson line is a scalar phase accumulation, and path-ordering is trivial (phases commute). This is mathematically impoverished.

The gauge fiber bundle paper (Theorem 2.3) proves that the *actual* gauge group of a Transformer is $G_{\max} = ((GL(d_k))^h \times (GL(d_v))^h) \rtimes S_h$ — a *non-Abelian* group. Non-Abelian gauge groups have nontrivial path-ordering:

$$U_\gamma = \mathcal{P}\exp\left(i \int_\gamma A\right) \neq \exp\left(i \int_\gamma A\right)$$

and the curvature $\Omega = dA + A \wedge A$ has an extra quadratic term $A \wedge A$ that vanishes in the Abelian case. This extra term is precisely the "interaction" between different gauge directions — the reason that the order of operations matters.

**Proposal:** Replace the $U(1)$ phase accumulation with an $SU(n)$ matrix accumulation. Each transport step applies a matrix exponential:

$$T_{t \to t+1} = \exp\left(i \sum_a A^a(x_t, q_t)\, \lambda_a\right)$$

where $\{\lambda_a\}$ are the generators of $SU(n)$. For $SU(2)$, this requires only 3 real parameters per transport step (the Pauli matrices). The parallel scan generalizes to matrix-valued states:

$$(A_t, b_t) \otimes (A_s, b_s) = (A_t A_s, A_t b_s + b_t)$$

which remains associative and supports the same scan algorithm.

The cost is modest: matrix exponentials in small dimensions ($n = 2, 3, 4$) are cheap, and the non-commutativity provides richer path-dependence than scalar phases.

### 6.2 Curvature-Aware Memory Bank

The current memory bank routing function $g(q) = \mathrm{k\text{-}WTA}(W_{\text{route}}\, q)$ is a learned linear projection followed by hard selection. It has no explicit notion of the manifold's local geometry.

**Proposal:** Make the memory bank selection depend not just on the manifold coordinate $q$, but on the local *curvature* of the manifold trajectory. The curvature at $q_t$ measures how sharply the manifold is turning — semantically, how rapidly the context is changing. High curvature (rapid topic shift, sentence boundary, speaker change) should activate a broader, more diverse set of memory atoms. Low curvature (continuation of a phrase, smooth narrative) should activate a narrow, focused set.

The discrete curvature at position $t$:

$$\kappa_t = \|q_t - 2q_{t-1} + q_{t-2}\| / \|q_t - q_{t-1}\|^2$$

Then the number of active atoms becomes curvature-dependent:

$$k_t = k_{\min} + \lfloor (\kappa_t / \kappa_{\max}) \cdot (k_{\max} - k_{\min}) \rfloor$$

High curvature → more atoms active → broader attractor landscape → more exploration.
Low curvature → fewer atoms → narrower, more decisive settling.

This connects directly to Gurnee et al.'s finding: the character count manifold has high curvature precisely where the model needs high resolution (near the line boundary). The memory bank should provide more atoms where the manifold curves sharply — more place cells where finer discrimination is needed.

### 6.3 Self-Synthesizing Geometry: Attention *as* the Manifold Generator

Rather than computing $q_t$ via a separate parallel scan module and then applying attention separately, merge them: **let the per-subbundle attention output define the manifold coordinate**.

The attention at position $t$ already computes a content-weighted summary of past positions — this *is* a contextual manifold coordinate. The idea:

$$q_t = \frac{1}{K} \sum_{k=1}^{K} \text{SubbundleAttn}_k(x_t, x_{0:t-1})$$

Now the manifold is not pre-computed; it is *recursively synthesized* by the attention mechanism at each Langevin step. The gauge connection governs the topology it operates upon. This creates a fixed-point structure: the manifold shapes the attention, and the attention shapes the manifold. Convergence of the Langevin loop simultaneously converges the manifold geometry and the token representation.

This is inspired by the Anthropic paper's finding that attention heads cooperatively construct the geometry they operate on. Rather than separating "build geometry" from "use geometry," we let the geometry emerge from the settling process itself.

### 6.4 Fiber-Bundle-Aware Training: Horizontal Gradient Projection

From Theorem 3.2, the natural gradient restricts updates to the horizontal subspace (function-changing directions). In Vega's parameter space, there exist gauge redundancies:

1. **Dictionary atom ordering**: permuting the atoms in a memory bank changes parameters but not the energy landscape
2. **Subbundle rotation**: rotating the basis within a subbundle changes the representation but not the decoded output (if the decoder is appropriately equivariant)
3. **Langevin trajectory equivalence**: different parameter settings can produce the same settled state

**Proposal:** Implement the Euclidean-proxy gauge-aware gradient decomposition from Algorithm 1 of the gauge fiber bundle paper. At each training step:

1. Compute the full Euclidean gradient $\nabla L$
2. Compute vertical generators $\{v_j\}$ from the symmetry group's Lie algebra
3. Project: $\nabla_V L = \sum_j c_j v_j$ where $(A^\top A)c = A^\top \nabla L$
4. Update with horizontal gradient: $\nabla_H L = \nabla L - \nabla_V L$

This eliminates wasted gradient norm on directions that don't change the function. For the memory bank, this means the optimizer spends zero effort on atom permutations — all gradient signal goes toward making the atoms *better*, not just differently arranged.

### 6.5 Riemannian Langevin Dynamics on the Contextual Manifold

The current Langevin dynamics uses Euclidean geometry: $x_{t+1} = x_t - \eta \nabla E + \text{noise}$. But the fiber $\mathcal{F}_q$ has a Riemannian metric inherited from the base manifold (the fiber metric depends on $q$). The correct Langevin dynamics on a Riemannian manifold is:

$$x_{t + \Delta t} = \text{Exp}_{x_t}\left(-\eta\, G^{-1}(q_t)\, \nabla_x E_q + \sqrt{2\eta / \beta_t}\, G^{-1/2}(q_t)\, \epsilon_t\right)$$

where $G(q_t)$ is the metric tensor of the fiber at manifold point $q_t$, and $\text{Exp}$ is the Riemannian exponential map. The metric tensor $G(q_t)$ can be parameterized as a function of the contextual manifold coordinate:

$$G(q_t) = I + W_{\text{metric}}^\top \text{diag}(\sigma(V_{\text{metric}}\, q_t))\, W_{\text{metric}}$$

This makes the settling process aware of the local geometry: in regions where the manifold has high curvature (rapid context change), the metric tensor stretches certain dimensions, making the settling more cautious and exploratory. In low-curvature regions, the metric is near-Euclidean and settling is fast.

### 6.6 The "Spectral Wilson Line": Fourier-Domain Context Accumulation

The original architecture executed gauge transport in the spectral domain: $\tilde{X}_q = X_p \odot \exp(-D\omega^2 - i\omega \int_\gamma A)$. This was applied to per-token transport but never to the context accumulation itself.

**Proposal:** Execute the Wilson line accumulation in the spectral domain. Instead of accumulating $q_t$ in the spatial domain (where the recurrence is sequential), transform the token sequence into the spectral domain and accumulate phases:

$$Q_t(\omega) = Q_{t-1}(\omega) \cdot \exp\left(-i\omega\, A(x_t)\right)$$

In the spectral domain, the content-dependent phase shift $A(x_t)$ acts as a frequency-dependent rotation, and the accumulated Wilson line $Q_t(\omega)$ is the product of all rotations. Pulling back to the spatial domain via IFFT gives the contextual manifold coordinate.

The advantage: the phase accumulation in the spectral domain is a pointwise product, naturally parallel across frequencies. Different frequencies accumulate context at different rates, providing multi-scale context sensitivity built into the architecture. This connects to Gurnee et al.'s finding that the character count manifold has Fourier structure — the ringing pattern is optimal for the truncated Fourier representation of a peaked similarity function.

### 6.7 Manifold-Aware Dictionary Learning

The dictionary coherence regularizer $\|(D^\top D - I)\|_F^2$ encourages atoms to be mutually orthogonal, but has no notion of the manifold topology. Atoms that serve adjacent manifold regions *should* have high cosine similarity (for smooth transitions), while atoms serving distant regions should be orthogonal.

**Proposal:** Replace the flat coherence loss with a manifold-aware version:

$$\mathcal{L}_{\text{topo}} = \sum_{i,j} w_{ij} \cdot \left(\cos(m_i, m_j) - \cos_{\text{target}}(d_{\mathcal{M}}(q_i, q_j))\right)^2$$

where $d_{\mathcal{M}}(q_i, q_j)$ is the geodesic distance on the contextual manifold between the regions primarily served by atoms $i$ and $j$, and $\cos_{\text{target}}$ is a decreasing function that maps small manifold distances to high similarity and large distances to near-orthogonality.

This encodes the topographic continuity requirement directly into the dictionary learning objective, rather than hoping it emerges from the routing function alone.

---

## Part VII: Diagnostic Framework — What to Measure

### 7.1 Is the Manifold Actually Contextual?

**Test 1: Same position, different contexts.** Feed two sequences that share position $t$ but differ in history. Measure $\|q_t^{(1)} - q_t^{(2)}\|$. In v1-v9, this is identically zero (positional embedding). In v10, it should be large when the histories diverge.

**Test 2: Manifold trajectory curvature.** Plot the trajectory $\{q_0, \ldots, q_T\}$ in the top 3 PCA components. In v1-v9, this is a monotone sequence of learned positional vectors. In v10, it should be a curved path whose shape reflects the content of the input. Compare to the helical counting manifolds found by Gurnee et al.

**Test 3: Memory bank divergence.** For two sequences that share a position, compute the Jaccard similarity of the active dictionary atoms at that position. In v1-v9, this is 1.0 (identical atoms). In v10, it should decrease with context divergence.

### 7.2 Is the Context Accumulation Meaningful?

**Test 4: Probing the manifold coordinate.** Train linear probes to predict features of the token history from $q_t$: (a) the last character seen, (b) the last word, (c) the speaker identity, (d) the sentiment. If $q_t$ is a useful context accumulator, these probes should achieve high accuracy.

**Test 5: Ablating the manifold.** Replace $q_t$ with $q_t^{\text{positional}} = \text{Embedding}(t)$ at test time and measure the accuracy drop. This quantifies how much the model relies on contextual vs. positional information in the manifold.

### 7.3 Are the Forces Cooperating?

**Test 6: Force magnitude analysis.** At each Langevin step, measure $\|\text{Hopfield gradient}\|$, $\|\text{causal force}\|$, $\|\text{attention force}\|$, $\|\text{inhibition}\|$. In v1-v9, the Hopfield gradient dominates by an order of magnitude. In v10, the forces should be more balanced, because the Hopfield gradient and the routing forces are now aligned (both context-aware).

**Test 7: Cosine similarity between forces.** Measure $\cos(\nabla E_q, f_{\text{attn}})$ at each step. In v9, this should be near zero (the forces are orthogonal — the Hopfield gradient pulls toward position-dependent attractors while attention pulls toward content-dependent ones). In v10, the cosine similarity should be positive (both forces are context-aware, pulling in similar directions).

### 7.4 Is the Gauge Curvature Nonzero?

**Test 8: Small-loop holonomy estimator.** Implement Algorithm 2 from the gauge fiber bundle paper. For two nearby sequences that differ only by a transposition of two adjacent tokens, measure the gauge displacement:

$$\Delta_\Omega = T_{A \to B \to D} - T_{A \to C \to D}$$

In a flat geometry, this is zero. In a curved geometry, it measures the curvature tensor $\Omega(u, v)$. The contextual manifold should produce nonzero holonomy — the order of tokens matters because $q_t$ is path-dependent.

---

## Part VIII: Open Research Questions

### 8.1 Does the Contextual Manifold Subsume the Cross-Position Forces?

If $q_t$ encodes the full sequence history, does the Hopfield gradient (parameterized by $M_{q_t}$) provide sufficient context-awareness to make the causal convolution and attention forces redundant? Or are they still needed for fine-grained refinement?

*Hypothesis:* The Hopfield gradient provides coarse-grained context (which attractor basin to fall into), while the cross-position forces provide fine-grained adjustment (where within the basin to settle). This is analogous to the two-phase dynamics of the brain: fast feedforward categorization followed by slower recurrent refinement.

*Experiment:* Train three variants — (A) contextual manifold + all 4 forces, (B) contextual manifold + Hopfield only (pure SDE), (C) contextual manifold + Hopfield + attention only. Compare accuracy and BPC. If (B) matches (A), the other forces are redundant and the architecture simplifies dramatically.

### 8.2 What is the Optimal Manifold Dimensionality?

Currently $d_m = 128$. A contextual accumulator might need more dimensions to encode rich history, or fewer if the history is compressible. The Anthropic paper found that character counts are represented in 6 dimensions. Word-level semantics likely require more, but how many?

*Experiment:* Sweep $d_m \in \{32, 64, 128, 256, 512\}$ and measure (a) probe accuracy on $q_t$, (b) downstream accuracy. If there's a phase transition (accuracy jumps sharply at some $d_m$), it reveals the intrinsic dimensionality of the contextual manifold for the task.

### 8.3 Should the Langevin Settling Still Enforce Numerical Sparsity?

v9 removed soft-thresholding, arguing that attention handles routing and Hopfield settling is "functionally sparse in attractor space" even with dense outputs. With the contextual manifold, the Hopfield gradient is itself context-aware. Does numerical sparsity (explicit zeros) still serve a purpose?

*Argument for sparsity:* The architectural philosophy is biologically inspired (lateral inhibition, sparse coding). Numerical sparsity enables the subspace routing of v6 — tokens with disjoint supports can't interfere. Without it, the subbundle decomposition loses its orthogonality guarantee.

*Argument against:* Soft-thresholding destroys gradient signal. The Langevin noise overwhelms the threshold in early steps. And with contextual manifold coordinates, the memory bank already provides implicit routing through atom selection — explicit sparsity is redundant.

*Resolution:* Reintroduce sparsity but only at the *final* Langevin step (after all noise has been removed), as v8 did. This preserves gradient flow during settling while producing sparse outputs for downstream use.

### 8.4 Can the Architecture Scale?

The most honest question. At 3.7M parameters, Vega produces BPC 2.65 on Tiny Shakespeare. A Transformer of similar size produces BPC ~1.5. The gap is large. Is it solely due to the context-blind manifold, or are there deeper capacity issues?

*Candidates for the gap beyond the manifold:*
1. **Sequential Langevin steps**: 4 blocks × 6 steps = 24 sequential operations per token, vs. 4 parallel layers in a Transformer. Each Langevin step is computationally equivalent to a shallow attention layer, but they can't be parallelized across steps.
2. **Memory bank capacity**: 8 subbundles × 128 atoms × 32-dim = 32K attractor dimensions, vs. an FFN with 256 × 1024 = 262K parameters. The memory bank may simply have less capacity.
3. **Gradient flow**: Backpropagation through 24 chained Langevin steps (even with deep supervision) is harder than through 4 residual Transformer layers. The effective gradient path length is much longer.
4. **Information bottleneck**: The Langevin settling is fundamentally lossy — projecting onto the nearest attractor discards information that doesn't align with any dictionary atom. The Transformer's residual stream preserves all information.

*Scaling experiments needed:* Match parameter count and FLOPs between Vega and a Transformer baseline on the same data. Measure where the capacity goes. If the contextual manifold closes most of the gap, the remaining difference is architectural; if not, the capacity issues need separate solutions.

### 8.5 Is There a Principled Way to Initialize the Contextual Manifold?

The parallel scan parameters $A(x), B(x), \psi(x)$ need initialization. Mamba uses specific schemes (e.g., HiPPO initialization for $A$) that encode mathematical priors about what the state should remember.

For a manifold coordinate, principled priors might be:

- **HiPPO-LegS**: Initialize $A$ to approximate the Legendre polynomial basis of the history — the state $q_t$ becomes a projection of the token history onto a polynomial basis, providing optimal approximation of continuous signals up to time $t$.
- **Slow decay**: $A(x) \approx 1 - \epsilon$ (small forget rate) — the manifold coordinate changes slowly, accumulating history with minimal loss. Each token adds a small perturbation.
- **Fourier initialization**: Initialize $A$ as rotation matrices in 2D blocks (cosine/sine pairs with different frequencies), so $q_t$ naturally encodes multi-scale temporal structure. This connects to the Fourier structure of the counting manifolds found by Gurnee et al.

---

## Part IX: Implementation Roadmap

### Phase 1: The Minimal Contextual Manifold (Restore the Wilson Line)

Replace `nn.Embedding(max_seq_len, manifold_dim)` with a parallel scan:

```python
class ContextualManifold(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.A_proj = nn.Linear(cfg.fiber_dim, cfg.manifold_dim)
        self.B_proj = nn.Linear(cfg.fiber_dim, cfg.manifold_dim)
        self.psi_proj = nn.Linear(cfg.fiber_dim, cfg.manifold_dim)

    def forward(self, x_sparse):
        # x_sparse: (B, T, D) — sparse token embeddings
        A = torch.sigmoid(self.A_proj(x_sparse))    # (B, T, d_m) — decay rates
        B = torch.sigmoid(self.B_proj(x_sparse))    # (B, T, d_m) — input gates
        psi = self.psi_proj(x_sparse)               # (B, T, d_m) — projected tokens

        # Parallel scan: q_t = A_t * q_{t-1} + B_t * psi_t
        q = parallel_associative_scan(A, B * psi)   # (B, T, d_m)
        return q
```

**Keep everything else from v9 unchanged.** The only modification is replacing the positional manifold embedding with the parallel scan. This isolates the effect of the contextual manifold.

**Expected result:** Significant accuracy improvement (break through 45% ceiling). The Hopfield gradient becomes context-aware, the routing forces are no longer fighting a deaf dominant force, and the memory bank selects atoms based on sequence history.

### Phase 2: Remove Redundant Components

With a contextual manifold, some v7-v9 mechanisms may be redundant:

1. **Test removing the causal convolution (Force 2):** If $q_t$ encodes history, the Hopfield gradient may provide sufficient context.
2. **Test removing the per-subbundle attention (Force 4):** If the manifold route already provides context, attention may be unnecessary overhead.
3. **Test simplifying the memory bank router:** With contextual $q_t$, a simpler router (e.g., single linear layer) may suffice, since $q_t$ already carries the information the router needs.

If Forces 2 and 4 become redundant, the architecture simplifies dramatically: each block becomes just a memory bank query (via $q_t$) followed by Langevin settling. This is closer to the original vision — pure geometric settling on a contextual energy landscape.

### Phase 3: Structural Innovations

Once the contextual manifold is validated:

1. Implement $SU(2)$ non-Abelian gauge group (Section 6.1)
2. Add curvature-aware memory bank (Section 6.2)
3. Experiment with self-synthesizing geometry (Section 6.3)
4. Implement horizontal gradient projection (Section 6.4)
5. Compare spectral vs. spatial Wilson line accumulation (Section 6.6)
6. Add manifold-aware dictionary learning (Section 6.7)

Each innovation is independently testable against the Phase 1 baseline.

### Phase 4: Scaling and Comparison

1. Match parameter count with a standard Transformer baseline
2. Compare BPC, accuracy, and perplexity on Tiny Shakespeare
3. Analyze what the manifold learns (probing experiments, Section 7.2)
4. Measure curvature (holonomy experiments, Section 7.4)
5. Scale to larger datasets if results are promising

---

## Part X: Summary of the Mathematical Evolution

```
Theoretical Framework (Architecture.md):
  Fiber bundles + gauge connections + Langevin-Hopfield + proximal sparsity
  q_t is a contextual coordinate on a Finsler manifold
  Wilson line accumulates sequence context as holonomy
  ↓
v1-v2: First implementation
  q_t = positional embedding [DEVIATION FROM THEORY]
  No cross-position mixing → 5-12%
  Wilson line accumulates position-only phases (meaningless)
  ↓
v3: Sequential ContextGate (GRU)
  Added GRU-style context accumulation at the TOKEN level
  h_t = gate(h_{t-1}, x_t) — implicitly contextualizes representations
  35% — best result, closest to the mathematical intent
  But sequential: can't parallelize training
  ↓
v4-v5: Parallel alternatives
  Replaced GRU with causal convolution (parallel but content-independent)
  v5 added attention but on wrong target (per-fiber gauge, not cross-position)
  18-20% — lost the content-dependent selectivity that v3 had
  ↓
v6: Sparsity as routing
  Hypothesis: sparse representations naturally route via subspace overlap
  Beautiful theory but presupposes the context-dependence it needs
  ~20%
  ↓
v7.0-v7.3: Unified Langevin
  Merged cross-position mixing into the Langevin loop
  Separated forces (v7.1), removed gauge (v7.2), added deep supervision (v7.3)
  36.5% — matched v3 via better gradient flow, not better representations
  ↓
v8.0-v8.2: Scaling to real text
  Per-subbundle memory, deeper manifold, more Langevin steps
  45% accuracy, BPC 2.65 on Shakespeare
  Plateaued — the representations are the bottleneck, not the routing
  ↓
v9: Per-subbundle attention
  Added content-dependent cross-position retrieval as 4th force
  98K attention parameters, operates inside Langevin loop
  45% — identical ceiling. Attention fights context-blind Hopfield gradient.
  ↓
v10: THE FIX
  q_t = parallel_scan(A(x_t), B(x_t), psi(x_t))
  The manifold coordinate is contextual. The Wilson line is restored.
  M_q_t is context-aware. The Hopfield gradient is context-aware.
  The routing forces can focus on refinement, not compensation.

  The manifold was always supposed to be contextual.
  Nine versions to find the real bottleneck.

  KEY DISCOVERY (all versions):
  Token construction = sparse manipulations on fiber bundle
  Per-subbundle top-k → quantifiable activation patterns
  Patterns = place cells tiling the fiber bundle
  Feature-geometry duality made concrete
  With contextual manifold: patterns become context-dependent fingerprints
```

### The Deepest Lesson

The original mathematical framework described a beautiful system: tokens as sparse sections of a fiber bundle, transported along a contextual manifold via gauge connections, settled into sparse attractors via Langevin dynamics. Every piece of this framework has been validated — by Gurnee et al.'s empirical observation that LLMs internally build curved manifolds with place-cell features, by the gauge fiber bundle paper's proof that attention *is* a connection with nonzero curvature, and by nine iterations of implementation showing that the routing mechanism is not the bottleneck.

The only piece that was never actually implemented was the foundation: the contextual manifold itself.

But alongside this diagnosis lies a key discovery that was hiding in plain sight across all nine versions: **the sparse token construction is itself a profound structural innovation**. Tokens in Vega are not dense vectors passed through a sparsity layer for efficiency — they are *sparse manipulations on the fiber bundle* whose composite activation patterns across orthogonal subbundles form a quantifiable, combinatorially vast, compositionally structured codebook. These patterns are the architectural realization of the place cells discovered by Gurnee et al., the concrete embodiment of the feature-geometry duality, and the bridge between discrete symbolic computation and continuous geometric transport. The sparsity is not a constraint; it is the *medium* in which representation occurs.

With the contextual manifold restored, these sparse patterns become *context-dependent fingerprints* — shaped not just by the current token but by the entire accumulated history encoded in $q_t$. The same token, in different contexts, crystallizes into different activation patterns across the fiber bundle. The pattern of which dimensions are active, in which subbundles, with what magnitudes, is a discrete, measurable summary of "this token, in this context" — something dense Transformer embeddings cannot provide.

The manifold was always the answer. It was written in the first line of Architecture.md:

> *"The system abandons the globally flat $\mathbb{R}^d$ topology of standard deep learning. Instead, computation occurs over a base manifold $\mathcal{M}$."*

But `nn.Embedding(128, 128)` is not a manifold. It is a lookup table wearing a manifold's name.

v10 gives the manifold its name back — and reveals that the sparse fiber bundle construction was always the architecture's most original contribution.

---

*Research document, March 15, 2026. David Ledbetter.*
*Supporting mathematics: arXiv:2601.04480v1 (Gurnee et al., Anthropic); ICLR 2025 submission 19168 (Anonymous).*
*The architecture was never wrong. The implementation was incomplete. The sparsity was never a compromise. It was the discovery.*
