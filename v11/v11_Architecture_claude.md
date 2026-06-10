# V11 Architecture: Sparse Source Field Reconstruction on a Context-Warped Bundle

## The Correction

Every version from v1 through v10.4 made the same fundamental error: treating
the token as a dense vector that gets processed through a pipeline. The sparse
embedding was a preprocessing step. The diffusion was a smoothing filter. The
Langevin was a per-token optimizer. Each component operated on "the token" as
a single object.

A token is not an object. A token is an **event** — many sparse activations
firing simultaneously across decoupled fiber channels. Each activation is a
point source of potential on the manifold. The token's identity is the *pattern*
of which sources fire and where, not a vector that summarizes them.

Everything in v11 follows from taking this literally.

---

## 1. Sparse Activations as Source Terms

A token at position t, in context q, is a sparse section of the fiber bundle:

    x_q = (S_q, a_q)    where S_q is the support, a_q the amplitudes

The fiber decomposes into K orthogonal subbundles:

    F_q = F_q^(1) ⊕ F_q^(2) ⊕ ... ⊕ F_q^(K)

Within each subbundle, only a small subset of dimensions are active. The rest
are not "small" — they are **absent**. They carry no information and seed no
field. The active dimensions are independent source terms, like point charges
on a conductor or heat sources on a plate.

This distinction matters computationally:

- **Dense vector view**: All 512 dimensions participate in every operation.
  Zeros are just values. Matmuls hit every element.
- **Sparse source view**: Only the ~128 active dimensions (across all
  subbundles) seed the diffusion. Inactive dimensions are unknowns to be
  solved for, not values to be processed.

The combinatorial identity space is enormous. With K=8 subbundles, each
selecting k=16 active dimensions from d_k=64, the number of possible support
patterns per subbundle is C(64,16) ≈ 4.9 × 10^13. Across 8 independent
subbundles, the total pattern space is (C(64,16))^8 — vastly larger than any
vocabulary. The sparse pattern IS the representation. It doesn't encode a
meaning; it is the meaning's fingerprint on the fiber.

### What "Simultaneous" Means

When we say a token is "many sparse activations happening simultaneously," we
mean:

- Subbundle 1 might activate dims {3, 17, 42, ...} — encoding syntactic role
- Subbundle 2 might activate dims {8, 11, 55, ...} — encoding semantic class
- Subbundle 3 might activate dims {1, 29, 60, ...} — encoding phonetic pattern
- ...

These are not facets of a single vector. They are independent signals on
independent channels. They propagate independently through the geometry (each
subbundle can have its own effective metric). They source independent
components of the field. And they settle into independent attractors during
the Langevin phase.

The subbundle orthogonality guarantee means they cannot interfere. This is
the mathematical enforcement of what cortical columns do biologically:
different feature channels (color, orientation, motion in V1; syntax,
semantics, pragmatics in language areas) process in parallel without crosstalk.

---

## 2. The Alcubierre Geometry

The fiber bundle's geometry — its metric, its connection — is not fixed. It
is sourced by context. The entire preceding sequence collectively determines
the shape of the manifold, like the energy-momentum tensor determines
spacetime curvature in GR.

The Alcubierre analogy is precise:

- **The warp bubble** = the context-dependent metric on the base manifold.
  Context contracts the geodesic distance between semantically related
  positions and expands it between unrelated ones.
- **The ship** = the current token's sparse activations. They don't move
  through the manifold. They exist at a point, and the geometry around
  that point determines how they influence (and are influenced by) every
  other point's activations.
- **The drive** = the accumulated gauge connection (Wilson line holonomy).
  It encodes the contextual history as geometric curvature.

The critical implication: **no explicit pairwise computation is needed.**
In attention, token i must explicitly query token j to discover their
relationship. On the warped manifold, their relationship is already encoded
in the metric. If context makes them geometrically close, diffusion will
automatically carry signal strongly between them. If context makes them
distant, their mutual influence is suppressed by geometric expansion. The
O(N²) pairwise comparison is replaced by O(N log N) field propagation on
the curved space.

### How Context Warps the Geometry

The metric at position q is not a fixed exponential decay. It is a function
of the accumulated context. Concretely, the diffusion coefficients D and the
gauge connection A in the transport operator:

    X̃_q = X_p ⊙ exp(-D(q)ω² - iω ∫_γ A(q))

are both functions of the context embedding at q. Different contexts produce:

- Different D(q): changing which frequency scales propagate (local vs global
  structure)
- Different A(q): changing the phase relationships between positions (which
  tokens are "aligned" in the warped space)

The Anthropic paper (2601.04480) empirically demonstrated that trained LLMs
already compute these geometric rotations — attention QK matrices act as
gauge connections that "twist" manifolds to align coordinate frames. V11
makes this the explicit forward pass rather than an emergent property.

---

## 3. Diffusion as Field Reconstruction

This is where v1-v10 went wrong. The "diffusion" in those versions was a
causal exponential-decay convolution — a 1D smoothing filter. It blurred
neighboring positions together with fixed weights. It did not solve a field
equation. It did not reconstruct anything.

The diffusion in v11 is fundamentally different. It is the **forward process**
that takes sparse source activations and estimates the full manifold state.

### The Physical Analogy

Imagine a metal plate (the manifold). You touch it with a few hot probes at
specific points (the sparse activations). Heat diffuses from the probes
through the metal, and after some time, every point on the plate has a
temperature that reflects the integrated influence of all probes, weighted
by their distance through the metal's geometry.

If the plate is uniform, nearby probes dominate. But if the plate has variable
conductivity — if some regions conduct heat well (contracted metric) and
others are insulating (expanded metric) — then the temperature field reflects
the geometry, not just spatial proximity.

The sparse activations are the probes. The context-dependent metric is the
variable conductivity. The diffusion solution is the temperature field. And
that field, evaluated at any point, tells you what the manifold "looks like"
at that point given all the evidence from all the sources.

### The Mathematical Pipeline

**Step 1: Sparse sources.** Each token contributes its active dimensions as
point sources on the manifold. Inactive dimensions contribute nothing.

**Step 2: Spectral projection.** Push the source field into the eigenbasis
of the context-dependent Laplace-Beltrami operator (or, as a tractable
approximation, the FFT over the fiber dimension):

    X_p = F(x_p)

**Step 3: Field propagation.** Apply the heat kernel + gauge transport in the
spectral domain. This is where context warps the geometry:

    X̃_q = X_p ⊙ exp(-D(q)ω² - iω ∫_γ A(q))

Fast modes (high ω) decay quickly — they capture local correlations (bigrams,
adjacent characters). Slow modes (low ω) persist — they capture long-range
dependencies (topic, speaker identity, narrative arc). The multi-scale
structure falls out of the physics.

**Step 4: Pullback.** IFFT back to the spatial fiber:

    x̃_q = F⁻¹(X̃_q)

This x̃_q is the **reconstructed field** — the dense, continuous estimate of
the manifold state at point q, given all sparse sources. It is not a smoothed
version of the input. It is a solution to the diffusion equation. Every
dimension now carries information, including the ones that were zero in the
original sparse source.

This x̃_q is the "blurry, dense superposition" / "continuous probability
cloud" described in CLMWithArch.md. It represents all the semantic concepts
that could validly exist at this point on the manifold, superimposed.

---

## 4. Langevin Settling: Cloud → Sparse Attractor

The diffused field x̃_q is dense and noisy. It must resolve into a sharp,
sparse state — a definite token with a definite support pattern.

This is the **reverse diffusion** phase: Langevin dynamics in the Hopfield
energy landscape.

**Step 1: Construct the attractor landscape.** A context-dependent memory
bank M_q defines the valid sparse attractors at point q. It is dynamically
constructed from a global overcomplete dictionary via k-WTA routing:

    g(q) = k-WTA(W_route · q)
    M_q = D[top-k indices]

Neighboring points on the manifold share overlapping atom sets (topographic
continuity / place cells). Distant points have orthogonal atoms.

**Step 2: Initialize from the diffused field.** The Langevin loop starts at
x̃_q — the reconstructed field, NOT the original sparse input. This is
critical. v10.3/v10.4 initialized Langevin from the raw sparse embedding,
completely disconnecting the forward and reverse processes.

**Step 3: Annealed descent with proximal sparsity.**

    for each step (β increasing):
        grad_E = -softmax(β · xᵀ M_q) · M_q    # Hopfield score (implicit attention)
        inhibit = -γ · W_inh · x                 # lateral cortical inhibition
        noise = √(2η/β) · ε                      # simulated annealing
        x = x - η·grad_E + inhibit + noise        # Langevin step
        x = sign(x) · max(|x| - λη, 0)           # proximal re-sparsification

The proximal operator fires at **every step**, not just the last one (as v10
incorrectly did). It progressively drives weak activations to exact zero,
enforcing sparsity throughout the reverse process. The annealing schedule
(increasing β) sharpens the softmax from a broad distribution to a peaked
one, collapsing the cloud into a single dominant attractor.

**Step 4: Output.** The settled state x_0 is a clean sparse section of the
fiber — a definite pattern of activations across subbundles. It is ready to:
- Serve as a source term for the next block's diffusion
- Be decoded into vocabulary logits for next-token prediction

---

## 5. The Full Forward Pass

For a sequence of tokens [t_1, t_2, ..., t_T]:

```
1. EMBED: Each token → sparse section of the fiber bundle
   - Token embedding + sinusoidal position
   - Per-subbundle top-k sparsification
   - Active dimensions = source terms; inactive = unknowns

2. For each block (× N_blocks):

   a. FORWARD DIFFUSION (field reconstruction):
      - Sparse sources → spectral domain (FFT)
      - Apply heat kernel + gauge transport: exp(-D(ctx)ω² - iω·A(ctx))
      - Pullback to spatial domain (IFFT)
      - Output: dense reconstructed field x̃ (the "cloud")

   b. MEMORY ROUTING:
      - Use x̃ (the field) to select relevant atoms from global dictionary
      - k-WTA routing constructs local attractor landscape M_q

   c. REVERSE DIFFUSION (Langevin settling):
      - Initialize from x̃ (NOT from original sparse input)
      - Annealed Hopfield descent + lateral inhibition + noise
      - Proximal soft-thresholding at EVERY step
      - Output: clean sparse section x_0

   d. x_0 becomes the sparse source for the next block

3. DECODE: Final sparse state → vocabulary logits
```

Each block performs a full forward→reverse diffusion cycle:
sparse sources → field reconstruction → attractor settling → sparse output.
The sparsity is maintained end-to-end. Dense representations only exist
transiently, inside the diffusion solution, as the "cloud" that must be
resolved.

---

## 6. What v1-v10 Got Wrong (Specific Failures)

### Diffusion was smoothing, not field reconstruction
The CausalDiffusionMixer in v10.3/v10.4 applied an exponential-decay causal
convolution: `kernel[t,d] = exp(-decay[d] * t)`, normalized and FFT-convolved.
This is a 1D low-pass filter across positions. It does not solve the heat
equation from sparse sources. It does not produce a field estimate. It
produces a blurred average of neighbors.

### Langevin was disconnected from diffusion
v10.3/v10.4 passed the original sparse embedding `h` to Langevin, not the
diffused output. The "forward" and "reverse" processes were operating on
different signals. There was no forward→reverse loop. The diffusion output
went only to the router, not to Langevin.

### Tokens were processed as dense vectors
Despite sparse embedding, every subsequent operation (mixer, routing,
settling, FFN, residual gate) operated on the full fiber dimension. No
distinction was made between active sources and inactive unknowns. The
matmuls didn't know or care which dimensions were zero.

### Proximal operator only fired once
v10.3/v10.4 applied soft-thresholding only on the final Langevin step.
Architecture.md and notes_claude_3101003pm.md both specify it should apply
at every step. Progressive re-sparsification throughout the reverse process
is the mechanism that enforces the sparse-section-of-a-fiber-bundle structure
at all times, not just at the output.

### No context-dependent geometry
The diffusion kernel was fixed (learned but position/content-independent).
The same exponential decay applied regardless of context. There was no
gauge connection A, no Wilson line, no context-dependent metric. The
"Alcubierre warp" — the thing that makes semantically related tokens
geometrically close — was completely absent.

### The residual was lazy
v10.3 used a sigmoid gate: `gate * settled + (1-gate) * x`. The gate
learned to stay near 0, making blocks pass-through. This defeated the
purpose of stacking blocks — the Langevin settling was being ignored in
favor of just passing the input through unchanged.

---

## 7. V11 Non-Negotiables

1. **Sparse in, sparse out.** Every block receives sparse sources and
   produces sparse outputs. Dense states exist only transiently inside the
   diffusion solution.

2. **Diffusion IS field reconstruction.** The forward pass solves (or
   approximates) the heat equation on the context-warped manifold with
   sparse boundary conditions. It is not a filter. It is not smoothing.

3. **Langevin starts from the field.** The reverse process must be
   initialized with the diffused field x̃, not the raw sparse input. The
   forward→reverse loop must be connected.

4. **Context warps the metric.** D and A are functions of context. The same
   sparse sources produce different fields under different contexts. This
   is the Alcubierre principle.

5. **Proximal sparsity at every Langevin step.** Not just the last one.
   Progressive re-sparsification is the mechanism.

6. **Subbundles are independent channels.** Different semantic features
   propagate independently through the geometry and settle into independent
   attractors. The orthogonality is structural.

7. **No pairwise attention.** Token relationships are discovered through
   field propagation on the curved manifold, not through explicit dot
   products. The geometry does the work.
