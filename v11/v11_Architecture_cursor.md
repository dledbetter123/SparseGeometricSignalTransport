# V11 Architecture: Sparse Source Diffusion on a Context-Warped Fiber Bundle

## The Core Claim

A token is not a dense vector. A token is a **simultaneous pattern of sparse activations** — a distribution of electric potentials firing across decoupled fiber channels. These activations are **source terms** on a manifold whose geometry is shaped by context. The model's job is not to push a vector through a pipeline. It is to solve for the **full field** that those sparse sources induce on the manifold, and read off the next token from that field.

Everything follows from taking this seriously.

---

## 1. What a Token Actually Is

In standard transformers, a token is a dense vector in R^d. Every dimension carries a value. The vector moves through layers as a single object.

In this architecture, a token at position t is a **sparse section** of a fiber bundle. The fiber F_q at context point q is decomposed into K orthogonal subbundles:

    F_q = F_q^(1) + F_q^(2) + ... + F_q^(K)

Within each subbundle, only a small subset of dimensions are active (non-zero). The token state is the pair (S, a) — support set and amplitudes. Most of the fiber is silent.

This is not a regularization trick. The zeros are not "small values we rounded down." The sparse pattern IS the token. Different tokens at different contexts activate different subsets of dimensions across the subbundles. The activation pattern is a combinatorial fingerprint — with K subbundles each choosing k active dimensions from d_k, the space of possible patterns is the product of C(d_k, k) across subbundles.

**A token is many sparse things happening at once across decoupled channels, not one dense thing.**

The biological analogy: a cortical column doesn't fire a single "meaning vector." It produces a sparse pattern of activity across layers and minicolumns. The pattern is the representation.

---

## 2. The Alcubierre Principle: Geometry Does the Transport

In an Alcubierre drive, the ship doesn't move through space. Space itself warps — contracting ahead, expanding behind — and the ship rides the geometry. The ship is locally stationary; the metric does the work.

The same principle governs this architecture. The sparse activations don't "travel" from one position to another. Instead, context modulates the geometry of the fiber bundle — the metric tensor, the connection — so that:

- **Semantically related tokens** are geometrically close, regardless of their temporal distance in the sequence. The metric contracts the space between them.
- **Irrelevant tokens** are geometrically distant. The metric expands the space, making their influence decay.

The context history (all preceding tokens) collectively defines the shape of this warp. It is the energy-momentum tensor that sources the geometry. A new token doesn't need to "look at" every previous token (attention). It just needs to exist on the manifold whose shape already encodes those relationships.

This is the key departure from attention: attention explicitly computes pairwise relationships between tokens. The Alcubierre principle says those relationships are already implicit in the geometry. You discover them by letting a physical process (diffusion) run on that geometry.

---

## 3. Diffusion as Manifold Estimation

This is the central computational mechanism and the thing that previous versions got wrong.

The sparse activations across all positions in the sequence are **source terms** for a diffusion process on the fiber bundle. They are not inputs to a pipeline. They are boundary conditions for a PDE.

The diffusion process — governed by the heat kernel / Laplace-Beltrami operator on the context-warped manifold — propagates the influence of every sparse source through the geometry. The solution at any point on the manifold integrates information from all sources, weighted by the geodesic distance (which is context-dependent, per the Alcubierre principle).

This is why any token can commute relationships with any other token: they all contribute to the same field. The diffusion solution at position t is not "the embedding at position t after some mixing." It is the **full manifold state estimated from all sparse observations**, evaluated at position t.

Concretely:

1. Each token contributes sparse activations — point sources on the manifold.
2. The context-dependent metric determines how those sources propagate.
3. The diffusion equation is solved (or approximated) over the bundle.
4. The solution at every point reflects the integrated influence of all sources, shaped by the geometry.

This replaces attention entirely. Attention asks: "how much should token i attend to token j?" Diffusion asks: "given all the sources and the geometry, what is the field here?" The second question is more physical, more parallel, and doesn't require explicit pairwise computation.

### What "Solving the Diffusion Equation" Means in Practice

The heat kernel on a Riemannian manifold with metric g is:

    du/dt = div_g(grad_g(u))

where div_g and grad_g are the metric-dependent divergence and gradient. The sparse activations provide initial/boundary conditions. The solution u(q, t) at manifold point q and diffusion time t gives the integrated field.

In the spectral domain (eigenbasis of the Laplace-Beltrami operator), this becomes diagonal — each eigenmode decays at a rate proportional to its eigenvalue. Fast-decaying modes capture local structure; slow-decaying modes capture global structure. Multi-scale information integration falls out naturally.

The context-dependent metric means the eigenbasis itself changes with context. Different contexts produce different geometries, different eigenmodes, different propagation patterns. The same sparse source pattern produces different fields under different contexts.

---

## 4. The Memory Landscape and Energy Settling

After diffusion estimates the full field, the signal at each point is a noisy, diffused mixture of influences. It needs to resolve into a sharp, sparse state — a definite token prediction.

This is where the Hopfield/Langevin settling operates. A context-dependent memory bank M_q defines the attractor landscape at point q. The diffused field state is the initialization. Langevin dynamics (reverse diffusion in the energy landscape) settles the state into a deep attractor:

    x_new = x_old - lr * grad E_q(x) + noise

The noise provides simulated annealing — escaping shallow basins (hallucinations) to find deep, globally consistent attractors.

The proximal operator (soft thresholding) enforces sparsity at each settling step, acting as lateral cortical inhibition. Weak activations are driven to exact zero. The settled state is a clean, sparse section of the fiber — ready to serve as a source term for the next layer or the next position.

---

## 5. What v1-v10 Got Wrong

### The "diffusion" was just smoothing
Previous versions used exponential-decay convolution as a "diffusion mixer." This applied the same blurring kernel regardless of which dimensions were active. It didn't treat sparse activations as sources. It didn't solve a PDE. It was a 1D causal convolution with a learned decay rate — essentially a low-pass filter across positions.

### Tokens were still treated as dense vectors
Despite the sparse embedding, the processing pipeline (mixing, settling, FFN) operated on the full vector. The zeros weren't meaningful to the computation — they were just values that happened to be zero. The architecture didn't distinguish "this dimension is active and carries information" from "this dimension is silent."

### No manifold estimation
The diffusion didn't estimate anything. It mixed. The Langevin settling didn't condition on the global field — it operated per-token. There was no shared field being computed from distributed sparse observations. Each token was processed independently after a local smoothing step.

### The geometry wasn't doing transport
The context-dependent metric (the Alcubierre warp) was either absent (v10.3-v10.4) or was a state-space model producing manifold coordinates (v10.2). Neither actually warped the propagation of signals. The geometry and the transport were disconnected.

---

## 6. V11 Design Principles

1. **Sparse activations are source terms, not vectors with zeros.** The architecture must distinguish active from inactive dimensions. Active dimensions seed the diffusion. Inactive dimensions are unknowns to be estimated.

2. **Diffusion solves for the full field.** The output of diffusion is not a "mixed" version of the input. It is the solution to the heat equation on the manifold, conditioned on all sparse sources across the sequence.

3. **Context warps the geometry.** The metric tensor (or equivalently, the Laplace-Beltrami operator, or equivalently, the graph Laplacian on a learned graph) is a function of context. Different contexts produce different propagation patterns.

4. **No explicit pairwise computation.** Token-to-token relationships are discovered through the field, not through dot products. If two tokens are geometrically close (metric contraction), they strongly influence each other's field values. If they're far (metric expansion), they don't.

5. **Settling produces sparse outputs.** The Langevin/Hopfield phase resolves the diffused field into sharp, sparse states. These become the source terms for the next stage. Sparsity is maintained end-to-end — it is the fundamental data type, not a post-hoc constraint.

6. **The bundle structure keeps channels decoupled.** Subbundles prevent representational collapse. Different feature channels (syntax, semantics, routing) propagate independently through the geometry, with different effective metrics if needed.
