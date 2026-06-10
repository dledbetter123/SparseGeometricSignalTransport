# Mathematical Foundations: Why This Architecture Should Work

## The Core Claim

Language representations have intrinsic curvature. GNN message passing destroys it (rank collapse). Spectral methods on fiber bundles make the geometry computationally tractable.

Standard transformers already build manifolds — Anthropic's "When Models Manipulate Manifolds" (Gurnee et al.) proved this empirically: 1D curved manifolds for counting (helical, 6-dimensional, discretized by "place cell" features), QK matrices as geometric rotations ("twists") aligning coordinate frames, orthogonal subspaces for decision boundaries. SGST's position: Anthropic had to use interpretability probes to reverse-engineer this from a trained Transformer. SGST builds an architecture where manifold manipulation IS the explicit forward pass.

---

## The Five Axioms (from Architecture.md)

### Axiom 1: Fiber Bundle Topology

Computation occurs over a base manifold M (possibly Finsler). At each contextual coordinate q in M, a local fiber F_q provides the representational space. The total space is a fiber bundle E = coprod_q F_q.

Tokens are sparse sections of this bundle, decomposed into K orthogonal subbundles:

```
F_q = F_q^(1) + F_q^(2) + ... + F_q^(K)
x_q = (S_q, a_q)  -- support and amplitudes
```

### Axiom 2: Spectral Gauge-Covariant Transport

Because fibers at different points are distinct vector spaces, transport requires a gauge connection A:

```
X_tilde_q = X_p * exp(-D*w^2 - i*w * integral_gamma A)
```

- The advective term `exp(-i*w * integral A)` is the U(1) holonomy (Wilson line) — content-dependent phase rotation encoding contextual history as geometric curvature.
- The diffusive term `exp(-D*w^2)` is the heat kernel — frequency-dependent damping providing natural scale separation.

### Axiom 3: Dynamic Memory Bank

Local attractor landscape M_q carved from a global dictionary via geometric gating:

```
g(q) = k-WTA(W_route * q)
```

### Axiom 4: Langevin-Hopfield Energy Descent

Annealed Langevin dynamics on the Modern Hopfield energy:

```
x_{t-dt} = x_t - eta * grad_x E_q(x_t; M_q) + sqrt(2*eta/beta_t) * epsilon_t
```

The Hopfield gradient is mathematically identical to softmax attention:

```
grad_x E = -(1/beta) * softmax(beta * M^T * x) * M = -weighted_sum(memories)
```

This is attention without QK projections — the energy landscape provides content-addressable retrieval.

### Axiom 5: Proximal Sparsity

Soft-thresholding enforces lateral inhibition at each Langevin step:

```
prox_lambda(x) = sign(x) * max(|x| - lambda, 0)
```

This is the ISTA algorithm (Daubechies et al. 2004) embedded inside the Langevin SDE — a stochastic, annealed variant of compressed sensing recovery.

---

## The Forward-Reverse Loop (Fourier Duality)

The architectural heartbeat. In v11 it was conceptualized as:

```
Sparse sources -> Field reconstruction (diffusion) -> Dense cloud -> Langevin settling -> Sparse output
```

V12 revealed this as Fourier duality:

```
Sparse spectral config  ->  IFFT  ->  Dense spatial field  ->  Spectral Langevin  ->  Sparse spectral config
    (few active w)        (field         (full manifold)       (attractor              (few active w)
                        reconstruction)                        descent)
```

The IFFT IS the "diffusion as field reconstruction." The dense spatial cloud IS the superposition of a few frequency modes. The Langevin settling IS projection back to the nearest sparse spectral attractor.

The heat kernel `exp(-D*w^2)` naturally occurs when spectral sources propagate through a finite-bandwidth medium. The gauge connection `exp(-i*w * integral A)` is the frequency-dependent phase shift through curved geometry. Both act directly on the spectral representation — no FFT/IFFT needed for transport.

**Why this matters:** In v11, forward (diffusion) and reverse (Langevin) were conceptually separate operations chained by initialization. In v12+, they are two halves of a single mathematical operation.

---

## Spectral Sparsity vs. Spatial Sparsity

### The Donoho-Stark Uncertainty Principle

```
|supp(x)| * |supp(x_hat)| >= d
```

A spatially sparse signal (few active spatial dimensions) must be spectrally spread (occupying all frequency bins). This means:
1. Transport is inefficient — must act on all frequency bins, O(d log d)
2. The heat kernel fights the representation — spatially sparse signals have flat spectra; the kernel destroys their structure
3. No natural scale separation

### Resolution via Spectral Sparsity (V12 Breakthrough)

| Property | Spatial Sparsity (v11) | Spectral Sparsity (v12+) |
|---|---|---|
| Token representation | Few active spatial dims | Few active frequency modes |
| Spatial extent | Localized (point sources) | Global (each mode spans all positions) |
| Transport efficiency | O(d log d) — all bins active | O(s) — only s << d bins active |
| Heat kernel effect | Destroys structure | Selectively modulates structure |
| Multi-scale hierarchy | Must be imposed externally | Intrinsic (each mode has a wavelength) |
| Training locality | Gradients hit all spatial dims | Gradients hit only active spectral modes |
| Routing locality | Requires proximity or attention | Spectral overlap determines interaction |

### Compressed Sensing Guarantee (Candes-Romberg-Tao 2006)

A signal with s-sparse spectral support can be perfectly recovered from O(s log d) spatial measurements. The Fourier and standard bases are maximally incoherent (mu = 1/sqrt(d)). The Restricted Isometry Property guarantees stable recovery under noise via basis pursuit denoising.

Langevin + spectral proximal approximates this optimization: it is a "stochastic, annealed, attractor-regularized variant of compressed sensing recovery."

---

## Constellations and Mode Fibers (V12.5)

David's core principle: "A few dots on the Fourier space. Mathematically, it's how many unique connections we can make."

### Constellation Geometry

A **constellation** is a token's identity: a sparse pattern of (magnitude, phase) pairs across spectral modes. With the current parameterization (8 subbundles x 17 modes per subbundle, 6/17 active per subbundle):

```
C(17,6)^8 ~ 5.7 x 10^32 possible support patterns for 65 characters
```

### Mode Fibers as Communication Channels

Each of the 136 modes carries a state that evolves across the sequence. When token A deposits its magnitude+phase at mode m, and token B later reads from mode m, they are geometrically connected through that mode — no attention needed.

### The Manifold Is Explicit

- **Topology** = which modes are active (discrete, changes at sparsification)
- **Metric** = values at shared modes (continuous, Parseval inner product)
- **Curvature** = mode activation/deactivation dynamics across the sequence
- **Geodesic** = constellation evolution through blocks

The network learns three things:
1. Which dots define each token (embedding)
2. What happens when dots overlap (update rule at shared modes)
3. When to move dots (how constellations shift through blocks)

### Parseval Inner Products Give Triple Geometric Content

`Re(h * conj(c))` simultaneously provides:
- **Metric**: magnitude overlap between accumulated past and current token
- **Connection**: phase rotation = parallel transport on S^1 per mode
- **Curvature**: how phase alignment varies across modes

One bilinear operation encodes three geometric objects.

---

## Key Theorems and Principles Referenced

1. **Donoho-Stark Uncertainty Principle**: `|supp(x)| * |supp(x_hat)| >= d`. Motivates spectral sparsity.

2. **Candes-Romberg-Tao (2006)**: s-sparse spectral recovery from O(s log d) measurements. Validates spectral sparsity as information-complete.

3. **Restricted Isometry Property**: Stable recovery under noise. Langevin + proximal approximates this.

4. **Parseval's Theorem**: Energy conservation under Fourier transform. Spectral inner products = spatial inner products.

5. **Modern Hopfield Energy (Ramsauer et al. 2021)**: `E_q(x; M_q) = -beta^{-1} log sum_j exp(beta * x^T * m_j)`. Gradient = softmax attention.

6. **Gauge Fiber Bundle Geometry of Transformers (NeurIPS 2025, Paper 19168)**: Attention induces Ehresmann connection with nonzero curvature (Theorem 4.1). Transport around a rectangle produces nontrivial gauge displacement = path dependence = context sensitivity.

7. **Rank Collapse in GNNs (Oono & Suzuki 2020)**: `d_M(X^(l)) = O((s*lambda)^l)`. Message passing exponentially collapses representational rank. Fundamental limitation of GNN approaches.

8. **Gibbs Phenomenon**: Ringing in Anthropic's counting manifolds = truncated Fourier series. V12 makes this explicit by defining tokens as sparse Fourier configurations.

---

## External Validation

**Anthropic's "When Models Manipulate Manifolds"** empirically proves standard transformers build:
- 1D curved manifolds for counting (helical, 6-dimensional)
- QK matrices as geometric rotations aligning coordinate frames
- Orthogonal subspaces for decision boundaries

The project's thesis: these geometric structures should be the architecture's primitives, not emergent artifacts discovered post-hoc through interpretability probes.
