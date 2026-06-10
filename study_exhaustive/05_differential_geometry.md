# Unit 05: Manifolds, Metrics, Curvature, and Geodesics

## Prerequisites
Units 01-02 (Linear Algebra, Calculus on Vector Spaces)

---

## Learning Objectives

By the end of this unit, you should be able to:

1. Define what a smooth manifold is and work with coordinate charts
2. Understand tangent vectors, tangent spaces, and vector fields
3. Define and compute Riemannian metrics and inner products on tangent spaces
4. Compute Christoffel symbols, geodesics, and curvature
5. Understand the manifold hypothesis for language (thesis Sec. 2.4)
6. See how curvature captures geometric structure that flat embeddings miss

---

## Readings

**Primary Texts:**
- do Carmo, *Riemannian Geometry*, Ch. 1-4 (or Lee's *Smooth Manifolds* Ch. 1-3, 5, 13)

**Video Lectures:**
- Eigenchris YouTube: "Tensor Calculus" playlist (episodes 1-15) -- outstanding visual intuition
- Frederic Schuller: "Geometrical Anatomy of Theoretical Physics" lectures 1-8 (YouTube)

**Thesis and Repo:**
- Thesis Sec. 2.4 (The Manifold Hypothesis for Language)
- Thesis Ch. 4 (The Geometric Turn: Finsler Transformer), especially Sec. 4.1-4.3
- Repo: `topology/metrics_and_transport.md` (comparative study of metric types)

**Papers:**
- `papers/gurnee2025_when_models_manipulate_manifolds.pdf`
- `papers/modell2025_origins_representation_manifolds.pdf`

---

## Key Concepts

### 1. Manifold

A **manifold** is a space that locally looks like $\mathbb{R}^n$. Think of the surface of the Earth: zoom in enough and it looks flat (like $\mathbb{R}^2$), even though globally it is curved and finite.

Formally, a smooth manifold $M$ of dimension $n$ is a topological space where every point has a neighborhood homeomorphic to an open subset of $\mathbb{R}^n$, and the transition maps between overlapping neighborhoods are smooth (infinitely differentiable).

Key intuition: a manifold has no preferred coordinate system. You can describe locations on Earth with latitude/longitude, or with UTM coordinates, or with any other system. The manifold exists independently of how you parameterize it.

### 2. Charts and Atlases

A **chart** $(U, \phi)$ is a pair: an open set $U$ in $M$ and a homeomorphism $\phi: U \to \mathbb{R}^n$. This is a local coordinate system.

An **atlas** is a collection of charts that covers all of $M$. Where two charts overlap, the transition map $\phi_2 \circ \phi_1^{-1}$ must be smooth. This is what makes $M$ a *smooth* manifold.

Example: The sphere $S^2$ cannot be covered by a single chart (you cannot flatten a sphere without tearing it). You need at least two charts -- for instance, stereographic projection from the north pole covers everything except the north pole, and stereographic projection from the south pole covers everything except the south pole. Together they form an atlas.

### 3. Tangent Space $T_pM$

At each point $p$ on a manifold $M$, the **tangent space** $T_pM$ is the vector space of all "directions you can move" from $p$. On a 2-sphere, $T_pM$ is the plane tangent to the sphere at $p$.

Dimension: if $M$ is $n$-dimensional, then $T_pM$ is also $n$-dimensional as a vector space.

Tangent vectors can be defined concretely as velocity vectors of curves through $p$, or abstractly as directional derivative operators on smooth functions.

### 4. Tangent Vectors as Directional Derivatives

The abstract definition: a tangent vector $v$ at $p$ is a linear map $v: C^\infty(M) \to \mathbb{R}$ satisfying the Leibniz rule $v(fg) = f(p)v(g) + g(p)v(f)$.

In coordinates $(x^1, \ldots, x^n)$, the tangent vector $v = v^i \frac{\partial}{\partial x^i}$ acts on functions by $v(f) = v^i \frac{\partial f}{\partial x^i}$. The partial derivatives $\left\{\frac{\partial}{\partial x^1}, \ldots, \frac{\partial}{\partial x^n}\right\}$ form a basis for $T_pM$.

This definition is coordinate-independent: the tangent vector is the derivation, not the coordinates.

### 5. Riemannian Metric $g$

A **Riemannian metric** assigns a smoothly varying inner product $g_p$ to each tangent space $T_pM$:

$$g_p : T_pM \times T_pM \to \mathbb{R}$$

In coordinates, $g$ is represented by a symmetric positive-definite matrix $g_{ij}(x)$:

$$g_p(u, v) = g_{ij}(p)\, u^i v^j$$

The metric lets you measure lengths, angles, and volumes on the manifold. Different metrics on the same manifold give different geometric structures. Euclidean space $\mathbb{R}^n$ has the constant metric $g_{ij} = \delta_{ij}$. The sphere and hyperbolic plane have non-constant metrics.

### 6. Arc Length

Given a curve $\gamma: [a,b] \to M$, its length is:

$$L(\gamma) = \int_a^b \sqrt{g(\gamma'(t), \gamma'(t))}\, dt$$

In coordinates:

$$L(\gamma) = \int_a^b \sqrt{g_{ij} \frac{dx^i}{dt}\frac{dx^j}{dt}}\, dt$$

This generalizes the Euclidean formula $L = \int \|\gamma'(t)\|\, dt$ to curved spaces.

### 7. Geodesics

A **geodesic** is a curve that locally minimizes length (or, more precisely, is a critical point of the length functional). Geodesics are the "straightest possible" curves on a curved manifold.

Examples:
- On $\mathbb{R}^n$ with Euclidean metric: straight lines
- On $S^2$: great circles (equator, meridians, etc.)
- On the Poincare half-plane: vertical lines and semicircles centered on the $x$-axis

### 8. Geodesic Equation

The geodesic equation is a system of second-order ODEs:

$$\frac{d^2 x^\mu}{dt^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{dt}\frac{dx^\beta}{dt} = 0$$

where $\Gamma^\mu_{\alpha\beta}$ are the Christoffel symbols. This says: acceleration in coordinate space is corrected by terms involving the Christoffel symbols, which account for the curvature of the coordinate system itself.

### 9. Christoffel Symbols

The **Christoffel symbols** encode how the coordinate basis vectors change from point to point:

$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\nu} \left(\partial_\alpha g_{\beta\nu} + \partial_\beta g_{\alpha\nu} - \partial_\nu g_{\alpha\beta}\right)$$

They are NOT tensors (they depend on the coordinate system), but they define how to take derivatives on the manifold (the covariant derivative, or Levi-Civita connection).

Key property: $\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$ (symmetric in lower indices for the Levi-Civita connection).

### 10. Riemann Curvature Tensor

The **Riemann curvature tensor** $R^i_{\ jkl}$ measures the failure of parallel transport to commute:

$$R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

Intuitively: if you parallel transport a vector around an infinitesimal loop in the $k$-$l$ plane, $R$ measures how much the vector rotates. On flat space, $R = 0$ identically. On curved space, $R \neq 0$.

The Riemann tensor has $\frac{n^2(n^2-1)}{12}$ independent components in $n$ dimensions: 1 in 2D, 6 in 3D, 20 in 4D.

### 11. Ricci Curvature

The **Ricci curvature** $\operatorname{Ric}_{jl} = R^i_{\ jil}$ is a contraction (trace) of the Riemann tensor. It measures how volumes distort compared to flat space.

- $\operatorname{Ric} > 0$: volumes shrink (sphere -- geodesics converge)
- $\operatorname{Ric} = 0$: volumes are Euclidean (flat space)
- $\operatorname{Ric} < 0$: volumes grow (hyperbolic space -- geodesics diverge)

### 12. Hyperbolic Space

**Hyperbolic space** $\mathbb{H}^n$ has constant negative curvature. Its key property is exponential volume growth: the volume of a ball of radius $r$ in $\mathbb{H}^n$ grows as $e^{(n-1)r}$, compared to $r^n$ for Euclidean space.

This makes hyperbolic space natural for embedding trees: a tree with branching factor $b$ has $b^d$ nodes at depth $d$, matching the exponential volume growth. Flat Euclidean space cannot embed trees without distortion because its polynomial volume growth cannot accommodate exponential branching.

Models: Poincare disk (unit disk with modified metric), Poincare half-plane (upper half-plane with modified metric), hyperboloid model (in Minkowski space).

### 13. The Manifold Hypothesis

The **manifold hypothesis** asserts that high-dimensional data (images, text embeddings, etc.) actually lives on or near a low-dimensional manifold embedded in the ambient space.

For language: token embeddings in $\mathbb{R}^{512}$ may live on a manifold of intrinsic dimension 20-50. The 512 dimensions are the ambient space; the manifold captures the actual degrees of freedom. This has profound implications for architecture design: operations should respect the manifold structure rather than treating all 512 dimensions as independent.

---

## Worked Problems

### Problem 1

**The 2-sphere $S^2$ of radius $r$** has metric $ds^2 = r^2(d\theta^2 + \sin^2\theta\, d\phi^2)$ in spherical coordinates ($\theta$ = polar angle from north pole, $\phi$ = azimuthal angle). Write the metric tensor $g_{ij}$ as a $2 \times 2$ matrix. Compute the area element $\sqrt{\det g}$.

**Solution:**

The coordinates are $(x^1, x^2) = (\theta, \phi)$. Reading off the metric:

$$g = \begin{bmatrix} r^2 & 0 \\ 0 & r^2 \sin^2\theta \end{bmatrix}$$

The determinant:

$$\det(g) = r^2 \cdot r^2 \sin^2\theta = r^4 \sin^2\theta$$

The area element:

$$\sqrt{\det g} = r^2 \sin\theta$$

The area form is $dA = r^2 \sin\theta\, d\theta\, d\phi$. As a sanity check, the total surface area is:

$$A = \int_0^\pi \int_0^{2\pi} r^2 \sin\theta\, d\phi\, d\theta = r^2 \cdot 2\pi \cdot [-\cos\theta]_0^\pi = r^2 \cdot 2\pi \cdot (1 - (-1)) = 4\pi r^2 \checkmark$$

This matches the known surface area of a sphere.

---

### Problem 2

**For the Poincare half-plane model** of hyperbolic space $\mathbb{H}^2$ with metric $ds^2 = \frac{dx^2 + dy^2}{y^2}$ ($y > 0$), compute the distance between $(0,1)$ and $(0,e)$. Show it equals 1.

**Solution:**

The two points lie on the vertical line $x = 0$. Along this path, $dx = 0$, so:

$$ds = \sqrt{\frac{dy^2}{y^2}} = \frac{dy}{y}$$

The distance is:

$$d = \int_1^e \frac{dy}{y} = [\ln y]_1^e = \ln e - \ln 1 = 1 - 0 = 1$$

In hyperbolic space, distances grow logarithmically with Euclidean coordinate distance. The Euclidean distance between the points is $e - 1 \approx 1.718$, but the hyperbolic distance is only 1.

This logarithmic scaling is exactly why hyperbolic space embeds trees efficiently: each "level" of a tree is at constant hyperbolic distance from the next, even though the Euclidean coordinates spread exponentially. A tree of depth $d$ requires only $O(d)$ hyperbolic distance but $O(b^d)$ Euclidean volume, and hyperbolic space has exactly that exponential volume growth.

---

### Problem 3

**Compute the Christoffel symbols** for the 2-sphere metric $g = \begin{bmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{bmatrix}$ (unit sphere, $r = 1$). There are 8 possible symbols $\Gamma^i_{jk}$ for $i, j, k \in \{\theta, \phi\}$.

**Solution:**

The metric components: $g_{\theta\theta} = 1$, $g_{\phi\phi} = \sin^2\theta$, $g_{\theta\phi} = g_{\phi\theta} = 0$.

The inverse metric: $g^{\theta\theta} = 1$, $g^{\phi\phi} = \frac{1}{\sin^2\theta}$, $g^{\theta\phi} = 0$.

Computing each Christoffel symbol using $\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\nu}(\partial_\alpha g_{\beta\nu} + \partial_\beta g_{\alpha\nu} - \partial_\nu g_{\alpha\beta})$:

**$\Gamma^\theta_{\theta\theta}$:** $\frac{1}{2} g^{\theta\theta}(\partial_\theta g_{\theta\theta} + \partial_\theta g_{\theta\theta} - \partial_\theta g_{\theta\theta}) = \frac{1}{2}(1)(0) = 0$

**$\Gamma^\theta_{\theta\phi}$:** $\frac{1}{2} g^{\theta\theta}(\partial_\theta g_{\phi\theta} + \partial_\phi g_{\theta\theta} - \partial_\theta g_{\theta\phi}) = \frac{1}{2}(1)(0) = 0$

**$\Gamma^\theta_{\phi\phi}$:** $\frac{1}{2} g^{\theta\theta}(\partial_\phi g_{\phi\theta} + \partial_\phi g_{\theta\phi} - \partial_\theta g_{\phi\phi}) = \frac{1}{2}(1)(0 + 0 - 2\sin\theta\cos\theta) = \mathbf{-\sin\theta\cos\theta}$

**$\Gamma^\phi_{\theta\theta}$:** $\frac{1}{2} g^{\phi\phi}(\partial_\theta g_{\theta\phi} + \partial_\theta g_{\phi\theta} - \partial_\phi g_{\theta\theta}) = 0$

**$\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta}$:** $\frac{1}{2} g^{\phi\phi}(\partial_\theta g_{\phi\phi} + \partial_\phi g_{\phi\phi} - \partial_\phi g_{\theta\phi}) = \frac{1}{2}\left(\frac{1}{\sin^2\theta}\right)(2\sin\theta\cos\theta) = \mathbf{\frac{\cos\theta}{\sin\theta} = \cot\theta}$

**$\Gamma^\phi_{\phi\phi}$:** $\frac{1}{2} g^{\phi\phi}(\partial_\phi g_{\phi\phi} + \partial_\phi g_{\phi\phi} - \partial_\phi g_{\phi\phi}) = 0$

Summary of nonzero Christoffel symbols:
- $\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta$
- $\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta$

All other symbols vanish.

---

### Problem 4

**Write the geodesic equations** on the sphere using the Christoffel symbols from Problem 3. Verify that the equator is a geodesic.

**Solution:**

The geodesic equation $\frac{d^2 x^\mu}{dt^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{dt}\frac{dx^\beta}{dt} = 0$ gives two equations:

**$\theta$ equation:**

$$\ddot{\theta} + \Gamma^\theta_{\phi\phi}\, \dot{\phi}^2 = 0$$
$$\ddot{\theta} - \sin\theta\cos\theta\, \dot{\phi}^2 = 0$$

**$\phi$ equation:**

$$\ddot{\phi} + 2\,\Gamma^\phi_{\theta\phi}\, \dot{\theta}\,\dot{\phi} = 0$$
$$\ddot{\phi} + 2\cot\theta\, \dot{\theta}\,\dot{\phi} = 0$$

**Verification that the equator is a geodesic:**

The equator has $\theta = \frac{\pi}{2}$ with $\dot{\theta}(0) = 0$, and $\phi(t) = \phi_0 + \omega t$ for some constant $\omega$.

Check the $\theta$ equation:

$$\ddot{\theta} - \sin\frac{\pi}{2}\cos\frac{\pi}{2}\, \omega^2 = 0 - (1)(0)\,\omega^2 = 0 \checkmark$$

Check the $\phi$ equation:

$$\ddot{\phi} + 2\cot\frac{\pi}{2}\cdot 0 \cdot \omega = 0 + 0 = 0 \checkmark$$

Both equations are satisfied, confirming the equator is a geodesic. More generally, all great circles are geodesics on the sphere.

---

### Problem 5

**Parallel transport and holonomy on the sphere.** A vector is parallel transported along a spherical triangle with one vertex at the north pole and two vertices on the equator separated by azimuthal angle $\alpha$. Compute the rotation angle of the vector after completing the loop.

**Solution:**

By the Gauss-Bonnet theorem, the holonomy (rotation angle) of parallel transport around a closed curve on a surface equals the integral of the Gaussian curvature $K$ over the enclosed region:

$$\Omega = \iint_R K\, dA$$

For the unit sphere, $K = 1$ everywhere. The spherical triangle described has:
- One vertex at the north pole ($\theta = 0$)
- Two vertices at $(\theta = \frac{\pi}{2}, \phi = 0)$ and $(\theta = \frac{\pi}{2}, \phi = \alpha)$

This triangle is bounded by two meridians and an equatorial arc. Its area is:

$$A = \alpha \quad \text{(for the unit sphere)}$$

This follows from the spherical excess formula: the sum of angles of the triangle is $(\frac{\pi}{2} + \frac{\pi}{2} + \alpha) = \pi + \alpha$, so the excess is $\alpha$, and area = excess for the unit sphere.

Therefore the holonomy angle is:

$$\Omega = A = \alpha$$

For $\alpha = \frac{\pi}{2}$: the vector rotates by 90 degrees after traversing the triangle. The vector returns to its starting point but is rotated, even though it was "held constant" (parallel transported) at every step. This rotation IS the curvature: it is detectable only via a closed loop.

This is a fundamental concept. In flat space, parallel transport around any loop returns a vector unchanged. On a curved manifold, the rotation is proportional to the enclosed curvature. This path-dependence is what makes curved geometry computationally richer than flat geometry.

---

### Problem 6

**The manifold hypothesis for language.** Modell et al. (2025) found that language models represent "color" as a circle (color wheel), "year" as a line, and "day of the year" as a circle. Explain why these are different manifolds with different topologies. What mathematical property distinguishes a circle from a line?

**Solution:**

**Circle $S^1$:** Compact (closed and bounded), no boundary, fundamental group $\pi_1(S^1) = \mathbb{Z}$ (the integers). The nontrivial fundamental group means there exist loops on $S^1$ that cannot be continuously shrunk to a point. Intuitively, you can "go around" and return to where you started.

**Line $\mathbb{R}^1$:** Non-compact (unbounded), no boundary, fundamental group $\pi_1(\mathbb{R}^1) = 0$ (trivial, simply connected). Every loop can be shrunk to a point. There is no way to "go around."

The mathematical property that distinguishes them is **topology** -- specifically, compactness and the fundamental group:

- **Colors** form a circle because the color wheel wraps around: red transitions through orange, yellow, green, blue, violet, and back to red. The representation MUST be topologically $S^1$ to capture this cyclic structure.

- **Years** form a line because they are ordered without wraparound: 2020 < 2021 < 2022, with no return to the beginning. The representation is topologically $\mathbb{R}^1$.

- **Days of the year** form a circle because December 31 transitions to January 1. The representation must be $S^1$ to capture this cyclic structure.

The deep insight is the "continuous correspondence hypothesis" (thesis Sec. 2.4): the topology of the learned representation matches the topology of the concept. The model discovers that colors are cyclic and years are linear, encoding this in the geometric structure of the representation manifold.

---

### Problem 7

**Intrinsic dimension and the ambient space.** The manifold hypothesis says token embeddings in a 512-dimensional space actually live on a much lower-dimensional manifold. If the intrinsic dimension is 20, how many degrees of freedom does each token actually have? What does this say about the 492 "extra" dimensions?

**Solution:**

**Degrees of freedom:** The intrinsic dimension of 20 means each token can be locally parameterized by 20 independent coordinates. A token's position on the manifold is determined by 20 numbers, not 512. The token has 20 effective degrees of freedom.

**The 492 extra dimensions** serve several purposes:

1. **Embedding room:** The manifold needs to "sit inside" a higher-dimensional space. A 2D surface (like a sphere) needs at least 3D space to embed without self-intersection (by the Whitney embedding theorem, a manifold of dimension $n$ generally needs $2n$ ambient dimensions for a smooth embedding).

2. **Curvature accommodation:** The manifold can bend and curve through the ambient space. More ambient dimensions allow more complex curvature without self-intersection.

3. **Multiple components:** Semantically different categories (verbs, nouns, punctuation) may occupy disconnected components of the manifold, each needing room in the ambient space.

4. **Varying local geometry:** Different regions of the manifold may have different curvature and local structure.

However, the extra dimensions also contribute to **representation degeneration**: most of the 512-dimensional vector encodes the embedding geometry rather than semantically useful features. The token representations tend to cluster in a narrow cone of the ambient space, wasting capacity. This is one motivation for geometric approaches that explicitly model the manifold structure rather than operating in the full ambient space.

---

### Problem 8

**From manifolds to fiber bundles.** Robinson et al. (2025) showed that token embeddings in GPT-2 are NOT smooth manifolds -- they have varying local dimension and singularities at polysemous tokens. Explain why a fiber bundle is a better model than a plain manifold.

**Solution:**

A **plain manifold** has the same local structure everywhere: every point has a neighborhood that looks like $\mathbb{R}^n$ for the same $n$. The manifold is smooth and has no singularities.

But polysemous tokens violate this:

- The word "bank" has at least 2 meanings (financial institution, river bank). In embedding space, this creates a **singularity**: the point where the two meaning branches meet. Near this point, the local structure is not $\mathbb{R}^n$ but rather a branching space (like two sheets meeting at a point).

- Different tokens may require different numbers of dimensions to describe their local meaning structure. A highly polysemous word needs more local dimensions than an unambiguous technical term.

A **fiber bundle** resolves this by separating the base structure from the local "decoration":

- The **base manifold $M$** encodes token positions/identities -- this is smooth and well-behaved.
- The **fiber $F_p$** at each point $p$ encodes the meaning/representation space -- and different points can have different fibers.
- At "bank," the fiber has a branching structure (two meaning branches).
- At an unambiguous word, the fiber is simpler (a single connected region).
- The **total space $E$** = union of all $F_p$ is NOT required to be a manifold -- it is a fiber bundle, which is a more general and appropriate mathematical object.

The additional structure of a fiber bundle (the connection, transition functions, and structure group) encodes HOW the meaning space changes from one token to the next -- which is precisely what attention computes. This is why the thesis (Ch. 2.5) argues for fiber bundles as the correct mathematical framework for language representations.

---

### Problem 9

**The Poincare disk model.** In the Poincare disk model (unit disk $\{x : \|x\| < 1\}$ with metric $ds^2 = \frac{4(dx^2 + dy^2)}{(1 - \|x\|^2)^2}$), points near the boundary represent "leaf" nodes of a tree and points near the center represent "root." Compute the metric coefficient $\frac{4}{(1 - r^2)^2}$ at $r = 0$, $r = 0.5$, $r = 0.9$, $r = 0.99$.

**Solution:**

The conformal factor $\lambda(r) = \frac{4}{(1 - r^2)^2}$:

| $r$    | $1 - r^2$  | $(1 - r^2)^2$ | $\lambda = \frac{4}{(1 - r^2)^2}$ |
|--------|------------|----------------|------------------------------------|
| $0$    | $1$        | $1$            | $4$                                |
| $0.5$  | $0.75$     | $0.5625$       | $7.11$                             |
| $0.9$  | $0.19$     | $0.0361$       | $110.8$                            |
| $0.99$ | $0.0199$   | $0.000396$     | $10{,}101$                         |

The metric coefficient **explodes** near the boundary. At $r = 0.99$, the metric is about 2,525 times larger than at the center. This means:

- A tiny Euclidean step (say, 0.001 in coordinate distance) near $r = 0.99$ corresponds to a hyperbolic distance of about $0.001 \times \sqrt{10101} \approx 0.1$, while the same step at the center corresponds to only $0.001 \times \sqrt{4} \approx 0.002$.

- The boundary $r = 1$ is "infinitely far away" in hyperbolic distance -- you can never reach it.

- This is how a bounded Euclidean disk represents the infinite hyperbolic plane. The boundary "has room" for exponentially many points, which is why trees embed naturally: leaf nodes spread along the boundary where the exponential volume growth provides room for the exponentially many leaves.

For a binary tree of depth $d$: $d$ levels of branching produce $2^d$ leaves. In hyperbolic space, the circumference at distance $d$ from the center is $2\pi \sinh(d) \sim \pi e^d$, which grows exponentially -- exactly matching the tree branching.

---

### Problem 10

**Curvature and path-dependence.** The Riemann curvature tensor $R^i_{\ jkl}$ measures how much a vector rotates when parallel transported around an infinitesimal loop in the $k$-$l$ plane. Explain intuitively why zero curvature means parallel transport is path-independent, and nonzero curvature means it is path-dependent.

**Solution:**

**If $R = 0$ everywhere (flat space):**

Parallel transport around any infinitesimal loop is the identity transformation (no rotation). Now consider two different paths from point $A$ to point $B$. Together, these paths form a closed loop. This loop can be subdivided into smaller and smaller loops by triangulation. Each small loop has trivial holonomy (zero rotation, since $R = 0$), and the total holonomy is the composition of all the small loops' holonomies. Since each is trivial, the total is trivial. Therefore, the transport from $A$ to $B$ along path 1 equals the transport along path 2. **Parallel transport is path-independent.**

**If $R \neq 0$ somewhere (curved space):**

There exist infinitesimal loops where parallel transport IS nontrivial -- the vector rotates by an amount proportional to $R$ times the loop area. Now consider two paths from $A$ to $B$ that enclose a region where $R \neq 0$. The closed loop formed by these paths has nontrivial holonomy (the vector rotates). This means the transport along path 1 and path 2 give DIFFERENT results. **Parallel transport is path-dependent.**

The amount of rotation is governed by the Ambrose-Singer theorem: the holonomy group (set of all possible rotations from transport around loops) is generated by the curvature values along the path.

**Why this matters for language:**

Path-dependent parallel transport is computationally richer than path-independent transport. In flat space, transporting a representation from token $A$ to token $B$ always gives the same result regardless of the intervening tokens. In curved space, the result depends on the path -- i.e., the context. This path-dependence is exactly what attention computes: the transformation of a token's representation depends on all the tokens between source and target.

This is why the thesis argues that curved geometric spaces (not flat embeddings) are the right model for language: the curvature enables context-dependent computation that flat spaces cannot express.

---

## Comprehension Questions

1. What is a manifold, intuitively? Give 3 examples from everyday life.

2. What is the tangent space at a point on a sphere? Draw it. How does the tangent space relate to "local linear approximation"?

3. Why can't a flat (Euclidean) metric represent hierarchical data efficiently? (Hint: think about how volume grows with radius in Euclidean vs. hyperbolic space.)

4. What does the Riemann curvature tensor tell you about the manifold? What does zero curvature imply about parallel transport?

5. Explain the manifold hypothesis as applied to language. Why do the thesis authors argue for fiber bundles rather than plain manifolds? (Reference Problems 6-8.)

---

## Bridge to Thesis

- **The manifold hypothesis** (Problems 6-8) underpins the entire thesis (Ch. 2.4). The thesis argues that token representations live on low-dimensional manifolds and that architectures should respect this structure.

- **Hyperbolic space** (Problems 2, 9) explains why flat embeddings are wasteful (Ch. 1.2). Euclidean space has polynomial volume growth; hierarchical language structure has exponential branching. The mismatch forces flat embeddings to use excessive dimensions.

- **Curvature and holonomy** (Problems 5, 10) become parallel transport in fiber bundles (Ch. 4.2). The key insight is that curvature enables path-dependent computation, which is exactly what contextual language representations require.

- **Geodesics** (Problems 3-4) become the Finsler Transformer's learned trajectories (Ch. 4.4). Instead of arbitrary attention patterns, representations flow along geodesics of a learned Finsler metric.

- **Fiber bundles** (Problem 8) are the correct geometric model for token representations (Ch. 2.5, `topology/gauge_connections_fiber_bundles.md`). The base manifold captures position/identity, the fiber captures meaning, and the connection captures how context transforms meaning.

- **Next unit** (Unit 06): Fiber bundles get their own detailed treatment, including connections, gauge theory, and Wilson lines -- the mathematical machinery that turns the geometric intuitions from this unit into computable operations.
