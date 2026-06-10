# Unit 06: Fiber Bundles, Connections, Gauge Theory, and Wilson Lines

## Prerequisites
Unit 05 (Manifolds, Metrics, Curvature, and Geodesics)

---

## Learning Objectives

By the end of this unit, you should be able to:

1. Define fiber bundles (trivial and non-trivial) and understand their components
2. Understand connections as "horizontal lifts" -- rules for parallel transport
3. Compute parallel transport along a path given a connection
4. Understand holonomy: what happens when you transport around a closed loop
5. Define and interpret gauge transformations
6. Connect Wilson lines to accumulated context in sequences
7. See attention as a gauge connection (thesis Sec. 2.5.2)

---

## Readings

**Primary Texts:**
- Nakahara, *Geometry, Topology and Physics*, Ch. 9-10 (Fiber Bundles)
- Baez & Muniain, *Gauge Fields, Knots and Gravity*, Ch. 1-4 (very accessible)

**Video Lectures:**
- Frederic Schuller: "Geometrical Anatomy of Theoretical Physics" lectures 15-20 (YouTube) -- mathematical rigor with physics intuition

**Thesis and Repo:**
- Thesis Sec. 2.5 (Fiber Bundles and Gauge Theory in Neural Networks)
- Thesis Ch. 4.2 (Attention as Parallel Transport)
- Repo: `topology/gauge_connections_fiber_bundles.md` (comprehensive treatment)
- Repo: `topology/holonomy_closed_manifolds.md`
- Repo: `topology/holonomy_open_manifolds.md`
- Repo: `topology/SYNTHESIS.md` (convergence of gauge theory + SSMs + OT)

**Papers:**
- `papers/gauge_fiber_bundle_geometry_transformers_iclr2025.pdf`

---

## Key Concepts

### 1. Fiber Bundle $(E, M, \pi, F)$

A **fiber bundle** consists of four ingredients:
- **Total space $E$** -- the "big" space containing all the data
- **Base manifold $M$** -- the "ground" space (e.g., sequence positions)
- **Projection $\pi: E \to M$** -- maps each point in $E$ to its "base" point
- **Fiber $F$** -- the space "above" each point of $M$; $\pi^{-1}(p)$ is a copy of $F$ for each $p \in M$

Intuition: think of $E$ as a building where $M$ is the ground floor plan and $F$ is the vertical structure above each point. The projection $\pi$ sends every point in the building down to its location on the ground floor. Each vertical column $\pi^{-1}(p)$ is a copy of the fiber $F$.

The key requirement: locally, $E$ looks like a product $M \times F$, but globally it might be "twisted."

### 2. Trivial vs. Non-Trivial Bundles

A **trivial bundle** is globally a product: $E = M \times F$. There is a consistent, global way to identify the fiber at one point with the fiber at any other point.

Example: the cylinder $S^1 \times \mathbb{R}$ (base = circle, fiber = real line). You can "unroll" it into a flat strip.

A **non-trivial bundle** has fibers that "twist" as you go around the base. There is no global, consistent identification between fibers.

Example: the Mobius band (base = circle, fiber = real line, but the fiber flips as you go around). You cannot unroll it without cutting.

The difference is encoded in the **transition functions**: how local trivializations (local product structures) are glued together. For the cylinder, all transition functions are the identity. For the Mobius band, one transition function is the reflection $x \mapsto -x$.

### 3. Section

A **section** $\sigma: M \to E$ is a map that picks one point in each fiber: $\pi(\sigma(p)) = p$ for all $p \in M$. In other words, a section assigns a "value" in $F$ to each point of $M$.

Examples:
- A vector field on a manifold is a section of the tangent bundle.
- A wave function in quantum mechanics is a section of a line bundle.
- In the SGST context: a token embedding sequence is a section of the representation bundle over the position manifold.

A **global section** is defined on all of $M$. Non-trivial bundles may not admit global sections (the Mobius band has no nowhere-zero global section).

### 4. Connection

A **connection** on a fiber bundle is a rule that defines "horizontal" directions in the total space $E$. At each point $e \in E$, the tangent space $T_eE$ splits into:

- **Vertical subspace $V_e$** = directions along the fiber (tangent to $\pi^{-1}(\pi(e))$). These change the "value" without moving along the base.
- **Horizontal subspace $H_e$** = directions "across" fibers, as defined by the connection. These move along the base while keeping the "value" as constant as possible.

The connection tells you: if you move from point $p$ to a nearby point $q$ in the base manifold, how should you "carry" the fiber element along? The horizontal lift of a tangent vector on $M$ is the unique direction in $E$ that moves along $M$ without changing the fiber value (according to the connection's definition of "without changing").

### 5. Parallel Transport

Given a connection and a path $\gamma$ in $M$ from $p$ to $q$, **parallel transport** carries an element of the fiber $F_p$ to an element of $F_q$ by following the horizontal lift of $\gamma$.

This gives a map $\tau_\gamma: F_p \to F_q$ that is:
- Linear (if the fibers are vector spaces)
- Invertible
- Path-dependent (in general, different paths from $p$ to $q$ give different maps)

The path-dependence is the crucial feature: parallel transport encodes how information transforms as it travels along a path, and the transformation depends on the path taken -- just as the meaning of a word depends on the context that precedes it.

### 6. Connection 1-Form $A$

The connection can be encoded as a **Lie-algebra-valued 1-form** $A$. If the structure group is $G$ (a Lie group), then $A$ takes values in the Lie algebra $\mathfrak{g}$ of $G$.

In coordinates, $A = A_\mu^a T_a\, dx^\mu$, where $T_a$ are the generators of the Lie algebra and $A_\mu^a$ are the component functions.

The parallel transport equation for a vector $v$ along a path $\gamma$ is:

$$\frac{dv}{dt} + A(\gamma'(t))\, v = 0$$

The solution involves the path-ordered exponential (see Wilson line below).

### 7. Curvature $F = dA + A \wedge A$

The **curvature 2-form** of a connection measures the failure of parallel transport to be path-independent:

$$F = dA + A \wedge A$$

For abelian groups (like $U(1)$), $A \wedge A = 0$ and $F = dA$.

Curvature is related to holonomy: for an infinitesimal loop enclosing area $dS$, the holonomy is approximately $I + F(dS)$. Zero curvature ($F = 0$, a "flat connection") means parallel transport is path-independent.

### 8. Holonomy

The **holonomy** of a connection around a closed loop $\gamma$ based at point $p$ is the transformation $H_\gamma: F_p \to F_p$ obtained by parallel transporting around the loop and returning to the starting point.

Even though the path is closed (returns to the same base point), the fiber element may have changed. This change is the holonomy.

Properties:
- For a flat connection ($F = 0$), holonomy around contractible loops is trivial (identity).
- Holonomy around non-contractible loops can be nontrivial even for flat connections (topological holonomy).
- The set of all holonomies forms a group, the **holonomy group** $\operatorname{Hol}(A)$, which is a subgroup of the structure group $G$.

The Ambrose-Singer theorem: the Lie algebra of the holonomy group is generated by the curvature values along all paths from the base point.

### 9. Wilson Line

The **Wilson line** along a path $\gamma$ from $s$ to $t$ is the path-ordered exponential:

$$U_\gamma = \mathcal{P} \exp\!\left(i \int_\gamma A\right)$$

For a matrix-valued connection, the path-ordering $\mathcal{P}$ is essential because matrices at different points may not commute. The Wilson line satisfies:

$$\frac{dU}{dt} = i\, A(\gamma(t))\, U(t), \quad U(s) = I$$

For an abelian group (like $U(1)$), path-ordering is unnecessary:

$$U_\gamma = \exp\!\left(i \int_s^t A(\tau)\, d\tau\right)$$

The Wilson line is the finite (not infinitesimal) parallel transport operator. It is gauge-covariant: under a gauge transformation $g$, it transforms as $U \to g(t)\, U\, g(s)^{-1}$.

In the SGST context: the Wilson line from position 0 to position $t$ encodes the total accumulated geometric transformation -- this is the "context" that has been built up from the beginning of the sequence.

### 10. Gauge Transformation

A **gauge transformation** is a change of local frame (basis) in the fiber at each point. If $g: M \to G$ is a smooth map, the gauge transformation acts on:

- Sections: $\psi(p) \to g(p)\, \psi(p)$
- Connections: $A \to g A g^{-1} + g\, dg^{-1}$ (or equivalently $A \to g A g^{-1} - (dg) g^{-1}$ depending on convention)
- Curvature: $F \to g F g^{-1}$

The key property: **physical observables are gauge-invariant.** The holonomy (trace of the Wilson line around a closed loop) does not change under gauge transformations. The curvature transforms covariantly but its trace is invariant.

Gauge invariance means that the choice of coordinate basis in each fiber is arbitrary -- only the relationships between fibers (encoded by the connection) are physically meaningful.

### 11. Gauge Groups

Common gauge groups and their meaning:

- **$U(1)$** (phases, $e^{i\theta}$): 1-dimensional, abelian. Rotates phases without changing magnitudes. Simplest non-trivial gauge group. RoPE (rotary position embeddings) uses $U(1)$.

- **$SO(n)$** (real rotations): preserves real inner products and norms. $\frac{n(n-1)}{2}$ parameters. Preserves Euclidean geometry of the fiber. The thesis proposes $SO(K)$ for SGST (see memory note: no complex numbers on MPS).

- **$SU(n)$** (special unitary): preserves complex inner products. $n^2-1$ parameters. Standard gauge group of particle physics ($SU(2)$ for weak force, $SU(3)$ for strong force).

- **$GL(n)$** (general linear): all invertible matrices. $n^2$ parameters. Maximum generality, no geometric structure preserved. Standard neural networks implicitly use $GL(n)$.

Choosing a smaller gauge group is an **inductive bias**: it restricts the allowed transformations, reducing parameters while imposing geometric structure.

### 12. Principal Bundle vs. Vector Bundle

A **vector bundle** has fibers that are vector spaces ($\mathbb{R}^n$ or $\mathbb{C}^n$). Sections are vector-valued functions. This is the natural setting for token representations.

A **principal bundle** has fibers that are copies of the structure group $G$ itself. Sections represent "frames" or "gauges." The connection lives most naturally on the principal bundle.

These are related: given a principal $G$-bundle and a representation $\rho: G \to GL(V)$, you get an associated vector bundle with fiber $V$. The connection on the principal bundle induces a connection on the vector bundle.

In practice, for neural networks, you mostly work with vector bundles (where the fiber is the representation space $\mathbb{R}^d$), but the principal bundle perspective clarifies gauge transformations and structure groups.

### 13. Ehresmann Connection vs. Levi-Civita Connection

An **Ehresmann connection** is the most general type: any smooth choice of horizontal subspace at each point of $E$. It requires only a fiber bundle, not a metric.

The **Levi-Civita connection** is specific to Riemannian geometry: it is the unique connection on the tangent bundle that is (a) compatible with the metric (parallel transport preserves inner products) and (b) torsion-free. It is determined entirely by the metric.

For neural networks, the relevant connection is typically an Ehresmann connection (or a principal connection) that is LEARNED, not determined by a pre-existing metric. The attention mechanism learns WHICH horizontal directions to use, rather than having them dictated by geometry.

---

## Worked Problems

### Problem 1

**Trivial vs. non-trivial bundles.** The cylinder $M \times \mathbb{R}$ (base = circle $S^1$, fiber = $\mathbb{R}$) is a trivial bundle. The Mobius band is a non-trivial bundle with the same base and fiber. Explain why the Mobius band cannot be written as $S^1 \times \mathbb{R}$ globally. What topological property distinguishes them?

**Solution:**

The **cylinder** $S^1 \times \mathbb{R}$ can be constructed by taking a rectangular strip $[0, 2\pi] \times \mathbb{R}$ and gluing the left and right edges with the identity map: $(0, y) \sim (2\pi, y)$. It has two sides (an "inside" and an "outside") and is **orientable**.

The **Mobius band** is constructed by gluing with a flip: $(0, y) \sim (2\pi, -y)$. Going around the circle, the fiber coordinate $y$ gets negated. It has only one side and is **non-orientable**.

If the Mobius band were globally $S^1 \times \mathbb{R}$, you could define a continuous function $f: S^1 \to \mathbb{R}$ that is "the $y$-coordinate" at each point. This function would need to satisfy $f(0) = f(2\pi)$ (continuity around the loop), but the twist requires $f(2\pi) = -f(0)$. The only solution is $f = 0$ everywhere, meaning there is no continuous, nowhere-zero section -- you cannot consistently choose an "upward" direction.

Formally, the distinction is encoded in the **transition functions**. Cover $S^1$ with two overlapping arcs $U_1, U_2$. Their overlap has two components. The transition function on one component is the identity $g_{12} = +1$, and on the other:
- Cylinder: $g_{12} = +1$ (trivial)
- Mobius band: $g_{12} = -1$ (nontrivial)

The structure group is $\mathbb{Z}/2\mathbb{Z} = \{+1, -1\}$, acting on the fiber $\mathbb{R}$ by multiplication. The cylinder has trivially-valued transition functions; the Mobius band does not. This is the simplest example of a non-trivial bundle.

---

### Problem 2

**Fiber bundles for language.** In a fiber bundle model for language, the base manifold $M$ represents token positions, and the fiber $F_p$ at position $p$ is the representation space. Explain why different tokens need different fibers (i.e., why a trivial bundle $E = M \times \mathbb{R}^d$ might not be the right model).

**Solution:**

In a **trivial bundle** $E = M \times \mathbb{R}^d$, there is a global, consistent identification between fibers. This means "dimension 42 at position 1" and "dimension 42 at position 5" refer to the same feature. There is a canonical way to compare representations at different positions without any dependence on the intervening context.

But in language, this is wrong:

1. **Context-dependent meaning.** The "meaning" of a representation dimension depends on what has come before. After "The bank of the..." dimension 42 might encode the feature "geography vs. finance." After "She walked to the..." the same dimension might encode "destination type." The relationship between fibers at different positions DEPENDS ON THE PATH (the intervening tokens).

2. **Non-trivial transport.** Moving a representation from position 1 to position 5 is not a fixed operation -- it depends on what tokens are at positions 2, 3, 4. A trivial bundle with a flat connection would make this transport path-independent, meaning context does not matter. This is clearly inadequate.

3. **Polysemy and branching.** As discussed in Unit 05 Problem 8, polysemous tokens create singularities that a smooth product structure cannot accommodate.

What is needed is a **non-trivial bundle with a non-flat connection**:
- The connection $A$ encodes how to transport representations between positions
- The connection is context-dependent (learned from data)
- The curvature $F = dA + A \wedge A$ is nonzero, meaning transport IS path-dependent
- Different paths through the sequence (different contexts) yield different transformations

This is precisely what the attention mechanism computes: for each pair of positions, attention determines HOW to transform the representation from one position to the other. The attention weights are the components of the connection 1-form.

---

### Problem 3

**$U(1)$ holonomy.** Consider $U(1)$ (circle group, phases $e^{i\theta}$) as the gauge group. A connection $A$ on a circle (base = $S^1$) is a 1-form $A = a(\phi)\, d\phi$. The holonomy of a full loop is $\exp\!\left(i \oint a(\phi)\, d\phi\right)$. If $a(\phi) = c$ (constant), compute the holonomy for $c = \frac{1}{2\pi}$ and $c = 1$.

**Solution:**

The holonomy of the full loop ($\phi$ from $0$ to $2\pi$) is:

$$\operatorname{Hol} = \exp\!\left(i \int_0^{2\pi} c\, d\phi\right) = \exp(i \cdot 2\pi c)$$

**Case $c = \frac{1}{2\pi}$:**

$$\operatorname{Hol} = \exp\!\left(i \cdot 2\pi \cdot \frac{1}{2\pi}\right) = \exp(i \cdot 1) = e^i$$

This is a rotation by 1 radian (approximately 57.3 degrees). In the complex plane:

$$e^i = \cos(1) + i\sin(1) \approx 0.540 + 0.841\, i$$

The holonomy is nontrivial: going around the loop rotates the phase by 1 radian.

**Case $c = 1$:**

$$\operatorname{Hol} = \exp(i \cdot 2\pi \cdot 1) = \exp(2\pi i) = 1$$

This is a full rotation by $2\pi$ (360 degrees), which returns to the identity. The holonomy is **trivial** even though the connection is nontrivial at each point. The total accumulated phase over the loop is exactly $2\pi$, a full revolution.

Physical analogy: the first case is like a sentence that shifts meaning as you read it, ending in a different "orientation" than where it started. The second case is like a sentence that "comes full circle," returning to its starting semantic state after a complete traversal.

Note: a trivial holonomy does NOT imply a trivial connection. The connection $a = 1$ is nonzero (it generates phase rotation at every point), but the total rotation over the full loop happens to cancel out to a multiple of $2\pi$.

---

### Problem 4

**Curvature of a $U(1)$ connection.** The curvature of a $U(1)$ connection $A$ on a surface is $F = dA$ (since $U(1)$ is abelian, $A \wedge A = 0$). For $A = x\, dy$ on $\mathbb{R}^2$, compute $F$. What does nonzero curvature mean for parallel transport?

**Solution:**

Compute the exterior derivative of $A = x\, dy$:

$$F = dA = d(x\, dy) = dx \wedge dy$$

This is the standard area 2-form on $\mathbb{R}^2$. The curvature is constant and nonzero everywhere.

**Meaning for parallel transport:**

For a small loop enclosing area $\Delta A$, the holonomy is approximately:

$$\operatorname{Hol} \approx \exp(i \cdot \Delta A)$$

The phase rotation is proportional to the enclosed area. Different loops enclosing different areas give different holonomies.

Consider two paths from $(0,0)$ to $(1,1)$:
- Path 1: go right to $(1,0)$, then up to $(1,1)$
- Path 2: go up to $(0,1)$, then right to $(1,1)$

Together they enclose a unit square of area 1. The holonomy around this loop is $\exp(i \cdot 1) = e^i$, which is nontrivial. Therefore, parallel transport along Path 1 and Path 2 give DIFFERENT results.

**Physical interpretation:** This is the **Aharonov-Bohm effect** in electromagnetism. A charged particle traveling through a magnetic field (the curvature $F = B\, dx \wedge dy$) accumulates a phase proportional to the enclosed magnetic flux. Different paths around the solenoid accumulate different phases.

**Neural network interpretation:** Nonzero curvature means the attention mechanism is genuinely path-dependent. Presenting tokens in the order A-B-C-D produces different transformations than A-C-B-D. The permutation sensitivity of attention is a manifestation of nonzero curvature in the representation bundle.

---

### Problem 5

**Wilson lines and accumulated context.** The Wilson line from token position $s$ to position $t$ is $U_{s \to t} = \mathcal{P} \exp\!\left(i \int_s^t A(\tau)\, d\tau\right)$. For an abelian ($U(1)$) gauge group with $A(\tau) = \alpha(\tau)$, this simplifies to $\exp\!\left(i \int_s^t \alpha(\tau)\, d\tau\right)$. If $\alpha(\tau) = \tau$ (linearly increasing), compute $U_{0 \to T}$ for $T = \pi$.

**Solution:**

Since the gauge group is $U(1)$ (abelian), the path-ordered exponential reduces to an ordinary exponential:

$$U_{0 \to \pi} = \exp\!\left(i \int_0^\pi \tau\, d\tau\right)$$

Evaluate the integral:

$$\int_0^\pi \tau\, d\tau = \left[\frac{\tau^2}{2}\right]_0^\pi = \frac{\pi^2}{2}$$

Therefore:

$$U_{0 \to \pi} = \exp\!\left(\frac{i\pi^2}{2}\right)$$

Since $\frac{\pi^2}{2} \approx 4.935$ radians $\approx 283$ degrees:

$$U_{0 \to \pi} = \cos\!\left(\frac{\pi^2}{2}\right) + i\sin\!\left(\frac{\pi^2}{2}\right) \approx 0.976 - 0.220\, i$$

The Wilson line accumulates phase **quadratically** with path length (because $\alpha(\tau) = \tau$ is linear, so its integral is quadratic).

**Interpretation for sequence models:**

The Wilson line $U_{0 \to t}$ represents the total accumulated geometric transformation from the start of a sequence to position $t$. It encodes ALL the context seen so far as a single transformation.

Key properties:
- **Composability:** $U_{0 \to t} = U_{s \to t} \cdot U_{0 \to s}$ for any intermediate position $s$. Context decomposes into segments.
- **Invertibility:** $U_{0 \to t}^{-1} = U_{t \to 0}$. You can "undo" context by transporting backward.
- **Content-dependence:** In the SGST, $\alpha(\tau)$ is not a fixed function but depends on the token at position $\tau$. Different input sequences produce different Wilson lines.

This IS the "KV cache" in geometric language. Standard transformers cache key-value pairs; the geometric version caches the Wilson line (accumulated transport operator). The Wilson line is more compact (a single group element vs. a growing list of KV pairs) and composes multiplicatively rather than additively.

---

### Problem 6

**Non-abelian parallel transport ($SO(2)$).** For the gauge group $SO(2)$ ($2 \times 2$ rotation matrices), a connection $A(t)$ at time $t$ is a skew-symmetric matrix $A(t) = \begin{bmatrix} 0 & -\omega(t) \\ \omega(t) & 0 \end{bmatrix}$. The parallel transport satisfies $\frac{dU}{dt} = -A(t)\, U$. For constant $\omega(t) = \omega_0$, solve for $U(t)$ with $U(0) = I$.

**Solution:**

The parallel transport equation is:

$$\frac{dU}{dt} = -A\, U = -\begin{bmatrix} 0 & -\omega_0 \\ \omega_0 & 0 \end{bmatrix} U = \begin{bmatrix} 0 & \omega_0 \\ -\omega_0 & 0 \end{bmatrix} U$$

Let $B = \begin{bmatrix} 0 & \omega_0 \\ -\omega_0 & 0 \end{bmatrix}$. Since $B$ is constant, the solution is the matrix exponential:

$$U(t) = \exp(t\, B)$$

To compute $\exp(tB)$, note that $B^2 = \begin{bmatrix} 0 & \omega_0 \\ -\omega_0 & 0 \end{bmatrix}^2 = \begin{bmatrix} -\omega_0^2 & 0 \\ 0 & -\omega_0^2 \end{bmatrix} = -\omega_0^2\, I$.

Using the Taylor series and collecting even/odd powers:

$$\exp(tB) = I\cos(\omega_0 t) + \frac{B}{\omega_0}\sin(\omega_0 t)$$

$$= \begin{bmatrix} \cos(\omega_0 t) & \sin(\omega_0 t) \\ -\sin(\omega_0 t) & \cos(\omega_0 t) \end{bmatrix}$$

This is a **rotation matrix** by angle $\omega_0 t$.

**Specific values:**
- At $t = 0$: $U = I$ (identity, no rotation). $\checkmark$
- At $t = \frac{\pi}{2\omega_0}$: $U = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$ (90 degree rotation).
- At $t = \frac{\pi}{\omega_0}$: $U = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} = -I$ (180 degree rotation).
- At $t = \frac{2\pi}{\omega_0}$: $U = I$ (full 360 degree rotation, back to identity).

**Interpretation:** This is parallel transport in the simplest non-trivial non-abelian case. A 2D vector is rotated at constant angular velocity $\omega_0$ as it is transported along the path. After time $t$, it has been rotated by angle $\omega_0 t$.

In the SGST, $\omega_0$ is NOT constant -- it depends on the token content at each position. Different tokens cause different rotation rates. The total rotation after traversing a sequence is the product of many small rotations, each determined by the local token. This product is the Wilson line, and it IS the context representation.

---

### Problem 7

**Gauge invariance.** Gauge transformations change the local frame. If we apply gauge transformation $g(p)$ at each point $p$, the connection transforms as $A' = gAg^{-1} + g\, dg^{-1}$. For $U(1)$, $g = e^{i\alpha(x)}$, show that $A' = A + d\alpha$. Then show the holonomy is gauge-invariant.

**Solution:**

For the $U(1)$ gauge group, elements are phases $e^{i\alpha}$. Since $U(1)$ is abelian:

$$g A g^{-1} = e^{i\alpha} A\, e^{-i\alpha} = A$$

(because all $U(1)$ elements commute.)

For the second term, compute $g\, dg^{-1}$:

$$g^{-1} = e^{-i\alpha}$$

$$dg^{-1} = d(e^{-i\alpha}) = -i\, (d\alpha)\, e^{-i\alpha}$$

$$g\, dg^{-1} = e^{i\alpha} \cdot (-i)\, (d\alpha)\, e^{-i\alpha} = -i\, d\alpha$$

With the convention $A' = gAg^{-1} - (dg)g^{-1}$ (which gives the connection in the new frame):

$$(dg) = i\, (d\alpha)\, e^{i\alpha}$$

$$(dg)\, g^{-1} = i\, (d\alpha)\, e^{i\alpha}\, e^{-i\alpha} = i\, d\alpha$$

So: $A' = A - i\, d\alpha$.

If we absorb the $i$ into the definition (writing $A = -i\, a$, the real-valued form), this becomes $a' = a + d\alpha$. The connection shifts by an exact 1-form.

**Gauge invariance of holonomy:**

The holonomy around a closed loop $\gamma$ is:

$$\operatorname{Hol}' = \exp\!\left(i \oint_\gamma a'\right) = \exp\!\left(i \oint_\gamma (a + d\alpha)\right) = \exp\!\left(i \oint_\gamma a + i \oint_\gamma d\alpha\right)$$

By Stokes' theorem (or simply because $\alpha$ is single-valued on the loop):

$$\oint_\gamma d\alpha = \alpha(\text{end}) - \alpha(\text{start}) = 0$$

(since the loop returns to the same point, and $\alpha$ is a well-defined function).

Therefore:

$$\operatorname{Hol}' = \exp\!\left(i \oint_\gamma a\right) = \operatorname{Hol}$$

The holonomy is **gauge-invariant**. Different choices of local frame (different gauge functions $\alpha$) give different-looking connections, but the physical observable (the holonomy) is the same.

**Significance:** Gauge invariance means the geometric content (parallel transport, holonomy) is independent of representation choices. For neural networks, this means the network's behavior should not depend on arbitrary choices like the basis of the embedding space. The thesis exploits this: by building gauge-invariant architectures, you ensure that the model's predictions depend only on geometric relationships, not on coordinate artifacts.

---

### Problem 8

**Horizontal-vertical decomposition in transformers.** Anonymous (2025) proved that trained transformers literally form principal fiber bundles. The attention sublayer's gradients and the feedforward sublayer's gradients are nearly orthogonal -- this is the horizontal-vertical decomposition of the Ehresmann connection. Explain what "horizontal" and "vertical" mean in fiber bundle language.

**Solution:**

At any point $e$ in the total space $E$ of a fiber bundle, the tangent space $T_eE$ decomposes as:

$$T_eE = V_e \oplus H_e$$

**Vertical subspace $V_e$:**
- Directions along the fiber $\pi^{-1}(\pi(e))$
- These change the representation WITHOUT moving to a new position in the base
- The projection $\pi$ maps $V_e$ to zero: $\pi_*(V_e) = 0$
- $V_e$ is intrinsically defined (it depends only on the bundle structure, not on the connection)

**Horizontal subspace $H_e$:**
- Directions "across" fibers, as defined by the connection
- These move to a new base position while keeping the representation "parallel"
- The projection $\pi$ maps $H_e$ isomorphically to $T_{\pi(e)}M$
- $H_e$ DEPENDS on the choice of connection (it IS the connection, in one formulation)

**In a transformer:**

The **attention sublayer** computes horizontal movement. Given a token at position $p$, attention looks at all other positions and determines how to transport information between them. It moves representations across the sequence (between fibers) while transforming them according to the connection. The output of attention is a "horizontal displacement" -- information gathered from other positions.

The **feedforward sublayer** computes vertical movement. It takes the representation at a fixed position and transforms it within the fiber (the representation space at that position). It does not look at other positions -- it only modifies the local representation.

The near-orthogonality of their gradients is precisely the $V \oplus H$ decomposition: the two sublayers contribute to orthogonal directions in the total space.

**Why this matters:**

This decomposition means transformer blocks naturally factor into:
1. Communication (attention = horizontal transport between positions)
2. Computation (feedforward = local transformation within each fiber)

This factoring is not imposed by architecture design -- it emerges from training. The fact that it matches the Ehresmann connection decomposition suggests that transformers are implicitly learning fiber bundle geometry.

---

### Problem 9

**Holonomy groups as inductive bias.** Berger's classification restricts which Lie groups can be holonomy groups of Riemannian manifolds. For neural networks, the relevant groups are: $U(1)$ (phase rotations), $SO(n)$ (real rotations), $U(n)$ (complex rotations), $GL(n)$ (general linear). Explain why choosing a smaller holonomy group is an inductive bias. What does each group restrict?

**Solution:**

The choice of holonomy group determines what transformations are allowed during parallel transport. Smaller groups restrict the allowed transformations more severely, providing a stronger inductive bias:

**$GL(n)$ -- General Linear Group ($n^2$ parameters):**
- Allows any invertible linear transformation
- No geometric structure is preserved
- Maximum expressivity, minimum inductive bias
- Standard neural networks implicitly operate here (dense linear layers are arbitrary invertible matrices)
- No guarantees about norm preservation, orthogonality, or geometric structure

**$U(n)$ -- Unitary Group ($n^2$ parameters, but constrained):**
- Preserves complex inner products: $\langle Uv, Uw \rangle = \langle v, w \rangle$
- Norms are preserved: $\|Uv\| = \|v\|$
- Prevents representation collapse (norms cannot shrink to zero)
- Also prevents explosion (norms cannot grow unboundedly)
- This is the gauge group of the Standard Model of physics

**$SO(n)$ -- Special Orthogonal Group ($\frac{n(n-1)}{2}$ parameters):**
- Preserves real inner products and orientations
- Norms, angles, and volumes are preserved
- Fewer parameters than $GL(n)$: for $n=64$, $SO(n)$ has 2016 parameters vs. $GL(n)$'s 4096
- Strong geometric guarantee: distances between representations are invariant
- The thesis proposes $SO(K)$ for the SGST (important: must use real $SO(K)$, not complex $U(K)$, because MPS does not support complex numbers)

**$U(1)$ -- Phase Rotations (1 parameter per dimension):**
- Only rotates the phase of each complex component independently
- Magnitude of each component is preserved
- Extremely parsimonious: 1 parameter per dimension instead of $n^2$
- RoPE (Rotary Position Embeddings) in standard transformers uses $U(1)$ -- this is the FLAT (zero-curvature) case
- The thesis (Sec. 6.7) identifies RoPE as $U(1)$ holonomy with zero curvature, and proposes $U(1)$ with nonzero curvature as a minimal extension

**The tradeoff:**

Smaller holonomy group = stronger inductive bias = fewer parameters + geometric guarantees, BUT less expressivity. The right choice depends on what geometric structure the data actually has. The thesis argues that $SO(K)$ is the right balance for language: it preserves distances (preventing collapse/explosion) while being expressive enough to capture contextual transformations, and it avoids the complex number issues that plague $U(K)$ on Apple Silicon.

---

### Problem 10

**The SYNTHESIS convergence.** The SYNTHESIS.md document describes 6 independent research lines converging on structured transport. Summarize how gauge theory (line 1: "attention IS gauge connection") and SSM research (line 3: "complex states enable computation") describe the same mathematical structure from different perspectives.

**Solution:**

**Gauge theory perspective (line 1):**

Attention applies a transformation $U_t$ (the Wilson line) to transport representations between positions. The transformation depends on the path (context) via the connection $A$. Specifically:
- The attention weights are components of the connection 1-form $A$
- The output of attention is the parallel-transported representation
- Multi-head attention corresponds to a reducible connection (direct sum of connections, one per head)
- The curvature $F = dA + A \wedge A$ measures the degree of path-dependence

**SSM perspective (line 3):**

The state-space model evolves a hidden state $x_t$ according to:

$$x_t = A_t\, x_{t-1} + B_t\, u_t$$

where $A_t$ is the (content-dependent) state transition matrix and $u_t$ is the input. The accumulated state from position 1 to position $t$ is:

$$x_t = A_t A_{t-1} \cdots A_1\, x_0 + (\text{input terms})$$

When $A_t$ is unitary or orthogonal, this product of matrices preserves norms.

**The mathematical equivalence:**

The key recognition is that the product $A_t A_{t-1} \cdots A_1$ IS the Wilson line $U_{1 \to t} = \mathcal{P} \exp\!\left(i \int_1^t A(\tau)\, d\tau\right)$. Both describe:

1. **Path-ordered product of local transformations:** The Wilson line is a product of infinitesimal rotations $\exp(i\, A(\tau)\, d\tau)$; the SSM state is a product of discrete transition matrices $A_t$. In the continuum limit, they are identical.

2. **Norm-preserving transport:** When $A_t$ is unitary (Mamba-3's "complex states"), the product preserves norms. This is exactly unitary parallel transport along a fiber bundle.

3. **Content-dependent:** In both frameworks, the transformation at each step depends on the local input (token content). In gauge theory, $A$ depends on the section (representation value). In SSMs, $A_t$ depends on the input $u_t$.

4. **Phase accumulation:** Mamba-3's "complex state" where each dimension evolves as $z_t = r_t\, e^{i\theta_t}\, z_{t-1}$ is exactly $U(1)$ holonomy. The phase $\theta_t$ at each step is the connection component, and the accumulated phase $\sum \theta_1 + \cdots + \theta_t$ is the Wilson line phase.

**The unexplored gap (from SYNTHESIS.md):**

Both frameworks independently discovered that:
- Norm-preserving (unitary/orthogonal) transport is essential
- Content-dependent transformations are essential
- The transport should compose multiplicatively (product of matrices, not sum)

But neither community has combined these with the **delta rule** (from SSM research line 4), which allows the accumulated state to be selectively updated rather than merely multiplied. The SYNTHESIS.md identifies that combining $U(K)$ transport (from gauge theory) with the delta rule (from SSM innovation) is the key unexplored direction -- this is the core of the SGST's proposed architecture.

---

## Comprehension Questions

1. What is a fiber bundle? Draw one with base = circle, fiber = interval $[0,1]$. Draw both a trivial version (cylinder) and a non-trivial version (Mobius band).

2. What is a connection on a fiber bundle? What does it "connect"? In what sense does it define parallel transport?

3. Explain holonomy in your own words. Why is it related to curvature? Can you have nontrivial holonomy with zero curvature? (Hint: think about flat connections on non-simply-connected spaces.)

4. What is a Wilson line? How does it relate to accumulated context in a sequence model? Why is it more compact than a KV cache?

5. The thesis claims "attention IS a gauge connection." Using the framework of this unit, explain this claim: what is the base manifold, what is the fiber, what is the connection 1-form, and what is the parallel transport?

6. Read `topology/gauge_connections_fiber_bundles.md`. What gauge group does the thesis propose, and why? How does the choice of gauge group relate to the inductive bias discussion in Problem 9?

7. Read `topology/SYNTHESIS.md`. What are the 6 research lines that converge on geometric transport? For each line, identify the key mathematical concept and how it maps to the fiber bundle framework.

---

## Bridge to Thesis

- **Fiber bundles** (Problems 1-2, 8) are the correct mathematical model for token representations (thesis Sec. 2.5). The base manifold is the sequence position space, the fiber is the representation space at each position, and the connection encodes how attention transforms representations.

- **Connections and parallel transport** (Problems 3-6) define how context transforms representations (thesis Sec. 4.2). The attention mechanism IS a connection: it determines horizontal directions in the representation bundle, and its output is the parallel-transported representation.

- **Wilson lines** (Problem 5) equal accumulated KV cache in geometric language (thesis discussion, `topology/topological_computation.md`). The Wilson line from position 0 to position $t$ is a single group element encoding all context, replacing the growing list of KV pairs in standard transformers.

- **Gauge groups** (Problem 9) determine the inductive bias (`topology/gauge_connections_fiber_bundles.md`). $U(1)$ for phases (RoPE), $SO(n)$ for real rotations (SGST on MPS), $U(n)$ for complex rotations (theoretical ideal). The choice of gauge group is a fundamental architectural decision.

- **Holonomy** (Problems 3, 10) is the key computational primitive -- path-dependent context accumulation (`topology/holonomy_closed_manifolds.md`, `topology/holonomy_open_manifolds.md`). The holonomy group determines what transformations the network can express.

- **The convergence from 6 research directions** (Problem 10) is the central intellectual contribution (`topology/SYNTHESIS.md`). Gauge theory, optimal transport, SSMs, the delta rule, spectral methods, and geometric deep learning all converge on the same mathematical structure: norm-preserving, content-dependent, compositional transport along fiber bundles.

- **Next steps:** With fiber bundles and gauge theory in hand, you are ready to study the specific architectures: Wilson line attention (`topology/topological_computation.md`), the Finsler Transformer (Ch. 4), and the SGST's spectral transport (V12-V16).
