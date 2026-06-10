# Unit 07: Finsler Geometry: Asymmetric Metrics for Language

## Learning Objectives

By the end of this unit, you will be able to:

1. Understand how Finsler geometry generalizes Riemannian geometry by allowing direction-dependent metrics
2. Compute distances using a Finsler metric (where $d(A,B) \neq d(B,A)$)
3. Understand why asymmetric metrics are natural for causal/sequential data
4. Connect Finsler geometry to the causal structure of language (no need for attention masks)
5. Understand why the Finsler Transformer (thesis Ch. 4) was theoretically correct but computationally prohibitive

## Prerequisites

- Unit 05 (Differential Geometry on Smooth Manifolds)

## Readings

- Bao, Chern, Shen: *An Introduction to Riemann-Finsler Geometry*, Ch. 1-2 (for formal treatment)
- Thesis Ch. 4 (The Geometric Turn: Finsler Transformer), all sections
- Thesis Sec. 4.1 (Riemann and Finsler comparison)
- Repo: `topology/metrics_and_transport.md` (comparative study of 7+ metric types)
- Repo: `topology/lit_review/finsler_information_geometry_ml.md` (Finsler in ML applications)

## Key Concepts

1. **Riemannian metric:** $g_p(u,v)$ -- a symmetric bilinear form on the tangent space at each point $p$. Always gives symmetric distances: $d(A,B) = d(B,A)$.

2. **Finsler metric:** $F(x,v)$ -- a norm on each tangent space $T_xM$ that depends on both position $x$ and direction $v$. Crucially, $F$ is NOT necessarily symmetric in $v$: $F(x,v) \neq F(x,-v)$ in general.

3. **The asymmetry:** Because $F(x,v) \neq F(x,-v)$, traveling from $A$ to $B$ along a path costs a different amount than traveling from $B$ to $A$ along the same path reversed. The distance function becomes a *quasimetric*: $d(A,B) \neq d(B,A)$.

4. **Why language is Finsler:** "A implies B" does NOT mean "B implies A". Causal sequences have a preferred direction. Predicting the next token from context (forward) is fundamentally easier than inferring context from a future token (backward). This asymmetry is intrinsic to language, not an artifact of the model.

5. **Geodesics in Finsler geometry:** A geodesic minimizes the length functional $\int F(\gamma(t), \gamma'(t))\, dt$. Because $F$ is asymmetric, the geodesic from $A$ to $B$ is generally a different curve (with different length) than the geodesic from $B$ to $A$.

6. **Geodesic deviation loss (thesis Sec. 4.2.1):** Measures how nearby geodesics diverge. If two points start close together, their geodesics separate at a rate determined by the curvature. The Finsler Transformer learns the metric by minimizing prediction error on this deviation.

7. **Matrix holonomy (thesis Sec. 4.2.2):** When you parallel-transport a vector around a closed loop on a curved manifold, it comes back rotated. The rotation matrix is the holonomy. In the Finsler Transformer, holonomy captures the path-dependent transformation accumulated along a token sequence.

8. **The metric-first principle (thesis Sec. 4.3):** Rather than hand-designing attention patterns, learn the metric tensor from data and derive all geometric quantities (geodesics, curvature, parallel transport) from it. The metric is the fundamental object.

9. **Scalability wall (thesis Sec. 4.5):** Computing Finsler quantities requires evaluating the metric at each direction, costing $O(d^2)$ per step (where $d$ is the embedding dimension). This made the Finsler Transformer impractical at scale.

10. **Quasimetric spaces:** A space where $d(A,B) \neq d(B,A)$ but the triangle inequality $d(A,C) \leq d(A,B) + d(B,C)$ still holds. This is a relaxed version of Finsler geometry that retains the essential asymmetry without requiring the full differential-geometric machinery.

---

## Worked Problems

### Problem 1: Computing Asymmetric Distances

**Problem:** Define a simple Finsler metric on $\mathbb{R}^1$: $F(x,v) = v$ if $v \geq 0$, and $F(x,v) = 2|v|$ if $v < 0$. Compute the distance from $0$ to $3$ and from $3$ to $0$.

**Solution:**

**$d(0,3)$:** We travel forward along the path $\gamma(t) = t$ for $t \in [0,3]$. The velocity is $\gamma'(t) = 1 > 0$. Since the velocity is positive, we use $F(x,v) = v$:

$$d(0,3) = \int_0^3 F(t, 1)\, dt = \int_0^3 1\, dt = 3$$

**$d(3,0)$:** We travel backward along the path $\gamma(t) = 3 - t$ for $t \in [0,3]$. The velocity is $\gamma'(t) = -1 < 0$. Since the velocity is negative, we use $F(x,v) = 2|v|$:

$$d(3,0) = \int_0^3 F(3-t, -1)\, dt = \int_0^3 2\, dt = 6$$

**Result:** $d(0,3) = 3$ but $d(3,0) = 6$. Going "backward" costs twice as much! This is the fundamental asymmetry of Finsler geometry. The metric penalizes travel in the negative direction, making the reverse path more expensive even though it covers the same spatial extent.

---

### Problem 2: Finsler Asymmetry as Causality in Language

**Problem:** In the language model context, the asymmetry of a Finsler metric corresponds to causality. If we define the "forward" direction as reading left-to-right, explain why $d(\text{token}_1, \text{token}_5)$ should be small (natural forward reading) while $d(\text{token}_5, \text{token}_1)$ should be large (unnatural backward inference).

**Solution:**

In causal language modeling, predicting token 5 from tokens 1-4 is the natural task -- the model has all the context it needs. This is "forward" transport along the causal direction, and should be cheap (small distance).

Predicting token 1 from token 5 requires "backward" reasoning -- inferring causes from effects, which is much harder and ill-posed (multiple possible causes could have led to the same effect).

The Finsler metric makes this formal:
- Forward transport is "downhill" along the metric: $F(x, v_{\text{forward}})$ is small
- Backward transport is "uphill": $F(x, v_{\text{backward}})$ is large

The asymmetry IS causality.

Standard attention enforces causality with a mask (setting certain weights to $-\infty$), but this is a crude binary mechanism: either information flows or it does not. In Finsler geometry, causality emerges from the geometry itself. The metric allows graded asymmetry -- some backward inferences might be merely expensive (not impossible), reflecting the fact that some effects are more diagnostic of their causes than others.

---

### Problem 3: Comparing Riemannian, Finsler, and Lorentzian Metrics

**Problem:** Compare Riemannian, Finsler, and Lorentzian metrics in terms of: (a) symmetry, (b) how they encode causality, (c) computational cost. Use the table from `topology/metrics_and_transport.md`.

**Solution:**

**(a) Symmetry:**
- **Riemannian:** Symmetric. $d(A,B) = d(B,A)$ always. The metric tensor $g_p$ is a symmetric bilinear form.
- **Finsler:** Asymmetric. $d(A,B) \neq d(B,A)$ in general. The norm $F(x,v) \neq F(x,-v)$.
- **Lorentzian:** Has signature $(1, n-1)$ rather than $(n, 0)$. Not positive definite -- some tangent vectors have negative "length squared." Distances are symmetric between spacelike-separated events but the causal structure introduces ordering.

**(b) How they encode causality:**
- **Riemannian:** No inherent causality. All directions are equivalent. Must add causal masks externally (as standard Transformers do).
- **Finsler:** Causality from asymmetric cost. Forward transport is cheap, backward is expensive. Causality is a continuous quantity, not binary.
- **Lorentzian:** Causality from causal structure. The light cone at each point divides tangent vectors into timelike (causal), null (lightlike), and spacelike (acausal). Only timelike and null directions allow causal influence.

**(c) Computational cost:**
- **Riemannian:** $O(d)$ for parallel transport (metric is a fixed bilinear form at each point).
- **Finsler:** $O(d^2)$ or higher because the metric depends on direction. At each step, you must evaluate $F(x,v)$ for the current direction $v$, and the Hessian of $F^2$ (which gives the induced inner product) is direction-dependent.
- **Lorentzian:** $O(d)$ but constrained to the causal cone. Must project onto the cone at each step.

**Verdict:** Finsler is the most expressive for language (continuous, graded causality) but the most expensive. This is why the thesis explored it but ultimately moved to spectral methods that capture the same asymmetry at lower cost.

---

### Problem 4: Geodesic Deviation and Curvature

**Problem:** The thesis's Finsler Transformer (Ch. 4) learns a metric from data on synthetic trajectories. The geodesic deviation loss measures how well the learned metric predicts how nearby geodesics separate. Write the intuition: if you have two nearby starting points $x$ and $x + \delta x$, how does the distance between their geodesics grow with time?

**Solution:**

The geodesic deviation equation governs how the separation vector $J$ (the vector pointing from one geodesic to a nearby one) evolves:

$$\frac{d^2 J}{dt^2} = -R(\gamma', J)\, \gamma'$$

where $R$ is the Riemann curvature tensor and $\gamma'$ is the tangent to the reference geodesic.

The behavior depends on curvature:

- **Flat space ($R = 0$):** Two parallel geodesics (straight lines) stay the same distance apart forever. $\frac{d^2 J}{dt^2} = 0$, so $J$ grows linearly at most.

- **Positive curvature (like a sphere):** Geodesics starting parallel converge. Think of lines of longitude on Earth: they start parallel at the equator but meet at the poles. $R$ acts as a restoring force.

- **Negative curvature (like a saddle):** Geodesics starting parallel diverge exponentially. Small initial differences amplify rapidly.

The Finsler Transformer learns the metric by minimizing the prediction error on geodesic separation. If the learned curvature tensor $R$ matches the data's actual curvature, the model correctly predicts how representations diverge with increasing context length.

For language: negative curvature regions correspond to contexts where small differences in input lead to very different continuations (high sensitivity). Positive curvature regions correspond to contexts where many inputs converge to similar continuations (low sensitivity, high predictability).

---

### Problem 5: The Scalability Wall

**Problem:** The Finsler Transformer proved theoretically correct on synthetic data (thesis Sec. 4.4): spirals converged, stochastic data was identified as noise-dominated, and holonomy was measurable. But it hit a "scalability wall." Explain what $O(d^2)$ per step means for $d = 512$.

**Solution:**

For embedding dimension $d = 512$, each step requires $O(d^2) = O(262{,}144)$ operations just for the geometric computation (evaluating the metric and its derivatives at each direction).

For comparison, standard attention on a sequence of length $T$ requires $O(T \cdot d) = O(T \cdot 512)$ per head.

At $T = 128$: attention costs $O(65{,}536)$ per head -- much cheaper than a single Finsler step.

And the Finsler computation happens at EVERY token position, not just once. The total cost becomes:
- **Finsler:** $O(T \cdot d^2) = O(T \cdot 262{,}144)$
- **Attention:** $O(T^2 \cdot d) = O(T^2 \cdot 512)$

For $T < d$ (which is typical at small scale), Finsler is MORE expensive than quadratic attention. At $T = 512$, they break even. Only for $T \gg d$ does Finsler become relatively cheaper, but at that point the constant factors still dominate.

The practical impact: training the Finsler Transformer on even modest-sized data required hours where standard Transformers required minutes. This is why the thesis pivoted to spectral methods: they provide the same geometric structure at $O(d \log d)$ cost via FFT-based operations.

---

### Problem 6: KL Divergence as a Finsler Metric

**Problem:** Information geometry uses the Fisher-Rao metric on probability distributions. This metric is naturally Finsler (asymmetric) when you consider KL divergence. Show that $\mathrm{KL}(P \| Q) \neq \mathrm{KL}(Q \| P)$ for $P = \text{Bernoulli}(0.7)$, $Q = \text{Bernoulli}(0.5)$.

**Solution:**

Recall $\mathrm{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$.

**$\mathrm{KL}(P \| Q)$:**

$$\mathrm{KL}(P \| Q) = 0.7 \cdot \log\frac{0.7}{0.5} + 0.3 \cdot \log\frac{0.3}{0.5}$$
$$= 0.7 \cdot \log(1.4) + 0.3 \cdot \log(0.6)$$
$$= 0.7 \cdot (0.3365) + 0.3 \cdot (-0.5108)$$
$$= 0.2356 - 0.1532$$
$$= 0.0824$$

**$\mathrm{KL}(Q \| P)$:**

$$\mathrm{KL}(Q \| P) = 0.5 \cdot \log\frac{0.5}{0.7} + 0.5 \cdot \log\frac{0.5}{0.3}$$
$$= 0.5 \cdot \log(0.7143) + 0.5 \cdot \log(1.6667)$$
$$= 0.5 \cdot (-0.3365) + 0.5 \cdot (0.5108)$$
$$= -0.1682 + 0.2554$$
$$= 0.0872$$

**Result:** $\mathrm{KL}(P \| Q) = 0.0824 \neq \mathrm{KL}(Q \| P) = 0.0872$.

The asymmetry is small here but can be large for very different distributions. For example, $\mathrm{KL}(P \| Q)$ diverges to infinity if $Q$ assigns zero probability to an event that $P$ assigns nonzero probability, but $\mathrm{KL}(Q \| P)$ remains finite.

This connects to the SYNTHESIS.md proposal of tokens as probability distributions on the Fisher-Rao manifold -- the natural metric on that manifold is inherently asymmetric, reflecting the Finsler structure of information space.

---

### Problem 7: Quasimetric Triangle Inequality

**Problem:** A quasimetric is a generalization of a metric where $d(x,y) \neq d(y,x)$ but the triangle inequality $d(x,z) \leq d(x,y) + d(y,z)$ still holds. Show that the Finsler metric from Problem 1 satisfies the triangle inequality.

**Solution:**

Take three points $a < b < c$ on $\mathbb{R}$. Using the metric $F(x,v) = v$ for $v \geq 0$ and $F(x,v) = 2|v|$ for $v < 0$:

**Case 1: All forward ($a$ to $c$ via $b$).**

$$d(a,c) = c - a$$
$$d(a,b) + d(b,c) = (b - a) + (c - b) = c - a$$
$$d(a,c) = d(a,b) + d(b,c) \quad \text{[equality holds]}$$

**Case 2: All backward ($c$ to $a$ via $b$).**

$$d(c,a) = 2(c - a)$$
$$d(c,b) + d(b,a) = 2(c - b) + 2(b - a) = 2(c - a)$$
$$d(c,a) = d(c,b) + d(b,a) \quad \text{[equality holds]}$$

**Case 3: Mixed direction (e.g., $a$ to $c$, detour through point $d > c$).**

$$d(a,d) + d(d,c) = (d - a) + 2(d - c) \quad \text{[forward then backward]}$$
$$d(a,c) = c - a$$

We need: $c - a \leq (d - a) + 2(d - c) = d - a + 2d - 2c = 3d - a - 2c$

Since $d > c$: $3d - a - 2c > 3c - a - 2c = c - a$ [true]

The triangle inequality holds in all cases. Quasimetric spaces are the natural setting for asymmetric embeddings in NLP, where the cost of traversal depends on direction.

---

### Problem 8: Finsler Dimensionality Reduction

**Problem:** The `topology/lit_review/finsler_information_geometry_ml.md` discusses Finsler-MDS, Finsler-tSNE, and Finsler-UMAP. These extend standard dimensionality reduction to handle asymmetric distances. Give an example of real-world data where asymmetric distances are essential.

**Solution:**

**Web page linking:** The "distance" from page A to page B (how easily you navigate from A to B) differs from B to A. Page A might link directly to B (distance 1), but B might not link back to A (distance = many hops through other pages). Standard MDS, which assumes symmetric distances, cannot faithfully embed this structure.

**Social media following:** A follows B but B does not follow A. The "influence distance" from a celebrity to a follower is large (the follower attends to the celebrity), but the reverse distance is effectively infinite (the celebrity does not attend to the follower).

**Natural language entailment:** Entailment is asymmetric. "A dog is an animal" is true (short distance from specific to general), but "An animal is a dog" is false (large or infinite distance from general to specific). Embedding words with symmetric distances (as in standard word2vec) cannot capture this.

**Biological networks:** A gene regulatory network has directed edges: gene A activates gene B, but B may not activate A. The "regulatory distance" is fundamentally asymmetric.

All of these require Finsler or quasimetric models. Standard Riemannian methods that assume $d(A,B) = d(B,A)$ lose essential structural information.

---

### Problem 9: Directional Fields and Causal Structure

**Problem:** The thesis (Sec. 4.4) shows the Finsler Transformer learning directional fields on synthetic data. A "directional field" is a vector field where the direction of arrows shows the preferred direction of flow. Explain how a learned directional field captures causal structure in a sequence.

**Solution:**

The directional field assigns a preferred direction to each point in the representation manifold. In language:

- The "flow" goes from cause to effect, from premise to conclusion, from subject to predicate.
- At each token position, the field indicates which direction is "downstream" (forward in the causal sense).

The Finsler metric interacts with this field:
- **Parallel transport along the flow direction is cheap:** $F(x, v)$ is small when $v$ aligns with the field. Information flows naturally in this direction.
- **Transport against the flow is expensive:** $F(x, -v)$ is large. Backward inference requires overcoming the metric's resistance.

The model learns this field from data. It discovers which directions are "natural" for the language's causal structure without being told. On synthetic spiral data, the learned field aligns with the spiral's direction of winding. On language data, it would align with the left-to-right causal flow.

This is more elegant than a binary attention mask (allowed/not allowed) because it allows **graded asymmetry**:
- Some backward inferences are merely expensive (e.g., inferring "the cat" from "sat on the mat" -- plausible backward inference).
- Others are nearly impossible (e.g., inferring the first word of a paragraph from the last).

The continuous nature of the directional field captures these gradations, whereas the binary mask treats all backward directions identically.

---

### Problem 10: From Finsler to Spectral Methods

**Problem:** Why did the thesis ultimately move away from explicit Finsler geometry to spectral methods, even though the Finsler framework was theoretically correct? What did spectral methods preserve from the Finsler insight?

**Solution:**

**Why move away:** The scalability wall ($O(d^2)$ per step) made the Finsler Transformer impractical for real language modeling tasks. The synthetic experiments validated the theory, but training on WikiText-103 or larger datasets was infeasible.

**What survived:** The KEY insight -- that language has intrinsic directional structure that geometry captures -- survived in spectral form:

1. **Asymmetric transport via spectral kernel:** The spectral transport kernel $\exp(-D \cdot \omega^2 - i \cdot A \cdot \omega)$ implements asymmetric transport. The real part ($-D \cdot \omega^2$) is symmetric diffusion. The imaginary part ($-i \cdot A \cdot \omega$) encodes direction-dependent phase shifts. This is the spectral analogue of the Finsler asymmetry: the gauge connection $A$ plays the role of the directional field.

2. **Causal recurrence:** The SSM context accumulator naturally has a preferred direction (it processes tokens left-to-right). This is "hardware causality" that matches the Finsler asymmetry without computing it explicitly.

3. **Fourier-geometry correspondence:** The forward-reverse spectral loop (FFT -> process in frequency domain -> IFFT) respects a deep correspondence between frequency-domain operations and fiber bundle operations. Multiplication by phases in frequency domain IS parallel transport. The geometry went from explicit (compute Christoffel symbols, evaluate the Finsler metric at each direction) to implicit (spectral operations that are geometrically equivalent but cost $O(d \log d)$ via FFT).

4. **Spectral sparsity as rank control:** The Finsler framework provided no natural mechanism for rank control. Spectral sparsity (few active frequency modes) gives explicit rank control by construction, addressing a problem the Finsler approach did not.

The evolution was: **correct geometry (Finsler) -> efficient geometry (spectral) -> same asymmetry at $\frac{1}{d}$ the cost.**

---

## Comprehension Questions

1. What is the key difference between Riemannian and Finsler geometry? Why does it matter for language modeling?

2. Give an example of an asymmetric distance in everyday life (not from the readings).

3. Why is $O(d^2)$ per step prohibitive for practical language models? At what sequence length does it become cheaper than quadratic attention?

4. How does spectral transport recover the asymmetry of Finsler geometry without computing the Finsler metric explicitly?

5. Read `topology/metrics_and_transport.md`. Which metric type does the thesis ultimately advocate for, and why?

---

## Bridge to Thesis

The Finsler Transformer (thesis Ch. 4) represents the "geometric turn" -- the moment the thesis committed to the principle that language has intrinsic geometry. The Finsler framework was theoretically correct: it captured causality, asymmetry, and curvature in a unified differential-geometric framework. The experiments on synthetic data validated every prediction.

But it could not scale. The $O(d^2)$ cost per step was a fundamental limitation of computing explicit Finsler quantities. This failure was productive: it showed exactly what geometric properties were needed (asymmetry, parallel transport, curvature sensitivity) and motivated the search for cheaper representations.

The spectral methods of Ch. 5 (SGST) deliver these properties at $O(d \log d)$ cost. The gauge connection provides asymmetry. The spectral transport kernel provides parallel transport. The mode structure provides curvature sensitivity. The Finsler chapter is not a dead end -- it is the theoretical foundation that the spectral architecture implements efficiently.

**Next unit:** Unit 08 covers compressed sensing and spectral sparsity -- the other half of the SGST's theoretical foundation.
