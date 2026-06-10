# Unit 08: Compressed Sensing, Sparsity, and the Donoho-Stark Uncertainty Principle

## Learning Objectives

By the end of this unit, you will be able to:

1. State and understand the Donoho-Stark uncertainty principle for DFT
2. Understand sparsity and why it enables efficient representation
3. Define the Restricted Isometry Property (RIP) and its role in compressed sensing
4. Understand basis pursuit and $\ell_1$ minimization for sparse recovery
5. Connect spectral sparsity to the SGST architecture's core design principle
6. Apply soft-thresholding as a proximal operator for sparsity enforcement

## Prerequisites

- Unit 03 (Fourier Analysis)
- Unit 02 (Optimization Foundations)

## Readings

- Candes & Wakin, "An Introduction to Compressive Sampling" (IEEE Signal Processing, 2008) -- accessible survey
- Thesis Sec. 2.7 (Spectral Methods in Deep Learning), especially 2.7.1-2.7.2
- Thesis Sec. 5.1 (Why Sparsity Belongs in Fourier Space) -- THE KEY SECTION
- Thesis Table 5.1 (Spatial vs Spectral Sparsity: Architectural Consequences)
- Repo: `v12/README.md` (the spectral shift discovery)

## Key Concepts

1. **Sparsity:** A signal $x \in \mathbb{R}^d$ is $s$-sparse if at most $s$ of its $d$ components are nonzero. The sparsity level $s$ measures the signal's intrinsic complexity. Most natural signals are sparse or approximately sparse in some basis.

2. **The Donoho-Stark Uncertainty Principle (thesis Eq. 2.6):** For a signal $x \in \mathbb{C}^d$ and its DFT $\hat{x}$:

$$|\operatorname{supp}(x)| \cdot |\operatorname{supp}(\hat{x})| \geq d$$

where $\operatorname{supp}$ denotes the support (set of nonzero entries). You CANNOT be sparse in BOTH time/space AND frequency simultaneously.

3. **Interpretation:** The uncertainty principle creates a fundamental tradeoff. If you choose to be sparse in one domain, you are forced to be spread out in the other. This is not a limitation -- it is a design principle.

4. **The thesis insight:** Spatial sparsity (few active dimensions) FORCES spectral spread, making frequency-domain transport expensive because you must handle all frequencies. Spectral sparsity (few active frequencies) forces spatial spread, which is what you WANT for language -- every token position is nonzero, giving long-range interaction by construction.

5. **Compressed sensing:** If a signal $x$ is $s$-sparse in some basis $\Psi$, you can recover $x$ exactly from $m = O(s \log(d/s))$ random measurements. This is far fewer than the $d$ measurements naive sampling would require.

6. **Restricted Isometry Property (RIP):** A measurement matrix $\Phi$ satisfies the RIP of order $s$ with constant $\delta_s$ if for all $s$-sparse vectors $x$:

$$(1 - \delta_s) \|x\|^2 \leq \|\Phi x\|^2 \leq (1 + \delta_s) \|x\|^2$$

This means $\Phi$ approximately preserves the norm (and therefore distances) of sparse signals.

7. **Basis Pursuit:** Recover a sparse signal by solving: minimize $\|x\|_1$ subject to $\Phi x = y$. This is a convex relaxation of the NP-hard $\ell_0$ minimization problem (minimize the number of nonzero entries).

8. **Soft-thresholding:** The proximal operator for the $\ell_1$ norm. For each component:

$$S_\lambda(x)_i = \operatorname{sign}(x_i) \cdot \max(|x_i| - \lambda, 0)$$

Components with magnitude below $\lambda$ are set to zero; components above $\lambda$ are shrunk toward zero by $\lambda$.

9. **Iterative Soft Thresholding (ISTA):** Alternates between a gradient step (move toward the data-fidelity minimum) and soft-thresholding (enforce sparsity). Converges to the sparse solution of the basis pursuit problem.

10. **Top-$k$ sparsification:** Keep only the $k$ largest magnitude components; zero out the rest. A hard sparsification operator used in the SGST's forward-reverse spectral loop.

---

## Worked Problems

### Problem 1: Uncertainty Principle -- Maximally Spatially Sparse

**Problem:** Verify the Donoho-Stark uncertainty principle for $x = [1, 0, 0, 0, 0, 0, 0, 0]$ (maximally spatially sparse, $N = 8$). Compute the DFT and count support sizes.

**Solution:**

The spatial support: $|\operatorname{supp}(x)| = 1$ (only one nonzero entry, at position 0).

Compute the DFT: $X[k] = \sum_{n=0}^{7} x[n] \cdot e^{-i 2\pi k n / 8} = x[0] \cdot e^0 = 1$ for all $k$.

So $X = [1, 1, 1, 1, 1, 1, 1, 1]$.

The spectral support: $|\operatorname{supp}(X)| = 8$ (all entries are nonzero).

**Check the uncertainty principle:**

$$|\operatorname{supp}(x)| \cdot |\operatorname{supp}(X)| = 1 \cdot 8 = 8 \geq 8 = N$$

The bound is tight. Maximally spatially sparse implies maximally spectrally spread.

**Interpretation:** A single spatial impulse (a delta function) contains ALL frequencies equally. This is the fundamental tradeoff: concentrating a signal in space forces it to spread across all frequencies. If you wanted to transport this signal in the frequency domain, you would need to handle all 8 frequency components -- no savings from sparsity.

---

### Problem 2: Uncertainty Principle -- Maximally Spectrally Sparse

**Problem:** Now verify for $x = [1, 1, 1, 1, 1, 1, 1, 1] / \sqrt{8}$ (maximally spatially spread, $N = 8$). Compute the DFT.

**Solution:**

The spatial support: $|\operatorname{supp}(x)| = 8$ (all entries are nonzero).

Compute the DFT:

$$X[0] = \frac{1}{\sqrt{8}} \sum_{n=0}^{7} 1 = \frac{8}{\sqrt{8}} = \sqrt{8}$$

$$X[k] = \frac{1}{\sqrt{8}} \sum_{n=0}^{7} e^{-i 2\pi k n / 8} = 0 \quad \text{for } k \geq 1$$

(The sum of all $N$th roots of unity is zero for $k \neq 0$.)

So $X = [\sqrt{8}, 0, 0, 0, 0, 0, 0, 0]$.

The spectral support: $|\operatorname{supp}(X)| = 1$ (only the DC component).

**Check the uncertainty principle:**

$$|\operatorname{supp}(x)| \cdot |\operatorname{supp}(X)| = 8 \cdot 1 = 8 \geq 8 = N$$

Again the bound is tight. Maximally spectrally sparse (one mode) implies maximally spatially spread (constant signal everywhere).

**Interpretation:** A constant signal has only the DC (zero-frequency) component. It is spectrally sparse but globally extended in space. For the SGST, this means tokens with few active frequency modes automatically have representations that extend across all spatial positions -- providing long-range interaction by construction.

---

### Problem 3: Compressed Sensing Bounds for SGST Parameters

**Problem:** The SGST uses spectral sparsity: tokens have $s = 10$ active frequency modes out of $d = 32$. Using the uncertainty principle, what is the minimum spatial extent? Using compressed sensing theory, how many measurements are needed to recover the token?

**Solution:**

**Uncertainty principle bound:**

$$|\operatorname{supp}(x)| \geq \frac{d}{|\operatorname{supp}(\hat{x})|} = \frac{32}{10} = 3.2$$

So at least 4 spatial positions must be nonzero (rounding up since support size is an integer).

**Compressed sensing bound:**

$$m = O(s \cdot \log(d / s)) = O(10 \cdot \log(32 / 10)) = O(10 \cdot \log(3.2)) = O(10 \cdot 1.163) \approx 12$$

So a 32-dimensional token with 10 active modes can be recovered from approximately 12 random projections. The compression ratio is $32/12 \approx 2.7\times$.

**For actual SGST parameters ($s = 10$, $d = 256$):**

$$m = O(10 \cdot \log(256 / 10)) = O(10 \cdot \log(25.6)) = O(10 \cdot 3.24) \approx 32$$

This gives a compression ratio of $256/32 = 8\times$. Only 32 random measurements suffice to recover a 256-dimensional token that has 10 active spectral modes.

**Significance:** This tells us that spectrally sparse tokens live in a much lower-dimensional subspace than the ambient dimension $d$ suggests. The SGST exploits this: transport operates on $s$ modes instead of $d$ dimensions, achieving $O(s \log d)$ cost instead of $O(d \log d)$.

---

### Problem 4: Spatial vs Spectral Sparsity Comparison Table

**Problem:** The thesis Table 5.1 compares spatial vs spectral sparsity. Fill in the key differences.

**Solution:**

| Property | Spatial Sparsity | Spectral Sparsity |
|----------|-----------------|-------------------|
| **What's sparse** | Few active dimensions (most entries zero) | Few active frequency modes (most spectral coefficients zero) |
| **Fourier dual** | Spectrally spread (uncertainty principle forces all frequencies active) | Spatially extended (uncertainty principle forces all positions nonzero) |
| **Transport cost** | $O(d \log d)$ -- must handle all frequencies in the spectral domain | $O(s \log d)$ where $s \ll d$ -- only active modes need transport |
| **Spatial extent** | Localized (few positions carry information) | Global (every position is nonzero by construction) |
| **Interaction range** | Short-range (spatially concentrated information cannot reach distant tokens cheaply) | Long-range (by construction, since the signal extends everywhere) |
| **Rank control** | No guarantee -- sparse but possibly correlated vectors can have any rank | Rank $= s$ by construction -- each active mode is an independent degree of freedom |
| **Neural analogy** | Sparse activations (ReLU zeros out negative values) | Grid cells in neuroscience (spectrally sparse, spatially periodic patterns) |

**The thesis argument:** Spatial sparsity is the wrong choice for language representations because it forces spectral spread (expensive transport) and provides only short-range interaction. Spectral sparsity is the right choice because it forces spatial extension (long-range interaction) and enables cheap transport (only $s$ modes to process).

---

### Problem 5: Soft-Thresholding in Action

**Problem:** Apply soft-thresholding with threshold $\lambda = 0.5$ to the spectral vector $X = [3.0, -0.2, 1.5, 0.4, -2.0, 0.1, 0.8, -0.3]$. Count the sparsity before and after.

**Solution:**

The soft-thresholding operator $S_\lambda(x)_i = \operatorname{sign}(x_i) \cdot \max(|x_i| - \lambda, 0)$:

$$X[0] = 3.0 \;\to\; \operatorname{sign}(3.0) \cdot \max(3.0 - 0.5, 0) = +1 \cdot 2.5 = 2.5$$
$$X[1] = -0.2 \;\to\; \operatorname{sign}(-0.2) \cdot \max(0.2 - 0.5, 0) = -1 \cdot 0 = 0$$
$$X[2] = 1.5 \;\to\; \operatorname{sign}(1.5) \cdot \max(1.5 - 0.5, 0) = +1 \cdot 1.0 = 1.0$$
$$X[3] = 0.4 \;\to\; \operatorname{sign}(0.4) \cdot \max(0.4 - 0.5, 0) = +1 \cdot 0 = 0$$
$$X[4] = -2.0 \;\to\; \operatorname{sign}(-2.0) \cdot \max(2.0 - 0.5, 0) = -1 \cdot 1.5 = -1.5$$
$$X[5] = 0.1 \;\to\; \operatorname{sign}(0.1) \cdot \max(0.1 - 0.5, 0) = +1 \cdot 0 = 0$$
$$X[6] = 0.8 \;\to\; \operatorname{sign}(0.8) \cdot \max(0.8 - 0.5, 0) = +1 \cdot 0.3 = 0.3$$
$$X[7] = -0.3 \;\to\; \operatorname{sign}(-0.3) \cdot \max(0.3 - 0.5, 0) = -1 \cdot 0 = 0$$

**Result:** $S_{0.5}(X) = [2.5, 0, 1.0, 0, -1.5, 0, 0.3, 0]$

**Before:** 8 nonzero entries (0% sparse).
**After:** 4 nonzero entries (50% sparse).

The threshold zeroed out 4 components with small magnitudes ($0.2, 0.4, 0.1, 0.3$ -- all below $0.5$) while shrinking the large ones toward zero by $0.5$. The energy is mostly preserved: the dominant modes ($3.0, 1.5, -2.0$) retain most of their magnitude. The small components, which likely carry noise rather than signal, are eliminated.

---

### Problem 6: Convexity of $\ell_1$ Norm and Sparsity Promotion

**Problem:** Prove that $\|x\|_1 = \sum_i |x_i|$ is a convex function. Then explain why minimizing $\|x\|_1$ subject to linear constraints promotes sparse solutions (compared to $\|x\|_2$).

**Solution:**

**Convexity proof:**

For any $x, y \in \mathbb{R}^d$ and $\lambda \in [0,1]$:

$$\|\lambda x + (1 - \lambda) y\|_1$$
$$= \sum_i |\lambda x_i + (1 - \lambda) y_i|$$
$$\leq \sum_i (\lambda |x_i| + (1 - \lambda) |y_i|) \quad \text{[triangle inequality]}$$
$$= \lambda \sum_i |x_i| + (1 - \lambda) \sum_i |y_i|$$
$$= \lambda \|x\|_1 + (1 - \lambda) \|y\|_1$$

This is exactly the definition of convexity. QED.

**Why $\ell_1$ promotes sparsity (geometric argument):**

Consider the constraint set $C = \{x : \Phi x = y\}$, which is a hyperplane (or affine subspace) in $\mathbb{R}^d$. We want the point in $C$ that minimizes the norm.

- The **$\ell_1$ ball** $\{x : \|x\|_1 \leq c\}$ is a cross-polytope (diamond shape in 2D, higher-dimensional analogue in $\mathbb{R}^d$). Its corners lie on the coordinate axes -- these are the sparsest points (only one coordinate nonzero per corner). As you inflate the $\ell_1$ ball (increase $c$), it first touches the constraint hyperplane at a corner or edge -- generically at a corner, which is a sparse point.

- The **$\ell_2$ ball** $\{x : \|x\|_2 \leq c\}$ is a smooth sphere with no corners. As you inflate it, it touches the constraint hyperplane at a point that is generically NOT aligned with any coordinate axis -- meaning all coordinates are nonzero.

This is the geometric reason $\ell_1$ minimization (basis pursuit) finds sparse solutions while $\ell_2$ minimization (least squares) does not. The corners of the $\ell_1$ ball act as attractors for sparsity.

---

### Problem 7: Spectral Sparsity and Rank Control

**Problem:** The thesis Sec. 2.3.3 argues spectral sparsity is a "structural remedy for representation degeneration." If a set of tokens all have exactly $s$ active modes, what is the maximum possible rank of the token embedding matrix?

**Solution:**

Each token is a vector in $\mathbb{R}^d$ with at most $s$ nonzero entries in the spectral (Fourier) basis. The token embedding matrix $X$ has shape $n \times d$ ($n$ tokens, $d$ dimensions).

In the spectral basis, each row of $X$ has at most $s$ nonzero entries. Therefore, all rows lie in the subspace spanned by at most $s$ basis vectors (the $s$ frequency modes that are used). The rank of $X$ is at most $s$.

**But crucially, rank is EXACTLY $s$ if the tokens choose different patterns of active modes.** If token 1 activates modes $\{1, 3, 7\}$ and token 2 activates modes $\{2, 5, 9\}$, their spectral representations span a 6-dimensional subspace. With enough tokens activating all $s$ modes, the rank reaches exactly $s$.

**Comparison to dense embeddings:**

Unlike dense embeddings (where rank can silently collapse to much less than $d$ due to training dynamics -- the "representation degeneration" problem), spectral sparsity gives **explicit rank control**:

$$\operatorname{rank}(X) \in [1, s] \quad \text{by construction}$$

Set $s = 10$ and you get at most 10 effective dimensions -- no more, no less. This prevents both:
- **Rank collapse** (rank $\ll s$): mitigated because each active mode contributes an independent degree of freedom
- **Rank explosion** (rank $\gg s$): impossible because only $s$ modes are active

This is a structural guarantee, not a soft regularization penalty that might or might not work during training.

---

### Problem 8: The Restricted Isometry Property

**Problem:** The RIP with constant $\delta_s$ means: for any $s$-sparse vector $x$, $(1 - \delta_s) \|x\|^2 \leq \|\Phi x\|^2 \leq (1 + \delta_s) \|x\|^2$. Explain intuitively what this means: "the measurement matrix $\Phi$ approximately preserves distances between sparse signals."

**Solution:**

For two $s$-sparse signals $x$ and $y$, their difference $x - y$ is at most $2s$-sparse (in the worst case, $x$ and $y$ have completely disjoint supports). If $\Phi$ satisfies the RIP of order $2s$ with constant $\delta_{2s} < 1$, then:

$$(1 - \delta_{2s}) \|x - y\|^2 \leq \|\Phi(x - y)\|^2 \leq (1 + \delta_{2s}) \|x - y\|^2$$

Since $\Phi$ is linear, $\Phi(x - y) = \Phi x - \Phi y$, so:

$$(1 - \delta_{2s}) \|x - y\|^2 \leq \|\Phi x - \Phi y\|^2 \leq (1 + \delta_{2s}) \|x - y\|^2$$

This means: **$\|\Phi x - \Phi y\|$ is approximately equal to $\|x - y\|$.**

Distances between sparse signals are approximately preserved after measurement. Two consequences:

1. **No collisions:** Two distinct sparse signals $x \neq y$ map to distinct measurements $\Phi x \neq \Phi y$ (because $\|\Phi x - \Phi y\| > 0$ whenever $\|x - y\| > 0$). So recovery is possible -- no information is lost.

2. **Stability:** Small perturbations in the signal produce small perturbations in the measurements, and vice versa. Recovery is robust to noise.

The smaller $\delta_{2s}$ is, the better the preservation. Random Gaussian matrices satisfy the RIP with high probability when $m = O(s \cdot \log(d/s))$ -- this is the fundamental theorem of compressed sensing. It says that $O(s \log(d/s))$ random projections suffice to preserve all pairwise distances between $s$-sparse signals in $\mathbb{R}^d$.

---

### Problem 9: The Forward-Reverse Loop and Re-Sparsification

**Problem:** In the SGST, the forward-reverse loop goes: sparse spectral -> IFFT -> dense spatial -> MLP -> FFT -> top-$k$ -> sparse spectral. The top-$k$ step is "re-sparsification." What information is lost in this step? When is this loss acceptable?

**Solution:**

The top-$k$ operator keeps the $k$ largest magnitude frequency components and zeros out the remaining $d - k$. The lost information resides in the small spectral coefficients that are discarded.

**When loss is acceptable:**

1. **The signal is approximately sparse.** If most energy is concentrated in a few modes, the small coefficients carry negligible information. Zeroing them changes the signal by a small amount (bounded by the sum of squared discarded coefficients).

2. **The MLP introduces noise in non-essential frequencies.** The nonlinear MLP operates in the spatial domain, where it processes all positions. When transformed back to the frequency domain, some of its output may land in frequencies that are not structurally important. Top-$k$ zeroing acts as denoising, removing these spurious spectral components.

3. **The sparsity pattern carries structural information.** WHICH modes are active encodes what "type" of token this is. Re-sparsification reinforces this structural identity by keeping the dominant modes and suppressing drift into inactive modes.

**When loss is NOT acceptable:**

If the small coefficients carry subtle but important information -- for example, fine-grained distinctions between similar tokens that are encoded in low-energy modes -- then top-$k$ discards discriminative features.

**Empirical evidence from the thesis:** Training shows that sparsity emerges naturally (up to 60% of modes go to zero without any sparsity loss). This suggests the model learns representations where the small coefficients genuinely are unimportant, validating the top-$k$ approach.

---

### Problem 10: Emergent Spectral Sparsity

**Problem:** The thesis's V12 experiments show spectral sparsity emerging without being explicitly optimized -- it goes from 22.9% to 60.2% over training. Explain why this is significant evidence for the "spectral sparsity hypothesis."

**Solution:**

**Why emergence matters more than enforcement:**

If you force sparsity via an explicit loss term (e.g., adding $\lambda \|\hat{x}\|_1$ to the training objective), the model sparsifies because you told it to -- not necessarily because sparsity is inherently useful for the task. The model might achieve better performance without sparsity but is forced into a suboptimal regime by the regularizer.

V12 has NO explicit sparsity loss. The only training objective is cross-entropy for next-token prediction. The fact that sparsity emerges spontaneously means the model DISCOVERS that spectral sparsity is useful for language modeling on its own.

**Why the model learns to be spectrally sparse:**

The model concentrates its spectral energy into fewer modes because:

1. **Fewer modes = more efficient transport.** Operations on $s$ active modes cost $O(s)$ instead of $O(d)$. The model implicitly optimizes for computational efficiency during training by not wasting capacity on modes that do not contribute to prediction.

2. **Sparsity prevents rank collapse.** With explicit rank control (rank $= s$), the model avoids the representation degeneration problem where dense embeddings silently lose effective dimensionality.

3. **The uncertainty principle guarantees long-range interaction.** Each active mode, by the Donoho-Stark principle, has global spatial extent. The model does not need many modes to achieve long-range information flow -- a few spectrally sparse modes suffice.

**Quantitative significance:** The 60.2% sparsity rate achieved is 88% of the theoretical ceiling of 68.75% (set by the architecture's top-$k$ parameter). The model nearly maximizes sparsity on its own, leaving only a thin margin where active modes are needed for prediction quality. This strongly supports the hypothesis that spectral sparsity is not merely compatible with language modeling -- it is the natural representation the model converges toward.

---

## Comprehension Questions

1. State the Donoho-Stark uncertainty principle. What are its implications for choosing WHERE to be sparse (spatial vs spectral)?

2. Why is spatial sparsity the wrong choice for language representations? Use the uncertainty principle to explain the consequences for transport cost and interaction range.

3. What is the Restricted Isometry Property, and why does it guarantee that sparse signals can be recovered from few measurements?

4. How does soft-thresholding differ from hard thresholding (top-$k$)? Which is preferred for gradient-based optimization and why? (Hint: consider the subdifferential of the $\ell_1$ norm versus the discontinuity of the $\ell_0$ "norm.")

5. The thesis reports that spectral sparsity emerges spontaneously during training (22.9% -> 60.2%) without an explicit sparsity loss. Why is this more convincing evidence for the spectral sparsity hypothesis than achieving the same sparsity via an $\ell_1$ penalty?

---

## Bridge to Thesis

This unit provides one of the two theoretical pillars of the SGST architecture (the other being differential geometry from Units 05-07).

The Donoho-Stark uncertainty principle is the foundational insight: you must choose WHERE to be sparse, and the choice has architectural consequences. The thesis (Sec. 5.1) argues that spectral sparsity is the correct choice for language because it simultaneously provides:
- Long-range interaction (spatial extension from the uncertainty principle)
- Cheap transport ($O(s)$ operations on few active modes)
- Rank control (rank $= s$ by construction)

Compressed sensing theory (RIP, basis pursuit) provides the mathematical guarantee that spectrally sparse signals can be faithfully represented and recovered from low-dimensional projections. Soft-thresholding provides the differentiable mechanism for enforcing sparsity during training.

The V12 experiments provide the empirical validation: spectral sparsity is not just theoretically motivated but emerges naturally when you give the model the right architectural scaffold. The model does not need to be told to be sparse -- it discovers sparsity as the optimal strategy for next-token prediction.

**Previous unit:** Unit 07 covered Finsler geometry -- the directional/asymmetric structure that spectral methods encode via the gauge connection.

**Next unit:** Unit 09 will cover fiber bundles and gauge theory -- the mathematical framework that unifies the geometric and spectral perspectives into the SGST's transport architecture.
