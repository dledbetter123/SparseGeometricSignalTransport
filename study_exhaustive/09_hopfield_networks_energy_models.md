# Unit 09: Hopfield Networks, Associative Memory, and the Attention Connection

## Learning Objectives

1. Understand classical Hopfield networks and their storage capacity
2. Derive the modern continuous Hopfield energy and its update rule
3. Prove that attention IS a one-step Hopfield update (Ramsauer et al. 2021)
4. Understand exponential storage capacity of modern Hopfield networks
5. Know sparse Hopfield networks and Fenchel-Young losses
6. Connect Hopfield energy to the SGST's Langevin settling phase

## Prerequisites

Units 01-02 (Linear Algebra Foundations, Multivariable Calculus and Optimization)

## Readings

- Ramsauer et al. 2021, "Hopfield Networks is All You Need" -- `papers/ramsauer2021_hopfield_is_all_you_need.pdf`
- Thesis Sec. 2.6 (Energy-Based Models and Hopfield Networks), all subsections
- Thesis Sec. 5.3.5 (Langevin-Hopfield Settling)
- `papers/santos2024_hopfield_fenchel_young.pdf` (sparse Hopfield via Fenchel-Young losses)
- `papers/santos2024_sparse_structured_hopfield.pdf`
- `papers/2024_optimal_hopfield_capacity_neurips.pdf`
- `papers/li2025_dynamic_manifold_hopfield.pdf`
- `papers/2025_hopfield_hidden_states_transformer.pdf`
- `topology/topological_computation.md` (Langevin dynamics section)

---

## Key Concepts

### 1. Classical Hopfield Network

A binary neural network with symmetric weights and no self-connections. Neurons take values $s_i \in \{+1, -1\}$. The network has energy:

$$E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j$$

Patterns are stored as energy minima. The update rule is asynchronous:

$$s_i \leftarrow \operatorname{sign}\left(\sum_j w_{ij} s_j\right)$$

This always decreases or preserves the energy (Lyapunov function), so the network converges to a fixed point.

### 2. Classical Storage Capacity

For $N$ neurons, the Hebbian learning rule $w_{ij} = \frac{1}{N} \sum_\mu p_i^\mu p_j^\mu$ can reliably store at most approximately $0.14N$ patterns. Beyond this threshold, pattern interference causes retrieval errors. This is a severe bottleneck -- 100 neurons store only ~14 patterns.

### 3. Modern Continuous Hopfield Energy

The continuous Hopfield energy (Ramsauer et al. 2021) is:

$$E(\xi) = -\frac{1}{\beta} \log\left( \sum_\mu \exp(\beta \, x^\mu \cdot \xi) \right) + \frac{1}{2}\|\xi\|^2$$

where $\xi$ is the state (query), $\{x^\mu\}$ are stored patterns, and $\beta$ is the inverse temperature. The first term is a soft minimum over pattern similarities; the second is a regularizer that prevents the state from diverging.

### 4. The Update Rule

Taking the gradient of $E$ and setting $\xi_{\text{new}} = \xi - \nabla E$:

$$\nabla_\xi E = -X \operatorname{softmax}(\beta X^T \xi) + \xi$$

$$\xi_{\text{new}} = \xi - \nabla E = X \operatorname{softmax}(\beta X^T \xi)$$

The new state is a softmax-weighted combination of stored patterns, where the weights depend on similarity to the current state.

### 5. Attention IS Hopfield Retrieval

The Hopfield update $\xi_{\text{new}} = X \operatorname{softmax}(\beta X^T \xi)$ is exactly the attention mechanism:
- Query $Q = \xi$ (the state to be updated)
- Keys $K = X$ (the stored patterns, used for similarity)
- Values $V = X$ (the stored patterns, used for retrieval)
- Temperature $\beta = \frac{1}{\sqrt{d_k}}$
- $\operatorname{softmax}(QK^T / \sqrt{d_k}) \, V = X \operatorname{softmax}(X^T \xi / \sqrt{d_k})$

One attention computation = one step of Hopfield energy minimization.

### 6. Exponential Storage Capacity

Modern continuous Hopfield networks store $P = O(\exp(d))$ patterns in $d$ dimensions -- exponential in the embedding dimension. This is because the exponential interaction function $\exp(\beta \, x \cdot \xi)$ creates exponentially narrow basins of attraction compared to the polynomial interactions in classical networks. For practical purposes, storage capacity is essentially unlimited; the bottleneck is the $O(n^2)$ computation of attention, not memory capacity.

### 7. Temperature and Sharpness

The inverse temperature $\beta$ controls retrieval behavior:
- $\beta \to \infty$: softmax becomes hard max. Retrieval returns a single stored pattern (nearest neighbor). Sharp, exact retrieval.
- $\beta \to 0$: softmax becomes uniform. Retrieval returns the average of all patterns. No discrimination.
- Intermediate $\beta$: soft retrieval, blending nearby patterns. Standard attention operates here.

### 8. Fenchel-Young Losses and Generalized Attention

The softmax function is the gradient of the log-sum-exp, which is the convex conjugate of the negative entropy. Fenchel-Young losses generalize this by using different convex functions:
- Softmax ($\alpha = 1$): dense attention, all weights nonzero
- Sparsemax ($\alpha = 2$): projects onto the simplex, producing exact zeros
- $\alpha$-entmax ($1 < \alpha < 2$): interpolates between dense and sparse

These yield sparse Hopfield networks where retrieval activates only a few stored patterns.

### 9. Sparse Hopfield Networks

Replacing softmax with entmax in the Hopfield update gives sparse retrieval: only patterns sufficiently similar to the query receive nonzero weight. Benefits include:
- Computational efficiency: $O(\text{nnz})$ instead of $O(n)$ for the weighted sum
- Noise reduction: irrelevant patterns contribute exactly zero
- Interpretability: the set of active patterns is explicit
- Sharper retrieval: closer to exact pattern recovery

### 10. Langevin-Hopfield Settling

Instead of a single update step, the SGST uses iterative Langevin dynamics on the Hopfield energy:

$$\xi_{t+1} = \xi_t - \eta \nabla E(\xi_t) + \sqrt{\frac{2\eta}{\beta_t}} \, \varepsilon_t$$

where $\varepsilon_t$ is Gaussian noise and $\beta_t$ increases over time (annealing). High temperature (early steps) allows exploration and escape from spurious minima. Low temperature (late steps) sharpens basins and settles into genuine memories. This is simulated annealing applied to associative memory retrieval.

---

## Worked Problems

### Problem 1: Classical Hopfield Fixed Point Verification

**Problem:** A classical Hopfield network with $N=4$ neurons stores pattern $p = [+1, -1, +1, -1]$. The weight matrix is $W = pp^T / N$ (with zero diagonal). Compute $W$ and verify that $p$ is a fixed point of the update rule $s_i \to \operatorname{sign}(\sum_j W_{ij} s_j)$.

**Solution:**

First, compute $pp^T$:

$$pp^T = \begin{bmatrix}+1\\-1\\+1\\-1\end{bmatrix} \begin{bmatrix}+1 & -1 & +1 & -1\end{bmatrix} = \begin{bmatrix} 1 & -1 & 1 & -1\\ -1 & 1 & -1 & 1\\ 1 & -1 & 1 & -1\\ -1 & 1 & -1 & 1 \end{bmatrix}$$

Divide by $N=4$ and zero the diagonal:

$$W = \frac{1}{4} \begin{bmatrix} 0 & -1 & 1 & -1\\ -1 & 0 & -1 & 1\\ 1 & -1 & 0 & -1\\ -1 & 1 & -1 & 0 \end{bmatrix}$$

Now verify the update rule with $s = p = [+1, -1, +1, -1]$:

For $s_1$: $h_1 = W_{12} s_2 + W_{13} s_3 + W_{14} s_4 = (-\frac{1}{4})(-1) + (\frac{1}{4})(1) + (-\frac{1}{4})(-1) = \frac{1}{4} + \frac{1}{4} + \frac{1}{4} = \frac{3}{4}$
$\operatorname{sign}(\frac{3}{4}) = +1 = p_1$ ✓

For $s_2$: $h_2 = W_{21} s_1 + W_{23} s_3 + W_{24} s_4 = (-\frac{1}{4})(1) + (-\frac{1}{4})(1) + (\frac{1}{4})(-1) = -\frac{1}{4} - \frac{1}{4} - \frac{1}{4} = -\frac{3}{4}$
$\operatorname{sign}(-\frac{3}{4}) = -1 = p_2$ ✓

For $s_3$: $h_3 = W_{31} s_1 + W_{32} s_2 + W_{34} s_4 = (\frac{1}{4})(1) + (-\frac{1}{4})(-1) + (-\frac{1}{4})(-1) = \frac{1}{4} + \frac{1}{4} + \frac{1}{4} = \frac{3}{4}$
$\operatorname{sign}(\frac{3}{4}) = +1 = p_3$ ✓

For $s_4$: $h_4 = W_{41} s_1 + W_{42} s_2 + W_{43} s_3 = (-\frac{1}{4})(1) + (\frac{1}{4})(-1) + (-\frac{1}{4})(1) = -\frac{1}{4} - \frac{1}{4} - \frac{1}{4} = -\frac{3}{4}$
$\operatorname{sign}(-\frac{3}{4}) = -1 = p_4$ ✓

All components match. The pattern $p$ is a fixed point (energy minimum) of the network.

---

### Problem 2: Modern Hopfield Energy Computation

**Problem:** For the modern Hopfield energy $E(\xi) = -\frac{1}{\beta} \log\bigl(\sum_\mu \exp(\beta \, x^\mu \cdot \xi)\bigr) + \frac{1}{2}\|\xi\|^2$ with stored patterns $X = \{x^1, x^2\}$ where $x^1 = [1,0]$, $x^2 = [0,1]$, compute $E$ at $\xi = [1,0]$ and $\xi = [0.5, 0.5]$ for $\beta = 1$.

**Solution:**

At $\xi = [1, 0]$:

$$x^1 \cdot \xi = 1 \cdot 1 + 0 \cdot 0 = 1$$
$$x^2 \cdot \xi = 0 \cdot 1 + 1 \cdot 0 = 0$$
$$\sum = \exp(1) + \exp(0) = e + 1 = 2.718 + 1 = 3.718$$
$$E = -\log(3.718) + \frac{1}{2}(1^2 + 0^2) = -1.313 + 0.5 = -0.813$$

At $\xi = [0.5, 0.5]$:

$$x^1 \cdot \xi = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$$
$$x^2 \cdot \xi = 0 \cdot 0.5 + 1 \cdot 0.5 = 0.5$$
$$\sum = \exp(0.5) + \exp(0.5) = 2 \times 1.649 = 3.297$$
$$E = -\log(3.297) + \frac{1}{2}(0.25 + 0.25) = -1.193 + 0.25 = -0.943$$

Comparing: $E([1,0]) = -0.813$, $E([0.5,0.5]) = -0.943$.

Since $-0.943 < -0.813$, the midpoint has LOWER energy. This is correct for low $\beta$: the log-sum-exp term is large when BOTH patterns are equally activated, and the regularizer $\frac{1}{2}\|\xi\|^2$ is smaller at the midpoint. At low temperature ($\beta$ close to 1), the minimum lies between patterns. At high $\beta$, the landscape sharpens and the minima separate to near each stored pattern individually.

---

### Problem 3: Deriving the Hopfield Update Rule

**Problem:** Derive the Hopfield update rule by computing $\nabla_\xi E$ and setting $\xi_{\text{new}} = \xi - \nabla E$ (one gradient step from the identity).

**Solution:**

Starting from the energy:

$$E(\xi) = -\frac{1}{\beta} \log\left( \sum_\mu \exp(\beta \, x^\mu \cdot \xi) \right) + \frac{1}{2}\|\xi\|^2$$

Compute the gradient term by term.

For the log-sum-exp term, let $Z = \sum_\mu \exp(\beta \, x^\mu \cdot \xi)$:

$$\nabla_\xi \left[-\frac{1}{\beta} \log Z\right] = -\frac{1}{\beta} \cdot \frac{1}{Z} \cdot \nabla_\xi Z$$

$$\nabla_\xi Z = \sum_\mu \beta \, x^\mu \exp(\beta \, x^\mu \cdot \xi)$$

Therefore:

$$\nabla_\xi \left[-\frac{1}{\beta} \log Z\right] = -\frac{1}{\beta} \cdot \frac{1}{Z} \cdot \sum_\mu \beta \, x^\mu \exp(\beta \, x^\mu \cdot \xi) = -\sum_\mu x^\mu \cdot \frac{\exp(\beta \, x^\mu \cdot \xi)}{Z} = -X \operatorname{softmax}(\beta X^T \xi)$$

For the regularizer:

$$\nabla_\xi \left[\frac{1}{2}\|\xi\|^2\right] = \xi$$

Combining:

$$\nabla_\xi E = -X \operatorname{softmax}(\beta X^T \xi) + \xi$$

The update rule $\xi_{\text{new}} = \xi - \nabla E$ gives:

$$\xi_{\text{new}} = \xi - \bigl(-X \operatorname{softmax}(\beta X^T \xi) + \xi\bigr) = X \operatorname{softmax}(\beta X^T \xi)$$

This IS the attention mechanism: the query $\xi$ attends to keys $X^T$ with inverse temperature $\beta$, obtains softmax weights, and retrieves a weighted combination of values $X$.

---

### Problem 4: Explicit Attention-Hopfield Equivalence

**Problem:** Show the equivalence explicitly: standard attention $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}(QK^T / \sqrt{d}) \, V$ equals one Hopfield update step when $Q = \xi$ (single query), $K = V = X$ (patterns), and $\beta = 1/\sqrt{d}$.

**Solution:**

Start from the Hopfield update for a single query $\xi$ (a row vector in $\mathbb{R}^{1 \times d}$):

$$\xi_{\text{new}} = \operatorname{softmax}(\beta \, \xi X^T) \, X$$

where $X$ is the $N \times d$ matrix of stored patterns (each row is a pattern), and the softmax is over the $N$ pattern indices.

Substituting $\beta = 1/\sqrt{d}$:

$$\xi_{\text{new}} = \operatorname{softmax}(\xi X^T / \sqrt{d}) \, X$$

Now write standard attention for a single query $q = \xi$:

$$\operatorname{Attention}(q, K, V) = \operatorname{softmax}(q K^T / \sqrt{d_k}) \, V$$

Setting $K = X$ and $V = X$ and $d_k = d$:

$$\operatorname{Attention}(\xi, X, X) = \operatorname{softmax}(\xi X^T / \sqrt{d}) \, X$$

This is identical to the Hopfield update. Term-by-term correspondence:

| Attention | Hopfield |
|-----------|----------|
| Query $q$ | State $\xi$ |
| Keys $K$ | Stored patterns $X$ (similarity computation) |
| Values $V$ | Stored patterns $X$ (retrieval targets) |
| $1/\sqrt{d_k}$ | Inverse temperature $\beta$ |
| softmax weights | Boltzmann distribution over patterns |
| Output | Updated state after one energy descent step |

QED: one attention computation is exactly one step of Hopfield energy minimization. Multi-head attention corresponds to multiple Hopfield networks with different pattern projections running in parallel. Stacking transformer layers corresponds to iterating the Hopfield update (multiple steps of energy descent).

---

### Problem 5: Exponential vs. Linear Storage Capacity

**Problem:** The classical Hopfield network stores at most approximately $0.14N$ patterns. The modern continuous Hopfield stores approximately $\exp(d)$ patterns. Explain WHERE this exponential capacity comes from (hint: the softmax/log-sum-exp energy).

**Solution:**

**Classical Hopfield (linear capacity):**

The sign activation function is binary -- it partitions the input space into two half-spaces per neuron. With $N$ neurons, the network can distinguish at most $2^N$ states total, but the Hebbian weight matrix creates interference between stored patterns. Each pattern's basin of attraction has width proportional to $\sqrt{N}$ in Hamming distance. When more than $\sim 0.14N$ patterns are stored, basins overlap and retrieval fails. The storage capacity is fundamentally limited by the linear (polynomial) nature of the energy: $E = -s^T W s$ is quadratic in $s$.

**Modern Hopfield (exponential capacity):**

The exponential interaction $\exp(\beta \, x \cdot \xi)$ changes everything. Consider the energy landscape:

$$E(\xi) = -\frac{1}{\beta} \log \sum_\mu \exp(\beta \, x^\mu \cdot \xi) + \frac{1}{2}\|\xi\|^2$$

When $\beta$ is large, the log-sum-exp is dominated by the largest term: the pattern with maximum similarity to $\xi$. The energy surface near each pattern looks like:

$$E(\xi) \approx -x^{\mu*} \cdot \xi + \frac{1}{2}\|\xi\|^2$$

which has a basin of attraction centered at $x^{\mu*}$.

The key: the exponential function $\exp(\beta \, x \cdot \xi)$ creates basins whose width shrinks exponentially as $\beta$ increases. In $d$-dimensional space, you can pack approximately $\exp(c \cdot d)$ well-separated directions (this is a sphere packing / Johnson-Lindenstrauss argument). Each direction can host a distinct pattern, and the exponentially narrow basins prevent interference.

Formally (Ramsauer et al. Theorem 3): if the patterns are drawn from a sphere of radius $M$, then the network can store and retrieve

$$P = O\!\left(\exp\!\left(\frac{M^2}{2}\right)\right)$$

patterns with exponentially small retrieval error. Since $M$ can grow with $d$, this gives exponential-in-dimension capacity.

**Practical implication:** Modern Hopfield (= attention) has essentially unlimited storage capacity. The true bottleneck is the $O(n^2)$ computation required to evaluate the softmax over all stored patterns, not the number of patterns the network can faithfully store.

---

### Problem 6: Sparse Hopfield and Language Modeling

**Problem:** Sparse Hopfield networks replace softmax with $\alpha$-entmax (Fenchel-Young loss). For $\alpha = 1.5$, entmax produces sparse attention weights (exactly zero for most patterns). Why is sparse retrieval useful for language?

**Solution:**

Standard softmax attention assigns nonzero weight to ALL stored patterns (tokens in context), even completely irrelevant ones. For a context window of 100K tokens, every query retrieves a weighted blend of all 100K memories. Most of these weights are tiny but collectively they form a noise floor that dilutes the signal from truly relevant memories.

Sparse entmax ($\alpha = 1.5$) zeros out all patterns whose similarity falls below a data-dependent threshold. Only the genuinely relevant memories receive nonzero weight.

**Benefits for language modeling:**

1. **Computational efficiency:** Sparse attention is $O(\text{nnz})$ for the weighted retrieval step instead of $O(n)$. If only 50 out of 100K tokens are relevant, the savings are 2000x.

2. **Signal quality:** No noise from irrelevant memories polluting the retrieved representation. When predicting the next word after "The capital of France is", you want strong weight on "France" and "capital" and zero weight on filler words.

3. **Interpretability:** You can inspect exactly which memories contributed to each prediction. The support set of nonzero weights is explicit and small.

4. **Sharper retrieval:** Closer to exact pattern recovery (the hard-max limit). In language, this means more decisive predictions rather than hedge-everything-softly behavior.

5. **Robustness to context length:** As context grows, softmax weights get diluted ($1/n$ effect). Sparse weights maintain their magnitude regardless of how many irrelevant tokens are added.

The thesis discusses sparse attention in Sec. 2.6.2, connecting it to the broader theme of sparsity as a guiding principle in SGST: the correct representation uses few active components (spectral modes or memory patterns), and sparsity-inducing mechanisms like entmax enforce this.

---

### Problem 7: Langevin Settling and Temperature Annealing

**Problem:** The Langevin-Hopfield settling in SGST runs $T$ steps of: $\xi_{t+1} = \xi_t - \eta \nabla E(\xi_t) + \sqrt{2\eta / \beta_t} \, \varepsilon_t$, with $\beta_t$ increasing (temperature decreasing). Explain why decreasing temperature helps avoid spurious memories.

**Solution:**

The Hopfield energy landscape contains two types of minima:

- **Genuine minima:** deep basins centered at (or very near) stored patterns. These are the memories we want to retrieve.
- **Spurious minima:** shallow basins created by interference between stored patterns. These are artifacts -- weighted combinations of multiple patterns that happen to be local minima.

**High temperature phase (early steps, small $\beta_t$):**

The energy landscape $E(\xi) = -\frac{1}{\beta} \log \sum \exp(\beta \, x^\mu \cdot \xi) + \frac{1}{2}\|\xi\|^2$ is smooth when $\beta$ is small. The log-sum-exp approaches a simple average, creating a single broad basin. The noise term $\sqrt{2\eta / \beta_t}$ is large, enabling the state to:
- Explore a wide region of state space
- Escape shallow minima easily (thermal fluctuations exceed the barrier height)
- Move toward the general region of the closest genuine pattern

Spurious minima are particularly shallow -- they exist because of interference, not because a stored pattern anchors them. High temperature washes them out entirely.

**Low temperature phase (late steps, large $\beta_t$):**

As $\beta_t$ increases:
- The energy landscape sharpens. Genuine minima become deep, narrow wells.
- Spurious minima become even shallower relative to genuine ones (the exponential interaction amplifies the deepest wells disproportionately).
- The noise term decreases, and the state performs near-deterministic gradient descent into the nearest deep well.

**The annealing schedule bridges these regimes:**

Early steps: explore widely, identify the correct basin.
Late steps: descend precisely into the basin minimum.

This is exactly simulated annealing applied to associative memory. The SGST uses this instead of one-shot attention (a single Hopfield update) to handle ambiguous tokens. When a token could plausibly match multiple stored patterns (ambiguous context), a single softmax step returns a blurry blend. Iterative annealed settling progressively resolves the ambiguity, first ruling out clearly wrong patterns (high temperature), then discriminating among close candidates (low temperature).

---

### Problem 8: Numerical Hopfield Update

**Problem:** Compute one step of the Hopfield update for query $\xi = [1, 0.5]$ with stored patterns $X = [[1,0], [0,1], [-1,0]]$ and $\beta = 2$.

**Solution:**

Step 1: Compute dot products $\beta \cdot X^T \xi$.

$$x^1 \cdot \xi = 1 \cdot 1 + 0 \cdot 0.5 = 1.0$$
$$x^2 \cdot \xi = 0 \cdot 1 + 1 \cdot 0.5 = 0.5$$
$$x^3 \cdot \xi = -1 \cdot 1 + 0 \cdot 0.5 = -1.0$$

$$\beta \cdot [1.0, 0.5, -1.0] = [2.0, 1.0, -2.0]$$

Step 2: Compute $\operatorname{softmax}([2.0, 1.0, -2.0])$.

$$\exp(2.0) = 7.389, \quad \exp(1.0) = 2.718, \quad \exp(-2.0) = 0.135$$

$$\text{sum} = 7.389 + 2.718 + 0.135 = 10.242$$

$$\operatorname{softmax} = [7.389/10.242, \; 2.718/10.242, \; 0.135/10.242] = [0.721, \; 0.265, \; 0.013]$$

Step 3: Compute weighted combination $\xi_{\text{new}} = \sum_\mu w_\mu x^\mu$.

$$\xi_{\text{new}} = 0.721 \cdot [1, 0] + 0.265 \cdot [0, 1] + 0.013 \cdot [-1, 0]$$
$$= [0.721, 0] + [0, 0.265] + [-0.013, 0]$$
$$= [0.721 - 0.013, \; 0 + 0.265]$$
$$= [0.708, \; 0.265]$$

**Interpretation:** The update moved $\xi$ toward pattern $x^1 = [1,0]$ (highest similarity, weight 0.721) and slightly toward $x^2 = [0,1]$ (moderate similarity, weight 0.265). Pattern $x^3 = [-1,0]$ (opposite direction to query) received near-zero weight (0.013).

After several more iterations, $\xi$ would converge to approximately $x^1 = [1,0]$, as $x^1$ dominates the retrieval. With higher $\beta$, convergence would be even faster (sharper softmax).

---

### Problem 9: From Hopfield Settling to Parseval Filtering

**Problem:** The thesis says the SGST's Hopfield settler was replaced in later versions (V15+) by a "Parseval spectral filter." Why was the Hopfield settler insufficient?

**Solution:**

From the V14 diagnosis and ablation studies, the Hopfield settler had several limitations:

1. **Iterative cost:** Langevin settling requires $T_{\text{settle}} = 2\text{-}3$ gradient steps per layer. Each step involves computing the full energy gradient (softmax over the memory bank). This multiplies the per-layer cost by $T_{\text{settle}}$, adding significant latency during both training and inference.

2. **Fixed memory bank:** The memory bank $\{x^\mu\}$ in each block is a fixed set of learned parameters. It cannot adapt dynamically to novel input combinations not seen during training. Attention, by contrast, computes pairwise similarities between ALL input tokens -- every token is simultaneously a query, key, and value. This is strictly more expressive than similarity against a fixed codebook.

3. **Limited relational capacity:** Hopfield retrieval computes similarity between a query and each stored pattern independently. It does not capture relationships BETWEEN patterns (token-token interactions). Standard attention's $O(n^2)$ pairwise computation captures these inter-token relationships. The Hopfield settler's $O(n \cdot P)$ computation ($n$ queries against $P$ patterns) misses this.

4. **Performance plateau:** The V13/V14 experiments showed cross-entropy stuck at approximately 2.17 on WikiText-103, unable to match attention baselines. The geometric machinery (gauge connections, Langevin dynamics) was expressive in theory but did not translate to better performance at the scales tested.

**The Parseval filter replacement (V15+):**

Instead of iterative settling, the Parseval spectral filter applies a single-step gating operation:

$$h_{\text{out}} = W_{\text{spectral}} \cdot h$$

where $W_{\text{spectral}}$ has eigenvalues bounded by 1 in magnitude (Parseval frame condition). This provides:
- Content-dependent energy redistribution without iteration (single matrix multiply)
- Guaranteed stability (no eigenvalue explosion)
- Spectral interpretation: each frequency mode is independently gated
- Compatible with FFT-based acceleration

The shift from iterative Hopfield settling to single-step spectral gating was a key architectural decision in SGST's evolution.

---

### Problem 10: Unification Results from Ramsauer et al. 2021

**Problem:** The paper "Hopfield Networks is All You Need" (Ramsauer 2021) unified multiple concepts. List and explain three things that are equivalent to Hopfield retrieval according to this paper.

**Solution:**

Ramsauer et al. 2021 demonstrated that several seemingly distinct mechanisms are all instances of the same mathematical operation -- energy minimization in a continuous Hopfield network:

1. **Attention mechanism:** The transformer's scaled dot-product attention $\operatorname{softmax}(QK^T/\sqrt{d}) \, V$ is exactly one step of Hopfield energy minimization. Each query is a state, keys/values are stored patterns, and the softmax weights are the Boltzmann distribution over pattern similarities. Multi-head attention corresponds to multiple independent Hopfield networks with different pattern projections.

2. **Dense associative memory (Krotov-Hopfield 2016):** The continuous energy framework generalizes the dense associative memories introduced by Krotov and Hopfield, which used polynomial interaction functions $F(x \cdot \xi) = (x \cdot \xi)^n$. The exponential interaction $F(x \cdot \xi) = \exp(x \cdot \xi)$ is a special case that yields softmax retrieval and achieves the best (exponential) storage capacity. The entire family of interaction functions corresponds to different attention kernels.

3. **BERT's masked language modeling:** The [MASK] token in BERT acts as the query state $\xi$. The contextual representations of surrounding tokens act as stored patterns $X$. BERT's prediction of the masked word is a Hopfield retrieval: the [MASK] embedding descends the energy landscape to retrieve the pattern (word) most compatible with the context. The classification head on top of [MASK] refines this retrieval into a vocabulary distribution.

Additionally, the paper connects to:

4. **Transformer layers as iterated settling:** Each attention layer performs one Hopfield update. A deep transformer with $L$ layers performs $L$ steps of energy minimization, progressively refining the representation toward deeper energy minima. This explains why deeper transformers produce better representations -- more iterations of the retrieval dynamics.

5. **Kernel methods and similarity functions:** The attention kernel $K(x, y) = \exp(x \cdot y / \sqrt{d})$ is the retrieval kernel of the Hopfield network. Different kernels (polynomial, RBF, etc.) correspond to different Hopfield interaction functions, each with different storage capacity and retrieval properties. This connects the attention literature to the extensive kernel methods literature.

---

## Comprehension Questions

1. Write the modern Hopfield energy from memory. Then derive the update rule by computing the gradient and performing one step of gradient descent.

2. Prove explicitly that one step of the Hopfield update is equivalent to one attention computation. State the exact correspondence between $Q$, $K$, $V$, $\beta$ and the Hopfield quantities.

3. The classical Hopfield network stores $O(N)$ patterns while the modern continuous version stores $O(\exp(d))$ patterns. What is the mathematical source of this exponential improvement? Why does the exponential interaction function create narrower basins of attraction?

4. What is the role of the inverse temperature $\beta$ in Hopfield retrieval? Describe the behavior in the limits $\beta \to \infty$ and $\beta \to 0$, and explain why intermediate values are used in practice.

5. Read the Ramsauer et al. 2021 paper (at minimum the introduction and main theorem). Summarize the key equivalence between Hopfield networks and attention in your own words, and explain why this equivalence matters for understanding transformers.

---

## Bridge to Thesis

The Hopfield-attention equivalence is foundational for understanding the SGST architecture:

- **V14's Langevin-Hopfield settling** (Sec. 5.3.5) used iterative energy descent as a replacement for standard attention. The settling dynamics $\xi_{t+1} = \xi_t - \eta \nabla E + \text{noise}$ aimed to resolve ambiguous tokens through annealing rather than one-shot softmax.

- **The V14 ablation** revealed that this approach, while theoretically elegant, did not outperform standard attention at practical scales. The geometric interpretation (energy basins as attractor manifolds) was correct but the computational overhead of iterative settling was too high.

- **The Parseval spectral filter** (V15+) replaced iterative settling with single-step spectral gating, preserving the energy interpretation (eigenvalue bounds = energy bounds) while eliminating iteration.

- **Sparse Hopfield** connects directly to the SGST's sparsity principle: just as entmax produces sparse attention weights (few active memories), the SGST's spectral sparsity activates few frequency modes. Both enforce the same inductive bias -- representations should be sparse -- but in different domains (memory space vs. frequency space).

- **The synthesis direction** (holonomic linear attention with delta rule) can be understood through the Hopfield lens: the delta rule is a content-addressable memory update (Hopfield write), and the $SO(K)$ transport provides geometric structure to the memory space (curved basins of attraction).
