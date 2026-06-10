# Unit 02: Calculus, Gradients, and Energy Landscapes

## Learning Objectives

By the end of this unit, you should be able to:

1. Compute gradients of multivariate functions and interpret them geometrically
2. Understand Hessians, convexity, and saddle points
3. Perform gradient descent and understand convergence conditions
4. Connect energy landscapes to Hopfield networks and Langevin dynamics
5. Understand the softmax function as a gradient of the log-sum-exp

## Prerequisites

- Unit 01 (Linear Algebra for Geometric Language Modeling)
- Single-variable calculus (derivatives, chain rule, Taylor series)

## Readings

- **Primary:** Boyd & Vandenberghe, *Convex Optimization*, Ch. 2 (Convex Sets), Ch. 3 (Convex Functions), Ch. 9 (Unconstrained Minimization) -- free at stanford.edu/~boyd/cvxbook
- **Visual:** 3Blue1Brown, "Essence of Calculus" (YouTube)
- **Thesis:** Sec. 5.3.5 (Langevin-Hopfield Settling)
- **Repo:** topology/topological_computation.md (Section on Langevin dynamics)

---

## Key Concepts

### 1. Gradient

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is the vector of partial derivatives:

$$\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$$

The gradient points in the direction of **steepest ascent** of $f$. Its magnitude $\|\nabla f\|$ tells you how steep the ascent is.

Geometrically, the gradient is perpendicular to the level set $\{x : f(x) = c\}$ at each point. Moving along the gradient increases $f$ as fast as possible; moving perpendicular to it (along the level set) does not change $f$ at all.

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$, the generalization is the **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$, where $J_{ij} = \frac{\partial f_i}{\partial x_j}$.

### 2. Hessian

The **Hessian** is the matrix of second partial derivatives:

$$H(x)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

For smooth functions, the Hessian is symmetric (by Clairaut's theorem). The Hessian encodes the **curvature** of $f$ at a point:

- **Positive definite $H$** (all eigenvalues $> 0$): the function curves upward in every direction -- a local minimum.
- **Negative definite $H$** (all eigenvalues $< 0$): curves downward in every direction -- a local maximum.
- **Indefinite $H$** (some positive, some negative eigenvalues): a **saddle point** -- the function curves up in some directions and down in others.

The eigenvalues of $H$ tell you the curvature magnitude in each principal direction. The eigenvectors tell you what those directions are.

### 3. Convex Functions

A function $f$ is **convex** if for all $x, y$ and all $\lambda \in [0, 1]$:

$$f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y)$$

Geometrically: the line segment between any two points on the graph of $f$ lies above the graph. The function is "bowl-shaped."

Key properties of convex functions:
- Every local minimum is a global minimum.
- The sublevel sets $\{x : f(x) \leq c\}$ are convex sets.
- The Hessian is positive semidefinite everywhere ($H \succeq 0$).
- Gradient descent converges to the global minimum (with appropriate step size).

Examples: $f(x) = x^2$, $f(x) = e^x$, $f(x) = \|x\|$, $f(x) = \max(x_1, \ldots, x_n)$.

Non-examples: $f(x) = \sin(x)$, $f(x) = x^3$, neural network loss landscapes (generally).

### 4. Saddle Points

A **saddle point** is a critical point ($\nabla f = 0$) where the Hessian is indefinite -- it has both positive and negative eigenvalues.

At a saddle point, the function looks like a minimum along some directions and a maximum along others. The classic example is $f(x, y) = x^2 - y^2$, which forms a hyperbolic paraboloid (saddle shape).

Saddle points are extremely important in deep learning:
- In high dimensions, most critical points of random functions are saddle points, not local minima (Baldi & Hornik 1989, Dauphin et al. 2014).
- Gradient descent can get stuck near saddle points because the gradient is small.
- Second-order methods (using Hessian information) can escape saddle points by identifying negative curvature directions.
- Stochastic gradient descent naturally escapes saddle points via noise.

### 5. Gradient Descent

**Gradient descent** iteratively moves in the direction of steepest descent:

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $\eta > 0$ is the **learning rate** (or step size).

Intuition: at each step, you take a step proportional to the negative gradient, which points toward decreasing $f$.

Convergence depends on the learning rate:
- Too large: the algorithm overshoots and diverges.
- Too small: convergence is extremely slow.
- Optimal (for convex, $L$-smooth functions): $\eta = 1/L$, where $L$ is the Lipschitz constant of the gradient.

### 6. Learning Rate

The learning rate $\eta$ controls the trade-off between speed and stability.

For a quadratic $f(x) = \frac{1}{2} x^T H x$, gradient descent gives:

$$x_{k+1} = (I - \eta H) x_k$$

The eigenvalues of $(I - \eta H)$ are $(1 - \eta \lambda_i)$ where $\lambda_i$ are eigenvalues of $H$. For convergence, we need $|1 - \eta \lambda_i| < 1$ for all $i$, which gives:

$$0 < \eta < \frac{2}{\lambda_{\max}}$$

The optimal rate is $\eta = \frac{2}{\lambda_{\max} + \lambda_{\min}}$, giving convergence rate:

$$r = \frac{\kappa - 1}{\kappa + 1}$$

where $\kappa = \lambda_{\max} / \lambda_{\min}$ is the condition number. Large $\kappa$ (ill-conditioning) means slow convergence, which is why preconditioning and adaptive methods (Adam, etc.) are important.

### 7. Log-Sum-Exp (LSE)

The **log-sum-exp** function is:

$$\operatorname{LSE}(z) = \log\left(\sum_i \exp(z_i)\right)$$

LSE is a smooth, convex approximation to the maximum function:

$$\max(z_i) \leq \operatorname{LSE}(z) \leq \max(z_i) + \log(n)$$

With a temperature parameter:

$$\frac{1}{\beta} \operatorname{LSE}(\beta z) = \frac{1}{\beta} \log\left(\sum_i \exp(\beta z_i)\right)$$

As $\beta \to \infty$, this converges to $\max(z_i)$. As $\beta \to 0$, it converges to the arithmetic mean.

Numerically stable computation: $\operatorname{LSE}(z) = \max(z) + \log\left(\sum_i \exp(z_i - \max(z))\right)$.

### 8. Softmax as Gradient of LSE

The **softmax** function is:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

The fundamental identity connecting softmax and LSE:

$$\operatorname{softmax}(z)_i = \frac{\partial \operatorname{LSE}}{\partial z_i}$$

Proof: $\operatorname{LSE}(z) = \log(\sum \exp(z_j))$. Differentiating: $\frac{\partial \operatorname{LSE}}{\partial z_i} = \frac{\exp(z_i)}{\sum \exp(z_j)} = \operatorname{softmax}(z)_i$.

This means softmax is the gradient of a convex function. This has deep implications:
- Attention weights (computed via softmax) are gradients of the convex LSE potential.
- The attention mechanism can be interpreted as performing a step of optimization on the LSE energy landscape.
- The Hopfield network connection: modern Hopfield networks use LSE as their energy, and the update rule is exactly one step of gradient descent on this energy.

### 9. Energy-Based Models

An **energy-based model** defines a scalar energy function $E(x)$ and associates probabilities via the Boltzmann distribution:

$$p(x) = \frac{1}{Z} \exp(-\beta E(x))$$

where $Z = \int \exp(-\beta E(x)) \, dx$ is the partition function and $\beta = 1/(k_B T)$ is the inverse temperature.

- Low energy states have high probability.
- $\beta \to \infty$ (low temperature): only the global minimum has nonzero probability.
- $\beta \to 0$ (high temperature): uniform distribution -- all states equally likely.

In the context of language modeling, the energy function encodes compatibility between representations. The attention mechanism with inverse temperature $\sqrt{d_k}$ can be viewed as evaluating energies between queries and keys.

### 10. Langevin Dynamics

**Langevin dynamics** combines gradient descent with Gaussian noise:

$$x_{t+1} = x_t - \eta \nabla E(x_t) + \sqrt{\frac{2\eta}{\beta}} \cdot \varepsilon_t$$

where $\varepsilon_t$ is sampled from a standard normal distribution $\mathcal{N}(0, I)$.

The two terms serve complementary purposes:
- **Gradient term** ($-\eta \nabla E$): pushes $x$ toward low-energy regions (exploitation).
- **Noise term** ($\sqrt{2\eta/\beta} \; \varepsilon$): allows $x$ to explore and escape local minima (exploration).

Under mild conditions, Langevin dynamics converges to sampling from the Boltzmann distribution $p(x) \propto \exp(-\beta E(x))$. Higher $\beta$ (lower temperature) means less noise, converging to gradient descent. Lower $\beta$ (higher temperature) means more noise, approaching random walk.

In SGST, Langevin dynamics is used in the "settling" phase: after diffusion spreads a token representation across the manifold, Langevin dynamics drives it toward a memory pattern (energy minimum) while maintaining enough stochasticity to avoid spurious local minima.

---

## Worked Problems

### Problem 1: Gradient and Hessian of a Quadratic

**Problem:** For $f(x, y) = x^2 + 4y^2$, compute $\nabla f$, the Hessian, and classify the critical point at $(0, 0)$.

**Solution:**

**Step 1: Gradient.**

$$\nabla f = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]^T = [2x, \; 8y]^T$$

At the origin: $\nabla f(0, 0) = [0, 0]^T$. So $(0, 0)$ is a critical point.

**Step 2: Hessian.**

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 8 \end{bmatrix}$$

**Step 3: Classification.**

The eigenvalues of $H$ are $2$ and $8$ (it is already diagonal). Both are strictly positive, so $H$ is positive definite. Therefore $(0, 0)$ is a **strict local minimum** (and since $f$ is convex, it is the global minimum).

**Geometric interpretation:** The function is a bowl, elliptical in cross-section. It is steeper in the $y$-direction (curvature 8) than the $x$-direction (curvature 2). The condition number is $\kappa = 8/2 = 4$, meaning gradient descent will be 4 times slower than optimal due to the eccentricity.

The level sets $f(x, y) = c$ are ellipses $x^2 + 4y^2 = c$, with semi-axes $\sqrt{c}$ in $x$ and $\sqrt{c/4} = \sqrt{c}/2$ in $y$.

---

### Problem 2: Saddle Point Analysis

**Problem:** For $f(x, y) = x^2 - y^2$, compute gradient and Hessian. Show why gradient descent starting at $(1, 1)$ will not converge to $(0, 0)$.

**Solution:**

**Step 1: Gradient and Hessian.**

$$\nabla f = [2x, \; -2y]^T$$

$$H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$$

At $(0, 0)$: $\nabla f = [0, 0]$, so it is a critical point.

**Step 2: Classify the critical point.**

Eigenvalues of $H$: $+2$ and $-2$. The Hessian is **indefinite** (one positive, one negative eigenvalue). Therefore $(0, 0)$ is a **saddle point**.

The positive eigenvalue ($+2$) corresponds to the $x$-direction: $f$ curves upward like a bowl. The negative eigenvalue ($-2$) corresponds to the $y$-direction: $f$ curves downward like an inverted bowl.

**Step 3: Gradient descent dynamics.**

Starting at $(x_0, y_0) = (1, 1)$ with step size $\eta$:

$$x_{k+1} = x_k - \eta(2 x_k) = (1 - 2\eta) x_k$$

$$y_{k+1} = y_k - \eta(-2 y_k) = (1 + 2\eta) y_k$$

For the $x$-component: $x_k = (1 - 2\eta)^k$. If $0 < \eta < 1$, then $|1 - 2\eta| < 1$, so $x_k \to 0$. Good.

For the $y$-component: $y_k = (1 + 2\eta)^k$. Since $\eta > 0$, we have $(1 + 2\eta) > 1$, so $y_k \to \infty$. The $y$-component **diverges**.

**Therefore, gradient descent DIVERGES from the saddle point along the negative-curvature direction.** The gradient pushes the iterate away from the saddle in the $y$-direction because the function is decreasing in that direction (and GD follows the negative gradient, which here increases $|y|$).

This illustrates a general principle: saddle points are unstable equilibria for gradient descent. Almost any perturbation will cause escape along the negative-curvature direction, which is actually desirable in neural network optimization.

---

### Problem 3: Softmax as Gradient of LSE

**Problem:** Prove that $\operatorname{softmax}(z)_i = \frac{\partial}{\partial z_i} \left[\log\left(\sum_j \exp(z_j)\right)\right]$. Then explain why this means attention weights are gradients of a convex function.

**Solution:**

**Step 1: Compute the partial derivative.**

$$\operatorname{LSE}(z) = \log\left(\sum_{j=1}^{n} \exp(z_j)\right)$$

Let $S = \sum_{j=1}^{n} \exp(z_j)$. Then $\operatorname{LSE}(z) = \log(S)$.

$$\frac{\partial \operatorname{LSE}}{\partial z_i} = \frac{1}{S} \cdot \frac{\partial S}{\partial z_i} = \frac{1}{S} \cdot \exp(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)} = \operatorname{softmax}(z)_i$$

**Step 2: Verify LSE is convex by computing its Hessian.**

The gradient is $\nabla \operatorname{LSE} = \operatorname{softmax}(z) = p$ (a probability vector).

The Hessian entry:

$$\frac{\partial^2 \operatorname{LSE}}{\partial z_i \partial z_j} = \frac{\partial p_i}{\partial z_j}$$

We need the Jacobian of softmax. For $i = j$:

$$\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$$

For $i \neq j$:

$$\frac{\partial p_i}{\partial z_j} = -p_i p_j$$

In matrix form:

$$H = \operatorname{diag}(p) - p p^T$$

This is the **covariance matrix** of a categorical distribution with probabilities $p$. It is positive semidefinite because for any vector $v$:

$$v^T H v = v^T \operatorname{diag}(p) v - (v^T p)^2 = \sum_i p_i v_i^2 - \left(\sum_i p_i v_i\right)^2 = \mathbb{E}[v^2] - (\mathbb{E}[v])^2 = \operatorname{Var}(v) \geq 0$$

Since $H \succeq 0$ everywhere, LSE is convex.

**Step 3: Implication for attention.**

In standard attention, the attention weights for query $q$ and key matrix $K$ are:

$$\alpha = \operatorname{softmax}(K q / \sqrt{d_k})$$

Since softmax is the gradient of the convex function LSE, the attention weights $\alpha$ are the gradient of:

$$E(q) = \operatorname{LSE}(K q / \sqrt{d_k}) = \log\left(\sum_j \exp(k_j^T q / \sqrt{d_k})\right)$$

This means computing attention weights is equivalent to evaluating the gradient of a convex energy landscape. The attention mechanism can be interpreted as:
1. Each key $k_j$ defines a "potential well" in query space.
2. The energy $E(q)$ is a soft-maximum of the query-key compatibilities.
3. The attention weights $\alpha_j$ tell you how much the query is "pulled" toward each key.

This is the bridge to Hopfield networks: modern continuous Hopfield networks use exactly this LSE energy, and their update rule (pattern retrieval) is one step of gradient descent on $E$.

---

### Problem 4: Hopfield Energy and the Attention Update

**Problem:** The Hopfield energy is $E(\xi) = -\frac{1}{\beta} \log\left(\sum_\mu \exp(\beta x^\mu \cdot \xi)\right) + \frac{1}{2}\|\xi\|^2$. Compute $\nabla_\xi E$ and show the update rule $\xi_{\text{new}} = X \operatorname{softmax}(\beta X^T \xi)$.

**Solution:**

**Step 1: Identify the terms.**

Let the stored patterns be $x^1, \ldots, x^M$ (rows of matrix $X \in \mathbb{R}^{M \times d}$), and $\xi \in \mathbb{R}^d$ is the probe (query).

$$E(\xi) = -\frac{1}{\beta} \log\left(\sum_{\mu=1}^{M} \exp(\beta x^\mu \cdot \xi)\right) + \frac{1}{2} \|\xi\|^2$$

The first term is $-\frac{1}{\beta} \operatorname{LSE}(\beta X \xi)$, and the second is a quadratic regularizer.

**Step 2: Gradient of the first term.**

Let $g(\xi) = -\frac{1}{\beta} \log\left(\sum_\mu \exp(\beta (x^\mu)^T \xi)\right)$.

$$\frac{\partial g}{\partial \xi} = -\frac{1}{\beta} \cdot \frac{\sum_\mu \beta x^\mu \exp(\beta (x^\mu)^T \xi)}{\sum_\mu \exp(\beta (x^\mu)^T \xi)} = -\frac{\sum_\mu x^\mu \exp(\beta (x^\mu)^T \xi)}{\sum_\mu \exp(\beta (x^\mu)^T \xi)}$$

$$= -\sum_\mu x^\mu \cdot \operatorname{softmax}(\beta X^T \xi)_\mu = -X^T \operatorname{softmax}(\beta X^T \xi)$$

(Here $X^T$ is $d \times M$ and $\operatorname{softmax}(\beta X^T \xi)$ is $M \times 1$, so the product is $d \times 1$.)

**Step 3: Gradient of the second term.**

$$\frac{\partial}{\partial \xi} \left[\frac{1}{2} \|\xi\|^2\right] = \xi$$

**Step 4: Full gradient.**

$$\nabla_\xi E = -X^T \operatorname{softmax}(\beta X^T \xi) + \xi$$

**Step 5: Derive the update rule.**

Setting $\xi_{\text{new}}$ to the result of one step starting from "equilibrium" (setting $\nabla E = 0$):

$$0 = -X^T \operatorname{softmax}(\beta X^T \xi) + \xi_{\text{new}}$$

$$\xi_{\text{new}} = X^T \operatorname{softmax}(\beta X^T \xi)$$

This is the **Hopfield update rule**.

Now compare with attention. In attention with queries $Q$, keys $K$, values $V$:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}(Q K^T / \sqrt{d}) \, V$$

For a single query $q$, with keys and values both equal to the stored patterns $X$:

$$\text{output} = X^T \operatorname{softmax}(X q / \sqrt{d})$$

This is structurally identical to the Hopfield update with $\beta = 1/\sqrt{d}$. The stored patterns play the role of both keys and values. The query $\xi$ retrieves a soft-weighted combination of stored patterns.

This is the Ramsauer et al. (2021) result: **attention IS Hopfield pattern retrieval**.

---

### Problem 5: Langevin Dynamics Step-by-Step

**Problem:** Implement one step of Langevin dynamics on $E(x) = (x - 3)^2 / 2$ starting at $x = 0$, with $\eta = 0.1$, $\beta = 1$. Show 5 steps with fixed noise $\varepsilon_k = [0.5, -0.3, 0.8, -0.1, 0.4]$.

**Solution:**

**Step 1: Set up the update rule.**

Energy: $E(x) = (x - 3)^2 / 2$.
Gradient: $\nabla E(x) = \frac{dE}{dx} = x - 3$.

Langevin update:

$$x_{k+1} = x_k - \eta \nabla E(x_k) + \sqrt{\frac{2\eta}{\beta}} \cdot \varepsilon_k$$

$$= x_k - 0.1(x_k - 3) + \sqrt{2 \cdot 0.1 / 1} \cdot \varepsilon_k$$

$$= x_k - 0.1(x_k - 3) + \sqrt{0.2} \cdot \varepsilon_k$$

$$= x_k - 0.1(x_k - 3) + 0.4472 \cdot \varepsilon_k$$

**Step 2: Execute 5 steps.**

**Step $0 \to 1$:** $x_0 = 0$, $\varepsilon_0 = 0.5$

$$x_1 = 0 - 0.1(0 - 3) + 0.4472(0.5) = 0 + 0.300 + 0.224 = 0.524$$

**Step $1 \to 2$:** $x_1 = 0.524$, $\varepsilon_1 = -0.3$

$$x_2 = 0.524 - 0.1(0.524 - 3) + 0.4472(-0.3) = 0.524 - 0.1(-2.476) + (-0.134) = 0.524 + 0.248 - 0.134 = 0.638$$

**Step $2 \to 3$:** $x_2 = 0.638$, $\varepsilon_2 = 0.8$

$$x_3 = 0.638 - 0.1(0.638 - 3) + 0.4472(0.8) = 0.638 - 0.1(-2.362) + 0.358 = 0.638 + 0.236 + 0.358 = 1.232$$

**Step $3 \to 4$:** $x_3 = 1.232$, $\varepsilon_3 = -0.1$

$$x_4 = 1.232 - 0.1(1.232 - 3) + 0.4472(-0.1) = 1.232 - 0.1(-1.768) + (-0.045) = 1.232 + 0.177 - 0.045 = 1.364$$

**Step $4 \to 5$:** $x_4 = 1.364$, $\varepsilon_4 = 0.4$

$$x_5 = 1.364 - 0.1(1.364 - 3) + 0.4472(0.4) = 1.364 - 0.1(-1.636) + 0.179 = 1.364 + 0.164 + 0.179 = 1.707$$

**Summary of trajectory:**

| Step | $x_k$ |
|------|--------|
| 0 | 0.000 |
| 1 | 0.524 |
| 2 | 0.638 |
| 3 | 1.232 |
| 4 | 1.364 |
| 5 | 1.707 |

The iterate is trending toward the minimum at $x = 3$. Without noise, this would be pure gradient descent: $x_{k+1} = x_k - 0.1(x_k - 3) = 0.9 x_k + 0.3$, which converges monotonically. The noise adds stochastic jitter that allows exploration and, in more complex landscapes, escape from local minima.

---

### Problem 6: Cross-Entropy Gradient

**Problem:** Show that the gradient of cross-entropy loss $L = -\sum_i y_i \log(\operatorname{softmax}(z)_i)$ with respect to $z$ is $\operatorname{softmax}(z) - y$.

**Solution:**

**Step 1: Set up notation.**

Let $p = \operatorname{softmax}(z)$, so $p_i = \exp(z_i) / \sum_j \exp(z_j)$. The loss is:

$$L = -\sum_i y_i \log(p_i)$$

where $y$ is the target distribution (typically one-hot: $y_c = 1$, $y_i = 0$ for $i \neq c$).

**Step 2: Compute $\frac{\partial L}{\partial z_j}$ using chain rule.**

$$\frac{\partial L}{\partial z_j} = -\sum_i y_i \cdot \frac{1}{p_i} \cdot \frac{\partial p_i}{\partial z_j}$$

We need $\frac{\partial p_i}{\partial z_j}$, the Jacobian of softmax (computed in Problem 3):

$$\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)$$

where $\delta_{ij}$ is the Kronecker delta ($1$ if $i=j$, $0$ otherwise).

**Step 3: Substitute.**

$$\frac{\partial L}{\partial z_j} = -\sum_i y_i \cdot \frac{1}{p_i} \cdot p_i \cdot (\delta_{ij} - p_j)$$

$$= -\sum_i y_i (\delta_{ij} - p_j)$$

$$= -\sum_i y_i \delta_{ij} + \sum_i y_i p_j$$

$$= -y_j + p_j \sum_i y_i$$

Since $y$ is a probability distribution, $\sum_i y_i = 1$:

$$\frac{\partial L}{\partial z_j} = -y_j + p_j \cdot 1 = p_j - y_j$$

**Step 4: Vector form.**

$$\nabla_z L = \operatorname{softmax}(z) - y = p - y$$

**Why this is remarkable:** The gradient has an incredibly clean form -- it is simply the difference between the predicted probabilities and the target. No messy chain rule terms, no Jacobian matrices in the final expression.

For a one-hot target (classification), if the true class is $c$:

$$\frac{\partial L}{\partial z_j} = p_j - \delta_{jc}$$

The gradient pushes the logit for the correct class UP (by $-1 + p_c$, which is negative since $p_c < 1$) and pushes all other logits DOWN (by $p_j > 0$). The magnitude of each push is proportional to the current prediction error.

This clean gradient is one reason why cross-entropy + softmax is the dominant choice for classification. Any other combination (e.g., mean squared error + softmax) yields a more complicated gradient with worse optimization properties.

---

### Problem 7: Convergence Guarantee for $L$-Smooth Functions

**Problem:** A function $f$ is $L$-smooth if $\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$. Show that gradient descent with step size $\eta = 1/L$ guarantees $f(x_{k+1}) \leq f(x_k) - \frac{1}{2L} \|\nabla f(x_k)\|^2$.

**Solution:**

**Step 1: The descent lemma.**

$L$-smoothness implies the following quadratic upper bound (the "descent lemma"). For all $x, y$:

$$f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|^2$$

This says $f$ is bounded above by a quadratic that touches $f$ at $x$ and has curvature $L$. (Proof: integrate the Lipschitz gradient condition along the line segment from $x$ to $y$.)

**Step 2: Apply to gradient descent step.**

Set $y = x_{k+1} = x_k - \frac{1}{L} \nabla f(x_k)$. Then $y - x_k = -\frac{1}{L} \nabla f(x_k)$.

$$f(x_{k+1}) \leq f(x_k) + \nabla f(x_k)^T \left(-\frac{1}{L} \nabla f(x_k)\right) + \frac{L}{2} \left\|-\frac{1}{L} \nabla f(x_k)\right\|^2$$

**Step 3: Simplify each term.**

Term 1: $\nabla f(x_k)^T \left(-\frac{1}{L} \nabla f(x_k)\right) = -\frac{1}{L} \|\nabla f(x_k)\|^2$.

Term 2: $\frac{L}{2} \cdot \frac{1}{L^2} \|\nabla f(x_k)\|^2 = \frac{1}{2L} \|\nabla f(x_k)\|^2$.

**Step 4: Combine.**

$$f(x_{k+1}) \leq f(x_k) - \frac{1}{L} \|\nabla f(x_k)\|^2 + \frac{1}{2L} \|\nabla f(x_k)\|^2 = f(x_k) - \frac{1}{2L} \|\nabla f(x_k)\|^2$$

**Interpretation:**

Each gradient descent step decreases the function value by at least $\frac{1}{2L} \|\nabla f(x_k)\|^2$. This guarantees:

1. **Monotone decrease:** $f(x_{k+1}) \leq f(x_k)$ always (since the decrease term is nonneg).
2. **Convergence:** Summing over $k$ steps: $\sum_{k=0}^{K-1} \frac{1}{2L} \|\nabla f(x_k)\|^2 \leq f(x_0) - f^*$. This means $\|\nabla f(x_k)\|^2 \to 0$, so the gradients vanish.
3. **Rate:** The minimum gradient norm in $K$ steps satisfies $\min_k \|\nabla f(x_k)\|^2 \leq \frac{2L(f(x_0) - f^*)}{K}$, giving an $O(1/K)$ rate. For convex functions, $f(x_k) - f^* = O(1/K)$; for strongly convex, $O(\exp(-k/\kappa))$.

This is the foundational convergence result for gradient-based optimization. The step size $1/L$ is optimal for the worst case.

---

### Problem 8: Proximal Operator and Soft-Thresholding

**Problem:** The proximal operator for $L_1$ regularization (soft-thresholding) is: $\operatorname{prox}_{\lambda \|\cdot\|_1}(x)_i = \operatorname{sign}(x_i) \max(|x_i| - \lambda, 0)$. Apply this to $x = [3.0, -0.5, 1.2, -0.1, 0.8]$ with $\lambda = 0.6$.

**Solution:**

**Step 1: Understand the proximal operator.**

The proximal operator of $g$ at $x$ is defined as:

$$\operatorname{prox}_g(x) = \arg\min_u \left\{ \frac{1}{2}\|u - x\|^2 + g(u) \right\}$$

For $g(u) = \lambda \|u\|_1 = \lambda \sum_i |u_i|$, the problem decomposes component-wise:

$$\operatorname{prox}_i = \arg\min_{u_i} \left\{ \frac{1}{2}(u_i - x_i)^2 + \lambda |u_i| \right\}$$

The solution is the **soft-thresholding** operator:

$$\operatorname{prox}_i = \operatorname{sign}(x_i) \max(|x_i| - \lambda, 0)$$

This shrinks each component toward zero by $\lambda$, and sets it to exactly zero if $|x_i| \leq \lambda$.

**Step 2: Apply component-wise with $\lambda = 0.6$.**

Component $x_1 = 3.0$: $|3.0| > 0.6$, so result $= \operatorname{sign}(3.0)(3.0 - 0.6) = +2.4$

Component $x_2 = -0.5$: $|-0.5| = 0.5 < 0.6$, so result $= 0$

Component $x_3 = 1.2$: $|1.2| > 0.6$, so result $= \operatorname{sign}(1.2)(1.2 - 0.6) = +0.6$

Component $x_4 = -0.1$: $|-0.1| = 0.1 < 0.6$, so result $= 0$

Component $x_5 = 0.8$: $|0.8| > 0.6$, so result $= \operatorname{sign}(0.8)(0.8 - 0.6) = +0.2$

**Result:**

$$\operatorname{prox}(x) = [2.4, \; 0, \; 0.6, \; 0, \; 0.2]$$

**Step 3: Interpret.**

Two of the five components ($-0.5$ and $-0.1$) were set to exactly zero. The vector became **sparser**: it went from 5 nonzero components to 3.

The larger components ($3.0$, $1.2$, $0.8$) survived but were shrunk by $0.6$. The threshold $\lambda$ controls the aggressiveness of sparsification: larger $\lambda$ kills more components.

**Connection to SGST:** In the SGST architecture, after Langevin dynamics settles a token representation into an energy minimum, soft-thresholding is applied to enforce spectral sparsity. The token's representation is expressed in the Fourier basis, and soft-thresholding zeros out weak frequency modes, leaving only the dominant spectral components. This is the mechanism by which SGST maintains sparse representations that avoid the degeneration problem of dense embeddings (thesis Sec. 5.3.7).

---

### Problem 9: Log-Sum-Exp Approximates Max

**Problem:** Show that as $\beta \to \infty$, $\frac{1}{\beta} \log\left(\sum \exp(\beta z_i)\right) \to \max(z_i)$. Compute for $z = [1, 3, 2]$ at $\beta = 1, 10, 100$.

**Solution:**

**Step 1: Theoretical argument.**

Let $z_{\max} = \max(z_i)$. Factor out $\exp(\beta z_{\max})$:

$$\frac{1}{\beta} \log\left(\sum_i \exp(\beta z_i)\right) = \frac{1}{\beta} \log\left(\exp(\beta z_{\max}) \sum_i \exp(\beta(z_i - z_{\max}))\right)$$

$$= z_{\max} + \frac{1}{\beta} \log\left(\sum_i \exp(\beta(z_i - z_{\max}))\right)$$

Now $z_i - z_{\max} \leq 0$ for all $i$, and equals $0$ for at least one $i$. So:

$$\sum_i \exp(\beta(z_i - z_{\max})) = 1 + \sum_{i:\, z_i < z_{\max}} \exp(\beta(z_i - z_{\max}))$$

As $\beta \to \infty$, each term $\exp(\beta(z_i - z_{\max})) \to 0$ for $z_i < z_{\max}$ (since the exponent is negative). So the sum $\to 1$, and:

$$\frac{1}{\beta} \log(1) = 0$$

Therefore the whole expression $\to z_{\max} + 0 = z_{\max} = \max(z_i)$.

**Step 2: Numerical computation for $z = [1, 3, 2]$.**

**$\beta = 1$:**

$$\frac{1}{1} \log(\exp(1) + \exp(3) + \exp(2)) = \log(2.718 + 20.086 + 7.389) = \log(30.193) = 3.408$$

**$\beta = 10$:**

$$\frac{1}{10} \log(\exp(10) + \exp(30) + \exp(20))$$

Use the numerically stable form: $z_{\max} = 3$, so $\beta \cdot z_{\max} = 30$.

$$= 3 + \frac{1}{10} \log(\exp(-20) + 1 + \exp(-10))$$

$$= 3 + \frac{1}{10} \log(1 + 2.061 \times 10^{-9} + 4.540 \times 10^{-5})$$

$$= 3 + \frac{1}{10} \log(1.0000454) = 3 + \frac{1}{10}(4.54 \times 10^{-5}) = 3.00000454 \approx 3.000$$

**$\beta = 100$:**

$$= 3 + \frac{1}{100} \log(\exp(-200) + 1 + \exp(-100)) = 3 + \frac{1}{100} \log(1 + \text{negligible}) \approx 3.000000\ldots$$

**Summary:**

| $\beta$ | Result | Notes |
|---------|--------|-------|
| 1 | 3.408 | smooth approximation, noticeably above $\max = 3$ |
| 10 | 3.000 | very close to max |
| 100 | 3.000 | indistinguishable from max |

As $\beta$ increases, the LSE becomes a sharper and sharper approximation to the max. In the attention mechanism, the temperature parameter (typically $1/\sqrt{d_k}$) controls this sharpness: small temperature (high $\beta$) gives peaked attention (nearly argmax), while high temperature (low $\beta$) gives diffuse attention (nearly uniform).

---

### Problem 10: The SGST Settling Mechanism (Conceptual)

**Problem:** In the SGST architecture, the "settling" step runs Langevin dynamics on a Hopfield energy landscape, then applies soft-thresholding. Describe in your own words what each step accomplishes physically, using the energy landscape metaphor.

**Solution:**

**The energy landscape:**

Imagine a mountainous terrain in high-dimensional space. The Hopfield energy $E(\xi)$ defines this landscape, with valleys (energy minima) located at each stored memory pattern. Deep valleys correspond to well-learned patterns; shallow valleys to weak or spurious memories. Between valleys, there are ridges, passes, and saddle points.

**Phase 1: Diffusion (before settling).**

The incoming token has been spread by the diffusion process across the manifold. In the energy landscape metaphor, this is like taking a precise point and blurring it into a cloud. The diffused token is at a HIGH-energy state -- it sits on the hillside, not in any valley. It is a superposition of many potential memories, none of them committed to.

**Phase 2: Langevin dynamics (gradient descent + noise).**

Langevin dynamics is like releasing a ball on the hillside with a bit of random jitter:

- The **gradient descent component** ($-\eta \nabla E$) rolls the ball downhill toward the nearest valley. The gradient points uphill, so the negative gradient pushes toward lower energy. At each step, the ball moves toward a more coherent, pattern-like representation.

- The **noise component** ($\sqrt{2\eta/\beta} \; \varepsilon$) randomly perturbs the ball. This serves two critical purposes:
  - **Escape from shallow valleys:** Spurious memories create shallow energy minima. Without noise, the ball would get trapped in the first valley it encounters, even if it is a poor match. Noise gives the ball enough energy to hop out of shallow valleys and continue rolling toward deeper, more appropriate ones.
  - **Exploration:** In a complex landscape with many valleys, noise ensures the system does not commit too early. It explores the local neighborhood before settling.

As the dynamics proceed, the ball descends into a valley. The depth of the valley (how much the energy decreases) reflects how well the retrieved pattern matches the input query.

**Phase 3: Soft-thresholding (proximal sparsity).**

After Langevin dynamics, the token representation has "settled" near a memory pattern, but it still has some residual noise and small, spurious components from the stochastic process. Soft-thresholding acts as a cleanup step:

- It zeros out weak components (spectral modes with amplitude below the threshold $\lambda$). These are the "noise" from Langevin dynamics and the remnants of patterns that the settling did not fully commit to.
- It shrinks the remaining components, pulling the representation into a sparser form.

The result is a **sparse spectral representation**: the token is described by a few dominant frequency modes, not a dense 512-dimensional vector. This directly combats the representation degeneration problem.

**The full cycle as an analogy to generative diffusion models:**

1. Forward diffusion: structure $\to$ noise (add noise to destroy information)
2. Reverse diffusion: noise $\to$ structure (denoise to reconstruct)

In SGST:
1. Forward (diffusion phase): sparse token $\to$ spread field (dissolve sharp representation into smooth manifold field)
2. Reverse (settling phase): spread field $\to$ sparse token (Langevin + thresholding collapses back to sharp, sparse representation)

The key insight is that the "reverse" process is not an arbitrary denoiser -- it is guided by the Hopfield energy landscape, which encodes the memory patterns (learned from data). The settling process naturally retrieves the most compatible memory pattern, analogous to how diffusion models use a learned score function to reverse the noise process.

---

## Comprehension Questions

Answer these after completing the readings and problems. Aim for deep, conceptual understanding.

1. Why is the Hessian matrix important for understanding the geometry of a loss landscape? What specific information do its eigenvalues and eigenvectors provide?

2. Explain the connection between softmax and the gradient of log-sum-exp. Why does this make attention interpretable as energy minimization?

3. What role does noise play in Langevin dynamics? What happens if you remove it (i.e., set the temperature to zero)?

4. How does soft-thresholding (proximal operator) differ from simply setting small values to zero (hard thresholding)? Why is the soft version preferred for gradient-based optimization?

5. The thesis uses Langevin dynamics to "settle" tokens into memory patterns. Draw the analogy to how a diffusion model generates images. What corresponds to the "score function" in SGST?

---

## Bridge to Thesis

The concepts in this unit connect directly to the thesis architecture and analysis:

- **Hopfield energy + Langevin settling** (Problems 4, 5, 10) form the core of the SGST settling mechanism (thesis Sec. 5.3.5). After diffusion spreads a token representation, Langevin dynamics on the Hopfield energy landscape retrieves the nearest memory pattern. The temperature parameter $\beta$ controls the sharpness of retrieval.

- **Soft-thresholding / proximal sparsity** (Problem 8) enforces spectral sparsity after settling (thesis Sec. 5.3.7). This is what makes SGST representations sparse rather than dense, directly addressing the representation degeneration problem diagnosed in Chapter 2.

- **Softmax as gradient of convex energy** (Problem 3) connects standard attention to the Hopfield energy minimization framework (thesis Sec. 2.6). This provides the theoretical bridge between conventional transformers and the energy-based formulation that SGST generalizes.

- **Energy landscapes and temperature** (Problem 9) explain the role of the inverse temperature $\beta$ throughout the architecture (thesis Sec. 2.6.1). High $\beta$ gives sharp, deterministic retrieval (like argmax attention); low $\beta$ gives soft, exploratory retrieval (like uniform attention). The SGST settling process uses an annealing schedule that starts with low $\beta$ (exploration) and increases it (exploitation).

- **Convergence of gradient methods** (Problem 7) provides the theoretical backbone for analyzing whether the settling process actually converges. The $L$-smoothness framework guarantees that each Langevin step makes progress toward the energy minimum, even with the added noise term.
