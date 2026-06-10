# Theory vs. Implementation Gap: What V13 Is Missing

## The Three Load-Bearing Mechanisms

The theory has three mechanisms that together provide the architecture's expressive power. Their implementation status in v13:

| Mechanism | Theory | V13 Status |
|---|---|---|
| **Gauge transport** (Wilson line) | Content-dependent phase rotation accumulates contextual curvature | **MISSING.** Scalar EMA, no content dependence |
| **Langevin settling** (Hopfield energy descent) | Iterative dynamics on attractor landscape collapse dense field to sparse | **MISSING.** One-shot MLP, no iteration |
| **Proximal sparsity** (soft-thresholding) | Lateral inhibition enforces spectral parsimony at each step | **MISSING.** No sparsification anywhere in the loop |

The constellation geometry (tokens as sparse spectral dots, shared modes as connections) is present and well-motivated. But it sits in a feedforward MLP wrapper instead of the dynamical system the theory prescribes.

---

## Gap 1: Gauge Transport (Wilson Line)

### What the theory says

Transport between fibers at different positions requires a gauge connection A. The spectral transport operator is:

```
X_tilde_q = X_p * exp(-D*w^2 - i*w * integral_gamma A)
```

The Wilson line `exp(-i*w * integral A)` accumulates contextual history as phase rotations. The connection A should be a function of the tokens being transported between — it is content-dependent. This is what gives the manifold its curvature: parallel transport around a closed loop in token space produces a nontrivial phase shift (holonomy), and this phase shift encodes the contextual relationship between the tokens on the loop.

### What V13 implements

```python
h[t] = alpha * h[t-1] + x[t]  # alpha is a learned constant per mode
```

A scalar exponential moving average. The decay alpha:
- Is learned but **fixed after training** — not content-dependent
- Is real-valued — no phase rotation component
- Is per-mode but not per-token — every token sees the same decay

This captures only the heat kernel `exp(-D*w^2)` (frequency-dependent damping), and only in degenerate form (a single scalar per mode rather than the full frequency-dependent kernel). The Wilson line — the content-dependent phase rotation that provides contextual curvature — is entirely absent.

### Why this matters

The GRU in v3 outperformed everything until v7.3 because the GRU's gates are content-dependent:

```
z_t = sigmoid(W_z * [h_{t-1}, x_t])  # update gate: CONTENT-DEPENDENT
r_t = sigmoid(W_r * [h_{t-1}, x_t])  # reset gate: CONTENT-DEPENDENT
```

The GRU was a de facto Wilson line accumulator — the very mechanism the architecture is designed around, implemented at the token-state level. V13's parallel scan removed this content dependence.

### What needs to change

The parallel scan's decay and phase rotation must be functions of the current token:

```
alpha_t = f_decay(c_t)     # content-dependent magnitude decay
theta_t = f_phase(c_t)     # content-dependent phase rotation
h[t] = (alpha_t * exp(i*theta_t)) * h[t-1] + x[t]
```

This makes the recurrence a proper gauge connection: the accumulated state depends on the *path* through token space (which tokens were seen, in what order), not just the exponentially-weighted sum. The holonomy (phase accumulated around a loop) becomes nonzero, giving the manifold genuine curvature.

Computationally, this is still compatible with parallel scan — the associative operation just uses token-dependent complex scalars instead of learned constants. Same O(n) complexity.

---

## Gap 2: Langevin Settling (Hopfield Energy Descent)

### What the theory says

After field reconstruction (the "forward" half of the loop), the dense spatial field should be iteratively settled onto an attractor landscape defined by the Hopfield energy:

```
E_q(x; M_q) = -beta^{-1} * log sum_j exp(beta * x^T * m_j)
```

The gradient descent on this energy is:

```
x_{t+1} = x_t + eta * softmax(beta * M^T * x_t) * M - eta * x_t + sqrt(2*eta/beta) * noise
```

This is mathematically identical to softmax attention over a memory bank M, but executed iteratively with annealing (beta increases over steps, reducing noise). The iteration provides:

1. **Content-addressable retrieval** — the softmax selectively amplifies memory atoms that match the current state
2. **Competition** — multiple memories compete; the attractor landscape resolves ambiguity
3. **Denoising** — annealing concentrates probability mass on the best attractor
4. **Multi-step refinement** — each step sharpens the representation

### What V13 implements

A single feedforward MLP pass:

```python
delta = self.net(torch.cat([c.mag, c.phase, messages], dim=-1))
new_mag = c.mag + sigmoid(self.gate) * delta_mag
new_phase = c.phase + sigmoid(self.gate) * delta_phase
```

One step. No energy landscape. No iteration. No attractor dynamics. No competition between interpretations. The MLP does not compute an energy gradient — it computes an arbitrary nonlinear map.

### Why this matters

The forward-reverse loop is the architectural heartbeat:

```
Sparse spectral -> IFFT (field reconstruction) -> Dense spatial -> Langevin settling -> Sparse spectral
```

V13 is running:

```
Sparse spectral -> fiber EMA -> Parseval read -> MLP -> Sparse spectral
```

The "reverse" half — the part that should collapse the dense field back to a sparse spectral configuration through energy descent — is replaced by a single feedforward pass. The model cannot iteratively refine its representation, cannot resolve ambiguity between competing interpretations, and cannot perform the stochastic search that Langevin dynamics provide.

### What needs to change

Implement even 2-3 Langevin steps per block:

```python
x = initial_state  # from fiber read
for k in range(K):
    beta_k = beta_min + (beta_max - beta_min) * k / K  # annealing schedule
    energy_grad = compute_hopfield_gradient(x, memory_bank, beta_k)
    noise = sqrt(2 * eta / beta_k) * torch.randn_like(x)
    x = x + eta * energy_grad + noise
    x = proximal_threshold(x, lambda_k)  # sparsification
```

The memory bank can be a fixed-size learned dictionary (Axiom 3), making each step O(|M| * d) — linear in sequence length.

---

## Gap 3: Proximal Sparsity (Soft-Thresholding)

### What the theory says

At each Langevin step, soft-thresholding enforces spectral sparsity:

```
prox_lambda(x) = sign(x) * max(|x| - lambda, 0)
```

This is the proximal operator for L1 regularization — it zeroes out small spectral coefficients and shrinks large ones toward zero. In the compressed sensing framework, this is the ISTA step that enforces the s-sparse prior.

The proximal operator provides:
1. **Lateral inhibition** — weak modes are suppressed, strong modes dominate
2. **Spectral parsimony** — the representation is forced to use few modes
3. **Submanifold selection** — the active support (which modes survive thresholding) defines the token's position in constellation space
4. **Gradient sparsity** — only active modes receive gradients, improving training efficiency

### What V13 implements

No sparsification anywhere in the processing loop. The constellation magnitudes and phases are updated by the MLP without any thresholding or sparsity enforcement. All 136 modes are active at all times.

### Why this matters

Without proximal sparsity, the "sparse spectral" representation is not actually sparse. The model has no mechanism to select a subset of modes as defining the token's identity. The constellation is dense, not a "few dots on Fourier space." The compressed sensing recovery guarantees (Candes-Romberg-Tao) require the sparse prior — without it, the Langevin dynamics have no reason to converge to sparse attractors.

### What needs to change

Apply soft-thresholding to constellation magnitudes after each update:

```python
# After MLP update
new_mag = torch.sign(new_mag) * F.relu(new_mag.abs() - lambda_thresh)
```

The threshold lambda can be:
- Fixed (simplest)
- Learned per block (adaptive sparsity depth)
- Annealed during training (start dense, progressively sparsify)

---

## The Compounding Effect

These three gaps compound. Without content-dependent transport, the fiber produces impoverished messages (exponentially blurred, no selectivity). Without Langevin settling, the model cannot iteratively refine from these impoverished messages. Without proximal sparsity, there is no spectral parsimony to constrain the search space. The MLP is left to do everything in a single feedforward pass — which is why 95.3% of parameters end up in the MLPs and the geometric machinery contributes nothing.

The SSM ablation result (SSM+MLP matched the full model) is the empirical signature of these combined gaps: the spectral machinery is decorative because none of its three load-bearing mechanisms are active.
