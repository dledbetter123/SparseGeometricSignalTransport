# V14 Design Requirements

## Guiding Principle

The mathematical skeleton is sound. The implementation must instantiate the theory's load-bearing structures, not just the scaffolding. V13 has the constellation geometry right. V14 must add the dynamics.

---

## Requirement 1: Content-Dependent Gauge Transport (Wilson Line)

**Priority: CRITICAL — most likely to break the plateau**

The parallel scan decay and phase rotation must be functions of the current token's constellation:

```python
# Current (V13): fixed decay, no phase
h[t] = alpha * h[t-1] + x[t]           # alpha is learned constant

# Required (V14): content-dependent complex recurrence
alpha_t, theta_t = gate_network(c_t)    # functions of current constellation
h[t] = (alpha_t * exp(i * theta_t)) * h[t-1] + deposit(c_t)
```

**Implementation notes:**
- `gate_network` should be lightweight — a small MLP or even a linear projection from the constellation to (decay, phase) per mode
- This is still compatible with parallel scan: the associative operation uses token-dependent complex scalars instead of constants. Same O(n) complexity.
- The phase rotation theta_t is the Wilson line: accumulated phase = holonomy = contextual curvature
- Consider per-subbundle or per-mode gating (not a single scalar for all modes)

**Validation:** After implementing, the fiber should have meaningfully more than 1,088 parameters. The gate network adds O(M * input_dim) parameters to the geometric path.

---

## Requirement 2: Langevin Settling with Hopfield Memory

**Priority: HIGH — provides content-addressable retrieval**

After the fiber read, apply K iterative Langevin steps on a Hopfield energy landscape:

```python
# Memory bank: fixed-size learned dictionary
memory_bank = nn.Parameter(torch.randn(num_atoms, constellation_dim))

x = initial_state  # from Parseval read + current constellation
for k in range(K):
    beta_k = beta_schedule(k, K)  # annealing: low -> high temperature

    # Hopfield gradient = softmax attention over memory bank
    scores = beta_k * (x @ memory_bank.T)
    weights = F.softmax(scores, dim=-1)
    energy_grad = weights @ memory_bank - x

    # Langevin step
    noise = math.sqrt(2 * eta / beta_k) * torch.randn_like(x)
    x = x + eta * energy_grad + noise

    # Proximal sparsity (Requirement 3)
    x = soft_threshold(x, lambda_k)
```

**Implementation notes:**
- Start with K = 2-3 steps. Even 2 steps provide iterative refinement.
- Memory bank size |M| = 64-256 atoms. Modern Hopfield capacity is exponential in d, so this should suffice.
- The memory bank can be shared across blocks or per-block.
- Beta annealing schedule: linear or cosine from beta_min to beta_max within the K steps.
- The Langevin steps operate on the constellation (mag, phase) representation, not raw embeddings.

**Validation:** The Langevin path should contribute measurable gradient magnitude. Compare gradient norms from the Langevin path vs. the MLP path — they should be comparable, not 100:1 as in v12.2.

---

## Requirement 3: Proximal Sparsity (Soft-Thresholding)

**Priority: HIGH — enforces the "few dots" constraint**

Apply soft-thresholding to constellation magnitudes:

```python
def soft_threshold(constellation, lambda_thresh):
    """Proximal operator for L1: enforces spectral sparsity."""
    new_mag = F.relu(constellation.mag.abs() - lambda_thresh) * torch.sign(constellation.mag)
    return Constellation(mag=new_mag, phase=constellation.phase)
```

**Implementation notes:**
- Apply after each Langevin step (within the settling loop)
- Also apply after the final constellation update (end of each block)
- Threshold lambda can be:
  - Fixed hyperparameter (simplest, start here)
  - Learned per block (adaptive sparsity depth)
  - Annealed during training (dense → sparse curriculum)
- Target sparsity: 6/17 modes active per subbundle (as in v12.5 constellation design) = 35% active
- Monitor actual sparsity during training to verify the threshold is effective

**Validation:** After training, constellation magnitudes should be bimodal — a cluster near zero (inactive modes) and a cluster of significant values (active modes). If all modes have similar magnitude, the threshold is too low.

---

## Requirement 4: Remove Deep Supervision

**Priority: MEDIUM — unblocks depth scaling**

Train only on the final block's output:

```python
# Current (V13): loss at blocks 2, 4, 6, 8 with weights 0.25, 0.5, 0.75, 1.0
# Required (V14): loss only at final block
logits = decode(constellation_final)
loss = F.cross_entropy(logits, targets)
```

**Rationale:** Deep supervision forces the shared decoder to produce good logits from shallow representations, preventing progressive geometric refinement through blocks. The theory says constellations evolve along geodesics — forcing good logits at block 2 is like requiring a geodesic to reach its destination at 25% of its arc length.

**Alternative (if training becomes unstable):** Use very small weights for early blocks — 0.01, 0.02, 0.05, 1.0 — so they provide gradient signal without dominating.

---

## Requirement 5: Separate Magnitude and Phase Normalization

**Priority: MEDIUM — fixes geometric correctness**

```python
# Current (V13): joint RMSNorm on [mag, phase] concatenation
x = c.to_flat()  # [mag, phase]
rms = (x ** 2).mean(-1, keepdim=True).sqrt()
x = x / rms * self.scale

# Required (V14): separate normalization respecting geometry
mag_rms = (c.mag ** 2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
normed_mag = c.mag / mag_rms * self.mag_scale

# Phase lives on S^1 — normalize by centering, not by dividing
# Or simply skip phase normalization (phases are bounded by definition)
normed_phase = c.phase  # no normalization needed for phases
```

**Rationale:** Phase has circular topology (pi and -pi are the same angle). RMSNorm divides by a scalar, which is not a valid operation on S^1. Magnitudes and phases have different scales and semantics — joint normalization conflates them.

---

## Requirement 6: Per-Mode or Per-Token Gates

**Priority: LOW — improves expressivity**

```python
# Current (V13): single scalar gate per block
self.gate = nn.Parameter(torch.tensor(-2.0))  # one number

# Required (V14): per-mode gate
self.gate = nn.Parameter(torch.full((num_modes,), -2.0))  # one per mode
# Or: per-token gate (computed from constellation)
gate_values = self.gate_net(constellation.to_flat())  # [batch, seq, num_modes]
```

Different modes should be able to update at different rates. A single scalar gate means "update everything by the same fraction" — too coarse for a 136-mode representation.

---

## Requirement 7: Reduce Learning Rate

**Priority: LOW — but easy win**

Reduce from 3e-3 to 1e-3. V13 showed instability during the hold phase (loss regression at steps 4000-5000). The cosine decay phase showed more consistent improvement, suggesting the peak LR is too high.

Also consider shortening the hold phase (warmup to 750, hold to 2000, decay from 2000-10000) to spend less time at the unstable peak.

---

## Implementation Order

1. **Content-dependent transport** (Req 1) — most impactful, enables selective retrieval
2. **Proximal sparsity** (Req 3) — enforces the sparse prior, lightweight to implement
3. **Remove deep supervision** (Req 4) — trivial code change, unblocks depth
4. **Separate mag/phase norm** (Req 5) — straightforward, fixes geometric bug
5. **Langevin settling** (Req 2) — most complex to implement, but high payoff
6. **Per-mode gates** (Req 6) — small expressivity gain
7. **LR adjustment** (Req 7) — tune after architecture changes

---

## Success Criteria

| Metric | V13 | Target | GPT-Nano Baseline |
|---|---|---|---|
| Val BPC | 3.19 | < 2.30 | 2.18 |
| Val Accuracy | 34.5% | > 50% | 55.8% |
| Steps to plateau | ~500 | > 3000 | N/A |
| Fiber gradient ratio | ~0.01x MLP | > 0.1x MLP | N/A |
| Spectral sparsity | None (all modes active) | ~35% active | N/A |
| Speed | ~2s/step | < 5s/step | ~0.3s/step |

The primary goal is not to beat GPT-Nano — it is to demonstrate that the spectral geometric machinery **contributes measurable value** beyond what SSM+MLP provides alone. The v12.2 ablation is the bar to clear: the full model must outperform the SSM+MLP ablation, not just match it.

---

## Ablation Plan

After implementing, repeat the v12.2 ablation structure:

| Model | Description | Purpose |
|---|---|---|
| A: Full V14 | All components | Baseline |
| B: No Wilson line | Replace content-dependent gate with fixed decay | Isolate transport contribution |
| C: No Langevin | Replace settling with single MLP pass | Isolate retrieval contribution |
| D: No sparsity | Remove proximal threshold | Isolate sparsity contribution |
| E: SSM+MLP only | Strip all spectral machinery | The bar to clear |

**If E matches A again, the spectral machinery is still decorative.** Each of B, C, D should be measurably worse than A for the corresponding mechanism to justify its existence.
