# V13 Diagnosis: Why Loss Plateaus Early

## Training Behavior

V13 starts at CE 4.20 (step 0), drops quickly to ~2.50 by step 500, then crawls from 2.50 to 2.17 over the remaining 9,500 steps. Validation BPC settles around 3.19 with 34.5% accuracy. For comparison, GPT-Nano reached BPC 2.245 (54% accuracy) by step 2,000.

The train/val gap is negligible throughout (2.17 vs 2.21 at end), confirming **underfitting, not overfitting**. The model saturates bigram-level statistics quickly (~step 500) but cannot learn longer-range dependencies.

---

## Root Cause 1: The ModeFiber Is a Pure Linear Recurrence

The parallel scan computes:

```
h[t] = alpha * h[t-1] + x[t]
```

This is a fixed exponential moving average. Each of the 136 modes has a single learned scalar decay `alpha`. There is no content-dependent gating — a token at position t gets a fixed exponentially-decayed average of all past tokens. It cannot selectively attend to or retrieve a specific earlier token.

The Parseval inner product `Re(h * conj(c))` is bilinear (h is linear in past inputs, multiplied by current input), but it cannot express selective retrieval. The entire cross-token communication channel is controlled by just **1,088 parameters** across 8 blocks (136 decay rates per block).

**Why this matters:** The theory prescribes transport via `exp(-Dw^2 - iw * integral_gamma A)` — a heat kernel modulated by a Wilson line (accumulated gauge connection encoding contextual history). The Wilson line `exp(-iw * integral A)` provides content-dependent phase rotation during transport: contextual curvature. The EMA has no phase rotation, no content dependence — it is a leaky integrator, not a gauge connection.

**Historical precedent:** V3's GRU outperformed everything until v7.3 because the GRU was a content-dependent state accumulator — a de facto Wilson line. V13's parallel scan is content-independent. This is the same failure mode diagnosed in v10 ("the manifold is fake") and v12.2 ("transport is mode-wise/linear").

---

## Root Cause 2: ConstellationUpdate Gate Initializes Suppressive

```python
self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2.0) = 0.119
```

Only ~12% of the MLP's delta passes through the gate. Combined with zero-initialized last linear layer:

```python
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

Each block starts as near-identity. With only 8 blocks and a single scalar gate per block (not per-mode, not per-token), the network has very limited capacity to escape the identity neighborhood. If the gate does not open sufficiently during training, blocks remain near-identity and the model cannot build deep representations.

---

## Root Cause 3: ConstellationNorm Conflates Magnitude and Phase

```python
def forward(self, c):
    x = c.to_flat()  # [mag, phase] concatenated
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    x = x / rms * self.scale
```

Phase values (radians, range [-pi, pi]) and magnitude values (unbounded) are concatenated and jointly normalized. Since phase values are typically larger in absolute value than magnitudes early in training, the RMS is dominated by the phase component, distorting magnitudes.

Worse: phase has circular topology. Dividing by a scalar breaks the angular structure (pi and -pi should be equivalent, but RMSNorm treats them as having different "energy").

---

## Root Cause 4: Deep Supervision Biases Toward Shallow Solutions

```python
weights = torch.zeros(cfg.n_blocks)
for i in range(cfg.n_blocks):
    if (i + 1) % 2 == 0:
        weights[i] = (i + 1) / cfg.n_blocks  # 0.25, 0.5, 0.75
weights[-1] = 1.0  # block 8 gets weight 1.0
```

Loss is computed at blocks 2, 4, 6, 8 with weights 0.25, 0.5, 0.75, 1.0, then normalized. Block 2's loss (after only 2 blocks of near-identity processing) contributes ~10% of the gradient.

The shared decoder must produce decent logits from minimally-processed representations AND from deeply-processed ones. This tension prevents the decoder from specializing for deep features and biases the entire system toward shallow solutions. The theory says constellations should evolve along geodesics through blocks — forcing good logits at block 2 prevents this progressive refinement.

---

## Root Cause 5: No Cross-Mode Interaction in the Fiber

All 136 modes are processed completely independently through the fiber. The ConstellationUpdate MLP is the only place where cross-mode interaction can happen:

```python
nn.Linear(3 * M, hd),   # 408 -> 384
nn.SiLU(),
nn.Linear(hd, 2 * M),   # 384 -> 272
```

At 384 hidden units for 136 modes (~2.8 hidden units per mode), this is a severe bottleneck. 95.3% of total parameters (2,094,216) live in the MLPs, yet the information they operate on (Parseval inner products from the linear EMA) is impoverished.

---

## Root Cause 6: Learning Rate Instability

Learning rate = 3e-3 with warmup to 750 steps, hold until step 4000, then cosine decay to 3e-4. The loss actually **regresses** at steps 4500-5000 (val CE jumps from 2.271 to 2.356), then slowly recovers. The cosine decay phase (steps 4000-10000) shows more consistent improvement, suggesting 3e-3 is at the edge of stability for this architecture.

---

## Summary: The Plateau Mechanism

The model's cross-token communication is fundamentally limited: a single-mode linear EMA cannot selectively retrieve past information. The only expressivity comes from the per-block update MLP, but it operates on impoverished inputs (exponentially-blurred inner products) and is further suppressed by the near-zero gate. Deep supervision with a shared decoder pushes toward shallow solutions. The result: quick learning of unigram/bigram statistics (CE ~2.5 by step 500), then plateau where longer-range dependencies would need to kick in.
