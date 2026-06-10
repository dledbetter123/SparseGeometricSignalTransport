# Handoff: V14 Implementation & Ablation Study

**Date**: 2026-03-31
**Context**: Full V14 design, implementation, debugging, and ablation study creation in one session.

---

## What Happened This Session

### 1. V13 Diagnosis

Analyzed `v12/architecture_v13.ipynb` to understand why loss plateaus at CE ~2.17 after ~500 steps (BPC 3.19, 34.5% accuracy). GPT-Nano comparison: BPC 2.18, 55.8% accuracy.

**Root causes identified (6):**
1. ModeFiber is a pure linear recurrence (fixed scalar EMA, no content dependence, no phase rotation — 1,088 params total across 8 blocks)
2. ConstellationUpdate gate initializes suppressive (sigmoid(-2) = 0.12, single scalar)
3. ConstellationNorm conflates magnitude and phase (joint RMSNorm breaks S^1 topology)
4. Deep supervision biases toward shallow solutions (loss at blocks 2,4,6,8)
5. No cross-mode interaction in the fiber (136 modes processed independently)
6. LR too high (3e-3 causes regression at steps 4000-5000)

**Core insight**: The theory prescribes three load-bearing mechanisms — none were implemented:
- Wilson line (content-dependent gauge transport) — MISSING, replaced by fixed EMA
- Langevin settling (Hopfield energy descent) — MISSING, replaced by one-shot MLP
- Proximal sparsity (soft-thresholding) — MISSING entirely

### 2. V14 Design & Implementation

Created `v14/gen_notebook_v14.py` → `v14/architecture_v14.ipynb` (24 cells, 15 code cells).

**Architecture (1.80M params):**

| Component | Params | % | What it does |
|---|---|---|---|
| Embedding | 17.7K | 1% | Token → (mag, phase) constellation |
| MagPhaseNorm | 1.1K | 0.1% | RMSNorm on magnitudes only (phases untouched) |
| WilsonFiber | 840K | 47% | Content-dependent complex EMA (gauge transport) |
| LangevinSettler | 854K | 48% | Hopfield memory bank + iterative settling + proximal sparsity |
| Decoder | 83K | 5% | irfft → LayerNorm → MLP → logits |

**Key classes:**
- `Constellation` — (mag, phase) tuple, unchanged from V13
- `MagPhaseNorm` — RMSNorm on magnitudes only; phases live on S^1, left untouched
- `ConstellationEmbedding` — unchanged from V13
- `complex_parallel_scan(a_re, a_im, b_re, b_im)` — NEW: complex associative scan for Wilson line
- `WilsonFiber` — content-dependent complex recurrence via wilson_proj MLP
- `LangevinSettler` — K=2 Hopfield energy descent steps + proximal threshold + ctx skip
- `V14Block` — norm → fiber → settler → residual
- `V14Model` — no deep supervision (loss only at final block)
- `ConstellationDecoder` — unchanged from V13

**Training config changes from V13:**
- LR: 1e-3 (was 3e-3)
- Hold steps: 1250 (was 3250)
- Per-mode gates (was single scalar)
- No deep supervision

### 3. Gradient Debugging (Critical Bug Found & Fixed)

**Bug**: Wilson fiber received exactly 0 gradient in the full model forward/backward.

**Root cause**: The only gradient path from loss to fiber passes through the Langevin settler:
```
loss → delta → (x - x0) → Langevin loop → ctx → msg_proj → messages → fiber
```
Six multiplicative attenuation factors compound: msg_proj (std=0.01) × F.normalize Jacobian × softmax (1/256 uniform) × memory (std=0.02) × eta (0.3) × gate (0.12) ≈ 2.8×10^-9, below float32 precision.

**Fix**: Add `ctx` directly to the output delta, creating a skip connection that bypasses the Langevin chain:
```python
# Before (broken):
delta = x - x0
# After (fixed):
delta = (x - x0) + ctx
```

**Result**: Fiber/settler gradient ratio = 1.89 (healthy gradients across all 8 blocks).

**Full debugging log**: `v14/gradient_debugging_260331_073500.md`

### 4. "Is This MLP Fakeness?" Analysis

David asked whether V14's improvements come from genuine geometry or hidden feedforward paths. Analysis:

**Two paths from fiber to output:**
- **Path A (ctx skip)**: messages → msg_proj (Linear) → add to delta → gate → output. This is a linear readout of the SSM. If this dominates, V14 reduces to "complex Mamba."
- **Path B (Langevin)**: messages → ctx biases query → Hopfield lookup → iterative settling → proximal threshold → (x - x0). This is the theoretically novel path.

**What's genuinely geometric (not MLP):**
- Complex phase rotation in recurrence (interference patterns, phase-sensitive selection)
- Parseval inner product (measures magnitude overlap AND phase alignment simultaneously)
- Proximal sparsity (if working: bimodal magnitude distribution)

**What's potentially decorative:**
- wilson_proj MLP (parameterizes recurrence, similar to Mamba's SelectiveSSM)
- Memory bank (if Langevin doesn't contribute)
- irfft decoder (fixed linear transform)

**Key diagnostics to check after training:**
1. `|θ|` > 0.01 → Wilson line is active (learning phase rotations)
2. `‖x-x₀‖ / ‖ctx‖` > 0.5 → Langevin is doing real work
3. Bimodal magnitude distribution → sparsity is enforcing "few dots"
4. Memory entropy < max → Hopfield using specific attractors

### 5. Ablation Study

Created `v14/gen_ablation_v14.py` → `v14/ablation_v14.ipynb` (25 cells, 13 code cells).

**5 models + GPT-Nano, 5000 steps each:**

| ID | Model | What's removed | Params | Tests |
|---|---|---|---|---|
| A | Full V14 | Nothing | 1.80M | Baseline |
| B | No Wilson | wilson_proj → fixed decay, θ=0 | 0.96M | Content-dependent transport |
| C | No Langevin | Settling loop removed, delta=ctx only | 1.24M | Iterative Hopfield settling |
| D | No Sparsity | Proximal threshold removed | 1.80M | Spectral parsimony |
| E | SSM+MLP | Real EMA + MLP (param-matched) | 1.80M | **The bar to clear** |

**Verdict interpretation:**
- A > E → geometry earns its keep
- A ≈ E → geometry is decorative (V12.2 repeats)
- A > C ≈ E → V14 is "complex Mamba" (Wilson line works, Langevin decorative)
- A > B ≈ E → Wilson line is the key mechanism
- A ≈ D → sparsity doesn't matter yet

---

## File Inventory (v14/)

| File | Description |
|---|---|
| `gen_notebook_v14.py` | V14 notebook generator (self-contained) |
| `architecture_v14.ipynb` | V14 training notebook (24 cells) |
| `gen_ablation_v14.py` | Ablation study notebook generator |
| `ablation_v14.ipynb` | Ablation study notebook (25 cells) |
| `README.md` | V14 overview and forward-reverse loop diagram |
| `v13_diagnosis.md` | 6 root causes of V13 plateau |
| `mathematical_foundations.md` | Five axioms, Fourier duality, theorems |
| `theory_vs_implementation_gap.md` | Three missing mechanisms with pseudocode |
| `scaling_analysis.md` | O(n) vs O(n^2) complexity tables, Hopfield retrieval |
| `version_history.md` | V1-V13 lessons learned |
| `v14_design_requirements.md` | 7 requirements, success criteria, ablation plan |
| `gradient_debugging_260331_073500.md` | Full gradient bug diagnosis and fix |

---

## What Needs to Happen Next

1. **Run `architecture_v14.ipynb`** — 10K steps, ~58 min. Check if loss keeps improving past V13's plateau at step 500.

2. **Run `ablation_v14.ipynb`** — 5K steps × 6 models, ~2-3 hours. This is the definitive test.

3. **Read the diagnostics cells** after training completes. The critical questions:
   - Is |θ| non-zero? (Wilson line active?)
   - Is ‖x-x₀‖/‖ctx‖ balanced? (Langevin contributing?)
   - Is magnitude distribution bimodal? (Sparsity working?)
   - Does A beat E? (Geometry earning its keep?)

4. **If A ≈ E** (geometry decorative again): The constellation representation and Fourier structure don't add value. Time to compile a new architecture as David mentioned.

5. **If A > E**: Identify which mechanism(s) contribute via B, C, D ablations. Double down on what works.

---

## Key Design Decisions to Remember

- **ctx skip connection is load-bearing for gradient flow.** Without it, the fiber gets 0 gradient because the Langevin chain attenuates below float32. If anyone modifies the settler, the `delta = (x - x0) + ctx` line MUST stay.

- **msg_proj must NOT be zero-initialized.** Zero init blocks all gradients to the fiber. Current init: `nn.init.normal_(weight, std=0.01)`.

- **wilson_proj last layer IS zero-initialized** (intentionally). This makes V14 start with V13's behavior (real decay=0.5, no phase rotation). Content-dependent transport emerges during training.

- **Memory bank init is small (std=0.02)** so initial Langevin settling barely moves the state. The gate (sigmoid(-2)=0.12) further limits early impact. Both grow as training progresses.

- **Proximal threshold only on last Langevin step** (V12.2 lesson: every-step threshold prevents exploration).

- **Phase wrapping** (`torch.remainder(phase + pi, 2pi) - pi`) applied after proximal step to respect S^1 topology.

---

## Mathematical Context (Brief)

The architecture implements the forward-reverse loop from the theory:

```
Sparse spectral (constellation)
    ↓
Wilson fiber: h[t] = z_t·h[t-1] + c_t     (gauge transport)
    where z_t = decay(c_t)·exp(i·θ(c_t))   (content-dependent)
    ↓
Parseval read: Re(h·conj(c))               (spectral metric)
    ↓
Langevin settling: K steps of               (Hopfield energy descent)
    x ← x + η·(softmax(β·q@M^T)@M - x) + noise
    ↓
Proximal: sign(mag)·relu(|mag|-λ)          (spectral sparsity)
    ↓
Sparse spectral (updated constellation)
```

The five axioms (fiber bundle topology, spectral gauge-covariant transport, dynamic memory bank, Langevin-Hopfield energy descent, proximal sparsity) are documented in `mathematical_foundations.md`.

The version history (v1-v13 lessons) is in `version_history.md`. The recurring pattern: every version that worked had content-dependent selectivity; every version that stripped it out plateaued.
