# V12.1 Session Summary — 2026-03-29

## Goal
Beat/match GPT-Nano (BPC 2.18, Acc 55.8%, 2.4M params) on character-level Tiny Shakespeare using the V12 spectral sparsity architecture.

## Starting Point
- **V12**: 2,093,637 params, 4 blocks, BPC 2.27, Acc 53.5%, 498.6ms/step, 5000 steps
- **GPT-Nano**: 2,412,544 params, 12 layers (n_embd=128, n_head=4), BPC 2.18, Acc 55.8%, 22.6ms/step
- Gap: 0.09 BPC, 2.3% accuracy. V12 is smaller but 22x slower.

## Root Cause Analysis
1. **No MLP** — V12 blocks had SSM + transport + memory + settling but NO nonlinear channel mixing (GPT has attention + MLP)
2. **Insufficient depth** — 4 blocks vs 12 layers
3. **Expensive settler** — 5 Langevin steps with FFT+topk+IFFT each = main speed bottleneck
4. **Bloated memory routing** — 8 per-subbundle MLP routers learning what dot-product already computes

## V12.1 Architecture (architecture_v12_1.ipynb)

### Changes from V12
| Change | V12 | V12.1 | Justification |
|--------|-----|-------|---------------|
| Channel mixing | None | SpatialMLP(256→384→256) per block | GPT's FFN equivalent |
| Langevin steps | 5 | 2 | Ramsauer 2021: Hopfield ≈ softmax attention |
| Memory routing | 8 MLP routers | Direct dot-product Hopfield | Saves ~130K params/block |
| Blocks | 4 | 6 | More representational depth |
| context_dim | 256 | 128 | Saves params for MLP + blocks |
| spectral_sparsity | 8/32 | 10/32 | V12 measured ~60% sparsity, model wants more modes |
| Deep supervision | All 4 blocks | Blocks 2, 4, 6 only | Halves decoder overhead |

### V12 Non-Negotiables (ALL preserved)
1. Sparse in spectral, dense only transiently in spatial
2. Field reconstruction IS the IFFT
3. Langevin starts from the reconstructed (IFFT) field
4. Context warps the spectral metric (D, A context-dependent)
5. Spectral proximal at every Langevin step
6. Subbundles are independent spectral channels
7. No pairwise token attention

### Block Flow
```
sparse spectral → IFFT → SSM context → spectral transport →
IFFT (field reconstruction) → 2-step Hopfield settling (spectral proximal each) →
LayerNorm → SpatialMLP → gated residual → FFT + re-sparsify → sparse spectral
```

### Parameters: 2,345,031 (under GPT's 2,412,544)
- SpatialMLP (6): 1,183,488 (50.5%)
- Context Accum (6): 592,128 (25.3%)
- Transport (6): 396,288 (16.9%)
- Memory (6): 49,152 (2.1%)
- Decoder+norms: 83,009 (3.5%)
- Rest: ~41K

### Speed: ~295ms/step (1.7x faster than V12's 499ms, still 6.1x slower than GPT's 48ms)
- Pairwise topk confirmed 49x faster than torch.topk on MPS
- Main cost is still FFT+topk+IFFT at spectral proximal calls (6 blocks × 3 calls each = 18 total)

## Run 1 Results (lr=5e-3, cosine to 0, warmup 750)
```
Step     0 | Val BPC: 6.04 | Val Acc:  1.6% | Sp: 22.9%
Step   500 | Val BPC: 2.68 | Val Acc: 45.5% | Sp: 25.2%
Step  1000 | Val BPC: 2.47 | Val Acc: 49.5% | Sp: 37.0%
Step  1500 | Val BPC: 2.38 | Val Acc: 51.4% | Sp: 43.4%
Step  2000 | Val BPC: 2.36 | Val Acc: 52.7% | Sp: 46.9%
Step  2500 | Val BPC: 2.33 | Val Acc: 53.2% | Sp: 49.3%
Step  3000 | Val BPC: 2.30 | Val Acc: 53.4% | Sp: 50.5%
```

### Observations
- **Accuracy plateauing** after step 2000 (+0.2% per 500 steps vs +4% early on)
- **Val CE still declining** (1.632→1.597) — model improving but not flipping predictions
- **Emergent sparsity increase** (22.9%→50.5%) — model self-organizing toward conjugate-symmetric spectra compatible with .real projection at every IFFT. This is the architecture's inductive bias working.
- At step 3000, already matching V12's final BPC (2.27 at step 3500)

### Diagnosis: LR Schedule
- Peak LR 5e-3 too conservative (V12 used 7e-3 successfully)
- Cosine decay started immediately after warmup at step 750 (only 7.5% into 10K training)
- Cosine decays to 0, wasting final 2K steps

## Pending Change: Trapezoidal LR Schedule (NOT YET RUN)

```
learning_rate: 7e-3      (was 5e-3)
min_lr: 1e-4             (was 0)
lr_hold_steps: 3250      (NEW — hold peak until step 4000)
```

Schedule:
- Steps 0-750: warmup 0 → 7e-3
- Steps 750-4000: hold at 7e-3 (40% of training at full power)
- Steps 4000-10000: cosine 7e-3 → 1e-4 (refinement phase)

## Files
- `v12/architecture_v12_1.ipynb` — Main notebook (self-contained, Run All)
- `v12/v12_1.py` — Architecture module (standalone, importable)
- `v12/train_v12_1.py` — Training script (CLI alternative to notebook)
- `v12/SESSION_SUMMARY_v12_1.md` — This file

## What to Do Next
1. **Run the notebook with the trapezoidal LR** — restart from config cell, Run All
2. **If accuracy still plateaus**, consider:
   - Increasing max_steps to 15K or 20K (V12.1 may just need more steps)
   - Reducing dropout (0.1 → 0.05) if train-val gap is small
   - Increasing mlp_ratio (1.5 → 2.0) for more mixing capacity
   - Trying warmup restart (cosine with warm restarts)
3. **If BPC is close but accuracy lags**, the model is improving confidence on already-correct predictions rather than flipping wrong ones — may need architectural change (more depth, larger context)
4. **Track all changes** in memory file `project_v12_1_changelog.md`

## Key Insight: Emergent Sparsity
The spectral sparsity increasing from 22.9% to 50.5% without being directly optimized is a strong signal. The architecture's .real projection at every IFFT creates pressure toward conjugate-symmetric spectra. As the model learns this symmetry, zeros in the sparse representation stay closer to zero through round trips. The spectral transport also learns frequency-selective damping (exp(-D*w^2)) that concentrates energy at fewer modes. Target ceiling is 68.75% (= 1 - 10/32).
