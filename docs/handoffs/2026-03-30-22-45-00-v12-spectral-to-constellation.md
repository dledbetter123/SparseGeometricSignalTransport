# Handoff: V12 Spectral Architecture Evolution (2026-03-30)

## Session Arc

Started with V12.3 (spectral-native, no intra-block irfft) smoke test and training.
Ended at V12.5 (sparse constellation architecture). This session was a conceptual
progression from "do computation in spectral domain" to "tokens ARE sparse spectral
patterns and shared modes ARE the connections."

## What Was Built and Tested

### V12.3 — Spectral-Native (already existed, tested this session)
- **File**: `v12/gen_notebook_v12_3.py` → `v12/architecture_v12_3.ipynb`
- SSM and MLP operate on spectral coordinates (real+imag = 272-dim real)
- No irfft inside blocks. irfft only at decoder.
- **Result**: BPC 2.584 at step 500, 347ms/step (B=16, T=512)
- **Problem identified**: SSM/MLP don't respect manifold structure. MLP dominates
  gradient flow (0.049 avg) vs SpectralInteraction (0.0004). Optimizer routes
  everything through the generic MLP, ignoring the geometric components.

### V12.4 — Geometric Transport (built this session)
- **File**: `v12/gen_notebook_v12_4.py` → `v12/architecture_v12_4.ipynb`
- SSM reads magnitudes only (136-dim, not 272)
- SpectralInteraction: context-dependent mixing via low-rank perturbation W(ctx) = W_base + U(ctx)@V(ctx), rank=2
- ComplexMLP with modReLU (nonlinear on magnitude, preserves phase)
- Cross-subbundle: fixed coupling matrix + context gates participation
- **Params**: 2,182,245 (under GPT-Nano's 2.4M)
- **Gradient balance**: Interaction (0.020) vs ComplexMLP (0.018) — much more balanced than V12.3
- **Critical bug found**: Initial implementation materialized full (B,T,K,17,17) complex matrix for U@V, causing 7+ minute step times. Fixed with factored form (W_base*x + U*(V*x)) but still ~1.6s/step due to complex einsum overhead.
- **David's feedback**: "This strict spectral analogue is slowing us. There has to be a unified architecture..."

### V12.5 — Sparse Constellation (built this session, NOT FULLY WORKING)
- **File**: `v12/gen_notebook_v12_5.py` → `v12/architecture_v12_5.ipynb`
- Radical simplification based on David's core insight
- **Params**: 1,175,229
- **Architecture**:
  - Tokens = sparse constellations (mag + phase + binary mask, 6/17 active per subbundle)
  - ModeFiber: per-mode causal accumulator via parallel scan
  - ConstellationUpdate: simple MLP on (active_mag, active_phase, messages) → delta
  - Sparsify (top-k) → new topology
- **Status**: Forward/backward work correctly, all 68 params have gradients
- **BLOCKING ISSUE**: 13 seconds/step at full size (B=16, T=512). The parallel scan
  over B×M×sd = 16×136×32 = 69,632 independent streams has too much overhead.
  Each scan does log2(512)=9 iterations of cat+mul+add on 69K-length batch dimension.

## Key Conceptual Decisions Made

### 1. Tokens are sparse spectral patterns (David's words)
"A few dots on the fourier space. Mathematically, it's how many unique connections
we can make." C(17,6)^8 ≈ 5.7×10^32 possible support patterns for 65 characters.
The surplus is room for constellations to shift during processing to reflect context.

### 2. Shared modes ARE connections
No attention mechanism needed. If token A and B both have mode m active, they're
geometrically connected through m. The mode carries a causal state that tokens
read from and write to.

### 3. The manifold is explicit
- Topology = which modes are active (discrete, changes at sparsification)
- Metric = values at shared modes (continuous, Parseval inner product)
- Curvature = mode activation/deactivation dynamics
- Geodesic = constellation evolution through blocks

### 4. The network learns three things
1. Which dots define each token (embedding)
2. What happens when dots overlap (update rule at shared modes)
3. When to move dots (how constellations shift through blocks)

### 5. What was removed and why
- SSM over full representation → mode fibers handle sequence
- Complex arithmetic → mag + phase as real vectors
- Transport kernels → mode gathering IS transport
- SpectralInteraction matrices → mode sharing replaces them
- MLP as primary computation → update rule is small

## What Remains To Do

### Immediate: Fix V12.5 Speed

The parallel scan approach is correct but the implementation is wrong.
69K independent scans is too many. Options:

**Option A: Reduce mode_state_dim to 1.** Each mode carries a single scalar state.
Deposit = magnitude at that mode. Scan over B×M = 2,176 streams (manageable).
The read_proj becomes identity. Messages are just "what was deposited here before."
This is the simplest version and should be tried first.

**Option B: Scan per-subbundle, not per-mode.** Group modes within each subbundle,
scan over B×K = 128 streams with state dim = k_active × sd. Preserves richer
per-mode information while being 540x fewer scans.

**Option C: Replace parallel scan with cumulative sum + exponential decay.**
For constant alpha (which ours is — alpha is learned but input-independent):
```python
# h[t] = alpha * h[t-1] + x[t]
# = sum_{s<=t} alpha^(t-s) * x[s]
# This can be computed via cumsum of alpha-weighted inputs
log_alpha = torch.log(alpha)
weights = torch.exp(log_alpha.cumsum(dim=T_dim))
h = (x / weights).cumsum(dim=T_dim) * weights
```
This is O(T) and fully parallel. No scan iterations needed.

### After speed fix: Train V12.5 and Evaluate

1. Run V12.5 for 10K steps on Tiny Shakespeare (seq_len=512, batch=16)
2. Compare against GPT-Nano baseline
3. Key diagnostics to watch:
   - `changed`: fraction of modes that switch per block (want 10-30%)
   - `shared_adj`: shared modes between adjacent tokens
   - Mode fiber decay values (how far back does each mode remember?)
   - Whether the manifold topology actually restructures across blocks

### Architecture Refinements (if V12.5 works)

1. **Informed sparsification**: Currently top-k by magnitude. Could use messages
   to inform which modes to keep (modes that received strong messages survive).
2. **Multi-head mode fibers**: Multiple state dimensions per mode for richer messages.
3. **Cross-subbundle mode interaction**: Currently subbundles are independent.
   Modes at the same frequency across subbundles could interact.

## File Inventory

```
v12/
  gen_notebook_v12_3.py    # Spectral-native (SSM+MLP on spectral coords)
  gen_notebook_v12_4.py    # Geometric transport (context-dependent mixing, ComplexMLP)
  gen_notebook_v12_5.py    # Sparse constellation (mode fibers, no SSM/MLP)
  architecture_v12_3.ipynb # Generated notebook, was trained (BPC 2.584 at step 500)
  architecture_v12_4.ipynb # Generated notebook, trained but hit 7min/step bug
  architecture_v12_5.ipynb # Generated notebook, NOT YET TRAINED (speed issue)
  v12_1.py                 # V12.2 fixes baked in (rfft, tanh*1.0, proximal-at-end)
  gen_notebook.py           # V12.2 notebook generator
  README_ABLATION_SSM.md   # V12.1 ablation results + 3 bug diagnoses

memory/
  project_v12_spectral_diagnosis.md  # V12.2 diagnosis: why spectral was passive
```

## David's Core Vision (preserved across all versions)

"Tokens live inside the brain as spectral activation maps and the submanifold
connection between the relevant portions of the sparse token allow for wide
ranging meaning."

V12.5 is the closest implementation to this vision: tokens ARE spectral patterns,
connections ARE shared modes, and the manifold restructures through sparsification.
The remaining work is making it fast enough to train.
