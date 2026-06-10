# V19 Design Proposal: Retrospective and Forward Path

**Purpose:** A systematic retrospective of architectures V5–V18, identifying load-bearing mechanisms, dropped decorative machinery, and a proposed V19 architecture that incorporates the thesis's future-work directions (Ch. 8.3) while directly addressing the state-capacity gap identified in the V18 audit.

---

## Part I — Retrospective: What Was Kept vs Dropped (V5–V18)

### 1.1 The version timeline in one table

| Ver | Headline change | Kept forward | Dropped (and why) | Fate |
|-----|-----------------|--------------|-------------------|------|
| V5 | Gauge + sparse sections + Hopfield + Langevin | Fiber bundle framework, spectral transport kernel, proximal sparsity | — (baseline) | ~20% synthetic |
| V6 | QK $\to$ sparse subspace routing | Framework | QK attention (too heavy at this scale) | ~20%, regression |
| V7 | Sequence Langevin, three additive forces, deep supervision | Per-subbundle decomposition, additive force structure | Pure sequence Langevin (gradient dies through 21 chained steps) | 36.5% synthetic |
| V8 | First real text; scale geometric components | Per-subbundle dictionaries, deeper manifold | Single global dictionary, shallow manifold | **45% / 2.65 BPC**, hits "45% wall" |
| V9 | Sparse per-subbundle attention inside Langevin | Lightweight attention as gauge connection | Full fiber QK | Still 45% — diagnosis "routing isn't the bottleneck" |
| V10 | "The manifold is fake" | Insight: $q_t$ must be context-dependent | Fixed positional embedding as manifold coordinate | Diagnostic, no code |
| V11 | Sparse events + Alcubierre field reconstruction | Forward–reverse loop as fundamental, proximal at every step, proper Langevin init from diffused field | Dense processing between blocks; "diffusion as smoothing" reading | Theoretical pivot |
| V12 | **Spectral sparsity via Donoho–Stark** | Spectral transport kernel $\exp(-D\omega^2 - iA\omega)$, context-dependent $D_k(q), A_k(q)$, forward–reverse spectral cycle, per-subbundle SSM | Spatial sparsity (v11); "diffusion as smoothing" semantics | **V12.1: 2.302 BPC / 53.4%** — best in project |
| V12.2 | Ablation | SSM + MLP as baseline | — | **SSM+MLP = 2.267 BPC** beats full V12.1 at 1/4.7 the compute. Spectral machinery adds zero. |
| V13 | Native complex, pure linear EMA fiber | Complex-valued state representation | Content-dependent state update, ConstellationUpdate MLP | Plateau at CE 2.17, BPC ~3.19 — content-independence relearned as cause |
| V14 | Wilson line + Hopfield + per-token FFN | **Content-dependent fiber** (restored), Hopfield as primary settler, per-token FFN as non-negotiable | Pure linear EMA, aggressive hard thresholding | WikiText-2 PPL 646 vs GPT 211 — geometry > SSM+MLP but gap to attention persists |
| V15 | Parseval spectral filter replaces Langevin settler | Energy-bounded filter $|W| \leq 1$, one-pass settling | Iterative Langevin over memory bank (too slow and narrow) | Misimplemented; becomes v16's baseline |
| V16 | **irfft is the mixing** (not a decoder trick) | Wilson fiber + Parseval filter + irfft round-trip + FFN, O($s$) spectral transport | — | **PPL 275 on WikiText-103**, still improving, no plateau |
| V16b | Position-axis FFT (SPECTRE-style) + Gaussian clouds + local conv | SPECTRE safeguards (global gate, positional phase injection), Gaussian cloud embedding (mag, phase, log_var) | Naive non-causal position FFT (leaked future tokens $\to$ PPL 36 cheating) | Multi-path mixing validated |
| V16c | Spectral-native (remove irfft) | — | irfft round-trip (removed) | **NaN at step 5500** — irfft is load-bearing for stability |
| V16d | Matrix-valued fiber (linear attention in spectral space) | $16 \times 8 \times 8$ matrix state per head | Scalar fiber (insufficient capacity) | 500 ms/step |
| V16e | Parallel matrix scan | $O(T \log T)$ fully parallel matrix accumulation | Sequential scan | 475 ms/step, state = 1,024 values |
| V17 | **Precision routing**: position-as-precision-pattern | Gaussian clouds, precision-weighted $q, k, v$, learned VarianceUpdate MLP, content-dependent variance evolution | Phase-based position encoding as sole position signal, fixed variance decay | Training in-flight at thesis close |
| V18 | Clean-sheet: strip to load-bearing essentials | Content-dependent transitions, FFN, irfft stability, matrix fiber (associative memory), complex state for holonomy, precision gating | Spectral constellation, Hopfield bank, Langevin iteration, local conv, position FFT, Wilson line (simplified to $q@S$) | Training in-flight; audit declares **state capacity the central constraint** |

---

### 1.2 Load-bearing mechanisms (survived every ablation)

Reading across V5–V18, six mechanisms survived every ablation and every clean-sheet rewrite. These are non-negotiable for V19.

1. **Content-dependent state transitions.** V3 (GRU), V12 (context-dependent $D_k, A_k$), V14 (Wilson fiber $h_t = \alpha(x_t) h_{t-1} + \beta(x_t) x_t$). Every version that removed content dependence (V6, V13) plateaued. The V13 plateau at CE 2.17 was resolved in V14 by restoring content dependence — this lesson was learned and relearned.

2. **Per-token nonlinearity (FFN).** Every version without an FFN, or with FFN removed from an ablation, underperformed. Attention is linear in values; spectral filtering is linear; transport is linear. The FFN is the sole nonlinear feature-interaction mechanism in the entire stack.

3. **irfft round-trip.** V16c removed it and hit NaN by step 5500. The DFT basis projection prevents magnitude–phase drift. Even when the representation "wants to be" spectral-native, the irfft cycle is structurally required for numerical stability.

4. **Associative memory via matrix fiber.** V16d/e's $q@S$ retrieval outperformed V13's scalar EMA. Matrix state (16 heads $\times 8 \times 8$) provides the capacity to encode pairwise-ish relationships without explicit attention's $O(T^2)$ cost.

5. **Complex-valued state.** Phase is how holonomy (path dependence) is encoded. Mamba-3 confirmed this independently: complex SSM states track parity that real states cannot (topology/lit_review/unitary_orthogonal_rnns.md).

6. **Sufficient state capacity.** V18's audit identified the blunt reality: 1,024 state values vs attention's 131,072 at $T=256$ is a **128$\times$ gap**. No routing cleverness has closed it. Every architecture below this threshold has plateaued below attention's quality.

---

### 1.3 Decorative machinery (dropped and should not return)

These components were explored multiple times, contributed no measurable quality improvement, and added cost or instability:

1. **Spatial sparsity** (V5–V11). The Donoho–Stark uncertainty principle guarantees that spatially sparse tokens are spectrally dense. Spatial sparsity therefore forces the transport kernel to pay full $O(d \log d)$ cost. V12 correctly moved sparsity to the spectral domain.

2. **Spectral constellation as primary representation** (V13–V17). Storing (magnitude, phase, log_var) separately and maintaining them through complex arithmetic roughly doubles the embedding overhead for no measurable quality improvement over dense embeddings + a complex-valued fiber. V18's dense + precision-gated design strips this out.

3. **Hopfield memory bank with fixed codebook** (V14). The Ramsauer equivalence (attention = Hopfield retrieval) is a description of attention, not a blueprint for beating it. A fixed memory bank with softmax retrieval is strictly less expressive than attention's exact pairwise similarity, because attention's "codebook" is the KV cache itself — dynamic and exactly the input. The ablation showed Hopfield + fiber > SSM + MLP, but Hopfield + fiber $<$ attention.

4. **Iterative Langevin settling** (V5–V14). Iterative settling adds latency proportional to step count, with no proven quality gain over a single-pass Parseval filter (V15/V16). Noise annealing was theoretically motivated but empirically unnecessary at this scale.

5. **Hard top-$k$ thresholding as the sparsity mechanism.** V14 proved hard thresholding hurts gradient flow. Soft-thresholding (L1 proximal) is acceptable as a mild regularizer, and V17's Gaussian-variance soft sparsity is strictly better (fully differentiable, content-dependent).

6. **Position-axis FFT without causal safeguards** (V16b naive). Non-causal FFT across positions mixes future information into past queries, which the model exploits as a cheating channel (PPL dropped to 36). The SPECTRE fix (global gate + positional phase injection) neutralizes the leakage but at the cost of making the path nearly inert.

7. **Spectral-native processing (no irfft)** (V16c). Loses stability — the representation drifts and blows up. Never remove the irfft cycle.

8. **Phase rotation as the sole position signal** (V5–V16). V17 replaced it with learned precision-pattern routing. The thesis's own primary contribution (CurvBias) shows that going beyond pure phase rotation by adding a curvature term improves attention by up to 9%. Phase alone is under-specified.

9. **The SSM context accumulator as-is.** From thesis §7.3.4: "The SSM-based context accumulator is the least geometrically motivated component of V12... a more geometrically principled alternative might use spectral transport itself for context accumulation." The SSM is a pragmatic choice but the audit treats it as a placeholder.

---

### 1.4 What the thesis explicitly proposes as future work (Ch. 8.3)

From the conclusion, the thesis lists four future directions. V19 should be evaluable against each:

- **§8.3.1 CurvBias at production scale.** Scale to 1B+ parameters on production datasets, establish whether the geometric signal grows or diminishes with scale and sequence length.
- **§8.3.2 Geometric enhancement of production models.** Apply CurvBias to LLaMA/Mistral-class models during continued pretraining. CurvBias is compatible with KV caching, Flash Attention, and existing inference infrastructure.
- **§8.3.3 Spectral sparsity: scaling and algorithmic improvements.**
  - **Sparse FFT**: compute transform in $O(s \log s)$ rather than $O(d \log d)$, exploiting known sparsity pattern.
  - **Learned sparsity patterns**: differentiable masks or learned frequency allocation; different blocks specialize for different frequency bands.
  - **Multi-scale spectral hierarchy**: wavelets instead of FFT to capture structure at multiple resolutions.
  - **100M+ parameter scaling study** on WikiText-103 / The Pile.
- **§8.3.4 Formal proofs.** Four theorems about the Fourier–geometry correspondence (DFT basis as parallel-transport eigenfunctions, spectral kernel as holonomy, IFFT as section reconstruction, top-$k$ as Stiefel retraction).

And from the V18 audit, the operative constraint is: **close the 128$\times$ state-capacity gap without paying attention's $O(T^2 d)$ cost.**

---

## Part II — V19 Design

### 2.1 V19 thesis statement

> V19 is the minimal architecture that (i) retains all six load-bearing mechanisms from V5–V18, (ii) closes the state-capacity gap by combining $U(K)$ unitary transport with the delta rule (the one unexplored combination in SYNTHESIS.md), (iii) replaces the SSM context accumulator with a geometric spectral running-summary, (iv) uses learned sparse-FFT frequency bands per block as the concrete realization of thesis §8.3.3, and (v) uses CurvBias as the attention bias on the sole attention path.

V19 is **not** another attempt to replace attention. Consistent with the thesis's final position, V19 treats attention as a local optimum and enhances it with geometric signals while using geometric transport for the paths where attention is cost-prohibitive.

### 2.2 The central idea: $U(K)$ transport + delta rule

From topology/SYNTHESIS.md, the one unexplored combination in the 6-way research convergence is:

$$
S[t] = U_t \, S[t-1] + k_t \bigl( v_t - (U_t S[t-1])^\top k_t \bigr)^\top
$$

where:
- $U_t \in U(K)$ is a content-dependent unitary matrix (parameterized as $U_t = \exp(-i H_t)$ with $H_t$ a skew-Hermitian content projection),
- The delta-rule correction updates only along the direction of the current key $k_t$, overwriting stale content rather than blending it in.

**Why this combination has never been tried:**
- Unitary-RNN research (uRNN, scoRNN, AUSSM) uses additive accumulation — information decays exponentially and cannot be selectively overwritten.
- DeltaNet research has the delta rule but uses real-valued transitions — no holonomy, no path structure.
- Combining the two gives simultaneously: norm-preserving transport, path-dependent holonomy, and sharp content-addressable memory writes.

**Why it addresses the state-capacity gap:**
- Unitary transitions preserve norms, so the model never loses information to exponential decay. Effective state capacity $\gg$ nominal state size.
- The delta rule gives selective overwrite — old slots can be reused cleanly when they stop being relevant, without contaminating other slots.
- With $K=32$ per head $\times$ 16 heads $=$ 16,384 state values per layer, V19 state is $16\times$ larger than V16e/V18 and closes most of the 128$\times$ gap to attention at $T=256$.

### 2.3 Architecture: block structure

```
Block(x):
    # 1. Normalization
    x  := RMSNorm(x)

    # 2. Content-dependent precision (V17 survivor)
    log_var := VarianceUpdate(log_var, x)
    precision := exp(-log_var)
    gate := sigmoid(precision - 1.0)
    x_eff := x * gate

    # 3. Geometric context accumulator (replaces SSM)
    #    Spectral running summary via sparse FFT on learned bands
    q_ctx := GeometricContextAccum(x_eff, band_ids)
        # Internally: sparse FFT on active modes, Parseval filter, IFFT.
        # Output is a running spectral summary of context, not a linear SSM state.

    # 4. Unitary + delta-rule fiber (the main innovation)
    H_t := SkewHermitian(q_ctx)            # content-dependent Hamiltonian
    U_t := matrix_exp(-1j * H_t)           # unitary transport matrix
    k_t, v_t := project(x_eff)
    S[t] := U_t @ S[t-1]                   # holonomic transport
    correction := k_t * (v_t - (S[t].T @ k_t)).T
    S[t] := S[t] + correction              # delta-rule write
    h_out := q_t @ S[t]                    # associative read

    # 5. Parseval spectral filter (V16 survivor)
    H_spec := rfft(h_out, band=band_ids)   # sparse rFFT on band_ids
    W := Parseval(|W|<=1)(q_ctx)           # content-dependent energy-bounded gate
    H_spec := W * H_spec
    y := irfft(H_spec, band=band_ids)      # sparse iFFT, preserves stability

    # 6. Single attention path with CurvBias (thesis contribution)
    kappa_t := CurvBias(q_ctx, x_eff)      # content-dependent curvature
    B_ij := -integrate(kappa_t, i to j)
    attn_out := softmax(Q K^T / sqrt(d) + B) V   # one head per block only

    # 7. Merge: fiber + attention (residual)
    mix := g * attn_out + (1 - g) * y      # learned scalar gate g
    x := x + mix

    # 8. FFN (non-negotiable V5-V18 survivor)
    x := x + FFN(RMSNorm(x))

    return x, log_var, S
```

### 2.4 Component justifications

**Component 1 — RMSNorm.** Identical to V16–V18. Survives.

**Component 2 — Precision-routed Gaussian clouds from V17.** Soft sparsity via variance is the only mechanism that gave us content+position dependent routing without a cheating channel or a hard threshold. Kept as-is.

**Component 3 — Geometric context accumulator (replaces SSM).** Thesis §7.3.4 identifies the SSM as "the least geometrically motivated component of V12." V19 replaces it with a sparse-FFT-based running spectral summary. Concretely:
- Each block owns a learned `band_ids` — a differentiable mask over which $s$ frequency modes are active in *this* block. Different blocks specialize for different bands (thesis §8.3.3).
- The spectral running summary accumulates context as a decaying complex sum over the active bands only. This is a genuine spectral state, not an abstract state vector.
- Cost: $O(T s)$ per block, vs SSM's $O(T d)$. For $s = 16, d = 256$, that's a 16$\times$ speedup on this component.

**Component 4 — $U(K)$ unitary + delta-rule fiber (the main innovation).** See §2.2. Implemented with $K = 32$ via $\exp(-iH)$ of a skew-Hermitian content projection. The matrix exponential at this size is cheap ($O(K^3) = 32{,}768$ flops per step, negligible). On MPS, use the Padé approximation path to avoid the hangs from feedback memo `feedback_no_complex_mps.md` (use real $SO(2K)$ embedding of $U(K)$ if complex ops misbehave).

**Component 5 — Parseval spectral filter.** V16 survivor. Energy-bounded content-dependent gate with $|W| \leq 1$. Runs on the same `band_ids` as the context accumulator, so the sparse-FFT cost applies here too: $O(s \log s)$ per filter.

**Component 6 — One attention path per block with CurvBias.** The thesis's primary empirical result is that CurvBias improves attention by up to 9% over RoPE. V19 has exactly one attention head per block — small enough that the $O(T^2)$ cost is tolerable, large enough that attention's exact pairwise expressivity is available on the path where it matters. This is the "hybrid" insight: don't replace attention, use it sparingly and augment it with geometry.

**Component 7 — Learned gate between fiber and attention paths.** A scalar $g \in [0, 1]$ per block decides how much of each path's output to use. If attention is the local optimum, $g$ will go high. If the fiber is genuinely contributing on a given block, $g$ will go low. This is how V19 *lets the data decide* which mechanism is load-bearing at each depth, rather than us pre-committing.

**Component 8 — FFN.** Non-negotiable. Kept identical to V16.

### 2.5 What V19 explicitly does NOT include

- **No Hopfield memory bank.** Superseded by delta-rule matrix fiber (content-addressable associative memory, but dynamic).
- **No iterative Langevin settling.** Single-pass spectral filtering proved sufficient (V15/V16).
- **No spectral constellation embedding.** Dense embeddings + complex fiber (V18's stripping). No mag/phase/log_var decomposition of the token itself.
- **No pure phase-based position encoding.** CurvBias replaces RoPE on the attention path; precision routing handles position on the fiber path.
- **No local causal conv.** V16b's conv was a duplicate mechanism with the fiber path. The fiber already captures short-range dependencies via the delta-rule update.
- **No position-axis FFT.** The cheating channel is too risky, and SPECTRE's safeguards neutered it anyway.

### 2.6 Addressing the thesis's future-work directions, item by item

| Thesis item | V19 realization |
|-------------|------------------|
| §8.3.1 CurvBias at production scale | V19 includes CurvBias on its attention path by default. Scaling V19 itself, or applying V19's CurvBias component to a LLaMA-class model, tests the scaling behavior. |
| §8.3.2 Geometric enhancement of production models | The CurvBias path in V19 is structurally identical to a drop-in CurvBias replacement for RoPE in any production attention stack. V19 ships it as a standalone module. |
| §8.3.3 Sparse FFT with learned sparsity patterns | `band_ids` are per-block learned differentiable masks. The context accumulator, Parseval filter, and spectral I/O all consume only the active bands $\to$ $O(s \log s)$ spectral ops instead of $O(d \log d)$. |
| §8.3.3 Multi-scale spectral hierarchy | Different blocks with different learned `band_ids` naturally specialize for different frequency bands. Shallow blocks: low frequencies (global topic). Deep blocks: high frequencies (local phrasing). This is a learned realization of the wavelet idea. |
| §8.3.3 100M+ parameter scaling study | V19's parameter count is dominated by FFN (same as transformers), so scaling is a straightforward knob. See §2.8 for the protocol. |
| §8.3.4 Formal proofs | V19 explicitly exposes the geometric operators as named modules (`UnitaryTransport`, `DeltaRuleWrite`, `SparseFFT`, `ParsevalFilter`) with unit-testable invariants. This makes the four thesis-§8.3.4 theorems directly auditable on the actual implementation. |

### 2.7 Expected efficiency improvements

Relative to V16e (the most recent baseline with measured numbers):

| Axis | V16e | V19 (target) | Mechanism |
|------|------|---------------|-----------|
| State capacity per layer | 1,024 values | 16,384 values | $K=32$ vs $K=8$ in matrix fiber |
| Spectral ops per block | $O(d \log d)$ | $O(s \log s)$ | Sparse FFT on `band_ids` |
| Context accumulator cost | $O(T d)$ | $O(T s)$ | Spectral running summary replaces SSM |
| Step time on GPU (WikiText-103, $T$=256, d=256) | 475 ms | ~250 ms (target) | Sparse-FFT savings + single-attention-head instead of fiber-only |
| PPL gap vs GPT (same params, same data) | 1.59$\times$ (V16) | $\leq 1.15\times$ (target) | Closing 128$\times$ state gap $\to$ 8$\times$; delta rule; CurvBias on attention path |

These targets are hypotheses, not guarantees. The V16e-to-V19 improvements are all independently motivated by ablation evidence, but stacking them is a new experiment.

### 2.8 Experimental protocol

1. **Ablation-first construction.** Before any full V19 run, verify each new component in isolation on Tiny Shakespeare against V18 as baseline:
   - V18 + $U(K)$ transport only (no delta rule)
   - V18 + delta rule only (no $U(K)$)
   - V18 + $U(K)$ + delta rule (the main claim)
   - V18 + sparse FFT context accumulator only
   - V18 + one attention head + CurvBias only
   The full V19 should beat the best single-component ablation; if it doesn't, the stacking isn't justified.

2. **Main run.** WikiText-103, $d$=512, 12 blocks, $T$=1024, 40K steps. Compare against:
   - GPT-Nano same config (baseline)
   - GPT-Nano + CurvBias (thesis's best)
   - V16 same config (previous SGST SOTA)
   - V18 same config (simplest recent)

3. **Scaling run.** If V19 is within 15% of GPT-Nano+CurvBias on WikiText-103 at 77M params, scale to 350M and 1B on The Pile or C4. This is the thesis §8.3.3 scaling study.

4. **Formal-proof audit.** Each of the four §8.3.4 theorems gets a unit test against V19's implementation:
   - `test_dft_basis_are_transport_eigenfunctions`
   - `test_spectral_kernel_is_holonomy`
   - `test_irfft_is_section_reconstruction`
   - `test_top_k_is_stiefel_retraction`
   These are numeric-sanity tests, not formal proofs, but they elevate the correspondence from heuristic to verified-in-practice.

### 2.9 Open risks

1. **MPS complex-ops hang** (from `feedback_no_complex_mps.md`). Mitigation: real $SO(2K)$ embedding of $U(K)$; all complex ops rewritten as real $2\times$ blocks. Lose 2$\times$ in memory, gain reliability.

2. **Delta-rule instability with unitary transitions.** The Lyapunov analysis for DeltaNet assumes real state; unitary $U_t$ preserves norms but the delta correction can briefly violate this inside a single step. Mitigation: apply the delta rule in the unitary-conjugate basis: $S'[t] = S[t-1] + U_t^\dagger k_t(v_t - (U_t S[t-1])^\top k_t)^\top$, then $S[t] = U_t S'[t]$.

3. **Learned `band_ids` degenerating.** If the differentiable mask collapses to always-active, we lose the sparse-FFT speedup. Mitigation: initialize with a fixed wavelet-like schedule across blocks (block $\ell$ starts with bands $\{2^{\ell-1}, \dots, 2^\ell\}$) and add a mild $L_1$ regularization on the mask.

4. **The 128$\times$ gap may not be closable by state capacity alone.** V18's audit says "no routing cleverness compensates for this." V19 closes the gap in state capacity but still uses matrix-fiber retrieval, which is associative rather than pairwise. If the residual gap is fundamentally due to pairwise-vs-associative expressivity, the only fix is the single attention head per block — which is exactly what V19 ships.

5. **CurvBias may not stack with the fiber path.** The 9% improvement is measured on pure attention. Whether it helps when fiber is doing most of the work is an empirical question the ablations in §2.8 will answer.

---

## Part III — Summary

### 3.1 The one-paragraph version

V19 preserves the six mechanisms that survived every V5–V18 ablation (content-dependent transitions, FFN, irfft stability, matrix fiber associative memory, complex state, sufficient capacity), drops every component that has been shown to be decorative (spatial sparsity, Hopfield bank, iterative Langevin, phase-only position, local conv, position FFT, unguarded spectral-native path), addresses the V18 audit's central constraint (state capacity, 128$\times$ gap) by combining $U(K)$ unitary transport with the delta rule — the single unexplored combination in the 6-way research convergence — replaces the SSM context accumulator with a geometric spectral running summary on learned per-block frequency bands (thesis §8.3.3 sparse FFT + learned sparsity + multi-scale hierarchy, all in one mechanism), and carries the thesis's primary empirical contribution (CurvBias) on a single attention head per block so that attention's local optimum is available where it matters.

### 3.2 The changes relative to the thesis's own future-work list

V19 implements every concrete direction in thesis §8.3:

- Sparse FFT $\checkmark$ — via per-block `band_ids`
- Learned sparsity patterns $\checkmark$ — the `band_ids` are differentiable masks
- Multi-scale hierarchy $\checkmark$ — different blocks own different bands
- CurvBias $\checkmark$ — on the single attention path
- Production compatibility $\checkmark$ — CurvBias component is a drop-in RoPE replacement
- Scaling study $\checkmark$ — protocol in §2.8

V19 additionally implements the one convergence gap from topology/SYNTHESIS.md that the thesis did not explicitly list: **$U(K)$ transport + delta rule**.

### 3.3 What V19 does not claim

- V19 does not claim to beat attention at equal-scale, equal-compute. The thesis's "attention is a local optimum" position stands.
- V19 does not claim to validate the spectral sparsity hypothesis. That remains an open empirical question at scale.
- V19 does not claim novelty of any single component. Each component is sourced from documented prior work (V14 for content-dependent fiber, V16 for irfft round-trip, V17 for precision routing, V18 for load-bearing minimalism, thesis Ch. 6.7 for CurvBias, SYNTHESIS.md for $U(K)$ + delta rule, thesis §8.3.3 for sparse-FFT directions). V19's contribution is the specific combination.

### 3.4 Success criteria

V19 is a success if any one of the following holds after the experimental protocol:

1. On WikiText-103 at 77M params, PPL gap to GPT-Nano+CurvBias $\leq 15\%$ at matched wall-clock time.
2. On WikiText-103 at 77M params, the V19 fiber path beats V18 by $\geq 10\%$ PPL (validating the $U(K)$ + delta-rule combination as a load-bearing mechanism).
3. The learned per-block `band_ids` empirically specialize for low/mid/high frequency across block depth, with measurable spectral sparsity $\geq 60\%$ per block (validating the learned multi-scale sparse-FFT hypothesis from §8.3.3).
4. At 1B scale on The Pile, CurvBias alone (extracted from V19, applied to a vanilla transformer) continues to outperform RoPE, replicating the thesis §6.7 result at production scale.

Any one of those is publishable. All four together would justify a V20.
