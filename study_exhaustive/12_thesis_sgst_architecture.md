# Unit 12: The SGST Architecture: Design, Evolution, and Informative Failure

## Learning Objectives

1. Trace the architectural evolution from V5 through V16
2. Understand each component of the V12 architecture and its geometric motivation
3. Analyze the V12.2 ablation results and what they reveal
4. Understand why certain architectural decisions failed and what was learned
5. See the trajectory from "replace attention" to "enhance attention" (CurvBias)

## Prerequisites

Units 03 (Fourier Analysis), 06 (Fiber Bundles), 08 (Signal Processing & Compressed Sensing), 09 (Hopfield Networks), 10, 11 (Spectral Methods in Deep Learning)

## Readings

- Thesis Ch. 5 (Spectral Methods) -- all sections
- Thesis Ch. 6 (Experiments) -- all sections
- Thesis Ch. 7.1 (What the Spectral Architecture Reveals About Attention)
- Repo: `v12/README.md`, `v12/README_ABLATION_SSM.md`
- Repo: `v12/v12_1.py` (actual implementation -- read and understand the code)
- Repo: `v14/v13_diagnosis.md` (why V13 plateaued)
- Repo: `v16/CHANGELOG.md` (evolution through V16)
- Repo: `Architecture.md` (formal reference)
- Repo: `CLMWithArch.md` (causal LM mapping)

---

## Key Concepts

1. **The 45% wall (V5-V9).** All spatial routing strategies -- message passing, Finsler metrics, learned distance functions -- hit the same performance ceiling at approximately 45% accuracy. The bottleneck was not the routing mechanism but the underlying representation.

2. **"The manifold is fake" (V10).** Positional embeddings define a fixed, context-blind geometry. A Riemannian manifold derived from these embeddings has no information about the actual content. The manifold must be learned from data, not imposed from positions.

3. **Sparse events + contextual manifold (V11).** Diffusion is reinterpreted as field reconstruction: sparse tokens are point sources and the diffusion process reconstructs the full manifold field, analogous to Alcubierre warp geometry. Langevin dynamics collapses the field back to sparse.

4. **The spectral shift (V12).** Donoho-Stark uncertainty principle implies that sparsity belongs in Fourier space (few active frequency modes), not in spatial space (few active tokens). Tokens become spectral wells; transport becomes $O(s)$ per step.

5. **V12 components.** SpectralTokenEmbedding (magnitude + phase), ContextAccumulator (SSM for causal context), SpectralTransport (learned diffusion + gauge kernel), MemoryBank (Hopfield atoms), HopfieldSettler (Langevin energy minimization), SpatialMLP (per-token nonlinearity).

6. **V12.1 refinements.** Separate SSMs per subbundle (specialization), per-subbundle memory banks, 4-block stack instead of 8 (depth vs width tradeoff).

7. **The devastating ablation (V12.2).** SSM+MLP alone beats the full spectral architecture at 4.7x lower cost. The spectral machinery contributed negative value.

8. **V13 plateau.** Cross-entropy stuck at 2.17. Root cause: the fiber (EMA context accumulator) is content-independent -- every past token gets the same exponential decay regardless of relevance.

9. **V14: Wilson line + Langevin + proximal sparsity.** Content-dependent fiber transport via $U(K)$ gauge group, but dynamics are wrong -- Hopfield memory bank is the only component that helps.

10. **V15-V16: Parseval filter, position FFT, Gaussian clouds.** Multi-path mixing covers all temporal scales. PPL 275 on WikiText-103, approaching but not matching GPT's PPL 173.

11. **The pivot.** From "replace attention" to "enhance attention with geometry." CurvBias adds content-dependent curvature to position encoding within standard attention, at negligible additional cost.

---

## Worked Problems

### Problem 1: Tracing the V12 Data Flow

Trace the V12 data flow for a single token. Starting from the character "a" (vocabulary index 0), describe what happens at each stage of a single block.

**Solution:**

**Stage 1: SpectralTokenEmbedding.**

Look up learned magnitude vector $m[0] \in \mathbb{R}^{d_{\text{model}}}$ and phase vector $\phi[0] \in \mathbb{R}^{d_{\text{model}}}$ from the embedding table. Form the complex spectral representation:

$$z[k] = m[k] \cdot \exp(i \cdot \phi[k]) \quad \text{for } k = 0, \ldots, d_{\text{model}} - 1$$

Add positional encoding (also in spectral domain). The result is a $d_{\text{model}}$-dimensional complex vector where each component represents one frequency mode's amplitude and phase.

**Stage 2: Top-$k$ sparsification.**

Keep only the $s$ largest $|z[k]|$ values, set the rest to exactly zero. Now have a sparse spectral token: only $s$ out of $d_{\text{model}}$ modes are "active." This implements the Donoho-Stark sparsity -- the token is concentrated on a few frequency modes.

**Stage 3: ContextAccumulator (SSM).**

Update the recurrent state: $q_t = A \cdot q_{t-1} + B \cdot z_t$, where $A$ is the state transition matrix and $B$ is the input projection. $q_t$ now encodes the causal history -- a compressed summary of all tokens seen so far. Each subbundle has its own SSM, allowing different subbundles to accumulate different aspects of context.

**Stage 4: SpectralTransport.**

Apply the context-dependent transport kernel to each active mode:

$$z'[k] = K[k](q_t) \cdot z[k] \quad \text{where } K[k] = \exp\!\bigl(-D_k(q_t) \cdot \omega_k^2 - i \cdot A_k(q_t) \cdot \omega_k\bigr)$$

The diffusion coefficient $D_k$ and gauge connection $A_k$ are computed from the context vector $q_t$ via learned projections. Each active mode is independently scaled (diffusion) and phase-rotated (transport).

**Stage 5: IFFT (field reconstruction).**

Apply the inverse FFT to the sparse spectral representation. This reconstructs a dense spatial signal from the sparse frequency components. The $s$ active modes "spread out" across all $d_{\text{model}}$ spatial positions. The output is a real-valued $d_{\text{model}}$-dimensional vector.

**Stage 6: HopfieldSettler.**

Run 2 steps of Langevin dynamics on the Hopfield energy landscape:

$$x_{t+1} = x_t - \eta \nabla_x E(x) + \sqrt{2 \eta T}\;\xi$$

where $E(x) = -\sum_\mu \frac{(x \cdot \xi_\mu)^2}{2N}$ is the Hopfield energy with memory patterns $\{\xi_\mu\}$. This "settles" the spatial representation into the nearest memory basin, implementing associative recall.

**Stage 7: SpatialMLP.**

Per-token nonlinear transformation:

$$y = W_2 \cdot \operatorname{GELU}(W_1 \cdot x + b_1) + b_2$$

where $W_1$ expands from $d_{\text{model}}$ to $4 \times d_{\text{model}}$ and $W_2$ contracts back. This is the "reaction" step -- local nonlinear processing without cross-token interaction.

**Stage 8: FFT + re-sparsify.**

Apply FFT to return to spectral domain, then select the top-$s$ modes again. The output is a sparse spectral token, ready for the next block. The new set of active modes may differ from the input set (mode selection resets per block -- this turns out to be a problem).

---

### Problem 2: Compute Efficiency Analysis of the V12.2 Ablation

The V12.2 ablation results are:

| Model | Val BPC | ms/step |
|-------|---------|---------|
| Full V12.1 | 2.302 | 321 |
| SSM+MLP only | 2.267 | 68 |
| No context | 3.589 | -- |
| No sparsity | 2.275 | -- |

Calculate the compute efficiency (BPC improvement per ms) for each variant relative to the "no context" baseline.

**Solution:**

**Full V12.1:**
- BPC improvement over no-context: $3.589 - 2.302 = 1.287$ BPC
- Compute cost: 321 ms/step
- Efficiency: $1.287 / 321 = 0.00401$ BPC per ms

**SSM+MLP only:**
- BPC improvement over no-context: $3.589 - 2.267 = 1.322$ BPC
- Compute cost: 68 ms/step
- Efficiency: $1.322 / 68 = 0.01944$ BPC per ms

**Efficiency ratio:** $0.01944 / 0.00401 = 4.85$

The SSM+MLP is 4.85x more compute-efficient than the full spectral model. It achieves BETTER quality ($1.322 > 1.287$ BPC improvement) at 4.7x less wall-clock cost.

**The spectral machinery's marginal contribution:**

Additional BPC from spectral components: $2.267 - 2.302 = -0.035$ (negative -- it hurts).
Additional cost: $321 - 68 = 253$ ms/step.
Marginal efficiency: $-0.035 / 253 = -0.000138$ BPC per ms.

The spectral machinery consumes 253 ms per step (79% of total compute) while making the model 0.035 BPC worse. This is the devastating finding: the theoretical crown jewels of the architecture -- spectral transport, Hopfield memory, FFT/IFFT cycles -- are net negative contributors.

---

### Problem 3: Diagnosing V12 Spectral Transport Failure

Why did V12 spectral transport fail as a "load-bearing mechanism"? List and explain the 4 diagnosis points from the V12.2 analysis.

**Solution:**

**Failure Mode 1: Transport was mode-wise/linear.**

Each frequency mode $k$ was processed independently:

$$z'[k] = K[k] \cdot z[k]$$

This is a diagonal operation in frequency space -- mode $k$ cannot influence mode $j$. It functions like an audio equalizer: you can boost or cut individual frequency bands, but you cannot create new frequencies from interactions between existing ones.

Real language requires cross-frequency interaction. Syntactic structure (captured at certain frequency scales) must influence semantic interpretation (at other frequency scales). A diagonal transport kernel cannot implement this.

**Failure Mode 2: No cross-mode spectral interactions.**

Formally, the transport matrix in frequency space is:

$$T = \operatorname{diag}(K[0], K[1], \ldots, K[d-1])$$

For expressive spectral processing, $T$ should be a full (or at least banded) matrix, allowing off-diagonal entries $T[j,k]$ that couple mode $j$ to mode $k$. The V12 design forced $T$ to be diagonal, sacrificing all cross-mode expressivity for computational efficiency.

**Failure Mode 3: Mode selection resets per block.**

At the end of each block, top-$k$ sparsification selects the $s$ most energetic modes from the output. This set can change completely from one block to the next. If block 1 activates modes $\{2, 5, 11, 17\}$ and block 2 activates modes $\{3, 7, 14, 22\}$, there is no continuity of the spectral submanifold.

This destroys the geometric picture: the "spectral well" that a token sits in should persist and evolve smoothly across blocks. Instead, it teleports randomly. The fiber bundle structure -- which requires smooth, continuous parallel transport along the base manifold -- breaks down when the fiber itself is discontinuous.

**Failure Mode 4: Sparsity was too mild.**

The learned sparsity was approximately 60% (keeping 40% of modes active). For the compressed sensing framework to provide meaningful guarantees, the sparsity must be much more aggressive -- typically keeping only 5-10% of components. At 40% sparsity, the spectral representation is not meaningfully different from the full representation; the "sparse geometry" intuition provides no practical benefit.

Moreover, the sparsification is a hard thresholding operation (top-$k$), which introduces discontinuities in the gradient. Modes near the threshold boundary jump in and out of the active set, creating noisy gradient signals.

---

### Problem 4: The V13 Plateau -- Content-Independent Fiber

The V13 diagnosis identified 6 root causes for the plateau at CE 2.17. Explain the most fundamental one: "Linear fiber is content-independent."

**Solution:**

The V12/V13 context accumulator uses an exponential moving average (EMA):

$$q_t = \alpha \cdot q_{t-1} + (1 - \alpha) \cdot x_t$$

where $\alpha$ is a learned but FIXED scalar (or diagonal matrix). Unrolling the recurrence:

$$q_t = (1-\alpha) x_t + \alpha(1-\alpha) x_{t-1} + \alpha^2(1-\alpha) x_{t-2} + \cdots$$

Every past token $x_j$ at position $j < t$ receives weight $\alpha^{t-j}(1-\alpha)$. This weight depends ONLY on the time gap $(t - j)$, not on the content of $x_j$ or $x_t$.

**The fundamental limitation:** The fiber cannot selectively attend to past tokens. Token $x_5$ always gets weight $\alpha^{t-5}(1-\alpha)$, whether $x_5$ is highly relevant to the current prediction or completely irrelevant.

**Comparison to attention:**

In standard attention, the weight of past token $j$ at position $t$ is:

$$w_j = \operatorname{softmax}\!\left(\frac{q_t \cdot k_j}{\sqrt{d}}\right)$$

This weight depends on BOTH the current query $q_t$ AND the past key $k_j$. The model can assign high weight to a distant but relevant token and zero weight to a nearby but irrelevant one.

**Why this is fundamental:** Language requires content-addressable memory. Consider: "The cat that the dog that the rat bit chased fled." To predict "fled," the model must attend to "cat" (the subject) despite the intervening clause. The EMA fiber assigns "cat" a small weight (it is far away) while "chased" gets a large weight (it is close) -- exactly the opposite of what is needed.

**The V14 fix:** Replace EMA with Wilson line transport, which uses content-dependent rotation matrices:

$$q_t = U(x_t, q_{t-1}) \cdot q_{t-1} + B \cdot x_t$$

where $U$ depends on both the current input $x_t$ and the current state $q_{t-1}$. Combined with a delta rule for content-addressable write/overwrite, this allows the fiber to selectively retain and recall relevant past context.

---

### Problem 5: V16 Multi-Path Mixing

V16 combines 4 mixing paths: (1) Wilson Fiber, (2) Parseval Filter, (3) Position-axis FFT, (4) Local Convolution. Explain why all 4 are needed by describing the temporal scale each covers.

**Solution:**

**Path 1: Wilson Fiber (complex EMA with content-dependent rotation).**

Temporal scale: long-range causal context via recurrence.

The Wilson fiber maintains a recurrent state that accumulates over the entire sequence. It uses $O(1)$ memory (fixed state size regardless of sequence length) and can in principle carry information from the first token to the last. However, information is compressed through the recurrence and earlier tokens are progressively "overwritten."

Covers: discourse-level patterns, long-range dependencies, running context.

**Path 2: Parseval Filter (spectral energy gating).**

Temporal scale: frequency-domain redistribution.

Operates on the spectral (FFT) representation of the sequence. Each frequency mode $k$ gets a content-dependent weight $W[k]$ with $|W[k]| \leq 1$. This amplifies or suppresses different frequency components of the sequence, effectively controlling which oscillatory patterns are visible.

Covers: periodic patterns, rhythmic structure, spectral features (e.g., alternating patterns at specific frequencies).

**Path 3: Position-axis FFT (SPECTRE-style mixing).**

Temporal scale: instant global token mixing.

Every token communicates with every other token in one operation. The position-axis FFT transforms the sequence into a position-frequency representation where all positions contribute to all frequencies. With causal safeguards (global gate, positional phase injection), this provides global information flow without violating causality.

Covers: global patterns, position-independent features shared across the sequence.

**Path 4: Local Convolution (causal depthwise, kernel = 7).**

Temporal scale: short-range patterns (7-token window).

A simple causal convolution that looks at the 7 most recent tokens. Extremely fast, no recurrence needed, captures local syntax: bigrams, trigrams, short phrases.

Covers: local syntax, n-gram patterns, adjacent-token dependencies.

**Why all 4 are necessary:**

Removing any single path leaves a gap in temporal coverage:
- Remove conv: local patterns (bigrams, trigrams) must be learned by slower/more expensive paths
- Remove fiber: long-range memory is lost; the model forgets distant context
- Remove FFT: global mixing requires the fiber to propagate information token-by-token (slow, lossy)
- Remove filter: spectral processing is lost; frequency-domain patterns must be learned in spatial domain (less efficient)

The 4 paths form a complete coverage of temporal scales from local (7 tokens) to global (full sequence).

---

### Problem 6: V16 Scaling Trajectory Analysis

V16 achieved PPL 275 on WikiText-103 vs GPT's PPL 173 (at 20K training steps). V16 at 20K steps matches GPT at approximately 7500 steps (same quality level). The V16 step time is 475 ms, GPT step time is 137 ms. Analyze the scaling trajectory.

**Solution:**

**Step-normalized comparison:**

At 20K steps, V16 PPL $= 275$, GPT PPL $= 173$. V16 is 1.59x worse in PPL.

V16 at 20K steps achieves the same quality as GPT at approximately 7500 steps. So V16 needs $20000/7500 = 2.67$x more steps to reach the same quality.

**Time-normalized comparison:**

V16 wall-clock time for 20K steps: $20000 \times 475\;\text{ms} = 9500$ seconds.
GPT wall-clock time for 7500 steps: $7500 \times 137\;\text{ms} = 1028$ seconds.

To match GPT quality, V16 needs $9500/1028 = 9.24$x more wall-clock time.

**FLOP-normalized comparison:**

Each V16 step costs approximately $475/137 = 3.47$x more compute than a GPT step. At equal total FLOPs (say, 20K GPT-equivalent steps of compute):

V16 can run: $20000 / 3.47 = 5764$ steps.
GPT runs: 20000 steps.

At 5764 V16 steps vs 20000 GPT steps: GPT achieves significantly better quality.

**Interpretation:**

V16's per-step quality improvement is somewhat better than GPT's (the geometric inductive bias helps each step teach more), but not enough to overcome the 3.47x per-step cost penalty. The constant factor overhead from spectral operations, multi-path mixing, and Wilson fiber transport dominates.

**Scaling trajectory implications:**

If V16's per-step advantage grows with scale (geometry becomes more valuable with more data/parameters), it might eventually catch up. But the thesis data does not show evidence of this -- the gap remains roughly constant across the training run. The conclusion: at current hardware scale and optimization, attention is more efficient per FLOP.

---

### Problem 7: Non-Causal FFT Leakage

The non-causal FFT in V16 caused PPL to drop to 36 (from 275) -- suspiciously good. Explain why this was "cheating" and how SPECTRE's safeguards fixed it.

**Solution:**

**Why non-causal FFT is cheating:**

In a causal language model, the prediction of token $t$ can only depend on tokens $0, 1, \ldots, t-1$. The model must NOT see tokens $t, t+1, \ldots, T$ (the "future").

The position-axis FFT computes:

$$X[k] = \sum_{n=0}^{T-1} x[n] \cdot \exp\!\left(\frac{-2\pi i k n}{T}\right)$$

Every frequency bin $X[k]$ depends on ALL positions $n = 0, \ldots, T-1$, including future positions. When the IFFT reconstructs the spatial signal, position $t$ contains information from positions $t+1, t+2$, etc.

In effect, the model can "see the answer" -- predicting token 5 is trivial when the FFT path provides a linear combination of all tokens including token 5 itself. PPL of 36 (compared to 275 with proper causality) measures the information leakage from the future.

**SPECTRE-style safeguards:**

Three mechanisms prevent future information from reaching past positions:

**(a) Global gate.** Multiply the FFT output by a learned scalar $g \in [0, 1]$:

$$\text{output} = g \cdot \text{FFT\_path}(x) + (1-g) \cdot x$$

The model can learn $g$ close to 0 to effectively disable the FFT path when it detects that useful information is being contaminated by future leakage. In practice, $g$ learns to be moderate (approximately 0.3-0.5), allowing some global mixing while limiting leakage.

**(b) Positional phase injection.** Before the FFT, add position-dependent phase shifts:

$$x'[n] = x[n] \cdot \exp(i \cdot \theta[n])$$

where $\theta[n]$ is a learned, position-dependent phase. This breaks the direct future-to-past information path by scrambling the phase relationship between positions. Position-independent features (common to all positions) survive the phase injection, while position-specific features (which carry the "answer" information) are disrupted.

**(c) Residual addition.** The FFT output is ADDED to the residual stream, not used directly:

$$\text{output} = x + \alpha \cdot \text{FFT\_path}(x)$$

where $\alpha$ is small. Even if some future information leaks through, it is attenuated by the residual weighting.

**Result with safeguards:** PPL rises from 36 back to 275. The safeguards successfully block position-specific future information while allowing global position-independent features to flow.

---

### Problem 8: Separate Magnitude and Phase Embedding

The SpectralTokenEmbedding in V12 stores magnitude and phase separately rather than using complex vectors directly. Explain the design rationale.

**Solution:**

Each token $i$ has two learned embedding vectors:

$$m_i \in \mathbb{R}^{d_{\text{model}}} \quad \text{(magnitude per frequency mode)}$$
$$\phi_i \in \mathbb{R}^{d_{\text{model}}} \quad \text{(phase per frequency mode)}$$

The complex spectral representation is formed as:

$$z_i[k] = m_i[k] \cdot \exp(i \cdot \phi_i[k])$$

**Why separate rather than a single complex vector $c_i[k] = a_i[k] + i \cdot b_i[k]$?**

**Reason 1: Independent learning of "what" and "where."**

Magnitude $m_i[k]$ encodes "how active is mode $k$ for token $i$" -- this is mode selection. A large magnitude means mode $k$ is important for representing token $i$. Phase $\phi_i[k]$ encodes "where in the oscillation cycle does mode $k$ start" -- this is timing/alignment.

These are conceptually independent: knowing that mode $k$ is important (magnitude) tells you nothing about its phase alignment. Separating them allows the gradient to update each independently.

**Reason 2: Compatible sparsification.**

Top-$k$ sparsification selects modes based on magnitude $|z[k]| = m[k]$. With separate parameterization, this simply means keeping the modes with the $s$ largest $m[k]$ values. The phase of the selected modes is preserved exactly. If you sparsify a complex vector $c[k] = a[k] + ib[k]$, you are thresholding based on $\sqrt{a[k]^2 + b[k]^2}$, and a gradient update that changes $a[k]$ simultaneously changes the threshold, creating entangled gradients.

**Reason 3: Gauge-compatible phase rotation.**

The gauge connection component of the transport kernel $\exp(-i A_k \omega_k)$ rotates the phase by $A_k \omega_k$ radians. With separate magnitude and phase, this is simply:

$$\phi'[k] = \phi[k] - A_k \omega_k \quad \text{(phase addition)}$$
$$m'[k] = m[k] \quad \text{(magnitude unchanged)}$$

The gauge connection affects ONLY the phase, not the magnitude. This clean separation is exactly the structure of a $U(1)$ gauge transformation: the magnitude (which determines mode selection) is gauge-invariant, while the phase (which determines alignment) transforms under gauge transport.

**Connection to V16 (Gaussian clouds):** V16 extends this by adding a third parameter -- variance $\sigma_i[k]$ -- to each mode. The token is no longer a point in spectral space but a Gaussian cloud: $z_i[k] \sim \mathcal{N}(m_i[k] \cdot \exp(i \cdot \phi_i[k]),\; \sigma_i[k]^2)$. This represents uncertainty about the spectral representation.

---

### Problem 9: Architecture Evolution as Informative Failure

The thesis presents the architecture evolution as a narrative of "informative failures." For each version from V5 through V16, state the key insight learned from the failure.

**Solution:**

**V5-V9: The 45% Wall.**
Failed: all spatial routing strategies (message passing, Finsler metrics, learned distances) hit the same performance ceiling.
Learned: the routing mechanism does not matter -- the bottleneck is the representation, not how information is routed between tokens.

**V10: "The Manifold is Fake."**
Failed: building a Riemannian manifold from positional embeddings produced a context-blind geometry with no information about actual content.
Learned: positional embeddings are insufficient for geometry. The manifold must be context-dependent, learned from data, not imposed from positions.

**V11: Diffusion is Field Reconstruction.**
Failed: treating diffusion as smoothing lost information.
Learned: the forward-reverse loop (diffuse then concentrate) is fundamental. Diffusion reconstructs the full field from sparse point sources; Langevin dynamics collapses back to sparse. This is not smoothing -- it is the Fourier transform/inverse transform duality.

**V12: The Spectral Shift.**
Failed: spatial sparsity could not achieve meaningful compression.
Learned: the Donoho-Stark uncertainty principle implies sparsity belongs in Fourier space (few active frequency modes), not spatial space (few active token positions). Tokens should be spectral wells.

**V12.2: The Devastating Ablation.**
Failed: the full spectral architecture performed worse than SSM+MLP alone.
Learned: spectral machinery needs cross-mode interaction (not just per-mode processing) to be useful. Mode-wise diagonal transport is too restrictive.

**V13: The Content-Independent Fiber.**
Failed: performance plateaued at CE 2.17.
Learned: the context accumulator (fiber) MUST be content-dependent. Linear EMA assigns weights based only on temporal distance, not relevance. Language requires content-addressable memory.

**V14: Geometry vs. Dynamics.**
Failed: correct geometry (Wilson line, $U(K)$ gauge group) combined with wrong dynamics (Langevin settling was too slow and noisy).
Learned: the Hopfield memory bank was the only component that contributed meaningfully. Iterative dynamics (Langevin) should be replaced by single-step operations.

**V15: Parseval Replaces Langevin.**
Failed: still not competitive with GPT at equal scale.
Learned: single-step spectral filtering (Parseval constraint $|W[k]| \leq 1$) is superior to iterative Langevin settling. Stability can be guaranteed algebraically rather than dynamically.

**V16: Multi-Path and Causality.**
Failed: non-causal FFT created information leakage (PPL 36). Even with safeguards, PPL 275 vs GPT's 173.
Learned: (a) ALL temporal scales must be covered (local, recurrent, spectral, global). (b) Causality safeguards are essential for global mixing. (c) At current hardware scale, attention remains more efficient per FLOP.

**The meta-lesson:** Each failure refined the understanding of what attention actually does and why it works. The failures were not wasted -- they mapped out the space of alternatives and identified the precise properties that make attention effective (content-addressable pairwise comparison, $O(T^2 d)$ expressivity, hardware-friendly dense matrix multiplication).

---

### Problem 10: "Attention is a Local Optimum"

The thesis ultimately concludes that "attention is a local optimum in the space of geometric sequence processors." Explain this claim using the evidence from the architecture evolution.

**Solution:**

**The claim:** In the space of all possible sequence-to-sequence operations, attention occupies a local optimum -- it may not be the global best, but every nearby alternative performs worse at the same computational cost. The SGST architecture exploration provides systematic evidence for this.

**Evidence from the architecture evolution:**

Every alternative to attention was implemented and tested:

| Mechanism | Version | Cost | Result vs Attention |
|-----------|---------|------|-------------------|
| Message passing | V5-V9 | $O(Ed)$ | 45% wall, far below attention |
| Explicit Riemannian metric | V10 | $O(Td^2)$ | Context-blind, failed |
| Diffusion + Langevin | V11 | $O(Td)$ iter | Slow, did not converge |
| Spectral transport | V12 | $O(Ts \log d)$ | Worse than SSM+MLP |
| SSM + Hopfield | V13-V14 | $O(Td)$ | Plateau at CE 2.17 |
| Parseval filter | V15 | $O(Td \log d)$ | Improvement but not competitive |
| Multi-path mixing | V16 | $O(Td \log d)$ | PPL 275 vs GPT's 173 |

None matched attention at the same parameter count and training budget.

**Why attention is hard to beat:**

1. **Pairwise expressivity.** Attention computes the exact pairwise similarity $q_i \cdot k_j$ for ALL $(i, j)$ pairs. This is the most expressive $O(T^2 d)$ operation: it can represent any pattern of inter-token relevance. Spectral methods achieve $O(Td \log d)$ but force the relevance pattern through the Fourier basis, which cannot represent arbitrary pairwise patterns.

2. **Content-addressability.** The attention weight $\operatorname{softmax}(q_t \cdot k_j / \sqrt{d})$ depends on BOTH the query (current token) and the key (past token). This is fully content-dependent retrieval. SSMs compress history into a fixed-size state, losing the ability to selectively recall specific past tokens.

3. **Hardware co-optimization.** GPUs are optimized for dense matrix multiplication. Attention's core operation ($QK^\top$) is a large dense matmul that achieves near-peak FLOP utilization. Spectral methods involve FFTs, scatter/gather operations, and conditional logic that achieve much lower hardware utilization despite having lower theoretical FLOPs.

**In what sense "local optimum":**

"Local" means: given the current hardware landscape (GPUs, dense matmul optimized) and the current scale (millions to billions of parameters), attention wins. A "global" optimum might exist on different hardware (e.g., neuromorphic chips where sparse operations are native) or at different scale (e.g., if spectral methods have better asymptotic scaling that only manifests at much larger sizes).

The thesis argues that the geometric understanding gained from the SGST exploration is not wasted -- it reveals WHY attention works (it implements parallel transport, it performs spectral filtering, it solves a reaction-diffusion equation). This understanding enables the CurvBias approach: use geometric insight to ENHANCE attention rather than replace it. CurvBias adds content-dependent curvature to position encoding, giving attention better geometric awareness at negligible additional cost. This is the productive outcome of understanding the local optimum: you cannot escape it cheaply, but you can improve it from within.

---

## Comprehension Questions

1. What was the "45% wall" and why did V5-V9 all hit it? What does this tell you about the relationship between routing and representation?

2. Explain the V12 architecture in your own words. What does each component (SpectralTokenEmbedding, ContextAccumulator, SpectralTransport, MemoryBank, HopfieldSettler, SpatialMLP) do, and what is its mathematical motivation?

3. What did the V12.2 ablation reveal? Why is this the most important experiment in the thesis? What were the 4 diagnosed failure modes?

4. Why did V13 plateau at CE 2.17? What is the fundamental limitation of an EMA-based context accumulator, and how does it compare to attention's content-addressable retrieval?

5. In what sense is "attention a local optimum"? What evidence supports this claim, and what does "local" vs "global" optimum mean in this context? How does the CurvBias pivot leverage this understanding?

---

## Bridge to Thesis

This unit synthesizes the entire SGST project arc (Thesis Chapters 5-7). The worked problems trace the mathematical reasoning behind each architectural decision, quantify the experimental results, and dissect the failure modes that ultimately led to the thesis's central conclusion. The "informative failure" narrative (Problem 9) maps directly to the thesis structure: each chapter presents a hypothesis, an implementation, an experiment, a failure, and a lesson. The final conclusion -- that geometric understanding should enhance attention rather than replace it (Problem 10) -- is the thesis's primary contribution. The study units in this series (01-12) provide the mathematical toolkit to understand every component, from the linear algebra of spectral transforms to the differential geometry of fiber bundles to the information theory of compressed sensing, all converging on the architecture and its evolution.
