# Unit 11: Spectral Methods in Neural Networks: FNO, Spectral Convolution, and Reaction-Diffusion

## Learning Objectives

1. Understand Fourier Neural Operators (FNO) and spectral convolution
2. See how pointwise multiplication in frequency domain implements global convolution
3. Understand the reaction-diffusion decomposition of transformers
4. Connect spectral methods to the SGST's V12 architecture
5. Understand the Parseval spectral filter and energy-constrained spectral operations

## Prerequisites

Units 03 (Fourier Analysis), 04 (Graph Theory & Spectral Methods), 10

## Readings

- Li et al. 2021, "Fourier Neural Operator for Parametric PDEs"
- Shi, Zhu, Liu 2025, "A Unified Geometric Field Theory Framework for Transformers" (thesis reference)
- Thesis Sec. 2.7 (Spectral Methods in Deep Learning), all subsections
- Thesis Sec. 5.3 (The V12 Architecture)
- Thesis Sec. 7.1.3 (The Reaction-Diffusion Decomposition)
- Repo: `v12/README.md` (spectral transport design)
- Repo: `v15/DESIGN.md` (Parseval spectral attention)

---

## Key Concepts

1. **FNO: learn a linear operator in Fourier space.** The core operation is $\mathcal{F}^{-1}(R \cdot \mathcal{F}(v))$ where $R$ is a learned spectral filter. The network parameterizes the kernel in frequency domain rather than spatial domain.

2. **Multiplication in Fourier = convolution in spatial.** This is the convolution theorem: pointwise multiplication of two spectra corresponds to circular convolution of their spatial signals. A single Fourier-domain multiply achieves a global receptive field in one layer.

3. **Spectral convolution formula.** $y[n] = \text{IFFT}(H[k] \cdot \text{FFT}(x[n]))$ where $H$ is a learned transfer function. Each frequency bin $k$ gets its own learned complex weight $H[k]$.

4. **Spectral efficiency.** FFT costs $O(N \log N)$ vs $O(N^2)$ for direct convolution. For sequence length $N = 1024$, this is roughly 10,000 vs 1,000,000 operations.

5. **Reaction-diffusion framework.** A transformer block decomposes as: diffusion (attention, spatial mixing across positions) + reaction (FFN, local per-position nonlinearity). This mirrors the PDE: $\frac{du}{dt} = D \nabla^2 u + f(u)$.

6. **The heat kernel in frequency domain.** $\exp(-D \omega^2 t)$ in frequency domain = Gaussian smoothing in spatial domain. High frequencies decay exponentially faster than low frequencies.

7. **Parseval frame.** $\|Wx\|^2 = \|x\|^2$ if $W$ is a Parseval frame. This is an energy-preserving transform: total signal energy is unchanged by the transformation.

8. **The Parseval spectral filter (V15).** $y = W \cdot h$ where $|W[k]| \leq 1$ ensures energy never increases. This replaces iterative Langevin settling with a single-step, provably stable filter.

9. **Spectral transport kernel.** $\exp(-D_k \omega_k^2 - i A_k \omega_k)$: the diffusion coefficient $D$ controls smoothing (real exponential decay), and the gauge connection $A$ controls phase rotation (transport along the fiber).

10. **The complete V12 forward pass.** embed $\to$ FFT $\to$ sparse spectral $\to$ transport $\to$ IFFT $\to$ MLP $\to$ FFT $\to$ re-sparsify. Each stage has a precise mathematical role in the geometric pipeline.

---

## Worked Problems

### Problem 1: FNO Forward Pass Computation

A Fourier Neural Operator learns spectral weights $R[k]$ for $k = 0, \ldots, K-1$. For input $x \in \mathbb{R}^8$ and $K = 4$ (keeping 4 lowest frequencies), compute the output $y = \text{IFFT}(R \cdot \text{FFT}(x))$ given $x = [1, 2, 3, 4, 5, 6, 7, 8]$ and $R = [1.0,\; 0.5 + 0.2i,\; 0.3 - 0.1i,\; 0.1]$.

**Solution:**

Step 1: Compute $\text{FFT}(x)$. Using the DFT formula $X[k] = \sum_{n=0}^{7} x[n] \exp(-2\pi i k n / 8)$:

- $X[0] = \sum x[n] = 1+2+3+4+5+6+7+8 = 36$
- $X[1] = -4 + 9.66i$ (computed from the DFT formula)
- $X[2] = -4 + 4i$
- $X[3] = -4 + 1.66i$
- $X[4] = -4$
- $X[5]$ through $X[7]$ are conjugate symmetric: $X[k] = X[8-k]^*$

Step 2: Apply the learned filter $R$ to the first $K = 4$ modes:

- $Y[0] = 36 \times 1.0 = 36$
- $Y[1] = (-4 + 9.66i)(0.5 + 0.2i) = (-2 - 1.932) + (4.83 - 0.8)i = -3.932 + 4.03i$
- $Y[2] = (-4 + 4i)(0.3 - 0.1i) = (-1.2 + 0.4) + (1.2 - 0.4)i = -0.8 + 0.8i$
- $Y[3] = (-4 + 1.66i)(0.1) = -0.4 + 0.166i$

Step 3: Set $Y[k] = 0$ for $k = 4, \ldots, 7$ (frequency truncation).

Step 4: IFFT of $Y$ gives the filtered output $y$.

**Key insight:** The FNO keeps only low-frequency components, each modulated by the learned weight $R[k]$. The truncation at $K = 4$ means the output is band-limited -- only smooth, slowly-varying features are retained. The learned $R[k]$ values determine how much each frequency contributes to the output.

---

### Problem 2: Heat Equation as Spectral Low-Pass Filter

The heat equation solution in Fourier domain is $\hat{u}(k, t) = \hat{u}(k, 0) \cdot \exp(-D k^2 t)$. For $D = 0.1$, initial condition $u(x, 0) = \delta(x)$ (Dirac delta, so $\hat{u}(k, 0) = 1$ for all $k$), compute $\hat{u}(k, t=1)$ for $k = 0, 1, 2, 5, 10$.

**Solution:**

$\hat{u}(k, 1) = \exp(-0.1 \cdot k^2)$:

| $k$ | $-D k^2$ | $\hat{u}(k, 1)$ |
|---|--------|-------------|
| 0 | 0      | 1.000       |
| 1 | -0.1   | 0.905       |
| 2 | -0.4   | 0.670       |
| 5 | -2.5   | 0.082       |
| 10 | -10.0 | 0.0000454   |

High frequencies (large $k$) are strongly suppressed. The decay is Gaussian in $k$: $\exp(-D k^2 t)$, which means the suppression grows quadratically with frequency. By time $t = 1$, only the lowest approximately 5 modes survive with appreciable amplitude.

In the spatial domain, this is equivalent to convolving the initial delta function with a Gaussian of width $\sigma = \sqrt{2Dt} = \sqrt{0.2} \approx 0.447$. The delta has "spread out" into a smooth bump.

**Connection to SGST:** The spectral transport kernel in V12 uses exactly this form: $\exp(-D_k \omega_k^2)$ implements diffusion per mode, but with learned, context-dependent $D_k$. Some contexts demand aggressive smoothing (large $D_k$ for noisy inputs), others demand preservation of high-frequency detail (small $D_k$ for structured inputs).

---

### Problem 3: Reaction-Diffusion Decomposition of a Transformer

The reaction-diffusion decomposition says a transformer block computes: $x \to x + \text{Attention}(x) + \text{FFN}(x + \text{Attention}(x))$. Identify the "diffusion" and "reaction" parts and connect them to the PDE framework.

**Solution:**

**Diffusion = Attention(x).** This mixes information across positions (spatial mixing). Each token receives a weighted sum from all other tokens:

$$\text{Attn}(x)_i = \sum_j \operatorname{softmax}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j$$

Information "spreads" from every position to every other position, weighted by relevance. This is analogous to the diffusion term $D \nabla^2 u$ in the PDE: it smooths local variations and propagates information globally.

**Reaction = FFN(.).** This applies a nonlinear transformation per position:

$$\text{FFN}(x)_i = W_2 \cdot \sigma(W_1 \cdot x_i + b_1) + b_2$$

No cross-position mixing occurs -- each token is processed independently. This is analogous to the reaction term $f(u)$ in the PDE: a local, nonlinear process that transforms the signal at each point.

**Together:** diffusion carries signal across positions, reaction processes it locally. The full PDE analogy is:

$$\frac{du}{dt} = D \nabla^2 u + f(u)$$

where $D \nabla^2 u$ is diffusion (spatial spreading) and $f(u)$ is reaction (local transformation).

The V12 SGST implements this literally:
- IFFT (diffusion/field reconstruction from sparse spectral to dense spatial)
- MLP (reaction: per-token nonlinear processing)
- FFT (re-concentration back to sparse spectral domain)

The residual connection $x \to x + \text{block}(x)$ corresponds to a forward Euler step of the ODE with step size $dt = 1$.

---

### Problem 4: Parseval Spectral Filter Energy Bound

The Parseval spectral filter constrains $|W[k]| \leq 1$ for all modes $k$. Show that this ensures $\|y\|^2 \leq \|h\|^2$ where $y[k] = W[k] \cdot h[k]$.

**Solution:**

We use Parseval's theorem, which states that for any signal $h$ and its DFT $H$:

$$\|h\|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |H[k]|^2$$

Now compute the energy of the filtered output $y$:

$$\|y\|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |Y[k]|^2 \quad \text{(by Parseval's theorem)}$$

$$= \frac{1}{N} \sum_{k=0}^{N-1} |W[k] \cdot H[k]|^2$$

$$= \frac{1}{N} \sum_{k=0}^{N-1} |W[k]|^2 |H[k]|^2$$

Since $|W[k]| \leq 1$ for all $k$, we have $|W[k]|^2 \leq 1$, so:

$$\leq \frac{1}{N} \sum_{k=0}^{N-1} |H[k]|^2$$

$$= \|h\|^2 \quad \text{(by Parseval's theorem)}$$

Therefore $\|y\|^2 \leq \|h\|^2$. The filter can only reduce energy, never amplify it.

**Why this matters:** No matter what the model learns for $W[k]$, the output energy is bounded by the input energy. This prevents the instability that plagues deep networks -- gradients can't explode through spectral layers. The V15 design uses this to replace Langevin settling: instead of iteratively descending an energy landscape (which requires multiple steps and is slow), a single Parseval filter step redistributes spectral energy while provably maintaining the energy bound.

**Equality condition:** $\|y\|^2 = \|h\|^2$ if and only if $|W[k]| = 1$ for all $k$ where $H[k] \neq 0$. In this case, the filter is a pure phase rotation (unitary transform), and the operation is energy-preserving rather than energy-reducing.

---

### Problem 5: Spectral Transport Kernel Components

In the V12 architecture, the spectral transport kernel is $K[k] = \exp(-D_k(q)\,\omega_k^2 - i\,A_k(q)\,\omega_k)$, where $q$ is the context vector, $D_k$ is the diffusion rate, and $A_k$ is the gauge connection. Explain what each component does to the signal.

**Solution:**

Write the kernel in polar form by separating real and imaginary parts of the exponent:

$$K[k] = \exp(-D_k \omega_k^2) \cdot \exp(-i A_k \omega_k)$$

**Component 1: $\exp(-D_k \omega_k^2)$ -- the diffusion component.**

This is a real, positive number less than or equal to 1 (since $D_k \geq 0$). For each mode $k$, it acts as an amplitude scaling factor. High-frequency modes (large $\omega_k$) decay faster -- the suppression grows quadratically with frequency. $D_k$ depends on context $q$ through a learned function $D_k = f(q)$, so the model learns context-dependent smoothing:
- Ambiguous or noisy inputs: large $D_k$, aggressive smoothing of high frequencies
- Clear, structured inputs: small $D_k$, preservation of high-frequency detail

In spatial domain, this corresponds to Gaussian blurring with a context-dependent width.

**Component 2: $\exp(-i A_k \omega_k)$ -- the gauge connection component.**

This is a pure phase rotation (unit complex number). Each mode $k$ gets phase-shifted by $A_k \omega_k$ radians. The magnitude $|X[k]|$ is unchanged -- only the phase angle changes. $A_k$ depends on context $q$, implementing parallel transport: the representation's "orientation" rotates based on contextual information.

In spatial domain, a linear phase shift $\exp(-i \alpha \omega)$ corresponds to a translation (shift) of the signal by $\alpha$. So $A_k$ implements a context-dependent, mode-specific spatial shift.

**Together:** the diffusion part smooths the signal (reduces high-frequency content), and the gauge part shifts it (reorients the representation). Context-dependent $D_k$ and $A_k$ mean the model learns to apply different amounts of smoothing and shifting depending on what it has seen so far.

---

### Problem 6: Subbundle Decomposition in V12

Why does the V12 use separate "subbundles" ($n_{\text{subs}} = 8$ groups of $s$ modes each) rather than one large spectral space?

**Solution:**

With $n_{\text{subs}} = 8$ subbundles of $s = 8$ modes each, the total spectral dimension is 64. Each subbundle is processed independently with its own:
- Transport kernel (separate $D_k$, $A_k$ per subbundle)
- Context accumulator (separate SSM state)
- Memory bank (separate Hopfield atoms)

**Benefit 1: Specialization.** Each subbundle can learn to track different linguistic features: one for syntax, another for semantics, another for entity tracking, another for discourse structure. This is directly analogous to multi-head attention where each head attends to different patterns.

**Benefit 2: Computational efficiency.** Processing 8 independent subbundles of size 8 is much cheaper than one subbundle of size 64. Matrix operations scale as $O(s^3)$ per subbundle, so $8 \times O(8^3) = 8 \times 512 = 4096$, versus $O(64^3) = 262144$ for one large space. This is a 64x reduction.

**Benefit 3: Orthogonality and representational diversity.** Different subbundles operate in orthogonal subspaces by construction. They cannot interfere with each other, preventing representational collapse (where all channels learn redundant features).

**Mathematical framing:** This is the orthogonal subbundle decomposition from the topology framework. The fiber at token position $q$ decomposes as:

$$F_q = F_q^{(1)} \oplus F_q^{(2)} \oplus \cdots \oplus F_q^{(8)}$$

where each $F_q^{(k)}$ is an independent $s$-dimensional feature channel. The full fiber bundle $E = M \times F$ decomposes into 8 independent sub-bundles, each with its own connection (transport law) and section (state trajectory).

---

### Problem 7: V12.1 Architectural Refinements

The V12.1 architecture changes include: (a) separate SSM per subbundle, (b) per-subbundle memory banks, (c) 4-block stack instead of 8. How do these changes affect parameter count and quality?

**Solution:**

**(a) Separate SSMs per subbundle.**

In V12.0: one shared SSM of dimension $d$. Cost: $O(d^2)$ parameters for state transition.

In V12.1: 8 separate SSMs, each of dimension $s$. Cost: $8 \times O(s^2) = 8 \times O(64) = O(512)$ vs $O(d^2) = O(65536)$ for $d = 256$.

Each subbundle's SSM can now specialize its context accumulation. The syntax-tracking subbundle accumulates syntactic history differently from the semantic-tracking subbundle.

**(b) Per-subbundle memory banks.**

V12.0: one bank of $d$-dimensional patterns, $n_{\text{atoms}}$ patterns total. Storage: $d \times n_{\text{atoms}}$.

V12.1: 8 banks of $s$-dimensional patterns. Storage: $8 \times s \times n_{\text{atoms}} = 8 \times 8 \times n_{\text{atoms}} = 64 \times n_{\text{atoms}}$.

For $d = 256$: this is a 4x reduction in memory bank parameters ($64$ vs $256$ per atom). Each bank stores patterns relevant to its subbundle's specialization.

**(c) 4 blocks instead of 8.**

Halves the depth of the network. Each block is more capable (better SSMs, better memory), so fewer blocks are needed. Reduces total sequential computation by 2x.

**Combined effect:** V12.1 has 2.35M parameters (vs V12.0's 2.09M -- slightly larger due to per-subbundle overhead). Quality improves from 2.27 BPC to 2.13 BPC, a significant reduction. The parameters are better distributed: more capacity where it matters (SSM specialization, memory specialization), less wasted on a single monolithic state.

---

### Problem 8: State Compression via Spectral Sparsity

The "field reconstruction via IFFT" step in V12 takes a sparse spectral representation and creates a dense spatial one. If $s = 10$ spectral modes are active out of $d = 32$, what is the spatial dimension after IFFT? What is the state compression ratio? Compare to GPT's KV cache.

**Solution:**

**Spatial dimension after IFFT:** The IFFT of a 32-dimensional spectral vector produces a 32-dimensional spatial vector. IFFT is a linear bijection -- it does not change the dimensionality.

**However, the representation is fundamentally different:**
- Spectral domain: only 10 of 32 values are nonzero (the rest are exactly 0)
- Spatial domain: generically all 32 values are nonzero

The uncertainty principle guarantees this: a signal that is sparse in frequency cannot also be sparse in space (and vice versa). The 10 spectral modes "spread out" across all 32 spatial positions.

**State compression ratio:** $10/32 = 31.25\%$ of the spectral coefficients are stored. But since the zeros are known to be zero, the effective state is just 10 complex numbers (20 real values).

**Comparison to GPT KV cache:**

SGST state per sequence position:
- Context vector: $d_{\text{context}} = 256$ values
- Spectral state: $n_{\text{subs}} \times s \times 2$ (real + imaginary) $= 8 \times 8 \times 2 = 128$ values
- Total: 384 values (constant, independent of sequence length)

GPT KV cache at sequence length $T = 128$:
- Per layer: $2 \times d_{\text{model}} \times T$  (K and V)
- Total for 12 layers: $2 \times 12 \times 128 \times 128 = 393{,}216$ values

Ratio: $384 / 393{,}216 = 0.000977$, approximately $1/1024$.

The SGST uses roughly 1000x less state than GPT at $T = 128$. At $T = 1024$, the ratio becomes approximately $1/8000$. This is the fundamental advantage of recurrent spectral architectures: $O(1)$ state vs $O(T)$ state.

---

### Problem 9: The SpatialMLP Parameter Budget

The SpatialMLP in V12 accounts for 50.5% of all parameters. Why is the per-token nonlinearity so important, and why does it need so many parameters?

**Solution:**

**Why nonlinearity is essential:**

Without the MLP, the entire V12 pipeline is linear:
- FFT: linear transform
- Spectral transport (multiply by $K[k]$): linear (pointwise multiplication)
- IFFT: linear transform
- Composition of linear transforms: still linear

A composition of any number of linear operations is itself linear. Linear models can only represent linear functions of the input, which is insufficient for language modeling (language has deeply nonlinear structure: negation, composition, ambiguity resolution).

The MLP introduces the essential nonlinearity:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

where $\sigma$ is ReLU or GELU. This allows the network to compute nonlinear features of the spatially-reconstructed signal.

**Why it needs many parameters:**

The MLP typically expands the dimension by a factor of 4:
- $W_1$: $d_{\text{model}} \times 4 d_{\text{model}}$ (maps to expanded hidden dimension)
- $W_2$: $4 d_{\text{model}} \times d_{\text{model}}$ (maps back)
- Total per block: approximately $8 \times d_{\text{model}}^2$ parameters

For $d_{\text{model}} = 256$ and 4 blocks: $4 \times 8 \times 256^2 = 2{,}097{,}152$ parameters.

**Comparison to standard transformers:** In a standard transformer, the FFN also accounts for approximately 2/3 of all parameters (the attention matrices Q, K, V, O contribute the other 1/3). The SGST's 50.5% for the MLP is actually LESS than the typical 67%.

**Interpretation:** The MLP is where the "intelligence" lives -- it learns the complex nonlinear mappings from spatial features to useful representations. The spectral/geometric machinery provides structured, principled input to the MLP; the MLP does the heavy lifting of nonlinear feature extraction.

---

### Problem 10: The Devastating V12.2 Ablation

The V12.2 ablation showed that SSM+MLP (without any spectral machinery) performs better than the full V12.1. Explain why this is a critical finding and what it means for the spectral architecture.

**Solution:**

**The numbers:**

| Model | Val BPC | ms/step |
|-------|---------|---------|
| Full V12.1 | 2.302 | 321 |
| SSM+MLP only | 2.267 | 68 |

SSM+MLP achieves better quality ($2.267 < 2.302$) at 4.7x lower computational cost (68 ms vs 321 ms).

The spectral machinery's net contribution to quality: $2.267 - 2.302 = -0.035$ BPC. It is negative -- the spectral components actually hurt performance while consuming 79% of the compute budget.

**Why this is critical:**

The entire V12 thesis was that spectral sparsity + geometric transport would be a superior alternative to attention. The ablation shows that removing the spectral transport, Hopfield memory, FFT/IFFT cycle, and sparsification -- keeping only the simplest components (SSM for context, MLP for nonlinearity) -- produces a better model.

**Diagnosis -- the 4 failure modes:**

1. **Transport was mode-wise/linear.** Each frequency mode $k$ was processed independently by its own kernel $K[k]$. There was no mechanism for mode $k$ to influence mode $j$. This is like an audio equalizer that adjusts volume per frequency band but cannot create new frequencies from combinations of existing ones. Real language requires cross-frequency interaction (syntactic structure at one frequency scale affects semantic content at another).

2. **No cross-mode spectral interactions.** The transport kernel is diagonal in frequency space: a diagonal matrix applied to the spectral vector. For the spectral transport to add value beyond what a simple SSM provides, it would need off-diagonal terms -- allowing mode $k$ to drive changes in mode $j$.

3. **Mode selection resets per block.** At the end of each block, top-$k$ sparsification selects the $s$ most energetic modes. This set can change completely from one block to the next, destroying any continuity of the spectral submanifold across blocks. The model cannot build up a consistent spectral representation over multiple blocks.

4. **Sparsity was too mild.** The learned sparsity was approximately 60% (keeping 40% of modes). For compressed sensing guarantees to kick in (exact recovery from sparse measurements), much higher sparsity is needed (keeping 5-10% of modes). At 40%, the spectral representation is not meaningfully different from the full representation.

**The lesson:** Theoretical elegance does not automatically translate to practical improvement. The spectral framework correctly describes aspects of what attention does, but implementing it with cheaper operations lost the expressivity that makes attention effective. The path forward: use geometric understanding to enhance attention (CurvBias), not replace it.

---

## Comprehension Questions

1. How does a Fourier Neural Operator achieve a global receptive field in one layer? What property of the Fourier transform makes this possible?

2. What is the reaction-diffusion decomposition? Identify the reaction and diffusion components in a standard Transformer block, and explain what each contributes.

3. Why does the Parseval constraint $|W[k]| \leq 1$ prevent instability? Give the mathematical argument and explain why this matters for deep networks.

4. Explain the V12 forward pass step by step. What is each stage (embed, FFT, sparse, transport, IFFT, MLP, FFT, re-sparsify) doing mathematically?

5. The V12.2 ablation showed spectral machinery added no value -- in fact it hurt. What were the 4 diagnosed failure modes, and what was the broader lesson for architecture design?

---

## Bridge to Thesis

This unit covers the mathematical foundations behind the SGST's spectral architecture (Thesis Ch. 5) and the reaction-diffusion interpretation of transformers (Thesis Sec. 7.1.3). The key concepts -- FNO, spectral convolution, Parseval filtering, and the reaction-diffusion framework -- directly motivate the design choices in V12-V16. The worked problems trace the mathematical reasoning from abstract spectral theory to concrete architectural decisions, and the V12.2 ablation analysis (Problem 10) connects to the thesis's central argument about attention as a local optimum. The next unit (Unit 12) builds on this by examining the full architectural evolution and the lessons learned from each version's failure.
