# Unit 03: Fourier Analysis: From Signals to Representations

## Learning Objectives

By the end of this unit, you will be able to:

1. Understand the Discrete Fourier Transform (DFT) as a change of basis
2. Compute DFT/IDFT by hand for small signals
3. Understand frequency, magnitude, phase, and their physical meaning
4. State and apply Parseval's theorem (energy conservation)
5. Understand the convolution theorem: multiplication in frequency = convolution in time
6. Connect FFT/IFFT to the SGST forward-reverse loop

## Prerequisites

- Unit 01 (complex vectors, unitary matrices)

## Readings

- Bracewell, *The Fourier Transform and Its Applications*, Ch. 1-4
- 3Blue1Brown: "But what is the Fourier Transform?" (YouTube)
- Thesis Sec. 5.1 (Why Sparsity Belongs in Fourier Space)
- Thesis Sec. 5.3.1-5.3.2 (Sparse Spectral Representation, Field Reconstruction via IFFT)
- Repo: v12/README.md (extensive Fourier duality discussion)

---

## Key Concepts

### 1. Periodic Signals and Their Frequency Content

Any discrete signal of length $N$ can be decomposed into a sum of $N$ complex sinusoids at equally spaced frequencies. Each sinusoid has a specific amplitude and phase. This decomposition is unique and invertible -- the signal and its frequency representation contain exactly the same information.

### 2. The DFT Formula

The Discrete Fourier Transform converts a length-$N$ signal $x[n]$ into its frequency-domain representation $X[k]$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-i 2\pi k n / N}, \quad k = 0, 1, \ldots, N-1$$

Each $X[k]$ measures how much of the $k$-th frequency sinusoid is present in the signal. $X[0]$ is always the sum of all samples (the "DC component").

### 3. The IDFT Formula

The Inverse DFT reconstructs the signal from its frequency components:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{i 2\pi k n / N}, \quad n = 0, 1, \ldots, N-1$$

The only differences from the forward DFT are the sign flip in the exponent (positive instead of negative) and the $\frac{1}{N}$ normalization factor.

### 4. The DFT Matrix $F$ -- It Is Unitary (Up to Scaling by $\sqrt{N}$)

Define the $N \times N$ DFT matrix $F$ by $F_{kn} = \exp(-i 2\pi kn / N)$. Then:

- $F^* F = N I$ (where $F^*$ is the conjugate transpose)
- $\frac{1}{\sqrt{N}} F$ is a unitary matrix
- The DFT is a rotation from the time basis to the frequency basis
- No information is created or destroyed -- just re-expressed

### 5. Magnitude $|X[k]|$ -- How Much of Frequency $k$ Is Present

The magnitude $|X[k]|$ tells you the amplitude of the $k$-th frequency component. A large $|X[k]|$ means frequency $k$ contributes strongly to the signal. The magnitude spectrum $|X[0]|, |X[1]|, \ldots, |X[N-1]|$ gives a "fingerprint" of the signal's frequency content.

### 6. Phase $\arg(X[k])$ -- Where in Its Cycle Frequency $k$ Starts

The phase angle $\arg(X[k])$ tells you the starting offset of the $k$-th sinusoid. Two signals can have identical magnitude spectra but sound/look completely different due to different phases. Phase carries structural information about how the frequency components are aligned.

### 7. Parseval's Theorem: Energy Conserved Between Domains

$$\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2$$

The total energy (sum of squared magnitudes) is the same whether computed in the time domain or the frequency domain. This means the DFT preserves the geometry of the signal space -- distances and angles are unchanged. No energy is lost or gained by changing representation.

### 8. Convolution Theorem: $\text{DFT}(x * y) = \text{DFT}(x) \cdot \text{DFT}(y)$

Circular convolution in the time domain corresponds to pointwise multiplication in the frequency domain:

$$\text{DFT}(x * y) = \text{DFT}(x) \cdot \text{DFT}(y)$$

This is enormously powerful: convolution is $O(N^2)$ directly, but via FFT it becomes $O(N \log N)$. Multiply in frequency, then IFFT back. This is the computational backbone of spectral methods.

### 9. Sparsity in the Frequency Domain

A signal is "spectrally sparse" if only a few of its DFT coefficients $X[k]$ are nonzero (or significantly large). This means the signal can be described by a small number of frequency modes. Many natural signals are spectrally sparse -- they have simple frequency structure even if their time-domain representation looks complicated.

### 10. The FFT Algorithm -- $O(N \log N)$ Instead of $O(N^2)$

The Fast Fourier Transform computes the DFT in $O(N \log N)$ operations instead of the naive $O(N^2)$. It exploits the recursive structure of the DFT matrix by splitting even and odd indices (Cooley-Tukey algorithm). This is what makes spectral methods practical for large $N$. Without FFT, the DFT would be no faster than direct convolution.

---

## Worked Problems

### Problem 1

**Compute the DFT of $x = [1, 0, -1, 0]$ by hand ($N=4$). Express each $X[k]$ in magnitude/phase form.**

**Solution:**

The twiddle factor is $W_N = \exp(-i 2\pi/N) = \exp(-i \pi/2) = -i$ for $N=4$.

Apply the DFT formula $X[k] = \sum_{n=0}^{3} x[n] \exp(-i 2\pi kn/4)$:

**$X[0]$:** $x[0] + x[1] + x[2] + x[3] = 1 + 0 + (-1) + 0 =$ **0**

**$X[1]$:** $x[0] + x[1]W^1 + x[2]W^2 + x[3]W^3$
$= 1 + 0(-i) + (-1)(-1) + 0(i)$
$= 1 + 0 + 1 + 0 =$ **2**

**$X[2]$:** $x[0] + x[1]W^2 + x[2]W^4 + x[3]W^6$
$= 1 + 0(-1) + (-1)(1) + 0(-1)$
$= 1 + 0 + (-1) + 0 =$ **0**

**$X[3]$:** $x[0] + x[1]W^3 + x[2]W^6 + x[3]W^9$
$= 1 + 0(i) + (-1)(-1) + 0(-i)$
$= 1 + 0 + 1 + 0 =$ **2**

So $X = [0, 2, 0, 2]$.

In magnitude/phase form:
- $|X[0]| = 0$, phase undefined
- $|X[1]| = 2$, phase $= \arg(2) = 0$
- $|X[2]| = 0$, phase undefined
- $|X[3]| = 2$, phase $= \arg(2) = 0$

The signal has only frequencies $k=1$ and $k=3$ active -- it is **spectrally sparse** with only 2 of 4 modes active.

---

### Problem 2

**Verify Parseval's theorem for $x = [1, 0, -1, 0]$ and its DFT $X = [0, 2, 0, 2]$.**

**Solution:**

**Time-domain energy:**

$$\sum |x[n]|^2 = |1|^2 + |0|^2 + |-1|^2 + |0|^2 = 1 + 0 + 1 + 0 = \textbf{2}$$

**Frequency-domain energy:**

$$\frac{1}{N} \sum |X[k]|^2 = \frac{1}{4}(|0|^2 + |2|^2 + |0|^2 + |2|^2) = \frac{1}{4}(0 + 4 + 0 + 4) = \frac{1}{4}(8) = \textbf{2}$$

Time-domain energy = Frequency-domain energy = 2. Parseval's theorem is verified.

Energy is conserved across the DFT. The signal's "power" is the same whether measured in time or frequency.

---

### Problem 3

**Given a signal $x$ with DFT $X = [5, 0, 0, 3+4i, 0, 0, 0, 3-4i]$, how many active frequency modes are there? What is the spectral sparsity? Reconstruct $x$ using IDFT.**

**Solution:**

**Active modes:** $X[k]$ is nonzero at $k = 0, 3, 7$. That is **3 active modes** out of 8 total.

**Spectral sparsity:** $5/8 =$ **62.5%** of modes are zero (inactive).

**IDFT reconstruction:**

$$x[n] = \frac{1}{8} \sum_{k=0}^{7} X[k] \exp(i 2\pi kn/8)$$

Since only $k = 0, 3, 7$ contribute:

$$x[n] = \frac{1}{8}\left[5 + (3+4i) \exp(i 2\pi \cdot 3n/8) + (3-4i) \exp(i 2\pi \cdot 7n/8)\right]$$

Note that $X[7] = X[8-1] = \overline{X[1]}$... but actually $X[7] = \overline{X[1]}$ only if $x$ is real and index $7 = N - 1$. Here $X[7] = 3-4i = \overline{X[3]} = \overline{3+4i}$, confirming $x$ is real (conjugate symmetry: $X[N-k] = \overline{X[k]}$).

Using $\exp(i 2\pi \cdot 7n/8) = \exp(-i 2\pi n/8)$:

$$x[n] = \frac{1}{8}\left[5 + (3+4i) \exp(i 6\pi n/8) + (3-4i) \exp(-i 6\pi n/8)\right]$$

$$= \frac{1}{8}\left[5 + 2 \operatorname{Re}\{(3+4i) \exp(i 3\pi n/4)\}\right]$$

$$= \frac{1}{8}\left[5 + 6 \cos(3\pi n/4) - 8 \sin(3\pi n/4)\right]$$

Computing each value:
- $x[0] = \frac{1}{8}[5 + 6(1) - 8(0)] = \frac{11}{8} = 1.375$
- $x[1] = \frac{1}{8}[5 + 6(-\frac{\sqrt{2}}{2}) - 8(\frac{\sqrt{2}}{2})] = \frac{1}{8}[5 - 3\sqrt{2} - 4\sqrt{2}] = \frac{5 - 7\sqrt{2}}{8}$
- $x[2] = \frac{1}{8}[5 + 6(0) - 8(-1)] = \frac{13}{8} = 1.625$
- $x[3] = \frac{1}{8}[5 + 6(\frac{\sqrt{2}}{2}) - 8(-\frac{\sqrt{2}}{2})] = \frac{5 + 7\sqrt{2}}{8}$
- $x[4] = \frac{1}{8}[5 + 6(-1) - 8(0)] = -\frac{1}{8}$
- $x[5] = \frac{1}{8}[5 + 6(\frac{\sqrt{2}}{2}) - 8(\frac{\sqrt{2}}{2})] = \frac{5 - \sqrt{2}}{8}$
- $x[6] = \frac{1}{8}[5 + 6(0) - 8(1)] = -\frac{3}{8}$
- $x[7] = \frac{1}{8}[5 + 6(-\frac{\sqrt{2}}{2}) - 8(-\frac{\sqrt{2}}{2})] = \frac{5 + \sqrt{2}}{8}$

All values are real, confirming the conjugate symmetry.

---

### Problem 4

**Show that the $N \times N$ DFT matrix $F$ with $F_{kn} = \exp(-i 2\pi kn/N)$ satisfies $F^* F = NI$ (i.e., $\frac{1}{\sqrt{N}} F$ is unitary). What does this mean about DFT as a "rotation"?**

**Solution:**

We need to compute $(F^* F)_{mn}$ where $F^*$ is the conjugate transpose.

$$(F^* F)_{mn} = \sum_{k=0}^{N-1} (F^*)_{mk} F_{kn} = \sum_{k=0}^{N-1} \overline{F_{km}} F_{kn}$$

$$= \sum_{k=0}^{N-1} \exp(i 2\pi km/N) \exp(-i 2\pi kn/N) = \sum_{k=0}^{N-1} \exp(i 2\pi k(m-n)/N)$$

**Case $m = n$:** Each term is $\exp(0) = 1$. The sum is $N$.

**Case $m \neq n$:** Let $r = \exp(i 2\pi (m-n)/N)$. Since $m \neq n$ and $0 \leq m,n < N$, we have $r \neq 1$. This is a geometric series:

$$\sum_{k=0}^{N-1} r^k = \frac{1 - r^N}{1 - r} = \frac{1 - \exp(i 2\pi (m-n))}{1 - r} = \frac{1 - 1}{1 - r} = \textbf{0}$$

Therefore $(F^* F)_{mn} = N$ if $m = n$, and $0$ if $m \neq n$. That is, $F^* F = NI$.

Dividing by $\sqrt{N}$: let $U = \frac{1}{\sqrt{N}} F$. Then $U^* U = \frac{1}{N} F^* F = I$, so **$U$ is unitary**.

**Interpretation:** The DFT is a unitary rotation (scaled by $\sqrt{N}$) from the time basis to the frequency basis. Since unitary transformations preserve inner products, distances, and angles, no information is lost or created -- the signal is merely re-expressed in a different coordinate system. This is why the DFT is called a "change of basis."

---

### Problem 5

**The convolution theorem states $\text{DFT}(x * y) = \text{DFT}(x) \cdot \text{DFT}(y)$. Use this to compute the circular convolution of $x = [1, 2, 3, 0]$ and $y = [1, 0, 1, 0]$ ($N=4$).**

**Solution:**

**Step 1: Compute $\text{DFT}(x)$ for $x = [1, 2, 3, 0]$ with $W = \exp(-i \pi/2) = -i$.**

$X[0] = 1 + 2 + 3 + 0 = 6$
$X[1] = 1 + 2(-i) + 3(-1) + 0(i) = 1 - 2i - 3 = -2 - 2i$
$X[2] = 1 + 2(-1) + 3(1) + 0(-1) = 1 - 2 + 3 = 2$
$X[3] = 1 + 2(i) + 3(-1) + 0(-i) = 1 + 2i - 3 = -2 + 2i$

**Step 2: Compute $\text{DFT}(y)$ for $y = [1, 0, 1, 0]$.**

$Y[0] = 1 + 0 + 1 + 0 = 2$
$Y[1] = 1 + 0(-i) + 1(-1) + 0(i) = 1 - 1 = 0$
$Y[2] = 1 + 0(-1) + 1(1) + 0(-1) = 2$
$Y[3] = 1 + 0(i) + 1(-1) + 0(-i) = 0$

**Step 3: Pointwise multiply.**

$Z[0] = X[0] \cdot Y[0] = 6 \times 2 = 12$
$Z[1] = X[1] \cdot Y[1] = (-2-2i) \times 0 = 0$
$Z[2] = X[2] \cdot Y[2] = 2 \times 2 = 4$
$Z[3] = X[3] \cdot Y[3] = (-2+2i) \times 0 = 0$

$Z = [12, 0, 4, 0]$

**Step 4: IDFT to get the convolution.**

$$z[n] = \frac{1}{4} \sum_{k} Z[k] \exp(i 2\pi kn/4) = \frac{1}{4}(12 + 4 \exp(i \pi n)) = \frac{1}{4}(12 + 4(-1)^n)$$

$z[0] = \frac{1}{4}(12 + 4) = 4$
$z[1] = \frac{1}{4}(12 - 4) = 2$
$z[2] = \frac{1}{4}(12 + 4) = 4$
$z[3] = \frac{1}{4}(12 - 4) = 2$

$\mathbf{z = [4, 2, 4, 2]}$

**Verification by direct circular convolution:**

$z[0] = x[0]y[0] + x[1]y[3] + x[2]y[2] + x[3]y[1] = 1(1) + 2(0) + 3(1) + 0(0) = 4$. Matches.
$z[1] = x[0]y[1] + x[1]y[0] + x[2]y[3] + x[3]y[2] = 1(0) + 2(1) + 3(0) + 0(1) = 2$. Matches.

---

### Problem 6

**In the SGST architecture, a token is represented as a sparse spectral vector with 10 active modes out of 32 total. What is the minimum spatial extent guaranteed by the uncertainty principle? ($|\operatorname{supp}(x)| \cdot |\operatorname{supp}(X)| \geq N$)**

**Solution:**

The discrete uncertainty principle states:

$$|\operatorname{supp}(x)| \cdot |\operatorname{supp}(X)| \geq N$$

where $|\operatorname{supp}(x)|$ is the number of nonzero time-domain samples and $|\operatorname{supp}(X)|$ is the number of nonzero frequency-domain coefficients.

Given: $|\operatorname{supp}(X)| = 10$, $N = 32$.

$$10 \cdot |\operatorname{supp}(x)| \geq 32$$

$$|\operatorname{supp}(x)| \geq \frac{32}{10} = 3.2$$

Since $|\operatorname{supp}(x)|$ must be an integer, $|\operatorname{supp}(x)| \geq 4$.

With only 10 spectral modes, the spatial signal MUST spread across **at least 4 spatial positions**. It cannot be more localized than that.

**Why this matters for SGST:** Spectral sparsity guarantees spatial spread. A spectrally sparse token cannot be spatially localized -- it inherently has long-range extent. This means that working in the spectral domain automatically provides long-range interaction without explicit attention mechanisms. The fewer spectral modes you use, the MORE spatially spread the signal must be.

---

### Problem 7

**Explain why pointwise multiplication in frequency domain equals circular convolution in the spatial domain. Draw what this means for the SGST "spectral transport kernel."**

**Solution:**

**The mathematical connection:**

If $Z[k] = X[k] \cdot H[k]$ (pointwise product in frequency), then in the spatial domain:

$$z[n] = \sum_{m=0}^{N-1} x[m] \, h[(n-m) \bmod N] = (x * h)[n]$$

This is circular convolution. Each output sample $z[n]$ is a weighted sum of ALL input samples $x[m]$, with weights determined by the kernel $h$.

**Proof sketch:** Apply IDFT to both sides of $Z[k] = X[k] H[k]$:

$$z[n] = \frac{1}{N} \sum_k X[k] H[k] \exp(i 2\pi kn/N)$$

$$= \frac{1}{N} \sum_k \left[\sum_m x[m] \exp(-i 2\pi km/N)\right] H[k] \exp(i 2\pi kn/N)$$

$$= \sum_m x[m] \left[\frac{1}{N} \sum_k H[k] \exp(i 2\pi k(n-m)/N)\right]$$

$$= \sum_m x[m] \, h[n-m]$$

**What this means for SGST's spectral transport kernel:**

The SGST transport kernel acts in the frequency domain as:

$$H[k] = \exp(-D_k \omega_k^2 - i A_k \omega_k)$$

This acts **pointwise** on each frequency mode:
- **Damping:** $\exp(-D_k \omega_k^2)$ scales the amplitude of mode $k$ (higher frequencies damped more -- diffusion)
- **Phase shift:** $\exp(-i A_k \omega_k)$ rotates the phase of mode $k$ (advection/transport)

In the spatial domain, this corresponds to **convolving the token with the kernel** -- a global operation that mixes information across all positions. Each spatial output depends on every spatial input, weighted by the kernel.

This is how transport happens without pairwise attention: the spectral kernel implicitly creates a global convolution. The cost is $O(N \log N)$ via FFT, compared to $O(N^2)$ for explicit pairwise computation.

---

### Problem 8

**For a real signal of length 8, the DFT has conjugate symmetry: $X[k] = \overline{X[N-k]}$. How many independent complex values are there? How does rfft exploit this?**

**Solution:**

For a real signal $x[n]$ of length $N = 8$, the DFT satisfies the **conjugate symmetry property:**

$$X[k] = \overline{X[N-k]} \quad \text{for all } k$$

Let us enumerate all 8 frequency bins and their constraints:

| Bin | Constraint | Type |
|-----|-----------|------|
| $X[0]$ | $X[0] = \overline{X[0]}$, so $X[0]$ is real | 1 real value |
| $X[1]$ | $X[1] = \overline{X[7]}$ | 1 complex value (2 reals) |
| $X[2]$ | $X[2] = \overline{X[6]}$ | 1 complex value (2 reals) |
| $X[3]$ | $X[3] = \overline{X[5]}$ | 1 complex value (2 reals) |
| $X[4]$ | $X[4] = \overline{X[4]}$, so $X[4]$ is real | 1 real value |
| $X[5]$ | determined by $X[3]$ | redundant |
| $X[6]$ | determined by $X[2]$ | redundant |
| $X[7]$ | determined by $X[1]$ | redundant |

**Independent values:**
- $X[0]$: 1 real number
- $X[1], X[2], X[3]$: 3 complex numbers = 6 real numbers
- $X[4]$: 1 real number

**Total: 8 real degrees of freedom** -- exactly matching the 8 real input values. No information is gained or lost.

**How rfft exploits this:**

The `rfft` function returns only $X[0]$ through $X[N/2]$, i.e., $X[0], X[1], X[2], X[3], X[4]$ -- just **5 complex values** for $N=8$. The remaining $X[5], X[6], X[7]$ are not stored because they are determined by conjugate symmetry.

The `irfft` function reconstructs the full signal from these $N/2 + 1$ values by implicitly filling in the conjugate-symmetric bins.

**In SGST:** The architecture uses `rfft`/`irfft` to exploit this symmetry, working only with the non-redundant half of the spectrum. This halves the memory footprint and ensures outputs are always real-valued.

---

### Problem 9

**A "low-pass filter" zeros out high-frequency components. Given $X = [10, 5, 3, 1, 0.5, 1, 3, 5]$, apply an ideal low-pass filter keeping only $k=0, 1, 7$ (i.e., the lowest 3 frequencies for a real signal). What does the filtered signal look like?**

**Solution:**

**Understanding the frequency ordering:** For a real signal of length $N=8$, the frequencies are ordered as:

$k = 0$ (DC), $1, 2, 3, 4$ (Nyquist), $5, 6, 7$

Due to conjugate symmetry, $k=7$ corresponds to the same physical frequency as $k=1$ (just the negative frequency). So "lowest 3 frequencies" means $k=0$ (DC) and the $k=1$/$k=7$ pair (lowest non-DC frequency).

**Applying the filter (zero out $k=2,3,4,5,6$):**

$X_{\text{filtered}} = [10, 5, 0, 0, 0, 0, 0, 5]$

**IDFT of the filtered signal:**

$$x_{\text{filt}}[n] = \frac{1}{8}(10 + 5 \exp(i 2\pi n/8) + 5 \exp(i 2\pi \cdot 7n/8))$$

$$= \frac{1}{8}(10 + 5 \exp(i 2\pi n/8) + 5 \exp(-i 2\pi n/8))$$

$$= \frac{1}{8}(10 + 10 \cos(2\pi n/8))$$

Computing values:
- $x_{\text{filt}}[0] = \frac{1}{8}(10 + 10) = 2.5$
- $x_{\text{filt}}[1] = \frac{1}{8}(10 + 10 \cos(\pi/4)) = \frac{1}{8}(10 + 7.071) = 2.134$
- $x_{\text{filt}}[2] = \frac{1}{8}(10 + 10 \cos(\pi/2)) = \frac{1}{8}(10 + 0) = 1.25$
- $x_{\text{filt}}[3] = \frac{1}{8}(10 + 10 \cos(3\pi/4)) = \frac{1}{8}(10 - 7.071) = 0.366$
- $x_{\text{filt}}[4] = \frac{1}{8}(10 + 10 \cos(\pi)) = \frac{1}{8}(10 - 10) = 0$
- $x_{\text{filt}}[5] = \frac{1}{8}(10 + 10 \cos(5\pi/4)) = 0.366$
- $x_{\text{filt}}[6] = 1.25$
- $x_{\text{filt}}[7] = 2.134$

The result is a **smooth sinusoid** -- all the sharp, rapidly varying features have been removed.

**Connection to GNNs:** This is exactly what GNN message passing does -- it acts as a low-pass filter on the graph Fourier transform (thesis Sec. 2.1.3). Repeated message passing suppresses high-frequency components, eventually smoothing all node features to the same value (over-smoothing). The SGST instead works WITH spectral components directly, manipulating individual frequency modes rather than filtering them away.

---

### Problem 10

**The SGST forward-reverse loop is: sparse spectral -> IFFT -> dense spatial -> processing -> FFT -> sparse spectral. Explain using Fourier concepts why this cycle is mathematically natural and what each step does to the representation.**

**Solution:**

The forward-reverse loop is the core computational cycle of SGST. Each step has a precise Fourier interpretation:

**(a) Sparse Spectral (starting point):**
The token lives as a few active frequency modes -- say 10 of 32 are nonzero. This is the "compressed" representation. By the uncertainty principle (Problem 6), these 10 modes MUST correspond to a spatially extended signal. The sparsity is the information bottleneck: only the essential spectral content is retained.

**(b) IFFT (field reconstruction):**
Applying the Inverse FFT transforms from frequency domain to spatial domain. The few active modes produce a globally-extended spatial signal. This is "field reconstruction" -- the sparse spectral data is expanded into a dense transient representation that fills all spatial positions. Mathematically, each mode $k$ contributes a sinusoid $\exp(i 2\pi kn/N)$ at every position $n$.

**(c) Dense Spatial (processing):**
In the spatial domain, the representation is dense -- all positions are populated. This is where pointwise nonlinearities (MLP, activation functions) can act. Nonlinearities are essential because they create NEW frequency content -- in the spectral domain, nonlinear operations couple different modes. This is the "reaction" in the reaction-diffusion analogy.

**(d) FFT (return to spectral):**
The FFT transforms back to the frequency domain. Any new spectral content created by the nonlinearity is now visible as new nonzero modes. Old spectral structure is preserved (Parseval's theorem guarantees energy conservation throughout the cycle).

**(e) Re-sparsification (top-$k$ selection):**
After the FFT, a top-$k$ operation selects only the largest-magnitude frequency modes, returning to a sparse spectral representation. This enforces the information bottleneck: only the most important spectral content survives.

**Why the cycle is natural:**
- The DFT/IDFT are exact inverses -- no information is lost in the domain transitions
- Parseval's theorem guarantees energy conservation at every step
- Sparsity is natural in the frequency domain (many signals have few dominant modes)
- Nonlinearities require the spatial domain (they are pointwise operations)
- The cycle respects the duality: sparsity for compression, density for processing

**Computational cost:** The entire cycle is $O(N \log N)$ due to FFT, compared to $O(N^2)$ for attention-based processing. This is the fundamental efficiency advantage of the spectral approach.

---

## Comprehension Questions

1. In your own words, what does the DFT do? Why is it described as a "change of basis"?

2. State the uncertainty principle for DFT. Why does spectral sparsity force spatial spread?

3. What is the convolution theorem, and why does it make spectral transport $O(N \log N)$ instead of $O(N^2)$?

4. Why does the thesis argue sparsity should be in the frequency domain rather than the spatial domain? (Relate to thesis Sec. 5.1)

5. What is Parseval's theorem and why does it guarantee the FFT/IFFT cycle preserves information?

---

## Bridge to Thesis

The Fourier concepts in this unit are the mathematical foundation for the entire SGST architecture. The key insight (thesis Sec. 5.1) is that **sparsity belongs in Fourier space**: rather than selecting a subset of spatial tokens (which destroys long-range information), SGST selects a subset of frequency modes (which preserves spatial extent via the uncertainty principle). The forward-reverse FFT/IFFT loop (thesis Sec. 5.3) is not an approximation -- it is an exact, energy-preserving change of basis that enables:

- **$O(s)$ transport** in the spectral domain (where $s$ is the number of active modes)
- **Pointwise nonlinear processing** in the spatial domain (where dense representations enable feature mixing)
- **Information-theoretic compression** via spectral sparsity (top-$k$ mode selection)

In the next unit (Unit 04), you will see how these ideas generalize from regular signals to signals on graphs, and why the graph Fourier transform reveals the fundamental limitations of GNN message passing that SGST overcomes.
