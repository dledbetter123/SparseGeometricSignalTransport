# Version History: What Each Version Taught

## The Full Arc

The project spans v1 through v13, each version diagnosing a failure and correcting it. The trajectory tells a story of progressively discovering which geometric structures are load-bearing.

---

## V1-V5: Routing Failures (12-35% accuracy on synthetic data)

Various approaches to routing tokens through geometric structures. V3's GRU worked best — **content-dependent selectivity is essential**. This lesson would be forgotten and relearned multiple times.

## V5-V9: The 45% Wall (Shakespeare)

Every content-independent routing mechanism (attention gauge, sparse subspace, diffusion, geometric attention) hits the same wall at 45% accuracy / ~2.65 BPC on Shakespeare, regardless of routing strategy.

The diagnosis from v10:

> "Nine architecture versions. Four different routing mechanisms. One consistent result: ~45% accuracy on Shakespeare, ~2.65 BPC, regardless of routing strategy. The routing isn't the bottleneck. The **representations** are."

## V9 → V10: "The Manifold Is Fake"

The smoking gun: `nn.Embedding(max_seq_len, manifold_dim)` made the manifold a positional lookup table. It was context-blind — the "contextual coordinate" q had no context in it.

**Lesson:** The manifold coordinate q_t must be a function of the input sequence q_t = f(x_0, ..., x_t), not a learned positional embedding.

## V10 → V11: Diffusion Is Field Reconstruction, Not Smoothing

The Alcubierre metaphor crystallized. Sparse activations are hot probes on a metal plate. Heat diffuses from the probes, and the temperature field reflects the integrated influence of all sources, weighted by the manifold geometry (variable conductivity = context-dependent metric).

A token is an EVENT (sparse activations across subbundles), not a dense vector. The forward-reverse loop:

```
Sparse sources -> Diffusion (field reconstruction) -> Dense cloud -> Langevin settling -> Sparse output
```

## V11 → V12: Donoho-Stark Kills Spatial Sparsity

The uncertainty principle `|supp(x)| * |supp(x_hat)| >= d` means spatial sparsity forces spectral spread. Since transport operates in Fourier space (heat kernel, Wilson lines), spatially sparse tokens require O(d log d) transport — touching all frequency bins.

**Resolution:** Move sparsity to spectral domain. Few active frequency modes. Transport becomes O(s).

The forward-reverse loop is revealed as Fourier duality:

```
Sparse spectral -> IFFT -> Dense spatial -> Spectral Langevin -> Sparse spectral
```

## V12.0-V12.2: Three Bugs

1. **Conjugate symmetry violation**: Using `fft` instead of `rfft`/`irfft` for real-valued signals
2. **Proximal every step**: Applying soft-thresholding at every Langevin step instead of only at the end, preventing the dynamics from exploring
3. **Transport only dampens**: Using `exp(-D*w^2)` without the signed diffusion coefficient `tanh(D)*w^2`, meaning transport could only reduce magnitudes, never amplify

After fixes, V12.1 reached **BPC 2.30 / 53.4% accuracy** at 3000 steps — the best result in the project's history, within 0.12 BPC of GPT-Nano.

## V12.2: The Devastating Ablation

| Model | Val BPC | Delta |
|---|---|---|
| A: Full V12.1 | 2.302 | baseline |
| B: Zero SSM Context | 3.589 | +1.287 (context essential) |
| C: SSM+MLP Only | 2.267 | **-0.035** (spectral adds NOTHING) |
| D: No Sparsity | 2.275 | -0.027 (sparsity adds nothing) |

The spectral machinery (FFT/IFFT, transport, Hopfield memory, sparsification) added zero measurable value at 4.7x compute cost. The SSM + MLP alone was slightly BETTER.

**Diagnosis** (v12.2 → v12.3):
1. Transport was mode-wise/linear — an equalizer, not a synthesizer. No cross-mode spectral interactions.
2. Mode selection resets every block — destroys submanifold continuity.
3. Sparsity too mild (10/17 = 59% active modes).
4. Sparse-dense-sparse cycle erases spectral structure.

## V12.3-V12.5: Constellation Architecture

David's insight: "A few dots on the Fourier space. Mathematically, it's how many unique connections we can make."

Tokens ARE sparse spectral patterns. Shared modes ARE connections. The manifold is explicit: topology = active modes, metric = values at shared modes, curvature = mode dynamics.

**V12.5 results:** BPC 2.784 / 43.2% at 6500 steps (1.15M params). Still improving but behind GPT-Nano.

**Blocking issues:**
- Speed: 13s/step (parallel scan over 69K streams)
- NaN instability
- Update MLP dominates at 94.6% of parameters

## V12.5 → V13: Native Complex Geometry

V12.5's mode fibers used learned linear maps over separated (mag, phase) reals — no complex structure, no geometric meaning. V13 replaces with native complex operations:

| | V12.5 | V13 |
|---|---|---|
| Deposit | Linear(mag, phase) → R^sd | mag * exp(i*phi) (spectral coefficient) |
| State | Real EMA, sd dims | Complex EMA |
| Read | dot(state, read_w) → R | Re(h * conj(c)) = Parseval inner product |
| Fiber params/block | ~22K | 136 (just decay) |

**V13 results:** BPC ~3.19 / 34.5% accuracy. Worse than V12.5. The Parseval inner product is elegant but the fiber is too impoverished (pure linear EMA with no content dependence).

---

## The Pattern

Every version that improved meaningfully added **content-dependent selectivity**:
- V3's GRU (content-dependent gates) was the first architecture that worked
- V12.1's SSM (content-dependent state updates) achieved the best results
- Every version that removed content dependence (fixed routing, learned positional manifold, constant EMA decay) plateaued

The mathematical theory provides the right framework (fiber bundles, spectral sparsity, Fourier duality). But the implementations keep stripping out the content-dependent parts — the Wilson line, the Hopfield retrieval, the Langevin dynamics — in favor of cleaner geometry. The geometry is necessary but not sufficient. The content dependence is what makes it work.
