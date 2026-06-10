# Audit: The State of Manifold Transport

**Date**: 2026-04-01
**Context**: V1-V18 architecture exploration. V17 and V16e training on WikiText-103.

---

## What We Set Out to Find

A transport mechanism native to the geometry of language — not a routing table (attention), not a memory bank (linear attention), but a dynamical system where tokens shape a manifold and meaning emerges from the dynamics of that manifold.

The original thesis: "Language representations have intrinsic curvature. The architecture should make manifold manipulation the explicit forward pass."

## What We Found Instead (V1-V16)

We found what DOESN'T work and why:

| Version | What we tried | What happened | Why |
|---|---|---|---|
| V1-V9 | Various routing mechanisms | All hit 45% wall on Shakespeare | Routing wasn't the bottleneck, representations were |
| V10 | Contextual manifold | "The manifold is fake" — positional lookup, no context | Manifold coordinates must be context-dependent |
| V11 | Diffusion as field reconstruction | Alcubierre metaphor crystallized | Correct intuition, wrong implementation |
| V12 | Spectral sparsity + transport | SSM+MLP matched full model in ablation | Spectral machinery was decorative |
| V13 | Native complex geometry (Parseval) | Plateaued — no content dependence in fiber | Content dependence is non-negotiable |
| V14 | Wilson line + Langevin + proximal | Geometry beats SSM+MLP; Hopfield memory is key contributor | First proof geometry adds value |
| V15 | "Parseval attention" (misimplemented) | Just a transformer with fancy embedding | Didn't implement the actual idea |
| V16 | True Parseval attention (irfft = mixing) | PPL 275 on WikiText-103. No plateau. | Best spectral result, but 1.6x behind GPT |
| V16b | + Position FFT (SPECTRE-style) | PPL 36 — CHEATING via non-causal information leakage | Position FFT sees future tokens during training |
| V16c | Spectral-native (no irfft) | NaN at step 5500 | irfft provides essential numerical stability |
| V16d/e | Matrix fiber (linear attention) | PPL ~560 at 12K steps, tracking toward V16's trajectory | Associative memory helps but state too small |
| V17 | Precision-routed Gaussian clouds | Training — results pending | Precision as routing mechanism |
| V18 | Clean-sheet: precision-gated linear attention | Training — results pending | Everything that survived, nothing that didn't |

## What We Know Works

1. **Content-dependent state transport.** Every version that added content-dependent gating improved. Every version that removed it plateaued. The GRU in V3, the SSM in V12.1, the Wilson fiber in V14 — all the same lesson.

2. **Per-token nonlinearity (FFN).** Can't be removed. Attention is linear in values. Spectral filtering is linear. The FFN provides the nonlinear feature interactions that language requires.

3. **The irfft round-trip as stabilizer.** V16c proved removing it causes NaN. The orthogonal DFT basis projection every block prevents magnitude/phase drift. This is an engineering finding, not a theoretical one.

4. **Associative memory (matrix fiber) over decayed averages (scalar fiber).** The matrix state S[t] with q@S retrieval outperforms scalar EMA. Tokens need to be STORED and RETRIEVED, not just averaged.

5. **Complex-valued state preserves path information.** The phase of the complex fiber state encodes the holonomy — the accumulated rotation from content-dependent transport along the path. Real-valued states lose this.

## What We Know Doesn't Work

1. **Spectral sparsity at small vocab.** Proximal thresholding hurts at vocab 65. At vocab 50K with 136 modes, the model never learned to use sparsity. The combinatorial argument requires vocab >> modes to create compression pressure, and even then the model preferred dense representations.

2. **Spectral representation over dense representation.** The constellation (mag, phase) costs 2x embedding parameters for no measurable benefit. Dense 256-dim embeddings achieve the same expressivity. The representation is not the contribution.

3. **Non-causal position FFT.** Sees future tokens during training. Model learns to cheat. SPECTRE argues global gate + positional phase prevents this; our experiments showed PPL 1.4 (nearly perfect copying). Anti-cheat mechanisms were insufficient in our setup.

4. **Spectral-native processing (no irfft).** Numerically unstable. The DFT basis change provides implicit normalization that raw (mag, phase) processing lacks.

5. **The Langevin settler as FFN replacement.** Iterative Hopfield energy descent is linear in values (softmax-weighted average). It doesn't replace per-token nonlinearity.

## The Central Constraint We Missed

**State capacity.** The entire V1-V17 arc optimized token REPRESENTATION (spectral vs dense, sparse vs dense, constellation vs vector) while the model's ability to REMEMBER was starved.

| Architecture | State per layer | Retrieval |
|---|---|---|
| Scalar fiber (V13-V16) | 136-272 values | Decayed average, no selectivity |
| Matrix fiber 16x8x8 (V16e-V18) | 1,024 values | Associative (q@S), selective but tiny |
| Attention KV cache (GPT) | 131,072 values at seq 256 | Exact pairwise, unlimited selectivity |

Our state is 128x smaller than attention's. No amount of clever routing compensates for 128x less memory.

The spectral thesis assumed the combinatorial space of mode patterns would provide an information advantage — C(17,6)^8 = 5.7 x 10^32 patterns. But combinatorial representation space is cheap; a 256-dim float32 vector has 10^77 distinguishable points. We were never bottlenecked on representation. We were bottlenecked on memory.

## The Holonomy Insight

What the spectral/geometric exploration DID produce is an insight about what attention is missing:

**Attention has no path dependence.** `score(t, t') = q_t @ k_{t'}` depends ONLY on the content at positions t and t'. It doesn't depend on what's BETWEEN them.

**Holonomic transport has path dependence.** The Wilson line product `z_3 * z_2 * z_1` applied to a deposit from position 0 encodes the ENTIRE PATH of intervening tokens in the accumulated phase rotation. The curvature enclosed by different paths gives different holonomies.

Example: "The cat that the dog chased ran away." When processing "ran", attention to "cat" gives a score based on content similarity alone. It doesn't know "the dog chased" intervened. The holonomy encodes the syntactic transformation: "cat" → relative clause → "cat" still subject. The accumulated phase rotation IS the clause structure.

This is the mechanism that attention DOESN'T have. Attention can learn to approximate path dependence through multiple layers of reprocessing (layer 1 finds "cat", layer 2 determines it went through a relative clause, layer 3 resolves coreference). But it takes multiple layers because each layer is path-independent. Holonomic transport encodes the path in a single accumulation.

## The Open Question

Can holonomic transport — complex-valued state with content-dependent phase rotation accumulating path information — serve as a practical cross-token mechanism that matches attention's quality?

The requirements:
- **Causal**: the Wilson line product is inherently causal (multiplicative scan, left to right)
- **Content-dependent**: each z_t depends on token content (check)
- **Path-preserving**: complex multiplication preserves phase = path information (check)
- **Composable**: complex multiplication is associative = parallel scan works (check)
- **Sufficient state capacity**: THIS IS THE OPEN PROBLEM

The scalar Wilson fiber (one complex number per mode) has 272 state values. The matrix fiber (kv^T outer products) has 1,024. Attention has 131,072.

The question: can holonomic transport with complex matrix state achieve sufficient effective capacity to match attention? The phase structure SHOULD provide an information advantage — it encodes not just what was stored but the path it took. But does this theoretical advantage translate to practical performance?

## What The Thesis Should Argue

Not: "we built an architecture that beats attention."
Not: "spectral representations are superior."

But: "attention is path-independent. Language is path-dependent. Holonomic transport on complex-valued state provides a natural path-dependent mechanism. We systematically explored the design space of geometric architectures (V1-V18), identified holonomic transport as the load-bearing mechanism, and showed it provides measurable value over path-independent SSM+MLP baselines. The gap to attention remains, primarily due to state capacity constraints, but the path-dependent inductive bias is a genuine architectural contribution independent of scale."

## The Deeper Vision

The prevailing thought: through some highly tuned geometric system, or evolving manifold, open or closed, we discover a vector transport system hidden in the math.

Attention was discovered, not designed. Bahdanau found soft alignment (2014). Vaswani realized it was all you need (2017). Nobody derived softmax(QK^T)V from theory — they found it empirically.

The transport mechanism we're looking for might be similarly waiting to be found. The holonomic fiber is the closest we've gotten — it provides something attention provably doesn't have (path dependence). But the full mechanism — "drop a token onto a manifold carved by past tokens and see where it rolls" — remains an open mathematical problem.

The experiments across V1-V18 constrain where this mechanism lives:
- It's not in the representation (spectral vs dense doesn't matter)
- It's not in the routing (attention/Hopfield/Parseval all approximate the same thing)
- It IS in the transport (complex phase, holonomy, path dependence)
- It REQUIRES sufficient state capacity (1K values isn't enough)
- It REQUIRES per-token nonlinearity (FFN is structural)
- It REQUIRES numerical stability (orthogonal projections or LayerNorm every block)

The search continues.

---

## Active Experiments

| Model | Status | What it tests |
|---|---|---|
| V16e (matrix fiber 16x8x8) | Training, WikiText-103, 20K steps | Baseline matrix fiber with Gaussian clouds |
| V17 (precision-routed) | Training, WikiText-103, 20K steps | Position as precision pattern, learned variance evolution |
| V18 (clean-sheet) | Training, WikiText-103, 20K steps | Dense embedding, precision-gated linear attention, no spectral |
| GPT-256d (baseline) | Training alongside V18 | Fair comparison: same d_model, same depth |

## Files

- `v18/audit_260401_manifold_transport.md` — this file
- `v18/gen_notebook_v18.py` — V18 clean-sheet architecture
- `v18/architecture_v18.ipynb` — V18 training notebook
- `v17/DESIGN.md` — V17 precision routing design
- `v16/CHANGELOG.md` — full V16 iteration history
- `v16/research_efficient_state_computation.md` — state capacity speed benchmarks
- `v16/findings_irfft_analysis.md` — why irfft is needed
- `v16/findings_learning_rate_state_capacity.md` — LR scales with state capacity
- `thesis/findings/11_spectral_architecture_findings_v1_v16.md` — formal thesis findings
