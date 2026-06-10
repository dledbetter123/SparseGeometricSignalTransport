# Sparse Geometric Signal Transport (SGST)

**A geometric theory of the transformer — and a search for a more efficient generative architecture built from it.**

SGST is a multi-year research program asking a single applied-math question:

> *Self-attention works extraordinarily well. **Why** does it work, what geometric object is it secretly computing, and is there a cheaper object that computes the same thing?*

The wager is that the answer lives in differential geometry and signal processing rather than in ever-larger dense matrices. If a transformer block is really performing **parallel transport of information along a learned connection on a fiber bundle**, then attention's $O(T^2)$ all-pairs computation is one — expensive — way to realize that transport, not the only way. This repository is the record of taking that hypothesis seriously across twenty-one architectural iterations (V1–V21), measuring it honestly against GPT baselines, and reporting both where it failed and the one place it produced a genuine, transferable win.

---

## 1. Motivation: attention as a gauge connection

Standard scaled dot-product attention scores token $i$ against token $j$ as $q_i \cdot k_j / \sqrt{d}$. Two structural facts about this operation drive the entire project:

1. **It is path-independent.** The score between positions $i$ and $j$ depends *only* on the content at $i$ and $j$ — never on what lies between them. In *"The cat that the dog chased ran away,"* attention from *ran* to *cat* is computed as if *the dog chased* were not interposed at all.
2. **It is $O(T^2)$ in sequence length and stores an $O(T)$ exact memory** (the KV cache). At sequence length 256 this is on the order of $1.3\times10^5$ stored values per layer — an exact, fully content-addressable record of the past.

Language, however, *is* path-dependent: the meaning that accrues to a token is a function of the entire route through the preceding clause, not just the endpoints. This is precisely the regime that **gauge theory** describes. Transporting a vector around a curved space and asking how it has rotated on return — the **holonomy** — is the canonical path-dependent quantity in physics. SGST's central conjecture is that the contextual transformation a token undergoes *is* a holonomy, and that attention is an expensive, path-independent approximation to it.

This places the project inside a remarkable **six-way convergence** of independent research lines that all arrive at *content-dependent transport of state*:

| Research line | The transport object |
|---|---|
| **Gauge theory / geometric deep learning** | Parallel transport along a connection $A$; holonomy as accumulated curvature |
| **State-space models (S4, Mamba, Mamba-3)** | Recurrent state $h_t = A_t h_{t-1} + B_t x_t$ with input-dependent transitions |
| **Linear / kernel attention** | Associative state $S_t = S_{t-1} + k_t v_t^\top$ updated online |
| **The delta rule** | *Selective overwrite* of state rather than mere decay |
| **Spectral methods / compressed sensing** | Transport realized as cheap frequency-domain filtering |
| **Optimal transport** | Moving probability mass along a cost geometry |

The product of input-dependent transition matrices in a modern SSM, $A_t A_{t-1}\cdots A_1$, **is literally the Wilson line** $U_{1\to t}$ of a gauge connection — a path-ordered product of local, content-dependent, norm-preserving maps. Mamba-3's complex-valued state $z_t = r_t e^{i\theta_t} z_{t-1}$ is exactly a $U(1)$ holonomy, with the accumulated phase $\sum_\tau \theta_\tau$ playing the role of the Wilson-line phase. SGST takes this equivalence as a design principle rather than a coincidence.

---

## 2. The architecture in one diagram

The mature formulation models computation over a base manifold $M$ (context) with a fiber $F_q$ of representational features attached at each contextual coordinate $q$, decomposed into $K$ orthogonal sub-bundles. Transport between coordinates is **gauge-covariant and performed in the spectral domain**:

$$\tilde{X}_{q} \;=\; X_{p}\,\exp\!\Big(\underbrace{-D\,\omega^2}_{\text{heat kernel}}\;\underbrace{-\,i\,\omega\!\int_\gamma A}_{\text{Wilson line / holonomy}}\Big)$$

- The **diffusive term** $e^{-D\omega^2}$ is a heat kernel: frequency-dependent damping that gives free, intrinsic multi-scale separation (low modes travel far, high modes decay fast).
- The **advective term** $e^{-i\omega\int_\gamma A}$ is a $U(1)$ Wilson line: a content-dependent phase rotation that *is* the path's holonomy. This is the term attention cannot express.

A single Parseval inner product $\mathrm{Re}(h \ast \bar{c})$ between an accumulated mode-state $h$ and the current token $c$ simultaneously yields three geometric objects: the **metric** (magnitude overlap), the **connection** (relative phase = parallel transport on $S^1$), and a reading of **curvature** (how phase alignment varies across modes). One bilinear operation, three geometric quantities — the efficiency promise in microcosm.

### The "constellation" reframing (V12.5)

The project's most evocative idea: a token *is* a **sparse constellation of active Fourier modes** — "a few dots in frequency space" — and **shared modes are the connections between tokens.** If tokens $A$ and $B$ both light up mode $m$, they are geometrically coupled through $m$'s causal state, which earlier tokens write to and later tokens read from. No attention matrix is instantiated; connectivity is an emergent property of which dots overlap. With 8 sub-bundles of 17 modes and 6 active each, the support space is $\binom{17}{6}^8 \approx 5.7\times10^{32}$ patterns — combinatorially enormous room for constellations to drift and re-form as context accrues. Stabilizing this required signed magnitudes (so $-|m| = $ a $\pi$ phase shift), a pre-norm residual highway, and the realization that *magnitude already encodes sparsity natively* — loud modes are active, quiet modes are silent, no discrete mask needed.

---

## 3. The efficiency argument

The case for why this *could* beat attention is asymptotic and capacity-theoretic:

- **Cost.** Spectral transport is $O(d \log d)$ dense, and $O(s)$ when only $s \ll d$ modes are active — against attention's $O(T^2 d)$. Compressed sensing (Candès–Romberg–Tao) guarantees an $s$-sparse spectral signal is recoverable from $O(s\log d)$ measurements, so sparsity in the *right* (frequency) domain is information-complete, not lossy.
- **Memory, honestly.** The hard constraint the project surfaced is not representation but **state capacity**. A scalar-fiber model carries $\sim\!10^2$ state values per layer; a matrix fiber $16\times8\times8$ carries $\sim\!10^3$; attention's KV cache carries $\sim\!10^5$. That is a **128× memory gap**, and the audit's blunt conclusion is *"no amount of clever routing compensates for 128× less memory."* The combinatorial richness of mode patterns is a red herring — a 256-d float vector already has $10^{77}$ distinguishable points. **We were never bottlenecked on representation; we were bottlenecked on memory.** This reframes the efficiency program precisely: close the capacity gap *without* paying attention's quadratic cost.

---

## 4. Honest results

This is a research repository, and the negative results are load-bearing.

- **The V12.2 ablation (the bar).** Plain SSM + MLP reached **BPC 2.267** on Tiny Shakespeare; the full geometric model reached **2.302** at **4.7× the compute.** The spectral machinery was *decorative* — it consumed 79% of step time to make the model 0.035 BPC **worse**. Every subsequent iteration had to clear this baseline before any geometric claim could be believed.
- **Best legitimate head-to-head.** V16 ("the irfft *is* the mixing") reached **PPL 275** on WikiText-103 versus GPT-Nano's **173** — within **1.6×**, the strongest honest spectral result, but still behind. One variant (V16b) appeared to crush the baseline at PPL 36 → traced to **non-causal information leakage** through a position FFT and disqualified.
- **Parity was never reached.** The closest-to-GPT number (V12.1, within 0.12 BPC) came from the *conventional* SSM+MLP components, not the geometry. The project's own settled position: **"attention is a local optimum in the space of geometric sequence processors"** — not the global best, but every nearby alternative tested so far performs worse at equal compute.
- **What genuinely survived every ablation:** content-dependent state transitions; a per-token nonlinearity (the FFN is the *only* nonlinear feature-mixer in the stack); the irfft round-trip (removing it NaNs by step 5500 — it is structurally load-bearing for stability); associative *matrix*-fiber memory ($q@S$ retrieval beats scalar EMA); and **complex-valued state**, because phase is how holonomy is stored and real states provably cannot track the parity/path information that complex states can.

### The most promising direction: a curvature-based positional encoding

The program's most promising applied direction is a **curvature-based positional encoding** that adds a content-dependent curvature term to the attention logits. These are early, encouraging results that require further study — especially at scale — but the direction is promising given where the geometry points:

$$\text{score}_{ij} = \frac{q_i\cdot k_j}{\sqrt{d}} \;-\; \int_i^j \kappa(t)\,dt$$

In the gauge hierarchy, absolute/relative encodings are flat connections and **RoPE is a flat $U(1)$ connection (zero curvature)**; this encoding is the first member with *nonzero, learned, content-dependent curvature*. It improved perplexity by **~9% over RoPE at 14M params** (48.6 vs 53.4) and remained ahead at 77M and at sequence length 3072 — while being a **drop-in** addition compatible with FlashAttention and KV caching, at $O(T^2)$ *scalar* overhead ($d\times$ cheaper than attention itself). These remain preliminary findings that warrant further study before any strong claim, but they suggest the geometric lens can *improve* attention rather than replace it — the more defensible scientific outcome to pursue.

---

## 5. The forward case: where more resources go

The honest framing of the negative results sharpens, rather than abandons, the efficiency thesis. The path-dependent inductive bias is real and architecture-independent; the gap to attention is a **capacity** gap, and capacity is exactly what scale and richer fiber operations buy. The concrete program (V19–V21):

1. **Expand the fiber bundle from scalar to matrix to operator.** Move state capacity from $\sim\!10^3$ toward $10^4$ per layer by widening the matrix fiber ($K=8 \to 32$) and giving each mode a genuine associative store, directly attacking the 128× gap.
2. **Go complex, deliberately.** Phase is the holonomy; complex/unitary state is the only representation that preserves path information. Combine **$U(K)$ unitary transport** (gauge theory) with the **delta rule** (selective overwrite from SSM research) — the single untried cell in the six-way convergence table, and the design's best shot at attention-level selectivity at sub-quadratic cost.
3. **Sparse and learned-band FFT.** $O(s\log s)$ transport on differentiable frequency masks, with blocks specializing to different bands — a true multi-scale spectral hierarchy via wavelets rather than a single global FFT.
4. **Ship the curvature encoding at scale.** Apply it to LLaMA/Mistral-class models during continued pretraining; it needs no architectural surgery and the only open question is whether the geometric signal grows or fades with scale.
5. **Formalize the correspondence.** Prove the conjectured theorems: DFT basis as parallel-transport eigenfunctions, the spectral kernel as holonomy, IFFT as section reconstruction, top-$k$ as a Stiefel retraction.

The bet is not "geometry beats attention tomorrow." It is that a **complex-valued, capacity-matched, sparse-spectral fiber bundle with selective (delta-rule) transport** is the most promising sub-quadratic generative architecture in the design space the convergence points at — and that the geometric framing is what tells you *which* operations to spend the next order of magnitude of compute on, instead of guessing.

---

## 6. Repository map

| Path | What it holds |
|---|---|
| `v5…v21/` | Iteration-by-iteration architectures, generators, and notebooks. Each `gen_notebook_*.py` is the source of truth for its `.ipynb`. |
| `study_exhaustive/` | The literature/foundations units: fiber bundles & gauge theory, Finsler geometry, spectral methods, compressed sensing, Hopfield/energy models, curvature-based position encoding. |
| `docs/handoffs/` | Dated session handoffs — the live reasoning trail, including the V12 spectral→constellation arc and its stabilization. |
| `v14/mathematical_foundations.md`, `v18/audit_*.md` | The formal axioms and the honest mid-project audit (the 128× capacity finding). |
| `v19/`, `v20/`, `v21/` | Forward design documents: $U(K)$ + delta rule, the capacity program, and the "landscape" reframing. |

Training artifacts, the masters-thesis LaTeX, the IP-constrained topology internals, and third-party papers are intentionally git-ignored.

---

*Part of a broader applied-math line on efficient generative AI. See also the [Finsler Transformer](https://github.com/dledbetter123/LedbetterFinslerTransformer) — a sister project that asks whether replacing attention with **geodesic flow on a learned Finsler manifold** can make context itself the geometry.*
