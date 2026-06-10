# V20 Design: The Spectral Return

**Status**: design, not code. Target: a short companion to `V19_DESIGN.md`
that explicitly treats V19 as a detour through dense attention-land and
returns the main architectural line to the V12/V12.1/V17 spectral
constellation idea. CurvBias is relegated to a diagnostic baseline; V19's
speed infrastructure and non-abelian transport become the core upgrades
over V12.1.

---

## Part 0 — Mandate

V20 answers one question explicitly:

> **Does the spectral-constellation architecture — sparse spectral tokens
> living on orthogonal subbundles, transported by a content-dependent
> non-abelian gauge connection, read back through a forward-reverse FFT
> loop — add measurable value over a standard transformer (or over
> SSM+MLP+attention) at matched compute, once the sparse-FFT + non-abelian
> + modern-training-infrastructure upgrades are in place?**

This is the question V12.2's ablation *left open*. V12.2 showed that V12.1's
particular implementation of the spectral machinery contributed zero net
value over SSM+MLP. V14 showed that geometry + Hopfield memory *did* beat
SSM+MLP alone. V19 showed that content-dependent SO($K$) transport can run
at reasonable wall-clock on H100 with the right infrastructure. V20 closes
the loop: it is the first architecture in the line that has *all* of the
V12.1 load-bearing structure plus the fixes that were missing the first
time around.

V20 is **not**:
- a literal Hopfield memory bank of stored sequence prototypes
- another attempt to replace attention at the same compute
- a CurvBias variant
- a dense-vector model like V18 or V19

V20 **is**:
- tokens as sparse spectral constellations
- $K$ orthogonal subbundles, each independent
- per-subbundle non-abelian SO($K$) transport with delta-rule writes
- the dynamic fiber state as the only associative memory (no stored bank)
- forward-reverse FFT/IFFT loop as a hard structural constraint
- sparse FFT (thesis §8.3.3) over learned per-block frequency bands
- trained with `torch.compile` + `bf16` autocast on H100

---

## Part I — What V12.1 captured that V19 lost

Five structural properties were present in V12.1 and dropped in V19. They
are the reason V19 is "linear attention with a geometric decoration" rather
than "the spectral architecture from the thesis":

### 1. Tokens as sparse spectral constellations

V12.1/V17 stored each token as a learned (magnitude, phase) pair per mode.
A token's identity *is* the pattern of which modes are active and the
complex values at those modes. This is the learned-optimal-fingerprint
mechanism: each vocabulary entry gets a complex spectral signature that
other tokens interact with through their own signatures.

V19 reverted to dense real embeddings with a precision-routing scalar per
channel. The constellation structure — the thing that makes tokens
*look* like sparse spectral patterns rather than dense vectors — was gone.

### 2. Orthogonal subbundle decomposition

V12.1 split the representation into 8 subbundles × 32 modes = 256 total
modes. Each subbundle was a genuinely independent feature channel: its own
SSM, its own transport kernel, its own memory bank. Tokens couldn't leak
between subbundles unless the architecture explicitly routed them across.

V19's "heads" in the `UnitaryDeltaFiber` share an input projection and a
single content-dependent $SO(K)$. They're slices of one shared transport,
not independent channels. Strictly less expressive per parameter and
strictly less interpretable.

### 3. Content-dependent spectral transport kernel with both $D_k(q)$ and $A_k(q)$

V12.1's per-mode transport had *two* learned content-dependent knobs:

$$K_k[\omega] = \exp\bigl(-D_k(q)\,\omega_k^2 \;-\; i\,A_k(q)\,\omega_k\bigr)$$

The diffusion rate $D_k(q)$ gave the model context-dependent damping — it
could selectively *kill* high-frequency content when the context called for
it. The gauge phase $A_k(q)$ gave it the rotation that encoded position.

V19's `UnitaryDeltaFiber` has only the rotation piece via $U_t =
\exp(\mathrm{skew}(x_t))$. There is no diffusion term. Transport is pure
norm-preserving rotation, which cannot selectively damp frequencies and
cannot implement the "reaction-diffusion" decomposition at all.

### 4. Forward-reverse FFT/IFFT loop as a structural constraint

V12.1's block flow was:

`sparse spectral → (FFT already done) → transport → IFFT → spatial MLP → FFT → proximal top-k → sparse spectral`

This loop is not a convenience. It is the mechanism that *forces* the
representation to stay sparse in Fourier space between blocks. The
`SpatialMLP` does arbitrary nonlinear work, then the FFT + top-k
proximally projects the result back onto the "sparse spectral" manifold.
Without this cycle, there is no such thing as a "spectral architecture" —
just a dense architecture with an FFT decorator in front.

V19 has no such loop. It runs in a dense representation throughout, with a
channel gate and a single attention head inserted mid-block.

### 5. Proximal re-sparsification

V12.1 enforced top-$k$ on the spectral coefficients at the end of every
block via a proximal operator. This is the step that makes "sparse
spectral" an *actual invariant* of the architecture, not a hope.

V19's `LearnedBandMask` is adjacent to this idea but only gates the
`GeometricContextAccum`, not the main representation.

---

## Part II — What V19 contributes that V12.1 lacked

Four things from V19 come forward into V20. They're the reason this isn't
just "copy V12.1 and hope":

### 1. Non-abelian transport per subbundle

V12.1's transport was abelian: $U(1)^M$, diagonal per mode, no order
dependence from holonomy alone (the loop $U_1 U_2 = U_2 U_1$ commutes, so
word order doesn't leave a structural trace in the accumulator). V12.1
relied on the SSM recurrence and the `SpatialMLP` to encode order.

V20 uses $SO(K)$ transport *per subbundle*, where $K$ is the number of
modes in that subbundle. Rotations within a subbundle don't commute, so
"dog bites man" and "man bites dog" accumulate to genuinely different
subbundle states. Word order is *structurally* encoded in the fiber,
independent of any sequential recurrence or MLP.

This is V19's actual contribution — the thing I was right to build, even
if I was wrong to strip the spectral machinery around it.

### 2. Speed infrastructure

The fixes from the V19 post-mortem all apply verbatim to V20:

- Vectorized `make_skew_symmetric` via `torch.triu_indices`
- `scalar_parallel_scan` (used inside `GeometricContextAccum`)
- `unitary_delta_parallel_scan` with `torch.matmul` (not `einsum`)
- `torch.compile(mode="reduce-overhead")` wrapping the full model
- `bfloat16` autocast on CUDA
- `TF32` enabled for non-autocast matmuls

Without these, V12.1 was 6–13× slower than GPT on H100 and V12.2's
ablation looked like "spectral machinery is just overhead." With them,
V20 has a chance to be speed-competitive before any more algorithmic
work.

### 3. Learned per-block frequency band masks

V19 has `LearnedBandMask`, a differentiable sparse mask over candidate
frequency bands per block, with an L1 penalty to push it toward sparsity
and a wavelet-like initialization across depth. V20 uses the exact same
module, but now it's the *primary* sparsity mechanism for the whole
representation, not just a gate on one path.

### 4. Multi-head `CurvBiasAttention` as a diagnostic baseline

CurvBias stays in the codebase as a baseline to run alongside V20, not as
part of V20's block stack. V20 ablations explicitly include a
`GPT-Nano + CurvBias` condition because that is the strongest non-spectral
attention baseline in the project and V20 needs to be measured against it.

### 5. All causality / Parseval / norm-preservation tests

The 17 unit tests from V19's `test_v19.py` apply to any V20 component that
shares an interface. V20's `test_v20.py` imports them directly.

---

## Part III — V20 architecture, block by block

### Overall block flow

```
V20Block:
    constellation = (mag, phase, log_var)     # input is spectral, not dense

    # 1. normalization over magnitude only (CloudNorm from V17)
    constellation = CloudNorm(constellation)

    # 2. per-subbundle unitary delta fiber
    #    — one fiber per subbundle, operating on that subbundle's modes only
    #    — content-dependent SO(K) rotation + delta-rule write + associative read
    for s in subbundles:
        q_s, k_s, v_s = project_from_constellation(constellation, subbundle=s)
        S_s[t] = U_s(q_s[t]) @ S_s[t-1] + k_s[t] v_s[t]^T    # delta rule
        read_s[t] = q_s[t] @ S_s[t-1]
    messages = concat across subbundles

    # 3. spectral transport kernel applied to the constellation
    #    — content-dependent diffusion and gauge phase, per mode
    #    — q = current fiber read, used to parameterize D_k(q) and A_k(q)
    constellation = apply_transport_kernel(constellation, messages, band_mask)

    # 4. sparse IFFT on active bands only — back to spatial
    spatial = sparse_ifft(constellation, band_mask)

    # 5. SpatialMLP — the reaction term, dominant nonlinearity
    spatial = spatial + SpatialMLP(LayerNorm(spatial))

    # 6. sparse FFT on active bands — back to spectral
    new_complex = sparse_fft(spatial, band_mask)
    new_mag = |new_complex|
    new_phase = arg(new_complex)

    # 7. proximal re-sparsification: top-k keep by magnitude per subbundle
    new_mag, new_phase = proximal_top_k(new_mag, new_phase, k_per_sub)

    # 8. learned variance evolution (V17)
    log_var = VarianceUpdate(log_var, messages)

    return Constellation(new_mag, new_phase, log_var)
```

The critical structural properties:

| Property | Mechanism |
|----------|-----------|
| Sparse spectral tokens | `ConstellationEmbedding`, enforced by proximal top-$k$ at step 7 |
| Orthogonal subbundles | Separate fiber, separate projections, separate transport kernel per $s$ |
| Non-abelian transport | $SO(K)$ per subbundle via `UnitaryDeltaFiber` |
| Content-dependent diffusion *and* phase | `SpectralTransportKernel` with both $D_k(q)$ and $A_k(q)$ |
| Forward-reverse loop | Steps 4–7 are the FFT/IFFT/MLP/FFT/proximal cycle |
| Sparse FFT (thesis §8.3.3) | `LearnedBandMask` per block, used in steps 3/4/6 |
| Dynamic associative memory | $S_s[t]$ is the only state; no stored bank |

### Components, each mapped to source

| Component | Role | Source |
|-----------|------|--------|
| `ConstellationEmbedding` | token → (mag, phase, log_var), per-subbundle layout | V17 verbatim |
| `CloudNorm` | RMS normalization of magnitudes with learnable per-mode scale | V17 verbatim |
| `LearnedBandMask` | per-block differentiable sparse frequency band mask | V19 verbatim |
| `PerSubbundleUnitaryDeltaFiber` | one $SO(K)$ unitary delta fiber per subbundle | new, adapts V19's `UnitaryDeltaFiber` to operate on sparse spectral input per subbundle instead of dense vector input, removes the shared projection and the cross-subbundle head mixing |
| `SpectralTransportKernel` | applies $\exp(-D_k(q)\,\omega_k^2 - iA_k(q)\,\omega_k)$ to the constellation, with both $D$ and $A$ as learned MLPs of the fiber read | new, ports V12.1's design but reparameterizes $q$ as the fiber read instead of a separate SSM state |
| `SparseFFT` / `SparseIFFT` | FFT/IFFT computed only on active mode indices from `LearnedBandMask` | new, thesis §8.3.3 realization (initial implementation can fall back to full FFT + mask; true sparse FFT kernel is a follow-up) |
| `SpatialMLP` | per-token nonlinear transformation in the spatial domain, the "reaction" term | V12.1 verbatim, sized so it is ≥ 40% of the block's parameter budget |
| `ProximalTopK` | top-$k$ by magnitude per subbundle, pass-through gradient for kept modes, zero for dropped | new, simple enough to implement directly |
| `VarianceUpdate` | learned log_var evolution driven by fiber messages | V17 verbatim |
| `CurvBiasAttention` | multi-head content-dep rotary with curvature bias — *not* in V20Block, only in the baseline models | V19 verbatim |

### Config knobs (`V20Config`)

```python
@dataclass
class V20Config:
    # constellation shape
    n_subbundles: int = 8          # orthogonal feature channels
    subbundle_dim: int = 32        # spatial size of each subbundle (power of 2)
                                   # => spectral_half_dim = subbundle_dim // 2 + 1
    n_modes: int = 136             # = n_subbundles * spectral_half_dim
    fiber_dim: int = 256           # = n_subbundles * subbundle_dim

    # sparse-FFT top-k
    active_modes_per_sub: int = 8  # proximal_top_k keeps this many per subbundle
    band_mask_l1: float = 1e-3     # L1 on LearnedBandMask to keep it sparse

    # per-subbundle unitary delta fiber
    # Each subbundle is one "head"; K is the SO(K) size.
    # Default K = spectral_half_dim so the fiber state matches the natural
    # subbundle shape. If K is too large for speed we can reduce.
    fiber_K: int = 17              # = subbundle_dim // 2 + 1 = 17 at default
    fiber_hidden_mult: int = 2

    # spectral transport kernel
    transport_hidden: int = 128    # MLP hidden size for D_k(q) and A_k(q)

    # SpatialMLP
    ffn_mult: int = 4              # standard 4x, keeps SpatialMLP as dominant compute

    # stack
    n_blocks: int = 6
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # training
    learning_rate: float = 3e-4    # higher than V19 default because sparse-FFT path is narrower
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    lr_hold_steps: int = 3000
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 20000
    eval_interval: int = 500
    eval_steps: int = 10
```

At this default, the state budget per block is
$n_{\text{sub}} \cdot K \cdot K = 8 \cdot 17 \cdot 17 = 2{,}312$ values per
block — significantly smaller than V19's 16,384 but far larger than V12.1's
scalar fiber (8 subbundles × 17 modes × 1 complex scalar ≈ 136 values).
The smaller state size reflects V20's acceptance that the state should not
be much larger than the natural "number of active modes per subbundle"
times itself.

---

## Part IV — Sparse FFT, concretely

The thesis §8.3.3 sparse-FFT plan is:

> *"A sparse FFT implementation could compute the transform in $O(s \log s)$
> rather than $O(d \log d)$, exploiting the known sparsity pattern."*

V20 implements this in three steps of increasing optimization, each
independently publishable:

### Level 0 — Full FFT + mask

Use `torch.fft.rfft` / `torch.fft.irfft` on the full $d$-dimensional
signal, then multiply by `LearnedBandMask` to zero out inactive bands.
Correctness-equivalent to the ideal sparse FFT; speed is the same as a
normal transformer (no worse than V12.1).

This is where V20 starts. Ablation #5 in Part V uses this level.

### Level 1 — Batched FFT on active subbundles only

Since `subbundle_dim` is a power of 2 and subbundles are independent, we
can batch the FFT across subbundles with `torch.fft.rfft(x, dim=-1)` on
shape `(B, T, n_subbundles, subbundle_dim)`. This is effectively an
$O(n_{\text{sub}} \cdot T \cdot d_{\text{sub}} \log d_{\text{sub}})$
transform and, for small `subbundle_dim` (32) with `n_subbundles = 8`, is
significantly cheaper than an $O(T \cdot 256 \log 256)$ transform on the
flat representation.

V12.1 already used this pattern implicitly; V20 makes it explicit and
`torch.compile`-friendly.

### Level 2 — True sparse FFT kernel

For the full thesis §8.3.3 realization, write a custom kernel (Triton or
compiled CUDA) that computes the FFT over only the active mode indices
per subbundle per block. This is the $O(s \log s)$ variant.

This is explicitly a **follow-up**: V20's initial release uses Level 0 or
Level 1. Level 2 is an optimization that should only be pursued if V20
demonstrates a quality win at Level 0/1, because writing custom FFT
kernels is expensive and the win matters only if the architecture itself
is worth investing in.

---

## Part V — Ablation protocol (the V12.2 open question, answered)

V12.2's devastating ablation compared full V12.1 to SSM+MLP and found the
spectral machinery added *negative* value. V14 later showed that adding a
Hopfield memory bank reversed this: geometry + memory > SSM+MLP. But V14's
result was on a different dataset and at a different scale, and the
follow-up never happened.

V20 runs the full matrix to give an unambiguous answer:

| # | Model | Tests |
|---|-------|-------|
| A0 | GPT-Nano (plain) | control |
| A1 | GPT-Nano + CurvBias (V19's multi-head attention) | thesis §6.7 baseline, strongest attention |
| A2 | SSM + MLP only (V12.2-style) | V12.2's baseline |
| A3 | V20 **without** per-subbundle fiber (constellation + FFT loop + SpatialMLP + proximal) | isolates the fiber contribution |
| A4 | V20 **without** constellation (dense input, V19 fiber, sparse FFT loop on the residual) | isolates the constellation contribution |
| A5 | V20 **without** non-abelian transport (per-subbundle abelian $U(1)^{K}$, V12.1-style) | isolates the non-abelian contribution |
| A6 | V20 **without** sparse FFT (full $d$-dim FFT every block) | measures the sparse-FFT speed benefit |
| A7 | V20 **without** SpatialMLP (or SpatialMLP at 1× instead of 4×) | isolates the reaction term |
| **A8** | **V20 full** | the actual proposal |
| A9 | V20 + `CurvBias` added to the SpatialMLP path | additive geometry |

### What each ablation decides

- **A8 vs A2**: does V20 beat SSM+MLP at matched compute? If yes, the
  spectral architecture is *worth something* post-V12.2.
- **A8 vs A1**: does V20 beat the strongest attention baseline (GPT +
  CurvBias)? If yes, the spectral architecture has surpassed attention,
  which is the thesis's original goal.
- **A8 vs A3**: is the fiber load-bearing? V14's result says yes; this
  re-runs the experiment at V20's scale.
- **A8 vs A4**: is the constellation load-bearing, or is V19 + sparse FFT
  enough?
- **A8 vs A5**: is non-abelian transport necessary, or does V12.1's
  abelian transport suffice? This directly tests whether my V19 claim
  about word-order-via-holonomy actually matters.
- **A8 vs A6**: does the sparse-FFT path buy anything in wall-clock?
- **A8 vs A7**: is the `SpatialMLP` still the dominant nonlinearity?
  V12.1's 50% budget should be reproduced.
- **A9 vs A8**: can CurvBias additively improve V20?

### Compute equalization

All ablations should be run at *matched wall-clock* on the same H100 with
`torch.compile` + `bf16` autocast, not at matched parameter count. V12.2's
mistake was comparing at matched parameters, which made the compute-
inefficient V12.1 look worse than it should have. V20 answers the right
question: *at the same H100-minute, which architecture wins?*

---

## Part VI — File layout

```
v20/
├── V20_DESIGN.md                 (this file)
├── v20_modules.py                (new components, imports from v19_modules.py)
├── test_v20.py                   (new tests, imports V19 test helpers)
├── benchmark_v20.py              (extends benchmark_v19.py with V20 components)
├── gen_notebook_v20.py           (notebook generator, full ablation matrix)
├── architecture_v20.ipynb        (generated training notebook)
└── README.md                     (quick-start + knobs + caveats)
```

### What `v20_modules.py` imports from `v19_modules.py`

Reused verbatim:
- `matrix_parallel_scan`, `scalar_parallel_scan`, `unitary_delta_parallel_scan`
- `make_skew_symmetric`, `fast_orthogonal`
- `LearnedBandMask`
- `VarianceUpdate`
- `CurvBiasAttention` (for the diagnostic baseline only)
- `count_params`
- `RMSNorm`

Reused as building blocks for new modules:
- `UnitaryDeltaFiber` — V20's `PerSubbundleUnitaryDeltaFiber` internally
  instantiates one `UnitaryDeltaFiber` per subbundle with a small `K`, or
  re-implements the scan directly to avoid the unnecessary per-subbundle
  instance overhead.

New in `v20_modules.py`:
- `V20Config`
- `ConstellationEmbedding`
- `CloudNorm`
- `PerSubbundleUnitaryDeltaFiber`
- `SpectralTransportKernel` (with learned $D_k(q)$ and $A_k(q)$)
- `SparseFFT`, `SparseIFFT` (Level 0/1 implementations)
- `ProximalTopK`
- `SpatialMLP`
- `V20Block`
- `V20Model`

### What `test_v20.py` covers

All V19 invariants extended to V20:
- shape correctness of every module
- causality of `PerSubbundleUnitaryDeltaFiber`, `SpectralTransportKernel`,
  and the full `V20Block`
- Parseval-style energy bound on the spectral transport kernel
  (`|K_k[ω]| ≤ 1` everywhere)
- orthogonality of each subbundle's $SO(K)$ fiber transport
- round-trip identity: `SparseFFT(SparseIFFT(x)) ≈ x` when `x` lives in
  the active band support
- proximal top-$k$ keeps exactly `active_modes_per_sub` modes per
  subbundle and preserves the kept values exactly
- full `V20Model` forward + backward + grad flow
- `V20Block` without any sub-module should roughly reduce to the V12.1
  block when its special sub-modules are ablated out

---

## Part VII — Risks, fallbacks, open questions

### Risk 1 — V14's Hopfield-bank result was load-bearing

V14's ablation said geometry + Hopfield memory bank > SSM+MLP, and V14's
bank was a *fixed learned* codebook, not the dynamic fiber state. V20
deliberately has no learned bank, relying entirely on the dynamic
`PerSubbundleUnitaryDeltaFiber` state as the associative memory. If A8
(V20 full) does **not** beat A2 (SSM+MLP), the fallback is:

> **V20.1** adds a small per-subbundle learned prototype bank (32 atoms
> per subbundle, dot-product retrieval) *alongside* the dynamic fiber
> state. This is the minimum intervention that reintroduces V14's
> load-bearing mechanism without falling back to a literal vector
> database.

The prototype bank is small ($n_{\text{sub}} \cdot 32 \cdot 2K$ reals per
block), and its retrieval cost is $O(T \cdot n_{\text{sub}} \cdot 32 \cdot 2K)$.

### Risk 2 — Sparse FFT at Level 0 is no faster than full FFT

Level 0 of the sparse-FFT plan uses `torch.fft.rfft` on the full signal
and then multiplies by the `LearnedBandMask`. This is numerically
identical to sparse FFT but has the same compute cost as a dense FFT.
If V20's wall-clock is dominated by the FFT, the speed claim is
unsupported until Level 1 or Level 2 is built.

**Mitigation**: measure this in `benchmark_v20.py` directly. If Level 0
dominates step time, move to Level 1 before the main run. Level 2 is
future work regardless.

### Risk 3 — Per-subbundle fiber is too small

V20's default $K = 17$ per subbundle with $n_{\text{sub}} = 8$ gives a
total state of 2,312 values per block — an order of magnitude smaller
than V19's 16,384 and two orders smaller than attention's 131 k at
$T = 256$. If the plateau we're seeing in V19 is fundamentally a state-
capacity issue, V20 may plateau even earlier.

**Mitigation**: the ablation matrix directly tests this. A8 vs A1 at
matched wall-clock tells us whether the structural inductive bias of
V20 compensates for its smaller state.

### Risk 4 — `torch.compile` doesn't handle the per-subbundle loop

A naive implementation of the per-subbundle fiber is a Python `for s in
range(n_subbundles)` loop, which will either cause a graph break or
produce `n_subbundles` independent compiled subgraphs. Both are bad.

**Mitigation**: implement the per-subbundle fiber as a single batched
operation by reshaping `(B, T, D) → (B*n_sub, T, K)` and using V19's
existing scan functions with `B*n_sub` as the outer batch. This keeps the
whole block in one compiled graph.

### Risk 5 — Content-dependent $D_k(q)$ is hard to stabilize

V12.1's $D_k(q)$ was a learned MLP of the context vector. If it ever goes
negative (exponentiates to amplification instead of damping), the state
blows up. V12.1 clipped with `softplus`; V20 does the same.

### Risk 6 — Proximal top-$k$ is not differentiable at the boundary

Top-$k$ is discrete at the mode boundary. V12.1 used straight-through
gradient. V20 uses either straight-through or a soft top-$k$ via
`sparsemax`/`α-entmax`. The choice is measured in `test_v20.py` on a
toy task where the true top-$k$ is known.

---

## Part VIII — Acceptance criteria

V20 is a success if *any one* of the following holds after the A0–A9
ablation matrix runs to completion:

1. **A8 beats A1** (V20 full beats GPT + CurvBias) on WikiText-103 at
   matched wall-clock. This is the original thesis goal: geometry beats
   attention at the same hardware budget.

2. **A8 beats A2 by ≥ 10%** (V20 full beats SSM + MLP at matched
   wall-clock) while A5 does *not* (ablating non-abelian gives up the
   win). This validates that non-abelian SO($K$) transport is
   load-bearing, which would be the first published evidence of this.

3. **A8 beats A3 by ≥ 10%** (V20 full beats V20 without the fiber) at
   matched wall-clock. This validates that the dynamic fiber state is
   the load-bearing associative memory, replicating V14's result *at
   scale* and without a literal bank.

4. **A9 beats A8** (V20 + CurvBias beats V20 alone) and both beat A1.
   This would justify a hybrid paper: geometry inside the model *and*
   on the attention path.

Any one of these is publishable. Two or more is a strong thesis
extension paper. Three or more justifies starting a V21 line around
whichever component the ablation identified as the load-bearing piece.

**Failure condition**: if A8 loses to both A1 and A2 at matched wall-
clock, then V20's architectural bet has been conclusively falsified at
this scale, and the follow-up is either V20.1 (add the prototype bank)
or returning the project's main line to CurvBias-based attention
enhancements.

---

## Part IX — Order of work

1. Write `v20_modules.py` with the minimum viable block: `ConstellationEmbedding`,
   `PerSubbundleUnitaryDeltaFiber`, `SpectralTransportKernel` (Level 0 sparse
   FFT), `SpatialMLP`, `ProximalTopK`. Skip `VarianceUpdate` initially.
2. Write `test_v20.py` (shape, causality, Parseval bound, round-trip).
3. Run unit tests; fix until all green.
4. Write `benchmark_v20.py` (component-level breakdown + compile-wrapped
   full model). Measure FFT cost fraction.
5. Decide Level 0 vs Level 1 sparse FFT based on step 4.
6. Write `gen_notebook_v20.py` with the full A0–A9 ablation matrix.
7. Syntax-check the notebook.
8. Run on H100 for the A2 (SSM+MLP) and A8 (V20 full) conditions first —
   these are the most important comparison. If A8 wins, run the rest.
9. If A8 loses to A2, implement V20.1 with the small prototype bank and
   re-run A8 vs A2 only.

Step 1 alone is enough to unblock further conversation and should be the
next thing we write. Everything after step 4 depends on H100 time.

---

## Appendix A — What we're explicitly *not* doing in V20

- No literal Hopfield memory bank of stored sequence prototypes. The fiber
  state is the memory, per the clarification in our design conversation.
  V20.1 is the fallback if this turns out to be necessary.
- No multi-scale wavelet hierarchy (thesis §8.3.3). This is a follow-up;
  V20 uses the simple single-resolution FFT per subbundle.
- No Level 2 sparse FFT kernel. Level 0 or Level 1 is sufficient to
  answer the correctness question; Level 2 is a post-V20 speed project.
- No 4 formal proofs from thesis §8.3.4. Those are paper work, not
  architecture work.
- No scaling to 100 M+ parameters. V20 runs at the V12.1 / V19 scale
  (roughly 10–70 M parameters) so the ablation matrix is feasible. If
  A8 wins, scaling is a V20-follow-up project.

## Appendix B — Glossary alignment with our design conversation

- **Constellation fingerprint** — the token's learned $(m_k, \phi_k,
  \log\sigma_k^2)$ tuple per mode, sparsified to a small set of active
  modes per subbundle. This is the "learned optimal fingerprint" from
  the design conversation.
- **Subbundle** — one of the orthogonal feature channels. $K$ independent
  subbundles, each with its own modes, its own fiber, its own transport
  kernel. In V20, "subbundle = head" in the V19 sense, but with
  independent projections rather than shared ones.
- **Connection** — the *operator* (content-dependent $U_s(q)$ and
  $K_k[\omega]$), not a link between two tokens. Two tokens "interact"
  when they both activate the same subbundle's modes and the transport
  operator propagates one's signature through the other's fiber state.
- **Holonomy** — the accumulated transport operator along the sequence.
  Non-abelian per subbundle in V20, so word order leaves a structural
  trace in the subbundle state.
- **Associative memory** — the dynamic per-subbundle fiber state
  $S_s[t]$. Not a stored codebook. Built during the forward pass, read
  by the current token's constellation, thrown away at the end of the
  sequence. Exactly analogous to attention's KV cache.
- **Recognition** — dot-product read of the dynamic fiber state by the
  current token's constellation, analogous to a Ramsauer Hopfield step
  where the "bank" is the fiber state of the current sequence.
