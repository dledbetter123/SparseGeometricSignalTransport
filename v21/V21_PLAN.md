# V21 Plan: The Landscape Return

**Status**: planning only — no code. This document is a handoff for the
next chat session. The previous chat designed and partially ran V20 (the
"spectral return") and is now pivoting to V21 because V20 does not
capture the architectural intent. This doc contains everything needed
to start V21 design in a fresh conversation without losing context.

---

## 0. What this document is

This is a **handoff**, not a design. When a fresh chat opens and reads
this, it should:

1. Understand the project's history (V5 → V20) at a summary level
2. Understand the *intent* the user has been trying to realize
3. See why V20 missed the intent (without re-deriving it)
4. Know which V20 code assets can be reused verbatim vs. which must be thrown away
5. See the three open design questions that must be answered **before**
   any V21 code is written
6. Have concrete architectural sketches for each branch of the decision tree,
   so the design conversation with the user can start from a specific
   proposal rather than from scratch

The first thing the new chat should do is **read this doc end-to-end
before proposing anything**. The second thing is to ask the user the three
questions in §6. Only after those are answered should V21 design start.

---

## 1. One-page project history

This project investigates whether an explicit geometric/spectral
architecture can compete with or beat standard transformer attention for
causal language modeling. The thesis is at
`/Users/davidledbetter/SparseGeometricSignalTransport/thesis_constrained/`
(LaTeX) and `/Users/davidledbetter/Downloads/thesis_constrained-5.pdf`
(PDF). It's David Ledbetter's Master's thesis, UMBC, 2026.

### Version timeline

- **V5–V9**: Fiber-bundle framework with various "spatial" routing
  mechanisms. All hit the same "45% wall" on character-level Shakespeare.
  Diagnosis: routing mechanism was not the bottleneck.
- **V10**: Diagnosed that the manifold coordinate was context-blind —
  `q_t = positional_embedding[t]` rather than a function of history.
- **V11**: Diffusion as field reconstruction (not smoothing). Sparse events,
  Alcubierre-inspired context-warped metric.
- **V12**: **Spectral shift.** Applied Donoho-Stark uncertainty: spatial
  sparsity forces spectral spread, so sparsity belongs in the frequency
  domain. V12.1 achieved **BPC 2.302 / 53.4% on Tiny Shakespeare**, best
  in the project at the time.
- **V12.2 ablation**: devastating. A plain SSM + MLP baseline (no
  spectral machinery at all) achieved BPC 2.267, slightly *better* than
  V12.1 at **4.7× less compute**. The spectral machinery contributed
  zero net value over a conventional baseline at matched capacity.
- **V13**: Pure linear EMA fiber, no content dependence. Plateaued at
  CE ~2.17 on WikiText.
- **V14**: Restored content-dependent fiber. Added Hopfield memory bank.
  Ablation showed **geometry + Hopfield memory > SSM + MLP** — the
  Hopfield bank was identified as the load-bearing piece.
- **V15**: Parseval spectral filter replaces iterative Langevin.
- **V16**: "irfft IS the mixing." Achieved PPL 275 on WikiText-103 at
  20K steps vs GPT-Nano PPL 173 — within 1.6× of attention.
- **V17**: Precision-routed Gaussian constellations. Position encoded as
  a learned precision pattern rather than phase rotation.
- **V18**: Clean-sheet minimal linear attention, "everything that
  survived V1-V17."
- **V19**: Combined V18 with non-abelian $SO(K)$ unitary + delta rule
  transport. Plus CurvBias (geometric content-dependent position encoding)
  which became the thesis's primary empirical contribution — it wins by
  up to 9% over RoPE across scales.
- **V20**: **The spectral return.** Went back to V12.1-style sparse
  spectral constellations with V19's non-abelian transport grafted on.
  `ProximalTopK` top-$k$ sparsification, forward-reverse FFT loop per
  block, `PerSubbundleUnitaryDeltaFiber` for associative memory on the
  post-IFFT signal.

Under-the-hood details are in `/Users/davidledbetter/SparseGeometricSignalTransport/v20/V20_DESIGN.md` and in the individual
version directories.

### V20's current state (as of 2026-04-10)

V20 trains but plateaus around **~20% accuracy / CE ~5.6–5.9 / PPL ~280–400**
on WikiText-103 at 10K steps, versus GPT-Nano which would reach
CE ~4.5–5.0 / 30–35% accuracy at the same compute. The gap is ~1 nat
of CE and persists across all fixes attempted in the session:

- The `softplus(0) = ln(2) ≠ 0` bug in `SpectralTransportKernel.D_head.bias`
  was causing high-frequency modes to be crushed at init. Fixed.
- `ProximalTopK` was cutting gradients exactly to zero on 53% of modes
  per block. Removed from `V20Block`.
- Fiber gate init was `sigmoid(-2) ≈ 0.12`, fiber projection scales were
  `std=0.02`, compound effect was `~1e-8` raw fiber contribution and a
  closed-gate collapse. Fixed: gate init `0.0` (sigmoid = 0.5), projection
  scales `0.1`.
- `log_var` field in `Constellation` was dead code — computed, passed
  through every module unchanged, contributed zero to the loss. Removed.
- The A8 vs A10 comparison (shared `SpatialMLP` vs `PerSubbundleMLP`)
  showed the FFN variant doesn't matter: 5% PPL difference at 9% extra
  params. The FFN is not where the bottleneck lives.

The plateau persists. The diagnosis from the previous chat is that V20
is structurally wrong — it's treating tokens as independent inputs
flowing through blocks, when the intended architecture treats tokens as
features of the model's landscape that are rung by input. V20 is
therefore misarchitected for the user's actual vision, and V21 should
not graft onto V20's block structure.

---

## 2. The intended architecture (user's own description)

Quoting the user directly from the previous chat, after they saw a
visual image in a dream:

> Tokens exist inside the model, or inside the landscape. They are
> simply activated by showing up in the input. The resulting agitation,
> like dropping 3 rocks into a pond (for a sentence with 3 words), or
> maybe activating 3 different frequency groups — what pattern is formed?
> The model should alter the global system's response to that. What
> resonance does the system find? What connection transports information
> encoded this way? Where a trajectory is formed.

And from an earlier message in the same design conversation, before the
pond metaphor:

> The entire point of this architecture was to allow the naturally sparse
> subbundles on the fiber that represent each token to interact. Each
> token eventually gets a learned optimal fingerprint on the fiber bundle
> and how the tokens interact via their connections represents a unique
> activation pattern which the model associatively recognizes naturally
> to represent a sequence, and the word order.

Combining these, the architectural intent is:

1. **Vocabulary tokens are fixed features of the model.** Not learned
   embeddings that get loaded into the hidden state each forward pass —
   permanent structural properties of the model's parameters. Showing a
   word to the model is not "injecting a vector" but "ringing a
   pre-existing bell."

2. **A sentence is a chord of simultaneous strikes.** Three tokens in a
   sentence cause three disturbances in a continuous medium. The
   simultaneous response of the medium — the interference pattern —
   is the computation.

3. **The landscape is a physical system.** It has resonant modes,
   standing waves, or fixed points. The same landscape across all inputs
   (its shape is a property of trained parameters). Inputs merely
   perturb it.

4. **Connections between tokens emerge from interference.** The model
   does not store "token A is related to token B." When A and B are
   both activated, their beams/ripples/waves naturally form a specific
   interference pattern. The connection between A and B *emerges* from
   the landscape's geometry.

5. **The trajectory is the sequence of evolving resonance patterns.**
   As each new token strikes the landscape, the resonance pattern
   changes. The "trajectory" is the path through the landscape traced
   out by successive patterns. Prediction = finding which next strike
   extends the trajectory most naturally.

### The dream image

The user described an image of lasers forming a ship-like or pyramidal
geometric structure in a dark space. The structural elements are:

- Lines of light (blue, green, pink — different colors) cross in 3D
- The structure is defined by the *connections* between vertices, not
  the vertices themselves
- Most of the space is dark — activation is sparse
- The lines look like they interfere/reflect — standing-wave-like

The image is suggestive of **interference patterns in a shared medium**,
not of point activations flowing through layers.

---

## 3. Why V20 is misarchitected for this vision

V20 keeps the transformer-style `(B, T, *)` layout: every token has
its own separate spectral constellation, and blocks process them with
parallel-but-mostly-independent operations. There is no "the landscape"
and no "the field." Specifically:

| User's intent | V20 reality |
|---|---|
| Tokens are fixed features of the model | `ConstellationEmbedding` is a learned `nn.Embedding` that copies per-token vectors into the hidden state each forward pass |
| Sentence is a simultaneous chord | `V20Block` processes `(B, T, M)` with causal per-position structure |
| Shared global landscape | No shared field; every position has its own constellation |
| Connections emerge from interference | Connections come from `PerSubbundleUnitaryDeltaFiber`'s per-position `q @ S_prev` query — which is linear attention, not interference |
| Resonance is the computation object | The computation object is a `(B, T, M)` tensor that flows through 6 blocks |

V20's `SpectralTransportKernel` does have the right *mathematical
form* (content-dependent $D_k(q)$ and $A_k(q)$ per mode), but it applies
it per-position, not to a shared global field. The forward-reverse FFT
loop operates on each position independently.

**V21 should not be "V20 with a few fixes."** The block structure itself
is the wrong object. V21 should build a different *computational model*
with the same goal (efficient sparse spectral language modeling).

---

## 4. Literature the new chat should know about

Before designing V21, the new chat should skim at least the abstracts of
these. The user's vision is closest to a combination of **resonator
networks + neural wave equations on a learned landscape**, which is not
standard in modern language modeling but is not unprecedented.

1. **Resonator networks** (Frady, Kent, Sommer 2018–2020) — each
   vocabulary item is a fixed high-dimensional fingerprint; a
   superposition of them is the input; iterative fixed-point dynamics
   decompose the superposition into its constituent items. This is
   nearly a literal implementation of the user's "rocks in a pond"
   picture.

2. **Hyperdimensional / vector symbolic architectures** (Kanerva, Plate,
   Gayler) — tokens are fixed high-dim vectors, binding is element-wise
   multiplication or circular convolution, composition is superposition.
   Meaning is geometric. Connections are literally geometric interference
   patterns.

3. **Fourier Neural Operators** (Li et al. 2021, ICLR) — learn linear
   operators in the spectral domain; the operator acts on a whole field
   at once, not per-position. Directly relevant to the "apply a global
   operator to a shared field" computation.

4. **Neural wave equations / neural ODEs on manifolds** — train a
   discretized wave or diffusion operator; treat inputs as source terms;
   read the field at "sensor" locations. Physically grounded.

5. **Reservoir computing / echo state networks** — a fixed random
   dynamical system (the landscape) is perturbed by inputs; only a
   linear readout is trained. The landscape is literally fixed across
   forward passes.

6. **Kuramoto oscillator networks** — $N$ phase oscillators with learned
   pairwise couplings; inputs nudge specific oscillators; the
   globally-synchronized phase-locked pattern encodes the input's
   meaning. Resonance is structural, not a metaphor.

7. **Dense Associative Memory / Modern Hopfield Networks** (Ramsauer
   et al. 2021) — the Ramsauer equivalence proves that attention is a
   one-step Hopfield retrieval. In V14 in this project, adding a
   Hopfield bank to the spectral architecture made it beat SSM+MLP.

8. **V12.1 architecture** in this project's own history
   (`thesis_constrained/Chapter5.tex`) — the closest prior version
   to what the user wants, and the one with the best empirical result
   (BPC 2.302 on Tiny Shakespeare) before the V12.2 ablation undermined
   the claim.

The new chat should treat this list as background research, not as an
exhaustive survey.

---

## 5. What V20 code assets can be reused

### Keep (useful regardless of architecture)

These are in `v20/v20_modules.py` and are general-purpose enough to
transfer into V21 directly:

- `RMSNorm`, `rms_norm` — standard normalization
- `make_skew_symmetric` — vectorized `triu_indices` implementation
- `fast_orthogonal` — 4-term Taylor approximation of `exp(A)` for small
  skew-symmetric `A`. MPS-safe, used for $SO(K)$ transport if needed
- `unitary_delta_parallel_scan` — $O(T \log T)$ Hillis–Steele sweep over
  the `(U, B)` semigroup. Useful if V21 still has a per-position linear
  recurrence (e.g., for an incremental field update)
- `count_params` — trivial but useful

### Maybe keep (depends on V21 architecture)

- `Constellation` as `(mag, phase)` NamedTuple — sparse spectral token
  representation. V21 might or might not use per-position constellations;
  if it uses a single global field, the NamedTuple becomes unused.
- `ConstellationEmbedding` — token → learned (mag, phase) per mode.
  V21 will probably have a different embedding mechanism (token →
  fixed fingerprint rather than learned embedding).
- `CloudNorm` — V17 RMS normalization of constellation magnitudes. Only
  relevant if V21 uses per-position constellations.
- `SpectralTransportKernel` — V12.1's $\exp(-D_k(q) \omega^2 - i A_k(q) \omega)$.
  The *form* is correct for a content-dependent spectral operator. If V21
  uses a shared global field, this may transfer — operating on the field
  rather than per-position — but the input signature needs rethinking.
- `SparseFFT` / `SparseIFFT` at Level 0 — full rFFT + band mask. Simple
  enough to reimplement in V21 if needed.
- `LearnedBandMask` — per-block differentiable sparse mask over modes.
  Transferable if V21 has per-block frequency allocation.
- `PerSubbundleUnitaryDeltaFiber` — unlikely to be useful. It's a
  per-position linear attention mechanism; V21 is probably not
  per-position.
- `SpatialMLP`, `PerSubbundleMLP` — standard FFNs; probably not needed.
- `ProximalTopK` — hard top-$k$ sparsification; V21 probably uses a
  softer mechanism (L1 penalty, soft `entmax`, or implicit sparsity via
  landscape geometry).

### Drop

- `V20Block` — wrong computation object, start fresh
- `V20Model` — wrong computation object, start fresh
- `gen_notebook_v20.py` — notebook generator for V20's ablation matrix;
  V21's experimental protocol will differ
- `test_v20.py` — most tests are tied to V20's per-position block
  structure; rewrite against V21's primitives
- `benchmark_v20.py` — same reason

### How to reuse (pattern)

V21 should be a **self-contained directory** like V20 is now. Copy the
utility kernels from V20 into a new `v21/v21_modules.py` rather than
importing across directories. This was a lesson from V20: cross-directory
imports via `sys.path` hacks are fragile and confusing.

---

## 6. Three open questions that must be answered before writing V21 code

These are the questions the previous chat asked the user but did not
receive answers to before the user asked for this handoff doc. The new
chat should ask them first.

### Question 1: Simultaneous or sequential strikes?

In the "rocks in a pond" metaphor, are the 3 rocks dropped:

**(a) Simultaneously**, and the only thing that position-encodes is
*where* on the pond each rock lands? Word order is encoded by
*spatial location* on the landscape, not by time.

**(b) Sequentially**, with the first rock's ripples already spreading
when the second rock lands? Word order is encoded by *time of arrival*,
and the landscape has its own dynamics that propagate each disturbance
between strikes.

Both are physically coherent but lead to very different architectures.
(a) is closer to a resonator network or a shared-field model where the
input is a single superposition vector. (b) is closer to a stateful
dynamical system with time-varying forcing terms — closer to a spectral
SSM.

### Question 2: One landscape per model, or one per sequence?

**(a) Fixed landscape.** The model parameters literally *are* the
landscape's geometry. Every forward pass rings the *same* landscape.
This matches the "tokens exist inside the model" framing most directly.

**(b) Input-conditioned landscape.** Each forward pass sets up a fresh
landscape whose shape depends on the input. More flexible but the
"tokens are fixed features" framing doesn't quite hold.

### Question 3: Readout mechanism

When the landscape has finished ringing, how does the model produce a
next-token prediction? Some candidates:

**(a) Correlation with fingerprints.** Compute an inner product between
the current resonance pattern and every vocabulary item's fingerprint;
the item with highest overlap is the prediction. The "head" is tied to
the token fingerprints by construction.

**(b) Next-strike prediction.** The model asks "what new strike would
extend the current resonance trajectory most consistently?" Generative
— the model hallucinates the next source term and matches it against
the vocabulary.

**(c) Sensor readout.** Fixed "sensor" locations in the landscape always
report the field value, and a linear layer maps sensor readings to
logits. Readout is independent of the vocabulary's fingerprints.

**(d) Trajectory continuation.** Treat the field's time evolution as a
trajectory in a latent space and predict where the trajectory goes next,
then project to logits via a small decoder.

### Until these are answered

No V21 code should be written. The architectural decisions cascade
from these three answers — specifically, (a) vs (b) in Question 1
determines whether V21 is a *static* resonance model or a *dynamical*
field model, and that is the single biggest fork.

---

## 7. Architectural sketches for the decision tree

Once the questions in §6 are answered, the new chat should design
against one of the sketches below rather than starting from scratch.
These are not complete specifications — they are starting points
showing what each decision branch looks like concretely.

### Sketch A — Simultaneous strikes, fixed landscape, fingerprint readout

*Closest to resonator networks.*

```
# Landscape: a learned shared complex-valued spectral basis
# (M modes, fixed across all forward passes — these are the trainable
# parameters that define the landscape's geometry)
Landscape: complex (M,) eigenvectors + (M,) eigenvalues

# Vocabulary: each token has a fixed fingerprint (a sparse complex vector
# in the landscape's basis). Init can be random, possibly orthogonal.
Fingerprints: (vocab_size, M) complex, PARAMETER (learnable but interpretable
              as "which modes does this token couple to")

forward(token_ids):
    # All tokens in the sentence strike simultaneously; superpose their
    # fingerprints, modulated only by position (e.g., a position-dependent
    # complex phase shift, not a scan).
    sources = Fingerprints[token_ids]         # (B, T, M) complex
    pos_phase = exp(1j * arange(T) * omega)   # (T, M) complex
    sources = sources * pos_phase             # position-modulated
    field = sources.sum(dim=1)                # (B, M) complex — the chord

    # Find the landscape's resonant response to this chord. This can be
    # (i) a closed-form spectral operator applied to field,
    # (ii) a few iterations of fixed-point dynamics (resonator-network style),
    # (iii) a content-dependent spectral transport kernel (V12.1 style)
    #      applied to the whole field (not per-position).
    response = landscape_operator(field)      # (B, M) complex

    # Readout by correlation with every token fingerprint.
    logits = |Fingerprints @ response.conj().T|  # (B, vocab_size)
    return logits
```

Note: this returns a **single prediction per sequence**, not per
position. For causal LM, this needs per-position output. Two options:

- **Incremental construction**: the field is built up one token at a
  time, and a prediction is made after each strike. See Sketch B for
  this.
- **Masking**: for each target position $t$, rebuild the field from
  only tokens 0..t-1 and predict the token at position $t$.

Sketch A is clean and fast but the per-position readout is awkward.

### Sketch B — Sequential strikes, fixed landscape, fingerprint readout

*Closest to a spectral SSM where the state vector is a shared "field"
over a learned basis.*

```
# Landscape: learned spectral operator (complex (M, M) matrix or its
# eigendecomposition)
LandscapeOp: complex (M, M), PARAMETER

# Vocabulary fingerprints
Fingerprints: (vocab_size, M) complex, PARAMETER

forward(token_ids):
    field = zeros(B, M, dtype=complex)
    logits_per_pos = []
    for t in range(T):
        # Strike the field with the next token's fingerprint
        strike = Fingerprints[token_ids[:, t]]                    # (B, M)

        # The landscape propagates the previous field AND receives the
        # new strike. This is literally a linear recurrence:
        #   field[t] = LandscapeOp @ field[t-1] + strike[t]
        field = einsum('mn,bn->bm', LandscapeOp, field) + strike   # (B, M)

        # Readout after each strike: correlate with all fingerprints
        logits_per_pos.append(|Fingerprints @ field.conj().T|)
    return stack(logits_per_pos, dim=1)
```

This is a **spectral linear recurrence with fixed fingerprints and
correlation readout**. Properties:

- **Parallel-scan friendly**: the linear recurrence can be computed in
  $O(T \log T)$ via the Hillis–Steele sweep reused from V20
  (`unitary_delta_parallel_scan` with the landscape op in place of
  content-dependent $U$).
- **Very fast**: no per-token projection (just fingerprint lookup),
  shared landscape operator, correlation readout is a single matmul.
- **Low parameter count**: `Fingerprints` is `vocab * M`, `LandscapeOp`
  is `M²`. If $M = 256$, landscape is 65K parameters plus $50K \times 256 = 12.8 M$
  for fingerprints plus a tiny input/output head. Maybe 15M total.
- **Deep version**: stack $L$ blocks where each block has its own
  landscape operator applied to the running field. This gives
  content-dependent transport (if the operator is made content-dependent
  via a small MLP from the current field).

This sketch is the most likely starting point in practice because it
maps cleanly to well-understood ML primitives (linear recurrence,
parallel scan, correlation head) while preserving the user's framing:
- Tokens are fixed (Fingerprints are permanent learned properties)
- The landscape is shared (one `LandscapeOp` per block, not per token)
- Inputs ring the landscape (additive source term in a linear recurrence)
- Trajectory is explicit (the sequence of `field[t]` values)
- Readout is via resonance overlap (correlation with fingerprints)

### Sketch C — Simultaneous strikes, fixed landscape, sensor readout

*Closest to reservoir computing on a learned spectral manifold.*

```
# Landscape: learned (M, M) complex operator
# Sensors: K fixed learned (M,) complex vectors that probe the field
LandscapeOp: complex (M, M)
Sensors: complex (K, M)
Readout: Linear(K * 2, vocab_size)   # real+imag of K sensor readings

forward(token_ids):
    # Build the chord
    sources = Fingerprints[token_ids].sum(dim=1) * pos_phase_modulation
    # Apply the landscape's closed-form response (e.g., inverse of I - LandscapeOp)
    field = solve(I - LandscapeOp, sources)  # steady-state resonance
    # OR: iterate N times with LandscapeOp until convergence

    # Read at fixed sensor locations
    readings = Sensors @ field.unsqueeze(-1)  # (B, K) complex
    logits = Readout(cat([readings.real, readings.imag]))
    return logits
```

Same caveats as Sketch A about per-position readout. This sketch is
worth considering if the user wants the readout path to be
vocabulary-agnostic (useful for transfer learning or for treating the
readout as a separate output head that can be swapped without
retraining the landscape).

### Sketch D — Sequential strikes, input-conditioned landscape

*Closest to a content-dependent spectral SSM that V20 was trying to be
but didn't quite achieve.*

This is the "keep the good parts of V20 but fix the per-position
framing" branch. The landscape operator is content-dependent per step,
but the object being transported is a **single shared field per sample**,
not a per-position constellation.

```
forward(token_ids):
    field = zeros(B, M, complex)
    for t in range(T):
        # Content-dependent spectral operator: D_k(x_t), A_k(x_t)
        D, A = transport_head(embedding[token_ids[:, t]])
        kernel = exp(-D * omega**2 - 1j * A * omega)  # (B, M)
        # Apply kernel to the field (element-wise, since the landscape
        # is parameterized in its eigenbasis)
        field = field * kernel + Fingerprints[token_ids[:, t]]
        logits_per_pos.append(|Fingerprints @ field.conj().T|)
    return stack(logits_per_pos, dim=1)
```

This is the closest sketch to "V20 rewritten correctly": the transport
kernel now operates on the shared field, not per-position constellations.
The field is the same object across all positions; each token both
strikes it and modulates its evolution. Fingerprint readout means
tokens are literally features of the vocabulary's geometry.

### Decision table

| Q1 | Q2 | Q3 | Sketch |
|---|---|---|---|
| Simultaneous | Fixed | Fingerprint | A |
| Sequential | Fixed | Fingerprint | **B (most likely starting point)** |
| Simultaneous | Fixed | Sensor | C |
| Sequential | Input-conditioned | Fingerprint | D |
| Any | Input-conditioned | Sensor | (need to design) |

---

## 8. Lessons from V20 that apply regardless of V21's architecture

These are hard-won lessons the previous chat learned. V21 should not
re-encounter them.

1. **`softplus(0) = ln(2) ≠ 0`.** Any init that uses
   `softplus(0)` as "the default value" will produce a
   non-zero default (≈0.693), which in the context of
   `exp(-softplus(0) · ω²)` is catastrophic damping on high-frequency
   modes. If V21 has any positivity-constrained content-dependent
   parameters, initialize their bias to −6 so `softplus(−6) ≈ 0`
   at init.

2. **Hard top-$k$ cuts gradients to zero on dropped modes.** This
   creates a self-reinforcing dead-mode problem. If V21 needs sparsity,
   use soft mechanisms: L1 penalty on amplitudes, differentiable
   `α-entmax`, or implicit sparsity from the landscape's spectral
   structure. Avoid `torch.topk`-based masking.

3. **Closed-gate collapse.** Any gated sub-module initialized with
   `sigmoid(-2) ≈ 0.12` or lower will almost never receive meaningful
   gradient because its contribution to the output is too small to
   influence the loss, so the gate never opens, so the sub-module
   never contributes. Initialize gates at `sigmoid(0) = 0.5` (the
   max-gradient point) or higher if the sub-module is trustworthy at
   init.

4. **Projection-init compounding.** If a forward path multiplies
   through $N$ weight matrices each with std $\sigma$, the compound
   output has magnitude $\sim \sigma^N$. For $\sigma = 0.02$ and $N = 4$
   this is $1.6 \times 10^{-7}$ — way below any meaningful gradient
   signal. Either use larger init (0.1) or use initialization schemes
   that account for the chain (e.g., fan-in based).

5. **Dead-code data structures compound in confidence.** V20 carried
   a `log_var` field through every module that *looked* like a V17
   precision-routing mechanism was active, but nothing downstream
   actually consumed it, so the `var_mix` layer received zero gradient
   and stayed at zero forever. If V21 adds a data field that isn't
   immediately consumed, add a test that verifies gradient reaches it.

6. **Cross-directory imports are landmines.** V20 originally imported
   utility kernels from `../v19/v19_modules.py` via a `sys.path.insert`
   hack. This created a `_LearnedBandMaskShim` adapter because V19 and
   V20 had different config attribute names for the same concept.
   Confusing, fragile, and easy to forget. V21 should be a
   self-contained directory.

7. **`torch.compile` was not reliable** on V19/V20's scan-based fiber.
   Eager + `bfloat16` autocast + TF32 matmul flags was the combination
   that actually worked on H100. Do not start with `torch.compile`
   enabled; add it only if it measurably helps and doesn't produce
   compile errors.

8. **The V12.2 ablation bar is real.** Any V21 architecture must beat
   a plain SSM + MLP baseline at matched compute on WikiText-103.
   That baseline is cheap to run and is the single most important
   control: if V21 doesn't beat it, the spectral machinery is
   providing no value (V12.2's original finding).

9. **V20's fiber projection gradient analysis was the real plateau
   diagnosis.** At some point in the V20 arc the user discovered the
   model was "plateauing at 20%" which corresponds to the unigram-
   frequency baseline — meaning no cross-token information flow was
   happening. The V21 diagnostic analog is: **if accuracy stays around
   the unigram frequency ceiling after several thousand steps, the
   architecture has no cross-token information flow**. Fix this
   urgently, not by adding more fancy machinery.

---

## 9. File layout for `v21/`

Follow the V20 pattern:

```
v21/
├── V21_PLAN.md                (this file)
├── V21_DESIGN.md              (to write AFTER user answers §6 questions)
├── v21_modules.py             (self-contained, copies utility kernels from v20/v20_modules.py)
├── test_v21.py                (shape, causality, Parseval bound, gradient flow, etc.)
├── benchmark_v21.py           (component + full-model breakdown)
├── gen_notebook_v21.py        (training notebook generator)
├── architecture_v21.ipynb     (generated)
└── README.md                  (quick-start)
```

The new chat should write `V21_DESIGN.md` as a separate document once
the three questions in §6 are answered and the sketch from §7 is
chosen. `V21_PLAN.md` (this doc) stays as the historical handoff.

---

## 10. Experimental protocol

V21's primary comparison is the same as V20's:

1. **Control: plain GPT-Nano** at matched `d_model` and `n_blocks`
   (no CurvBias).
2. **SSM+MLP baseline** — the V12.2-style control that V12.1 failed to
   beat. If V21 loses to this, the spectral direction is conclusively
   falsified.
3. **V21 full** — the proposed architecture.

All three trained on **WikiText-103**, `seq_len = 256`, at matched
wall-clock on H100 with `bf16` autocast (no `torch.compile`). Each
eval reports: BPC, PPL, accuracy, ms/step, peak memory.

The **minimum win condition** for V21 is:

> V21 full beats SSM+MLP by ≥10% PPL at matched wall-clock on
> WikiText-103 at 20K steps.

Stretch goal:

> V21 full matches or beats GPT-Nano at matched wall-clock.

Publishable outcome:

> Either of the above, plus a clean ablation that isolates which
> component is load-bearing (the fingerprint structure, the landscape
> operator, or the readout mechanism).

---

## 11. How to pick up from here (new chat bootstrap)

When the new chat opens, the user will describe the project briefly
and point at this doc. The new chat should:

1. **Read** `V21_PLAN.md` (this file), all of §1–10.
2. **Read** `V20_DESIGN.md` for the full retrospective of V5-V20 if more
   context is needed.
3. **Read** `v20/v20_modules.py` to see what utility kernels are
   available for transfer (skip the V20-specific block/model code).
4. **Ask** the user the three questions in §6 of this doc. Do not
   propose code before getting answers.
5. **Pick a sketch** from §7 based on the answers.
6. **Write** `V21_DESIGN.md` as a concrete design doc (one sketch,
   one experimental protocol, one file layout). Get user approval.
7. **Start** implementing in `v21/v21_modules.py` with tests in
   `test_v21.py`.

The new chat should **not**:

- Graft onto V20's `V20Block` or `V20Model` — the computation object is
  wrong.
- Propose incremental tweaks to V20's architecture.
- Re-derive the three questions from scratch; they're already framed
  here.
- Start with `torch.compile` enabled (it was unreliable in V20).
- Ship a `ProximalTopK` or any hard-masking sparsity mechanism.
- Use cross-directory imports (`sys.path.insert`); copy utility
  kernels into `v21/v21_modules.py`.

---

## Appendix A — Specific context from the V20 session that the new chat might want

### Current V20 file state (as of handoff)

`v20/v20_modules.py` is self-contained (no V19 imports), has no
`log_var` or `ProximalTopK` in `V20Block`, has the softplus bias fix,
the fiber gate open-at-init fix, and the projection-init scale fix.
All 22 tests in `v20/test_v20.py` pass.

### Results from the most recent V20 run

A10 V20 strict-subsp after the softplus + gate fixes but before the
projection-init scale fix:

| Step | Val CE | BPC | PPL | Acc |
|---:|---:|---:|---:|---:|
| 0 | 10.920 | 15.754 | 55,251 | 0.0% |
| 500 | 7.048 | 10.168 | 1,150 | 12.5% |
| 1000 | 6.713 | 9.685 | 823 | 13.5% |
| 5000 | 5.952 | 8.588 | 385 | 18.6% |
| 10000 | 5.632 | 8.126 | 279 | 20.7% |

Comparison: GPT-Nano at matched scale would reach CE ~4.5–5.0, PPL
~150–200, accuracy ~30–35% at 10K steps. V20 has ~1 nat of CE gap
and ~10–15 percentage points of accuracy gap — consistent with
"no cross-token information flow" (the unigram frequency ceiling is
around 18–22% on WikiText-103).

### The image the user saw

The user described a mental image from a dream: lasers forming a
ship-like or pyramidal 3D structure against a dark background. Lines
of colored light (blue, green, pink) cross and interfere at the
vertices. The structure is defined by the interconnections, not by
the vertices themselves. Most of the space is dark; activation is
sparse. The lines give a strong sense of standing waves or resonance.

This image is the closest visual summary of the user's architectural
intent. When in doubt, the new chat should ask the user to
re-describe this image because it is the clearest reference point.

### What the user explicitly said CurvBias was

> CurvBias was a pitstop.

The new chat should not propose adding CurvBias to V21. It is
available in `v19/v19_modules.py` as a baseline for comparison, but
it is not part of V21's main line.

### What the user said V20 was (briefly)

> We're returning to trying to make the spectral model efficient.

V21 is the next iteration of that goal. The scope is: **efficient
spectral language modeling** where "efficient" means speed-competitive
with attention and "spectral" means in the frequency domain with a
shared landscape.
