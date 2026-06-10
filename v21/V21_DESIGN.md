# V21 Design: Holonomic Linear Attention on a Sparse Multi-Landscape Bundle

**Status**: design document, approved for implementation.
Predecessor: `V21_PLAN.md` (the handoff doc with project history and the
three open questions). This document answers those questions concretely
and specifies V21's architecture, file layout, and experimental protocol.

---

## 0. One-paragraph summary

V21 is a **fiber-bundle forward-model network trained by state
prediction**. The model holds $K$ parallel orthogonal landscapes (each
an $SO(M)$ recurrent system) connected by a learned $SO(M)$ parallel
transport on a base graph (initially a line graph). Each vocabulary
token has a permanent learned **sparse fingerprint** across the $K$
landscapes — an L1 penalty during training discovers which small subset
of landscapes each token "lives in." A forward pass processes tokens
sequentially: at each step the accumulated field is propagated through
landscape dynamics and bundle transport, and a small learned head
predicts the next strike directly from the propagated field. The
training signal is an **L2 distance between the predicted strike and
the actual next token's fingerprint**, augmented with an InfoNCE
contrastive term against sampled negative tokens to prevent
fingerprint collapse. There is no global vocab-wide scoring during
training — only the target token and a handful of negatives per
position participate in the loss. The entire dynamical core is linear;
the only nonlinearity is the contrastive softmax over the small
negative set. The sparsity of fingerprint occupancy across landscapes
*is* the sparsity mechanism that V12 originally identified as "sparsity
belongs in Fourier space" — reinterpreted here as "sparsity belongs in
the bundle's section space."

---

## 1. Mapping to V21_PLAN.md's three questions

From `V21_PLAN.md` §6:

- **Q1 (simultaneous vs. sequential strikes)**: **Sequential.** Tokens
  strike one at a time; the landscape has its own linear dynamics
  between strikes.
- **Q2 (fixed vs. input-conditioned landscape)**: **Fixed.** The
  landscapes and their transports are trained parameters — one global
  *bundle* per model, not conditioned on input. This is consistent with
  "tokens exist inside the model." The resolution of the user's "many
  landscapes" extension is that the bundle *is* one global object,
  realized as a collection of local landscapes glued by the transport.
- **Q3 (readout)**: **Self-supervised state prediction.** The model
  predicts the next strike's pattern directly from the accumulated
  field state, and the training signal is an L2 distance between the
  predicted strike and the actual next token's fingerprint. There is
  no vocab-wide scoring during training. The user's framing is "the
  system is learning to predict a series of states... the last token
  being in its locations should modify the fingerprint. When this does
  not happen correctly we adjust the landscapes and the tokens'
  fingerprints across the landscapes." The next token *emerges* from
  the landscape's current pattern as a forward-model prediction; it is
  not *chosen* by scoring all candidates.

V21 therefore starts from **Sketch B** in `V21_PLAN.md` §7 and extends
it with (a) multiple landscapes connected via bundle transport, (b)
sparse per-token fingerprint allocation, and (c) a state-prediction
training objective that bypasses vocab-wide readout entirely (a
mechanism not present in any of the four original sketches — call it
"Sketch B′").

---

## 2. Architecture

### 2.1 Hyperparameters

```
K          : number of landscapes in the bundle              (default 32)
M          : per-landscape field dimension                   (default 32)
V          : vocabulary size                                 (50257, GPT-2 tokenizer)
T          : context length                                  (256)
L          : number of bundle layers stacked (blocks)        (default 1; ablation: up to 4)
alpha_init : initial bundle transport blending coefficient   (0.25)
l1_coef    : L1 penalty on fingerprint k-norms               (1e-3, tuned)
sigma_fp   : init std for fingerprints                       (0.02)
N_neg      : number of contrastive negatives per position    (32)
tau        : InfoNCE temperature                             (0.1)
lambda_ctr : weight of contrastive term in total loss        (1.0)
lambda_l1  : weight of L1 sparsity term in total loss        (1.0)
```

Parameter count at defaults: `Fingerprints` dominates at
$V \cdot K \cdot M = 50257 \cdot 32 \cdot 32 \approx 51.5\text{M}$.
`LandscapeOps` and `Transports` each contribute $\approx K M (M-1)/2
\approx 16\text{K}$ skew-symmetric parameters (the matrix exp is
computed on the fly, not stored). `PredictStrikeHead` adds $K \cdot M^2
\approx 32\text{K}$ parameters. Total $\approx 51.6\text{M}$ per bundle
layer, dominated entirely by the fingerprint tensor.

Note: the per-landscape Hopfield temperature `beta_k` from the previous
design is **removed**. The new training signal is L2 distance, not a
correlation-based softmax, so per-landscape temperatures are not
needed. The InfoNCE `tau` is a single scalar shared across the loss.

### 2.2 Parameters

```python
# One bundle layer (V21Block)
LandscapeOps      : (K, M*(M-1)/2)         # skew-symmetric params; exp'd to SO(M)
Transports        : (K-1, M*(M-1)/2)       # line graph: one transport per edge
alpha             : scalar, learnable      # bundle transport blending coefficient
PredictStrikeHead : (K, M, M)              # per-landscape linear map field→strike
                                           # (W_k : R^M → R^M, one per landscape)

# Shared across all bundle layers (model-wide)
Fingerprints      : (V, K, M)              # dense, L1-regularized
```

Tied input/target via `Fingerprints`: a token's "embedding" when it
strikes the field is the same tensor as its "target" during
state-prediction loss. This tying is what lets gradients flow through
`Fingerprints` in both the forward direction (via strikes landing on
the field) and the backward direction (via the L2 state-prediction
loss). The user's description — "the last token being in its locations
should modify the fingerprint" — is literally this: the L2 loss between
the predicted strike and the target fingerprint pushes the
fingerprint toward the prediction and the prediction toward the
fingerprint simultaneously.

### 2.3 Forward pass

The forward pass produces **predicted strikes** — one $(K, M)$-shaped
prediction per position — not vocab logits. Vocab logits only exist at
eval time (see §4.3).

```python
def forward(token_ids):                                 # (B, T) int64
    B, T = token_ids.shape
    fields = zeros(B, K, M)                             # bundle state
    
    # Compute SO(M) from skew-symmetric params once per forward
    U_k   = fast_orthogonal(LandscapeOps)               # (K, M, M), each SO(M)
    T_k   = fast_orthogonal(Transports)                 # (K-1, M, M), each SO(M)
    a     = sigmoid(alpha)                              # blending in (0, 1)
    
    predicted_strikes = []                              # one (B, K, M) per position
    for t in range(T):
        # --- 1. Propagate the accumulated field WITHOUT seeing token[t] ---
        # (a) Per-landscape temporal dynamics (pure linear SO(M))
        fields = einsum('kij,bkj->bki', U_k, fields)
        # (b) Bundle transport on the line graph (pure linear)
        left_transported = zeros_like(fields)
        left_transported[:, 1:] = einsum('kij,bkj->bki', T_k, fields[:, :-1])
        fields = (1 - a) * fields + a * left_transported
        
        # --- 2. Predict what strike should arrive at step t ---
        # Per-landscape linear forward model: predicted_strike[b, k] = W_k @ fields[b, k]
        pred_strike = einsum('kij,bkj->bki', PredictStrikeHead, fields)
        predicted_strikes.append(pred_strike)
        
        # --- 3. Actually land the real strike (training-time supervision) ---
        # Use the ground-truth token's fingerprint as the strike. Gradients
        # on this strike flow back to Fingerprints.
        real_strike = Fingerprints[token_ids[:, t]]     # (B, K, M)
        fields = fields + real_strike
    
    # predicted_strikes[t] is the model's guess at Fingerprints[token_ids[:, t]]
    # BEFORE seeing token_ids[:, t]. This is the causal state-prediction target.
    return stack(predicted_strikes, dim=1)              # (B, T, K, M)
```

**Design notes.**

1. The forward pass splits each step into three actions: **propagate**
   (apply landscape and bundle dynamics to the existing field),
   **predict** (use `PredictStrikeHead` to guess what the next strike
   should be), and **strike** (actually add the ground-truth fingerprint
   to the field for the next step's propagation). The order matters:
   the prediction at step $t$ must be made *before* the model sees
   token $t$ — that's how causality is enforced.

2. `PredictStrikeHead` is a per-landscape linear map $W_k \in
   \mathbb{R}^{M \times M}$. At landscape $k$, the predicted strike is
   $W_k \cdot \text{field}[k]$. This gives each landscape independent
   control over its forward model while keeping the cost tiny
   ($K \cdot M^2$ parameters, same flops per step as landscape dynamics).
   **Cross-landscape information already flows through bundle
   transport** in step 1, so `PredictStrikeHead` doesn't need its own
   cross-landscape mechanism.

3. The per-landscape temporal dynamics and the bundle transport are
   both orthogonal operators. The convex blend `(1-a) fields + a ·
   left_transported` is a learnable diffusion rate; at `a=0` landscapes
   are decoupled, at `a=1` each landscape is entirely replaced by its
   transported left neighbor.

4. The left-only directional transport matches a causal, sequential
   "sweep" along the line graph. A bidirectional variant is a future
   ablation.

5. **Parallel scan**: the propagation step (landscape dynamics + bundle
   transport) is linear in `fields`, so in principle a Hillis–Steele
   scan (V20's `unitary_delta_parallel_scan`) over the combined
   $(KM, KM)$ operator can compute the $T$-step recurrence in
   $O(T \log T)$. For V21's initial implementation we use the simple
   unrolled Python loop — parallel scan is a Tier-2 optimization.

6. **Position encoding**: none. Position is implicit in the sequential
   order of strikes, as in a standard RNN.

7. **Causality**: automatic. Each `predicted_strikes[t]` is computed
   from `fields` *before* the `Fingerprints[token_ids[:, t]]` strike is
   added, so it depends only on `token_ids[:, :t]` (strictly less than
   $t$). At $t=0$ the field is zero, so `predicted_strikes[0]` is
   whatever `PredictStrikeHead` applied to the zero-field — a constant.
   This is fine; the state-prediction loss at $t=0$ just pushes that
   constant toward the fingerprint of the most likely first token,
   which converges to the unigram prior.

8. **Teacher forcing during training**: in the loop above, the "strike"
   added to the field at step $t$ is the **ground-truth**
   `Fingerprints[token_ids[:, t]]`, not the predicted strike. This is
   teacher forcing — standard for CLM. At inference time, the predicted
   strike is decoded to a token (see §4.3) and that token's fingerprint
   is used as the strike for the next step.

### 2.4 Stacking (optional)

For $L > 1$ bundle layers, each layer has its own `LandscapeOps`,
`Transports`, `alpha`, and `PredictStrikeHead`. `Fingerprints` is
**shared across all layers** to preserve the "tokens exist in the
model" framing and keep parameter count manageable. Only the final
layer's `PredictStrikeHead` produces the predicted strike that
contributes to the loss; earlier layers' propagated fields are passed
up the stack at each position.

The V21 initial experiment uses $L = 1$ (one bundle layer). Ablation
up to $L = 4$ is planned.

### 2.5 Training loss

The loss has three parts: a state-prediction L2 term, a contrastive
InfoNCE term, and an L1 sparsity term.

**State prediction (primary signal).**
At each position $t$, the predicted strike $\hat{s}_t \in
\mathbb{R}^{K \times M}$ is compared against the target strike $s_t =
\text{Fingerprints}[\text{token}_t]$:

$$
\mathcal{L}_\text{state}(t) = \frac{1}{KM} \| \hat{s}_t - s_t \|_2^2
$$

This is averaged over $t$ and over the batch. Cost per position:
$O(KM)$, not $O(V \cdot KM)$. This is the primary gradient signal for
both the dynamics (`LandscapeOps`, `Transports`, `alpha`,
`PredictStrikeHead`) and the fingerprints.

**Contrastive anti-collapse term (InfoNCE over sampled negatives).**
Without anti-collapse, the L2-only loss has trivial minima (all
fingerprints = 0, or all identical). To prevent this, at each position
we sample $N_\text{neg}$ random token ids from the vocab (with
replacement, per-position resampled) and require the predicted strike
to be closer to the target fingerprint than to the negatives:

$$
\mathcal{L}_\text{ctr}(t) = -\log \frac{\exp(-\|\hat{s}_t - s_t\|^2 / \tau)}
{\exp(-\|\hat{s}_t - s_t\|^2 / \tau) + \sum_{j=1}^{N_\text{neg}} \exp(-\|\hat{s}_t - s^{(j)}_\text{neg}\|^2 / \tau)}
$$

where $s^{(j)}_\text{neg} = \text{Fingerprints}[\text{neg\_ids}^{(j)}]$
and $\tau$ is the InfoNCE temperature. Cost per position: $O(N_\text{neg}
\cdot K \cdot M)$. At $N_\text{neg} = 32$ this is 1000× cheaper than
global scoring, and it gives gradients to the negative fingerprints
(pushing them *away* from the predicted strike) which is exactly the
mechanism that prevents collapse.

**L1 sparsity term (group lasso on fingerprints).**

$$
\mathcal{L}_\text{L1} = \lambda_1 \sum_{v=1}^{V} \sum_{k=1}^{K}
\| \text{Fingerprints}[v, k, :] \|_2
$$

This is the group lasso norm: the L2 norm of each $(v, k)$ slice,
summed over $v$ and $k$. Produces block-sparse solutions where entire
$(v, k)$ slices go to zero, so sparsity is at the "token $v$ does /
doesn't live in landscape $k$" level.

**Total loss.**

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{state} + \lambda_\text{ctr} \mathcal{L}_\text{ctr} + \mathcal{L}_\text{L1}
$$

Defaults: $\lambda_\text{ctr} = 1$, $\lambda_1 = 10^{-3}$, $N_\text{neg} =
32$, $\tau = 0.1$.

**Gradient flow summary.** The user's description of how learning
should work — "the last token being in its locations should modify the
fingerprint; when this does not happen correctly we adjust the
landscapes and the tokens' fingerprints across the landscapes" — is
implemented as follows. For a mismatch between $\hat{s}_t$ and $s_t$:

- Gradient through $\hat{s}_t$ → `PredictStrikeHead`, `LandscapeOps`
  (via `fields`), `Transports` (via `fields`), and earlier
  `Fingerprints[token_{<t}]` (via the strikes that built `fields`).
- Gradient through $s_t$ → directly to `Fingerprints[token_t]`
  (pulling it toward the prediction).
- Contrastive term → pushes negative tokens' fingerprints *away* from
  $\hat{s}_t$.

All three flows happen in one backward pass.

---

## 3. File layout

```
v21/
├── V21_PLAN.md               (handoff, unchanged)
├── V21_DESIGN.md             (this file)
├── v21_modules.py            (the architecture itself — self-contained)
├── test_v21.py               (correctness tests)
├── benchmark_v21.py          (component-level and full-model timing)
├── gen_notebook_v21.py       (ablation-matrix notebook generator)
├── architecture_v21.ipynb    (generated)
└── README.md                 (quick-start)
```

**Cross-directory import rule**: none. Utility kernels are *copied* from
`v20/v20_modules.py` into `v21/v21_modules.py` rather than imported.
This is a lesson from V20 (see `V21_PLAN.md` §8.6).

### 3.1 Modules in `v21_modules.py`

```
# Kernels copied from v20/v20_modules.py (real arithmetic only)
make_skew_symmetric     : vectorized triu_indices skew-symmetric assembly
fast_orthogonal         : 4-term Taylor exp(A) for small skew-sym A
RMSNorm                 : standard RMS normalization (unused in core but
                          provided for ablation)
count_params            : trivial parameter counting

# New V21 modules
BundleLayer             : one (LandscapeOps, Transports, alpha,
                          PredictStrikeHead) step
V21Model                : BundleLayer stack + Fingerprints; forward()
                          returns predicted_strikes of shape (B, T, K, M)
StatePredictionLoss     : L2 + InfoNCE contrastive + L1 — given
                          predicted_strikes, target token_ids, and the
                          Fingerprints parameter, computes the total loss
                          and a per-term breakdown for logging.
EvalDecoder             : at eval time, turn predicted_strikes into
                          vocab logits via negative-L2 distance to every
                          fingerprint. Only called during evaluation.

# Config dataclass
V21Config               : K, M, V, T, L, alpha_init, l1_coef, sigma_fp,
                          N_neg, tau, lambda_ctr, lambda_l1
```

### 3.2 Tests in `test_v21.py`

Minimum test matrix (15+ tests, matching V20's rigor):

1. **Shape tests**: `V21Model.forward` produces `(B, T, K, M)` for all
   sensible `B, T`
2. **Causality test**: `predicted_strikes[:, t]` depends only on
   `token_ids[:, :t]` — strictly less than $t$. Perturb
   `token_ids[:, t]` and verify `predicted_strikes[:, t]` is unchanged;
   perturb `token_ids[:, t+1]` and verify `predicted_strikes[:, :t+1]`
   unchanged.
3. **Orthogonality tests**: `U_k @ U_k.T ≈ I` for all `k`; same for `T_k`
4. **Fingerprint gradient flow**: after one forward/backward, verify
   `Fingerprints[target_token, :, :]` receives gradient (from the L2
   term pulling it toward $\hat{s}$), and `Fingerprints[neg_token, :,
   :]` receives gradient (from the InfoNCE term pushing it away).
5. **Landscape gradient flow**: verify `LandscapeOps` receives gradient
   via the propagation path (confirm that a mismatch at step $t$ with
   $t > 0$ updates the landscape operators through the accumulated
   `fields`).
6. **L1 block-sparsity emergence**: train on a toy task (copy-sequence
   with small vocab, $V=32$, $T=16$) with $\lambda_1 = 0.01$; verify
   some $(v, k)$ norms drop below $10^{-4}$ while others grow above
   $0.1$, and the diagnostic `s_avg` drops from $K$ to a smaller
   number over training.
7. **Collapse prevention test**: train the same toy task with
   $\lambda_\text{ctr} = 0$; verify that **all fingerprints collapse
   to nearly identical values** (the failure mode). Then retrain with
   $\lambda_\text{ctr} = 1.0$ and verify collapse is prevented (the
   variance across fingerprints stays bounded away from zero).
8. **Toy-task learnability**: on a copy-sequence task ($V=32$, $T=16$),
   verify training loss decreases monotonically over 500 steps to
   below some threshold. This is the smoke test that training works at
   all.
9. **EvalDecoder correctness**: given a synthetic predicted_strike and
   a synthetic fingerprint tensor, verify that `EvalDecoder` returns
   the vocab-wide negative-L2 logits and that `argmax` of those logits
   matches the closest fingerprint.
10. **Field magnitude bound**: forward on a long sequence ($T=1024$),
    verify `||fields||` stays bounded at $\mathcal{O}(\sqrt{T \cdot
    s_\text{avg}})$ where $s_\text{avg}$ is the observed average
    fingerprint occupancy.
11. **Transport blending correctness**: with `alpha=0` (sigmoid at
    -inf), per-landscape evolution should be independent; with
    `alpha=large` (sigmoid ≈ 1), each field should equal the
    left-transported neighbor after one step.
12. **Holonomy test (ablation-ready)**: on a cycle topology, verify
    that transporting a fingerprint around the cycle via the learned
    `Transports` does **not** generally return to the original
    (non-trivial holonomy), and that the transport is approximately
    orthogonal (`||transported|| ≈ ||original||`). Relevant once cycle
    topology is added in ablation.
13. **No-NaN test** at all intermediate steps
14. **InfoNCE sanity**: when `predicted_strike == target_fingerprint`,
    the contrastive loss should equal $-\log(1 / (1 + N_\text{neg} \cdot
    \mathbb{E}[\exp(-\text{neg\_dist}^2/\tau)]))$, which for reasonable
    negatives is close to zero. Verify empirically.
15. **Dtype consistency**: everything stays `float32` (or `bfloat16` in
    autocast); no accidental `cfloat`

Plus V20-pattern tests for module composition and config validation.

### 3.3 Benchmark in `benchmark_v21.py`

Component-level timing on an H100 in `bfloat16` autocast:

- `fast_orthogonal(LandscapeOps)` and `(Transports)`: per-forward cost
  (paid once at the start of each forward)
- Per-step landscape dynamics `einsum('kij,bkj->bki', U_k, fields)`
- Per-step bundle transport `einsum('kij,bkj->bki', T_k, fields)`
- Per-step `PredictStrikeHead` application (same einsum shape)
- Per-step strike embedding lookup `Fingerprints[token_ids[:, t]]`
- State-prediction L2 loss computation (cheap — just a difference
  squared reduced to scalar)
- InfoNCE negative sampling + `Fingerprints[neg_ids]` lookup + pairwise
  distance computation
- L1 regularization term (full scan of `Fingerprints`, dominated by
  the fingerprint tensor size — can be amortized across steps)
- Full forward + backward for one batch at matched `B, T` to GPT-Nano

**Expected bottleneck**: the full forward over `T` positions is
$O(T \cdot B \cdot K \cdot M^2)$ — dominated by the landscape dynamics,
bundle transport, and strike head, all of which are the same shape.
For $B=32, T=256, K=32, M=32$: about $2.7 \times 10^8$ flops per
forward, roughly comparable to a small transformer's attention path.
**There is no vocab-wide scoring during training**, so the only
vocab-scale cost at training time is the `Fingerprints[token_ids]`
gather (an embedding lookup, cheap) and the L1 regularization scan
(which can be amortized or computed once per optimizer step, not per
forward).

**Eval-time bottleneck**: the `EvalDecoder` step computes $B \cdot T
\cdot V \cdot K \cdot M$ flops per eval batch (comparable in scale to
a vocab-scale logit projection in GPT-Nano, but applied per position
and per eval batch). This is acceptable because eval runs infrequently
(every 500–1000 training steps).

---

## 4. Experimental protocol

### 4.1 Comparison matrix

Per `V21_PLAN.md` §10, V21 is compared against:

| Model | Purpose |
|---|---|
| GPT-Nano | Upper bound — standard attention baseline |
| SSM+MLP | Lower bound — the V12.2 ablation that V12.1 failed to beat |
| V21 (L=1) | V21's one-bundle-layer variant |
| V21 (L=4) | V21's stacked-bundle variant |

All trained on **WikiText-103**, `seq_len=256`, batch size matched to
saturate H100 memory, `bf16` autocast, TF32 matmul, no `torch.compile`
(V20 lesson).

### 4.2 Ablation matrix

Holding the architecture from §2 fixed, the following axes are varied
one at a time:

1. **K** ∈ {8, 16, 32, 64} — landscape count
2. **M** ∈ {16, 32, 64} — per-landscape dimension
3. **L** ∈ {1, 2, 4} — number of stacked bundle layers
4. **Topology** ∈ {line, cycle} — adds cycle as a second ablation
   point to test whether holonomy around a loop matters
5. **Sparsity L1 coef** ∈ {0, 1e-4, 1e-3, 1e-2} — tests whether L1
   sparsity is actually load-bearing
6. **Blending alpha init** ∈ {0, 0.25, 0.5, 1.0} — tests whether
   bundle transport matters

The default V21 uses K=32, M=32, L=1, line, L1=1e-3, alpha_init=0.25.

### 4.3 Metrics

Each run reports:
- **Validation L2 state-prediction loss** (primary metric for V21
  itself; comparable across V21 runs but not across architectures)
- **Validation cross-entropy, BPC, PPL, accuracy** (derived via
  `EvalDecoder` — cross-architecture comparable)
- ms/step train (wall-clock)
- ms/step eval (separate, because eval has the $V$-scale decode step)
- Peak memory (train and eval separately)
- Average fingerprint occupancy $s_\text{avg} = \text{mean}_v
  \sum_k \mathbb{1}[\|\text{Fp}[v,k,:]\| > \varepsilon]$ (sparsity
  diagnostic)
- Fingerprint variance diagnostic: $\text{Var}_v
  (\text{Fingerprints}[v])$ — if this drops below a threshold during
  training, fingerprints are collapsing

**How eval CE is computed from predicted strikes.** The predicted
strikes have shape `(B, T, K, M)`. `EvalDecoder` reshapes to
`(B*T, K*M)`, flattens `Fingerprints` to `(V, K*M)`, and computes

$$
\text{logit}[b, t, v] = -\| \hat{s}_{b,t} - \text{Fingerprints}[v] \|^2 / \tau_\text{eval}
$$

The eval temperature $\tau_\text{eval}$ is swept on a held-out split
to maximize validation likelihood (it's not trained; it's a
post-training calibration scalar). Cross-entropy is then computed
against the true token id as usual:

$$
\mathcal{L}_\text{CE} = -\log \text{softmax}(\text{logit})[b, t, \text{target}_{b, t}]
$$

This is a purely post-hoc decoder — it does **not** appear in the
training loss. It exists only to produce metrics comparable to
GPT-Nano / SSM-MLP and to the broader LM literature.

### 4.4 Win conditions

Per `V21_PLAN.md` §10:

- **Minimum viable**: V21 beats SSM+MLP by ≥10% PPL at matched
  wall-clock at 20K steps on WikiText-103.
- **Stretch**: V21 matches or beats GPT-Nano at matched wall-clock.
- **Publishable**: either of the above, plus an ablation showing which
  component (landscape count, bundle transport, fingerprint sparsity)
  is load-bearing.

---

## 5. Known risks and open questions

### 5.1 Risk: fingerprint collapse (PRIMARY RISK)

The L2 state-prediction loss has trivial global minima at (all
fingerprints = 0) and at (all fingerprints identical). Without the
InfoNCE contrastive term, the model would discover one of these
solutions and stop learning — BYOL, SimSiam, and VICReg all had to
solve this problem in the visual self-supervised literature.

**Mitigation (baked into §2.5)**: InfoNCE contrastive term with
$N_\text{neg} = 32$ sampled negatives per position. This gives
gradient pressure to every sampled negative fingerprint to move
*away* from the current predicted strike, keeping the fingerprint
cloud spread out. The variance diagnostic from §4.3 directly monitors
collapse: if $\text{Var}_v(\text{Fingerprints}[v])$ drops below a
threshold, collapse is happening.

**Escalation path**: If InfoNCE at $N_\text{neg} = 32$ is insufficient,
try in order: (1) increase $N_\text{neg}$ to 128, (2) add VICReg-style
variance/covariance regularization, (3) project fingerprints onto a
unit sphere after each optimizer step. Do not silently add
cross-entropy — that defeats the entire "emerges from the pattern"
point of the architecture.

### 5.2 Risk: pure linear core may underfit

With no pointwise nonlinearity in the dynamics, V21's representational
capacity depends entirely on: (a) per-landscape orthogonal recurrence,
(b) bundle transport with holonomy on non-trivial topologies, (c) the
contrastive softmax at the loss level, (d) the sheer number and depth
of the landscapes. V15/V16 achieved competitive results with
mostly-linear architectures, and "holonomic linear attention" is the
stated conceptual target. But there is a real chance this is too
linear for language modeling.

**Mitigation**: If V21 (L=1) underfits substantially vs SSM+MLP, add
a gated bundle transport (from the pre-decided Sub-Q2 option 2) as
the first extension. Do **not** add pointwise nonlinearities on the
landscape dynamics — that breaks the architectural story.

### 5.3 Risk: L1 sparsity never kicks in

The L1 coefficient is a tunable. Too small → fingerprints stay dense,
the "sparse sections" framing is vacuous. Too large → fingerprints
collapse to zero (conflated with the collapse in §5.1). The sparsity
diagnostic $s_\text{avg}$ and the variance diagnostic must be monitored
jointly.

**Mitigation**: If L1 is not producing sparsity at
$\lambda_1 = 10^{-3}$, increase it by 10× stepwise. If fingerprints
collapse at higher L1, lower the L1 and raise $\lambda_\text{ctr}$ in
parallel.

### 5.4 Risk: dense fingerprint tensor too large

51.5M parameters for `Fingerprints` alone. This is ~10× GPT-Nano's
param count. For fair comparison, GPT-Nano must be run at matched
*wall-clock* not param count (per `V21_PLAN.md` §10 which already
specifies this).

**Mitigation**: If the comparison is criticized as unfair, add a
parameter-matched GPT (GPT-Small scale, ~50M params) as a second
comparison row. Note also that a large fraction of V21's fingerprints
will go to near-zero under L1, so the *effective* parameter count is
much smaller than the raw tensor size.

### 5.5 Open question: multi-block stacking semantics

The design currently proposes that `Fingerprints` is shared across
stacked blocks, and that only the final block's `PredictStrikeHead`
produces the predicted strike. But the handoff of `fields` between
blocks is underspecified — options include (a) fields from layer
$\ell-1$ become the initial field for layer $\ell$ at every position,
(b) residual connection across layers, (c) layerwise readouts summed.
The exact choice is left for the $L > 1$ ablation.

**Mitigation**: Keep $L = 1$ for the initial V21 experiment. Address
stacking only if V21 (L=1) beats the minimum viable bar.

### 5.6 Open question: teacher-forcing vs. scheduled-sampling

During training the forward pass lands the ground-truth fingerprint
on the field at every step (teacher forcing). At inference the model
has to use its own predictions. There is potentially an
exposure-bias problem — the model has never seen its own predictions
as inputs during training.

**Mitigation**: Start with pure teacher forcing. If generation quality
lags the training PPL, try scheduled sampling (with some probability
$p$, replace the ground-truth fingerprint with the decoded prediction
during training). This is a well-known technique and easy to add.

---

## 6. Implementation order

1. **Copy V20 utility kernels** into `v21_modules.py` (no cross-dir
   import)
2. **Implement `BundleLayer`** with the forward pass from §2.3
   (propagate → predict → strike)
3. **Implement `V21Model`** wrapping `BundleLayer` + `Fingerprints`;
   returns predicted strikes of shape `(B, T, K, M)`
4. **Implement `StatePredictionLoss`** with the L2 + InfoNCE + L1
   breakdown from §2.5, returning a scalar total loss and a dict of
   named sub-losses for logging
5. **Implement `EvalDecoder`** for the post-hoc CE/PPL computation at
   eval time
6. **Write `test_v21.py`** with the test matrix from §3.2; iterate
   until all pass. The two critical tests to pass before anything
   else: **gradient flow** (test 4) and **collapse prevention** (test
   7)
7. **Smoke-train on a toy task** (copy-sequence or next-char on Tiny
   Shakespeare) to verify learning happens at all. Watch the variance
   diagnostic and $s_\text{avg}$ closely — these reveal collapse and
   sparsity emergence before the loss curve tells the full story.
8. **Write `benchmark_v21.py`** and measure per-component timing
9. **Write `gen_notebook_v21.py`** for the experimental notebook
10. **Run V21 (L=1) vs SSM+MLP vs GPT-Nano on WikiText-103** at the
    default hyperparameters
11. **Ablate** the axes in §4.2 one at a time

Only step 10 tells us whether V21 is worth pursuing. If it fails the
minimum-viable bar, the project pivots; if it passes, the ablations
in step 11 establish which components are load-bearing.

**Critical early-exit**: if at step 7 (toy-task smoke training) the
variance diagnostic shows fingerprints are collapsing despite the
InfoNCE term, **stop and fix the collapse before proceeding**. A
collapsed V21 cannot possibly learn anything, and there is no point
running WikiText-103 on it. This is the most likely place V21 fails
early.

---

## 7. Glossary of terms specific to V21

- **Landscape**: one of the $K$ orthogonal recurrent systems. Each
  landscape has its own $SO(M)$ operator and its own field (a vector in
  $\mathbb{R}^M$).
- **Fingerprint**: a token's permanent learned feature in a specific
  landscape. Token $v$ in landscape $k$ is `Fingerprints[v, k, :]` $\in
  \mathbb{R}^M$. If the L1 penalty drives this slice to zero, token $v$
  does not "live in" landscape $k$. Fingerprints are used both as
  *strikes* (added to the field when a token arrives) and as *targets*
  (the state-prediction loss pulls the predicted strike toward them).
- **Strike**: the event of a token arriving at a specific position in
  the sequence. A strike instantaneously adds the token's fingerprints
  to the current fields.
- **Predicted strike**: the model's guess at what strike should arrive
  next, produced by `PredictStrikeHead` from the current propagated
  field. Compared against the actual target fingerprint via L2 to give
  the primary training signal.
- **Propagated field**: the field after landscape dynamics and bundle
  transport have been applied but **before** the next strike lands.
  The `PredictStrikeHead` reads this propagated field to produce its
  prediction — this is what makes the prediction causally valid (it
  does not depend on the current token).
- **State-prediction loss**: the L2 distance between predicted and
  target strikes, averaged over positions and batch. This is the
  primary gradient signal in V21.
- **Contrastive (InfoNCE) term**: an anti-collapse loss term that
  pushes the predicted strike closer to the target fingerprint than
  to sampled negative fingerprints. Necessary because pure L2 has
  trivial zero-everywhere solutions.
- **Bundle transport** (also "connection" or "parallel transport"):
  the $SO(M)$ operator that maps one landscape's field to a neighboring
  landscape's basis. In V21 initial, the base graph is a line, so each
  landscape has one left and one right neighbor; only the left-transport
  is active (causal sweep).
- **Holonomy**: the net transformation a field undergoes when
  parallel-transported around a closed loop in the base graph. In a
  trivial bundle this is the identity; in V21 (line topology) there are
  no loops, so holonomy is moot for the initial run. The cycle-topology
  ablation introduces real loops and exercises the holonomy path.
- **Bundle layer**: the `BundleLayer` module implementing one step of
  (propagate → predict → strike) for a single position.
- **EvalDecoder**: the post-hoc module that turns predicted strikes
  into vocab logits via negative-L2 distance to fingerprints. Only
  used at eval time — never appears in the training loss.

---

## 8. Readings for implementer context

In rough order of usefulness:

1. `V21_PLAN.md` (this directory) — the handoff doc; §1–§8 still apply.
2. `v20/v20_modules.py` — to lift `fast_orthogonal`,
   `make_skew_symmetric`, `RMSNorm`, `count_params`.
3. `project_v21_decisions.md` in memory — the "three questions answered"
   pointer.
4. `project_synthesis_260401.md` in memory — the "holonomic linear
   attention" conceptual target.
5. Oord, Li, Vinyals 2018, "Representation Learning with Contrastive
   Predictive Coding" — the original InfoNCE paper and the closest
   conceptual match for V21's "predict next state, contrastive
   against negatives" objective.
6. Grill et al. 2020, "BYOL" and Chen & He 2021, "SimSiam" — case
   studies in L2-style self-supervised prediction and how they avoid
   collapse (important to read *before* debugging V21 collapse).
7. Bardes, Ponce, LeCun 2022, "VICReg" — variance/invariance/covariance
   regularization; backup mitigation if InfoNCE isn't enough.
8. Bodnar et al. 2022, "Neural Sheaf Diffusion" — for the
   sheaf-cohomology interpretation of the bundle transport.
9. Frady, Kent, Sommer 2020, "Resonator networks" — for the
   fixed-fingerprint framing, though V21's loss is different from
   resonator-network iterative decoding.
10. Ramsauer et al. 2021, "Hopfield Networks is All You Need" —
    historical / V14-era reference; V21 does **not** use Hopfield
    retrieval at readout, but the paper's "modern Hopfield" framework
    is worth knowing because an earlier version of this design did.
