# Unit 13: CurvBias — Geometric Position Encoding That Enhances Attention

**Prerequisites:** Units 06, 10

This is the thesis's PRIMARY CONTRIBUTION -- the practical outcome.

---

## Learning Objectives

1. Understand the hierarchy of geometric position encodings (flat $\to$ curved $\to$ content-dependent)
2. Know how CurvBias works as a lightweight attention bias
3. Interpret CurvBias through gauge theory (curvature of a gauge connection)
4. Understand the experimental results: CurvBias vs RoPE across 3 scales
5. Explain why geometry enhances attention rather than replacing it

---

## Readings

- Thesis Sec. 6.7 (The Geometric Position Encoding Ablation: Back to Attention) -- THIS IS THE KEY SECTION
- Thesis Sec. 7.5 (Interpreting the CurvBias Results)
- Thesis Tables 6.6, 6.7, 6.8 (results at all 3 scales)
- Su et al. 2021: RoPE (Rotary Position Embeddings)
- Repo: `topology/gauge_connections_fiber_bundles.md` (gauge theory background)
- Paper: `gauge_fiber_bundle_geometry_transformers_iclr2025.pdf`

---

## Key Concepts

### 1. Position Encoding Hierarchy (Thesis Sec. 6.7.2)

- **Level 0: Absolute position** (sinusoidal/learned) -- no geometric structure
- **Level 1: Relative position** (ALiBi) -- translation-invariant bias
- **Level 2: RoPE** -- flat $U(1)$ gauge connection (constant rotation per position)
- **Level 3: HoloRoPE** -- non-abelian $SO(K)$ rotation (richer but still content-independent)
- **Level 4: CurvBias** -- content-dependent gauge curvature (full geometric structure)

### 2. RoPE as a Flat Connection

$R_\theta^m$ rotates by angle $m \cdot \theta$. The "connection" $A = \theta$ is constant. Curvature $F = dA = 0$ (flat). Transport is the same everywhere.

### 3. CurvBias Adds Curvature

The bias depends on accumulated geometric distance. Not just position difference, but content-dependent curvature integrated along the path.

### 4. Implementation

A lightweight bias $B_{ij}$ added to attention logits:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} + B_{ij}$$

### 5. Computing $B_{ij}$

$B_{ij}$ is computed from cumulative curvature: integral of learned curvature field along positions $i$ to $j$.

### 6. Negligible Overhead

$B$ is computed once per layer, added as a scalar to each attention pair.

### 7. Results

Up to 9% PPL improvement over RoPE at small scale, consistent 3-4% at large scale.

---

## Worked Problems

### Problem 1

**Problem:** RoPE applies rotation $R_\theta^m$ to query at position $m$, where $R_\theta^m = \text{block-diagonal matrix of 2D rotations by angle } m \cdot \theta_k$ for frequency $k$. For $d=4$ (two frequency bands) with $\theta_1=1$, $\theta_2=0.1$, compute the rotation applied to position $m=5$.

**Solution:**

$$R_\theta^5 = \operatorname{blockdiag}(R(5 \cdot 1),\; R(5 \cdot 0.1)) = \operatorname{blockdiag}(R(5),\; R(0.5))$$

where $R(\alpha) = \begin{bmatrix} \cos \alpha & -\sin \alpha \\ \sin \alpha & \cos \alpha \end{bmatrix}$.

For the first block:

$$R(5) = \begin{bmatrix} \cos 5 & -\sin 5 \\ \sin 5 & \cos 5 \end{bmatrix} = \begin{bmatrix} 0.284 & 0.959 \\ -0.959 & 0.284 \end{bmatrix}$$

For the second block:

$$R(0.5) = \begin{bmatrix} \cos 0.5 & -\sin 0.5 \\ \sin 0.5 & \cos 0.5 \end{bmatrix} = \begin{bmatrix} 0.878 & -0.479 \\ 0.479 & 0.878 \end{bmatrix}$$

The rotation $R_\theta^5$ rotates the first 2 dims by 5 rad and last 2 dims by 0.5 rad.

This is position-dependent but content-INdependent.

---

### Problem 2

**Problem:** Show that RoPE makes $q_m \cdot k_n$ depend only on $m - n$ (relative position). Then explain why this is a "flat gauge connection."

**Solution:**

$$q_m = R_\theta^m \, q$$
$$k_n = R_\theta^n \, k$$

$$q_m \cdot k_n = (R_\theta^m \, q)^\top (R_\theta^n \, k) = q^\top R_\theta^{-m} R_\theta^n \, k = q^\top R_\theta^{n-m} \, k$$

Only depends on $n - m$. Check.

**Gauge theory interpretation:** The connection is $A = \theta \, d\phi$ (constant 1-form). Parallel transport from position $m$ to $n$ is $R_\theta^{n-m}$. This transport ONLY depends on the displacement, not on position or content.

Curvature: $F = dA = d(\theta \, d\phi) = 0$ ($\theta$ is constant).

Zero curvature = flat connection. A flat connection means "position information is the same everywhere" -- no position-dependent structure.

---

### Problem 3

**Problem:** CurvBias adds a bias $B_{ij}$ to the attention logits. If $B_{ij} = -\int_i^j \kappa(t)\, dt$ where $\kappa(t)$ is a learned curvature field, compute $B$ for positions $i=2$, $j=5$ if $\kappa(t) = 0.1t$ (linearly increasing curvature).

**Solution:**

$$B_{2,5} = -\int_2^5 0.1t \, dt = -0.1 \left[\frac{t^2}{2}\right]_2^5 = -0.1(12.5 - 2) = -0.1(10.5) = -1.05$$

$$B_{5,2} = -\int_5^2 0.1t \, dt = -0.1 \left[\frac{t^2}{2}\right]_5^2 = -0.1(2 - 12.5) = +1.05$$

Note $B_{ij} \neq B_{ji}$ -- the bias is ASYMMETRIC!

Looking backward costs more than looking forward (causality from geometry, not from a mask). The magnitude increases with distance (farther tokens get more negative bias), naturally implementing attention decay.

---

### Problem 4

**Problem:** The thesis reports these perplexity results at small scale (14M params, $d=256$, $T=512$):

| Method   | PPL  |
|----------|------|
| RoPE     | 53.4 |
| ALiBi    | 52.1 |
| HoloRoPE | 51.8 |
| CurvBias | 48.6 |

Compute the percentage improvement of CurvBias over each baseline.

**Solution:**

- vs RoPE: $(53.4 - 48.6) / 53.4 = 4.8 / 53.4 =$ **9.0% improvement**
- vs ALiBi: $(52.1 - 48.6) / 52.1 = 3.5 / 52.1 =$ **6.7% improvement**
- vs HoloRoPE: $(51.8 - 48.6) / 51.8 = 3.2 / 51.8 =$ **6.2% improvement**

CurvBias improves over ALL baselines. The improvement is largest vs RoPE (the most widely used) -- 9.0% at small scale.

The hierarchy is clear: flat $<$ constant-bias $<$ non-abelian $<$ curved (CurvBias wins at every level).

---

### Problem 5

**Problem:** At large scale (77M params, $d=512$, $T=1024$): CurvBias PPL 37.3 vs RoPE 38.9 ($-4.1\%$). At long sequence (77M, $T=3072$, 40K steps): CurvBias PPL 24.5 vs RoPE 25.4 ($-3.5\%$). Why does the improvement shrink with scale?

**Solution:**

At larger scale, the model has more parameters to compensate for geometric limitations. A sufficiently large attention model can LEARN position-dependent patterns implicitly -- it does not need them handed via position encoding. But it still cannot learn them as efficiently.

- **At small scale:** the model MUST rely heavily on position encoding (limited capacity), so better encoding = much better performance (9%).
- **At large scale:** the model can partially compensate with its extra capacity, but geometry still helps (4%).

The improvement does not vanish -- it persists across all scales. This suggests CurvBias provides genuine structural value, not just a better initialization.

---

### Problem 6

**Problem:** Interpret CurvBias through gauge theory. RoPE is a flat $U(1)$ connection. What does "adding curvature" mean physically?

**Solution:**

**RoPE:** connection $A = \theta$ (constant). Transport from $m$ to $n$: always the same rotation regardless of content. It is like a conveyor belt at constant speed.

**CurvBias:** connection $A(t)$ depends on content (learned from token representations). Transport from $m$ to $n$ integrates the varying $A(t)$ along the path. It is like walking on terrain with hills and valleys -- the effort depends on the path's geometry.

Curvature $F = dA \neq 0$ means the connection CHANGES with position/content. Transporting around a loop gives non-trivial holonomy (the representation rotates based on enclosed content).

This allows attention to capture content-dependent positional relationships: "the distance between subject and verb" depends on what is between them, not just how many tokens apart they are.

---

### Problem 7

**Problem:** CurvBias is implemented as a simple bias term added to attention logits. It does NOT modify the Q/K/V projections, does NOT add layers, and has negligible parameter count. Why is this important for practical adoption?

**Solution:**

1. **Drop-in replacement:** can be added to any existing transformer without architectural changes. Just replace the position encoding computation.
2. **Negligible overhead:** the bias $B_{ij}$ is a single scalar per attention pair, computed once per layer. Vs. attention's $O(T^2 d)$ computation, the bias adds $O(T^2)$ -- a factor of $d$ smaller.
3. **No training modifications:** same optimizer, same hyperparameters, same everything.
4. **Compatible with existing optimizations:** FlashAttention, KV caching, etc. all work with attention biases.
5. **Immediately applicable to production:** can enhance GPT, LLaMA, etc. without retraining from scratch (just fine-tune with CurvBias).

This is why CurvBias is the thesis's primary practical contribution.

---

### Problem 8

**Problem:** The thesis also tests CurvBias with GLA (Gated Linear Attention), not just softmax attention. CurvBias improves GLA too. Why is this significant?

**Solution:**

GLA is a linear attention variant ($O(Td^2)$ instead of $O(T^2 d)$). If CurvBias only worked with softmax attention, it would be limited to architectures that are already $O(T^2)$ -- the most expensive ones. But CurvBias improving GLA means:

1. The geometric insight is **architecture-agnostic** -- it helps ANY attention mechanism, not just softmax.
2. CurvBias can make efficient linear attention models better, combining the efficiency of linear attention with the geometric awareness of curved position encoding.
3. The improvement is not an artifact of softmax's specific properties but reflects genuine geometric structure in language.

---

### Problem 9

**Problem:** Compare CurvBias to the concurrent CARoPE method (Veisi & Fartoot, 2025). Both add content-dependence to rotary embeddings. What distinguishes them?

**Solution:**

**CARoPE:** makes the rotation FREQUENCY content-dependent ($\theta$ varies with token content). Still operates within the RoPE framework (frequency $\times$ position).

**CurvBias:** makes the CURVATURE of the gauge connection content-dependent. Operates as an additive bias to attention logits (not within the RoPE rotation).

**Key difference:** CARoPE modifies how fast the rotation goes (content-dependent $\theta$). CurvBias adds a path-dependent geometric distance (curvature integral).

CurvBias is more general: it captures the full gauge-theoretic structure (holonomy, non-abelian rotation, curvature). The thesis results show CurvBias outperforms CARoPE at all tested scales, though both beat standard RoPE.

---

### Problem 10

**Problem:** Explain the complete "geometric position encoding hierarchy" from the thesis (Sec. 6.7.2). For each level, state the gauge group, whether curvature is zero or nonzero, and whether it is content-dependent.

**Solution:**

**Level 0 -- Absolute Position (sinusoidal/learned):**
No gauge group. No transport structure. Position is just an index. Curvature: N/A.

**Level 1 -- Relative Position (ALiBi):**
"Group" $= (\mathbb{R}, +)$. Bias $= -m|i - j|$ (linear penalty for distance). Curvature: zero (constant slope). Content-independent.

**Level 2 -- RoPE:**
Gauge group $= U(1)^{d/2}$ (product of phase rotations). Connection $A = \theta$ (constant per frequency). Curvature $F = 0$ (flat). Content-independent.

**Level 3 -- HoloRoPE:**
Gauge group $= SO(K)$ (non-abelian rotation). Connection $A$ encodes richer rotation. Curvature can be nonzero but is still content-independent.

**Level 4 -- CurvBias:**
Gauge group = content-dependent. Connection $A(t)$ varies with token content. Curvature $F = dA \neq 0$ in general. Fully content-dependent.

Each level adds geometric structure. The experimental results show a monotonic improvement: more geometry = better performance.

---

## Comprehension Questions

1. What is the geometric position encoding hierarchy? Name all 5 levels.
2. Why is RoPE a "flat gauge connection"? What does flat mean here?
3. How is CurvBias implemented? Why is the overhead negligible?
4. Why does CurvBias outperform RoPE? What geometric information does it capture that RoPE misses?
5. What makes CurvBias the thesis's primary practical contribution?

---

## Bridge to Thesis

CurvBias is where the entire geometric journey pays off in practice. Units 01-12 built the mathematical machinery: manifolds, fiber bundles, gauge connections, curvature, spectral analysis, Finsler geometry, Hopfield energy. The Finsler Transformer (Unit 10) and spectral SGST (Unit 12) showed that full geometric architectures cannot yet beat attention. But CurvBias shows that geometric INSIGHT -- specifically, the gauge-theoretic interpretation of position encoding -- can enhance the architectures we already use. The thesis's narrative arc: "we tried to replace attention with geometry, discovered we could not, but in the process learned enough geometry to make attention better." CurvBias is the proof that the geometric detour was worth taking.
