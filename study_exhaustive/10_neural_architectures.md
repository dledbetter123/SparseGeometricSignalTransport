# Unit 10: Transformers, GNNs, and State-Space Models

## Learning Objectives

1. Understand the Transformer architecture (attention, FFN, residual connections, layer norm)
2. Know how GNNs work (message passing, aggregation, update)
3. Understand State-Space Models (SSM): S4, Mamba, selective scan
4. Know the parallel scan algorithm for $O(T \log T)$ SSM computation
5. Understand linear attention and its relationship to SSMs
6. Compare computational costs: $O(T^2 d)$ attention vs $O(Td)$ SSM vs $O(s \, T \log d)$ spectral

## Prerequisites

Units 01 (Linear Algebra), 02 (Multivariable Calculus), 04 (Graph Theory and Spectral Methods), 09 (Hopfield Networks)

## Readings

- Vaswani et al. 2017, "Attention Is All You Need" (original transformer paper)
- Gu & Dao 2024, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Thesis Ch. 2.1 (GNNs), Ch. 3 (Failure of Spatial Approaches)
- Thesis Ch. 7.4 (Relationship to Existing Architectures)
- `topology/lit_review/linear_attention_ssm_sota.md` (comprehensive SSM survey)
- `topology/lit_review/unitary_orthogonal_rnns.md` (historical context)

---

## Key Concepts

### 1. Self-Attention

The core operation of the transformer:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- $Q$ (queries), $K$ (keys), $V$ (values) are linear projections of the input: $Q = X W_Q$, $K = X W_K$, $V = X W_V$
- $QK^T$ computes pairwise similarity between all token pairs ($T \times T$ matrix)
- Division by $\sqrt{d_k}$ prevents dot products from growing with dimension (keeps softmax gradients healthy)
- softmax normalizes each row to a probability distribution over keys
- Multiplication by $V$ retrieves a weighted combination of value vectors

Cost: $O(T^2 d)$ where $T$ = sequence length, $d$ = embedding dimension.

### 2. Multi-Head Attention

Instead of one large attention computation, use $h$ parallel attention heads with smaller projections:

$$\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O$$

$$\text{where } \text{head}_i = \operatorname{Attention}(Q W_Q^i, \; K W_K^i, \; V W_V^i)$$

Each head operates in a $d/h$ dimensional subspace. Different heads can attend to different types of relationships (syntactic, semantic, positional). The output projection $W_O$ mixes head outputs.

### 3. Position Encoding

Transformers are permutation-equivariant without position information. Position must be injected explicitly:

- **Sinusoidal (Vaswani 2017):** $PE(pos, 2i) = \sin\!\bigl(pos / 10000^{2i/d}\bigr)$, $PE(pos, 2i+1) = \cos\!\bigl(pos / 10000^{2i/d}\bigr)$. Fixed, no learned parameters. Relative positions encoded in dot products.
- **Learned embeddings:** A lookup table of $d$-dimensional vectors, one per position. Simple but doesn't generalize to unseen lengths.
- **RoPE (Rotary Position Embeddings):** Applies position-dependent rotation to $Q$ and $K$: $q_m = R_\theta^m q$. The dot product $q_m \cdot k_n$ depends only on relative position $m - n$. The thesis identifies RoPE as a flat $U(1)$ gauge connection (Sec. 6.7.2).
- **ALiBi (Attention with Linear Biases):** Adds $-m|i - j|$ to attention logits, where $m$ is a head-specific slope. Simple linear decay with distance.

### 4. Feed-Forward Network (FFN)

Applied independently to each token after attention:

$$\operatorname{FFN}(x) = W_2 \cdot \operatorname{activation}(W_1 x + b_1) + b_2$$

Typically $W_1$ projects from $d$ to $4d$ (expansion), and $W_2$ projects back from $4d$ to $d$ (compression). The activation is usually GELU or SiLU. The FFN performs per-token nonlinear transformation -- it is the "reaction" in the reaction-diffusion decomposition.

### 5. Residual Connections

Every sublayer (attention, FFN) is wrapped in a residual connection:

$$x_{\text{out}} = x + \operatorname{Sublayer}(x)$$

Benefits:
- Gradient flow: gradients pass directly through the identity path, preventing vanishing gradients in deep networks
- Incremental refinement: each layer adds a small correction rather than computing a completely new representation
- Implicit ensemble: the network is effectively an ensemble of paths of different depths

### 6. Layer Normalization

Normalizes activations across the embedding dimension:

$$\operatorname{LayerNorm}(x) = \gamma \cdot \frac{x - \operatorname{mean}(x)}{\sqrt{\operatorname{var}(x) + \varepsilon}} + \beta$$

Applied before each sublayer (Pre-LN, now standard) or after (Post-LN, original). Prevents activation drift across layers and stabilizes training.

### 7. GNN Message Passing

Graph Neural Networks operate on graph-structured data through iterative message passing:

$$h_v^{(l+1)} = \operatorname{UPDATE}\!\left( h_v^{(l)}, \; \operatorname{AGGREGATE}\!\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right) \right)$$

- $h_v^{(l)}$: embedding of node $v$ at layer $l$
- $\mathcal{N}(v)$: neighbors of $v$ in the graph
- AGGREGATE: permutation-invariant function (sum, mean, max) over neighbor messages
- UPDATE: combines the node's own embedding with the aggregated message (typically an MLP)

Examples:
- **GCN:** $h_v = \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}} W h_u\right)$. Normalized sum with shared weight matrix.
- **GAT:** Uses attention weights between neighbors: $\alpha_{vu} = \operatorname{softmax}\!\bigl(\operatorname{LeakyReLU}(a^T [Wh_v \| Wh_u])\bigr)$.
- **GIN:** $h_v = \operatorname{MLP}\!\bigl((1+\varepsilon) \, h_v + \sum_{u \in \mathcal{N}(v)} h_u\bigr)$. Maximally powerful among 1-WL equivalent GNNs.

The thesis (Ch. 3) demonstrates that spatial GNN approaches fail for SGST's goals due to over-smoothing and the 1-WL expressivity bottleneck.

### 8. State-Space Models (SSMs)

Continuous-time linear dynamical systems:

$$\frac{dx}{dt} = A x(t) + B u(t)$$
$$y(t) = C x(t) + D u(t)$$

where $x$ is the hidden state, $u$ is the input, $y$ is the output, and $A, B, C, D$ are system matrices.

Discretized (zero-order hold) for sequence modeling:

$$x_k = \bar{A} \, x_{k-1} + \bar{B} \, u_k$$
$$y_k = C x_k + D u_k$$

where $\bar{A} = \exp(\Delta A)$, $\bar{B} = A^{-1}(\bar{A} - I) B$, and $\Delta$ is the discretization step.

**S4 (Structured State Spaces for Sequences):** Uses a specific parameterization of $A$ (HiPPO matrix) that enables efficient long-range memory. The HiPPO matrix projects the input history onto a basis of Legendre polynomials, providing optimal approximation of past inputs.

### 9. Selective Scan (Mamba)

Mamba makes the SSM parameters input-dependent:

$$A_t = f_A(u_t), \quad B_t = f_B(u_t), \quad C_t = f_C(u_t), \quad \Delta_t = f_\Delta(u_t)$$

This is crucial: fixed $A$, $B$ mean the same transition regardless of input content. Input-dependent parameters allow the model to:
- Selectively store relevant information (large $B_t$ for important tokens)
- Selectively forget stale information (fast decay via $A_t$ for noise tokens)
- Adapt the state dynamics to the current context

The cost of selectivity: input-dependent parameters break the convolution form of classical SSMs, requiring the parallel scan algorithm instead.

### 10. Parallel Scan Algorithm

The recurrence $x_k = a_k x_{k-1} + b_k$ can be computed for ALL $k$ simultaneously using the associative operation:

$$(a, b) \oplus (c, d) = (a \cdot c, \; a \cdot d + b)$$

This encodes "apply transition $(a_k, b_k)$ after transition $(a_{k-1}, b_{k-1})$":

$$x_2 = a_2 x_1 + b_2 = a_2(a_1 x_0 + b_1) + b_2 = (a_2 a_1) x_0 + (a_2 b_1 + b_2)$$

Since $\oplus$ is associative, we can use parallel prefix scan:
- Round 1: combine adjacent pairs $\to$ $T/2$ operations
- Round 2: combine results $\to$ $T/4$ operations
- ...
- Round $\log(T)$: final result

Total: $O(T)$ work, $O(\log T)$ parallel time. This enables efficient GPU computation of SSM recurrences.

### 11. Linear Attention

Replace softmax attention with a kernel decomposition:

$$\text{Standard:} \quad \operatorname{Attention} = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V \qquad [O(T^2 d)]$$

$$\text{Linear:} \quad \operatorname{Attention} \approx \phi(Q) \bigl(\phi(K)^T V\bigr) \qquad [O(T d^2)]$$

where $\phi$ is a feature map. The key trick is changing the order of matrix multiplication:
- Standard computes the $T \times T$ attention matrix first (expensive when $T$ is large)
- Linear computes the $d \times d$ outer product $\phi(K)^T V$ first (expensive when $d$ is large, but $d \ll T$ for long sequences)

Common feature maps: $\phi(x) = \operatorname{elu}(x) + 1$ (Katharopoulos et al.), random Fourier features (Performer), polynomial features.

### 12. The SSM-Attention Duality

There is a deep connection between SSMs and linear attention:

- An SSM with state $x_t \in \mathbb{R}^d$, scalar input $u_t$, and output $y_t = C x_t$ can be written as: $y_t = \sum_{j=1}^{t} C A^{t-j} B u_j$. This is a weighted sum of past inputs with exponentially decaying weights.

- Linear attention computes: $y_t = \phi(q_t)^T \bigl(\sum_{j=1}^{t} \phi(k_j) v_j^T\bigr) = \sum_{j=1}^{t} \phi(q_t)^T \phi(k_j) \, v_j$. This is also a weighted sum of past values.

When $A$, $B$, $C$ are input-dependent (selective SSM), the connection tightens: Mamba's selective scan is essentially linear attention with a specific parameterization of the kernel function. Both maintain an $O(d^2)$ or $O(d \cdot d_{\text{state}})$ running summary of the past, updated incrementally.

---

## Worked Problems

### Problem 1: Numerical Attention Computation

**Problem:** For $Q = \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}$, $K = \begin{bmatrix}1 & 1\\0 & 1\\1 & 0\end{bmatrix}$, $V = \begin{bmatrix}1 & 2\\3 & 4\\5 & 6\end{bmatrix}$, compute $\operatorname{Attention}(Q, K, V)$ with $d_k = 2$.

**Solution:**

Step 1: Compute $QK^T / \sqrt{d_k}$.

$$QK^T = \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix} \begin{bmatrix}1 & 0 & 1\\1 & 1 & 0\end{bmatrix} = \begin{bmatrix}1 & 0 & 1\\1 & 1 & 0\end{bmatrix}$$

$$\frac{QK^T}{\sqrt{2}} = \begin{bmatrix}0.707 & 0 & 0.707\\0.707 & 0.707 & 0\end{bmatrix}$$

Step 2: Apply softmax to each row.

Row 1: $\exp([0.707, 0, 0.707]) = [2.028, 1.000, 2.028]$
Sum $= 5.056$
Weights $= [2.028/5.056, \; 1.000/5.056, \; 2.028/5.056] = [0.401, \; 0.198, \; 0.401]$

Row 2: $\exp([0.707, 0.707, 0]) = [2.028, 2.028, 1.000]$
Sum $= 5.056$
Weights $= [2.028/5.056, \; 2.028/5.056, \; 1.000/5.056] = [0.401, \; 0.401, \; 0.198]$

Step 3: Multiply by $V$.

Output row 1: $0.401 \cdot [1,2] + 0.198 \cdot [3,4] + 0.401 \cdot [5,6] = [0.401, 0.802] + [0.594, 0.792] + [2.005, 2.406] = [3.000, 4.000]$

Output row 2: $0.401 \cdot [1,2] + 0.401 \cdot [3,4] + 0.198 \cdot [5,6] = [0.401, 0.802] + [1.203, 1.604] + [0.990, 1.188] = [2.594, 3.594]$

Final output:

$$\operatorname{Attention}(Q,K,V) = \begin{bmatrix}3.000 & 4.000\\2.594 & 3.594\end{bmatrix}$$

Token 1 (query $[1,0]$) attends equally to keys 1 and 3 (both have dot product 0.707), retrieving their average. Token 2 (query $[0,1]$) attends equally to keys 1 and 2 (both 0.707), with less weight on key 3 (dot product 0).

---

### Problem 2: RoPE and Relative Position

**Problem:** RoPE (Rotary Position Embeddings) applies position-dependent rotation: $q_m = R_\theta^m q$, $k_n = R_\theta^n k$, where $R_\theta^m$ is a block-diagonal rotation matrix. Show that the dot product $q_m \cdot k_n$ depends only on the relative position $m - n$.

**Solution:**

RoPE applies a rotation $R_\theta^m$ to both queries and keys at position $m$. For a 2D subspace, $R_\theta^m$ is:

$$R_\theta^m = \begin{bmatrix}\cos(m\theta) & -\sin(m\theta)\\ \sin(m\theta) & \cos(m\theta)\end{bmatrix}$$

The dot product between query at position $m$ and key at position $n$:

$$q_m \cdot k_n = (R_\theta^m q)^T (R_\theta^n k) = q^T (R_\theta^m)^T R_\theta^n k$$

Since $R_\theta^m$ is an orthogonal rotation matrix, $(R_\theta^m)^T = (R_\theta^m)^{-1} = R_\theta^{-m}$. Therefore:

$$(R_\theta^m)^T R_\theta^n = R_\theta^{-m} R_\theta^n = R_\theta^{n-m}$$

This uses the group property of rotations: $R^a R^b = R^{a+b}$.

So:

$$q_m \cdot k_n = q^T R_\theta^{n-m} k$$

The result depends on:
- The content vectors $q$ and $k$ (semantic similarity)
- The relative position difference $n - m$ (positional bias)
- NOT the absolute positions $m$ or $n$ individually

This gives translation-invariant positional attention: the attention pattern between two tokens depends on their distance, not their absolute location. This is crucial for length generalization.

**Connection to SGST (thesis Sec. 6.7.2):** RoPE is a FLAT $U(1)$ gauge connection. The rotation $R_\theta^m$ is parallel transport by distance $m$ along a connection with constant curvature zero. The SGST's CurvBias generalizes this by allowing nonzero curvature (content-dependent rotation rates), enabling position-dependent attention patterns that adapt to the input.

---

### Problem 3: SSM Unrolling and Context Accumulation

**Problem:** The SSM $x_k = A x_{k-1} + B u_k$ can be unrolled: $x_k = A^k x_0 + \sum_{j=0}^{k-1} A^{k-1-j} B u_j$. For scalar $A = 0.9$, $B = 0.1$, compute $x_1$ through $x_5$ starting from $x_0 = 0$ with inputs $u = [1, 1, 1, 1, 1]$.

**Solution:**

Apply the recurrence step by step:

$$x_1 = 0.9 \cdot 0 + 0.1 \cdot 1 = 0.100$$
$$x_2 = 0.9 \cdot 0.1 + 0.1 \cdot 1 = 0.090 + 0.100 = 0.190$$
$$x_3 = 0.9 \cdot 0.19 + 0.1 \cdot 1 = 0.171 + 0.100 = 0.271$$
$$x_4 = 0.9 \cdot 0.271 + 0.1 \cdot 1 = 0.244 + 0.100 = 0.344$$
$$x_5 = 0.9 \cdot 0.344 + 0.1 \cdot 1 = 0.310 + 0.100 = 0.410$$

Verify with the unrolled formula for $x_5$:

$$x_5 = \sum_{j=0}^{4} 0.9^{4-j} \cdot 0.1 \cdot 1 = 0.1 \cdot (0.9^4 + 0.9^3 + 0.9^2 + 0.9^1 + 0.9^0) = 0.1 \cdot (0.6561 + 0.729 + 0.81 + 0.9 + 1.0) = 0.1 \cdot 4.0951 = 0.410 \; \checkmark$$

**Interpretation:** The state accumulates information from all past inputs with exponential decay. Input $u_1$ contributes $0.1 \cdot 0.9^4 = 0.066$ to $x_5$, while input $u_5$ contributes the full $0.1 \cdot 0.9^0 = 0.1$. Recent inputs matter more.

**Steady state ($t \to \infty$ with constant input $u = 1$):**

$$x_\infty = \frac{B}{1 - A} = \frac{0.1}{1 - 0.9} = \frac{0.1}{0.1} = 1.0$$

This is the geometric series sum. The state converges to 1.0, representing a running average with exponential forgetting. This is the "context accumulation" mechanism in SSMs and in the SGST -- the hidden state summarizes the entire history with a recency bias controlled by $A$.

---

### Problem 4: Parallel Scan Associativity

**Problem:** The parallel scan computes $x_k = a_k x_{k-1} + b_k$ for all $k$ simultaneously using $O(T \log T)$ parallel time. The key insight is associativity: $(a, b) \oplus (c, d) = (ac, \; ad + b)$. Verify: if $x_1 = a_1 x_0 + b_1$ and $x_2 = a_2 x_1 + b_2$, then $x_2 = (a_2 a_1) x_0 + (a_2 b_1 + b_2)$. Also verify associativity.

**Solution:**

**Verification of the combination formula:**

$$x_1 = a_1 x_0 + b_1$$

$$x_2 = a_2 x_1 + b_2 = a_2 (a_1 x_0 + b_1) + b_2 = (a_2 a_1) x_0 + (a_2 b_1 + b_2)$$

The combined transition from $x_0$ to $x_2$ has parameters:

$$(a_{\text{combined}}, b_{\text{combined}}) = (a_2 a_1, \; a_2 b_1 + b_2) = (a_2, b_2) \oplus (a_1, b_1) \; \checkmark$$

**Verification of associativity:**

We need $\bigl((a,b) \oplus (c,d)\bigr) \oplus (e,f) = (a,b) \oplus \bigl((c,d) \oplus (e,f)\bigr)$.

Left side:
$(a,b) \oplus (c,d) = (ac, \; ad + b)$
$(ac, \; ad+b) \oplus (e,f) = (ace, \; acf + ad + b)$

Right side:
$(c,d) \oplus (e,f) = (ce, \; cf + d)$
$(a,b) \oplus (ce, \; cf+d) = (a \cdot ce, \; a(cf+d) + b) = (ace, \; acf + ad + b)$

Both sides equal $(ace, \; acf + ad + b)$. $\checkmark$

**Why associativity enables parallelism:**

Since $\oplus$ is associative, we can regroup the computation in any order. For a sequence of $T$ transitions:

Sequential: $((t_1 \oplus t_2) \oplus t_3) \oplus \cdots \oplus t_T$ [$O(T)$ serial steps]

Parallel prefix:
- Round 1: combine adjacent pairs $(t_1 \oplus t_2), (t_3 \oplus t_4), \ldots$ [$T/2$ parallel ops]
- Round 2: combine results from round 1 pairwise [$T/4$ parallel ops]
- ...
- Round $\log_2(T)$: final result [1 op]

Total: $O(T)$ work distributed over $O(\log T)$ parallel rounds. With $T$ processors, this computes ALL prefix results $x_1, x_2, \ldots, x_T$ in $O(\log T)$ time. The SGST uses this for efficient context accumulation on GPU.

---

### Problem 5: State Size Comparison

**Problem:** Compare the state size of a Transformer (KV cache) vs an SSM (hidden state) for a sequence of length $T = 1024$, $d = 256$, 12 layers.

**Solution:**

**Transformer KV cache:**

Each layer stores keys and values for all tokens processed so far:
- Keys: $T \times d$ per layer
- Values: $T \times d$ per layer
- Total per layer: $2Td$ values
- Total across 12 layers: $12 \times 2 \times T \times d$

$$= 12 \times 2 \times 1024 \times 256 = 6{,}291{,}456 \text{ values}$$

This grows LINEARLY with sequence length $T$.

At $T = 1{,}000{,}000$ (million-token context):

$$12 \times 2 \times 1{,}000{,}000 \times 256 = 6{,}144{,}000{,}000 \text{ values (approximately 6.1 billion)}$$

At float16 (2 bytes per value): 12.3 GB of KV cache alone.

**SSM hidden state:**

Each layer maintains a fixed-size hidden state regardless of sequence length:
- State dimension: $d_{\text{state}}$ (typically equal to $d$ or a fraction of $d$, here 256)
- Total across 12 layers: $12 \times d_{\text{state}}$

$$= 12 \times 256 = 3{,}072 \text{ values}$$

This is CONSTANT regardless of $T$.

**Comparison:**

| Metric | Transformer | SSM | Ratio |
|--------|-------------|-----|-------|
| $T=1024$ | 6,291,456 | 3,072 | 2,048x |
| $T=100\text{K}$ | 614,400,000 | 3,072 | 200,000x |
| $T=1\text{M}$ | 6,144,000,000 | 3,072 | 2,000,000x |
| Scaling | $O(T \cdot d \cdot L)$ | $O(d \cdot L)$ | $O(T)$ |

The SSM achieves $O(1)$ state size because each recurrence step $x_k = A x_{k-1} + B u_k$ compresses the entire history into a fixed-size vector. The cost: information from early tokens decays exponentially, and the fixed-size state cannot store arbitrary pairwise relationships between tokens. This is the fundamental memory-computation tradeoff that the SGST navigates: SSM-like $O(1)$ state vs. attention-like $O(T)$ memory.

---

### Problem 6: Why Selective Scan Matters

**Problem:** Mamba's selective scan makes $A_t$ and $B_t$ input-dependent. Explain why this is crucial (compared to fixed $A$, $B$ in classical SSMs like S4).

**Solution:**

**Fixed $A$, $B$ (classical SSM, S4):**

The state transition $x_k = \bar{A} \, x_{k-1} + \bar{B} \, u_k$ applies the same dynamics regardless of input content. This means:
- The decay rate (eigenvalues of $A$) is predetermined
- What gets stored ($B$ maps input to state) follows a fixed projection
- How information is forgotten follows a fixed exponential decay

This is a fixed-bandwidth linear filter. It processes all inputs identically: important tokens and noise tokens receive the same treatment. The model can learn good average-case dynamics, but cannot adapt to specific inputs.

**Input-dependent $A_t$, $B_t$ (Mamba):**

With selectivity, the model can make per-token decisions:

1. **Selective storage ($B_t$):** When a token is important (e.g., a key entity, a surprising word), the model sets $B_t$ large, strongly writing this token into the state. When a token is predictable noise (e.g., filler words like "the", "of"), $B_t$ is small, effectively ignoring it.

2. **Selective forgetting ($A_t$):** The decay rate adapts to content. When the model encounters a topic change or paragraph boundary, $A_t$ can be set to decay quickly (clearing stale context). When accumulating related information, $A_t$ decays slowly (preserving relevant context).

3. **Selective output ($C_t$):** What information is read from the state depends on the current input, enabling content-dependent retrieval.

**Concrete example in language:**

For the sentence "The cat, which was a beautiful orange tabby that had been rescued from the shelter last Tuesday, sat on the mat."

- Fixed SSM: Every word decays at the same rate. By the time we reach "sat", the representation of "cat" has decayed by $0.9^{13} = 0.25$. Significant information loss.
- Selective SSM: "cat" gets large $B_t$ (important subject). The relative clause tokens get small $B_t$ (details, not essential for the main clause). When "sat" arrives, $A_t$ is set to preserve the "cat" state, so the model correctly predicts that the cat is the one sitting.

This selectivity is what makes Mamba competitive with transformers: it provides content-dependent context management within the $O(T)$ computational budget. The SGST's spectral transport kernel implements an analogous selectivity: the transport parameters (diffusion coefficient $D$, gauge connection $A$) are input-dependent, controlling which spectral modes are amplified or attenuated based on context.

---

### Problem 7: Linear Attention Complexity

**Problem:** Linear attention replaces $\operatorname{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^T V)$, changing the computation order. Show that this changes complexity from $O(T^2 d)$ to $O(T d^2)$.

**Solution:**

**Standard attention:**

$$\operatorname{Attention} = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Step 1: Compute $QK^T$. $Q$ is $T \times d$, $K^T$ is $d \times T$.
$QK^T$ is $T \times T$. Cost: $O(T^2 d)$.

Step 2: Apply softmax (row-wise). Cost: $O(T^2)$.

Step 3: Multiply $(T \times T)$ by $V$ $(T \times d)$.
Cost: $O(T^2 d)$.

Total: $O(T^2 d)$.

**Linear attention:**

$$\operatorname{Attention} = \phi(Q) \cdot \bigl(\phi(K)^T V\bigr)$$

The key insight: change the order of multiplication using associativity.

Standard order: $\bigl(\phi(Q) \, \phi(K)^T\bigr) V$ -- forms the $T \times T$ matrix first.
Linear order: $\phi(Q) \bigl(\phi(K)^T V\bigr)$ -- forms the $d \times d$ matrix first.

Step 1: Compute $S = \phi(K)^T V$. $\phi(K)^T$ is $d \times T$, $V$ is $T \times d$.
$S$ is $d \times d$. Cost: $O(T d^2)$.

Step 2: Compute $\phi(Q) \, S$. $\phi(Q)$ is $T \times d$, $S$ is $d \times d$.
Result is $T \times d$. Cost: $O(T d^2)$.

Total: $O(T d^2)$.

**When is linear attention faster?**

$$O(T d^2) < O(T^2 d) \quad \text{when} \quad d < T$$

For long sequences ($T = 100\text{K}$, $d = 256$): $T d^2 = 100\text{K} \times 65\text{K} = 6.5\text{G}$ vs $T^2 d = 10\text{G} \times 256 = 2.56\text{T}$. Linear attention is approximately 400x faster.

For short sequences ($T = 64$, $d = 256$): $T d^2 = 64 \times 65\text{K} = 4.2\text{M}$ vs $T^2 d = 4\text{K} \times 256 = 1.0\text{M}$. Standard attention is actually 4x faster here.

**The tradeoff:** Linear attention approximates softmax attention. The feature map $\phi$ introduces approximation error. Softmax attention can represent sharp, selective patterns (attending to 1-2 tokens out of 100K). Linear attention struggles with sharp patterns because $\phi(q)^T \phi(k)$ cannot approximate very peaked distributions well.

**SGST context:** The SGST operates in the spectral domain where the effective "sequence length" for operations is $d$ (embedding dimension) and sparsity reduces it to $s \ll d$ active modes, giving $O(Ts)$ cost -- potentially better than both standard and linear attention.

---

### Problem 8: SGST vs. Mamba

**Problem:** The thesis (Ch. 7.4) relates SGST to Mamba, Hyena, FNO, and Associative Transformers. For Mamba: both use state-space recurrence for context. What is the key difference?

**Solution:**

**Similarities between SGST and Mamba:**

Both architectures:
- Use recurrent state-space dynamics for context accumulation (vs. attention's $O(T^2)$ pairwise computation)
- Maintain an $O(d)$ hidden state that summarizes past context
- Use input-dependent parameters (selective transitions)
- Achieve $O(T)$ sequential complexity for training

**Key differences:**

1. **State structure:**
   - Mamba: real/complex-valued vector state with element-wise recurrence. Each state dimension evolves independently: $s_k^i = a_k^i s_{k-1}^i + b_k^i u_k$. Cross-dimension interaction comes ONLY from the subsequent MLP/projection layers, not from the recurrence itself.
   - SGST: spectral state with geometric structure. The transport kernel $\exp(-D \omega^2 - i A \omega)$ operates in Fourier space, where $\omega$ indexes frequency modes. The kernel implements diffusion ($D \omega^2$ term) and phase rotation ($A \omega$ term), with cross-mode interactions through the nonlinear MLP sandwiched between forward and inverse FFT.

2. **Interpretability:**
   - Mamba: the recurrence parameters $a_k$, $b_k$ are learned black boxes. There is no prescribed physical or geometric meaning to what each state dimension represents.
   - SGST: the recurrence has explicit geometric meaning. $D$ is a diffusion coefficient (information spread rate), $A$ is a gauge connection (parallel transport phase). The spectral modes have frequency interpretation. This interpretability guides architecture design, even though the V12 ablation showed it didn't improve performance at small scale.

3. **Mixing mechanism:**
   - Mamba: state mixing is purely through element-wise gating. The selectivity is per-dimension: each state dimension independently decides to remember or forget.
   - SGST: state mixing is through spectral transport. The FFT/IFFT pair enables global communication in a single step (all positions interact through shared frequency modes). This is fundamentally different from element-wise gating.

4. **Ablation finding (V12):**
   The V12 ablation study showed that Mamba-style SSM + MLP slightly outperformed the full SGST spectral machinery at the tested scale. The geometric interpretation and spectral structure did not translate to better perplexity. This is a central challenge the thesis addresses: the geometric framework is theoretically richer but has not yet demonstrated a clear empirical advantage over simpler selective SSMs.

---

### Problem 9: GLA vs. DeltaNet Update Rules

**Problem:** GLA (Gated Linear Attention) uses gated state updates: $S_t = (1 - g_t) S_{t-1} + g_t k_t v_t^T$. DeltaNet uses delta rule: $S_t = S_{t-1} + k_t (v_t - S_{t-1}^T k_t)^T$. Explain the difference between these two update rules.

**Solution:**

**GLA (Gated Linear Attention):**

$$S_t = (1 - g_t) S_{t-1} + g_t \, k_t v_t^T$$

This is a gated interpolation. The gate $g_t \in [0, 1]$ controls the blend:
- $g_t = 0$: $S_t = S_{t-1}$ (complete retention, no new information)
- $g_t = 1$: $S_t = k_t v_t^T$ (complete replacement with new key-value pair)
- $0 < g_t < 1$: smooth blend of old state and new information

Properties:
- Old information decays exponentially: contribution of $S_0$ after $t$ steps is $(1-g)^t$
- The decay is GLOBAL: all entries of $S$ decay at the same rate $g_t$
- New information is blended uniformly: $k_t v_t^T$ overwrites all of $S$ proportionally
- Simple and stable, but not content-addressable

**DeltaNet (Delta Rule):**

$$S_t = S_{t-1} + k_t (v_t - S_{t-1}^T k_t)^T$$

The term $(v_t - S_{t-1}^T k_t)$ is the prediction error: the difference between the target value $v_t$ and what the current state predicts for key $k_t$ (which is $S_{t-1}^T k_t$ = the value associated with $k_t$ in the current state).

Properties:
- If the state already predicts $v_t$ correctly from $k_t$, the update is zero (delta $= 0$). No unnecessary modification.
- The update acts ONLY along the key direction $k_t$. Other key-value associations in $S$ are preserved exactly.
- This is content-addressable: it updates a specific "slot" in the associative memory (the entry for key $k_t$) without disturbing other slots.
- Closely related to the Widrow-Hoff learning rule from classical adaptive filtering.

**Comparison:**

| Property | GLA | DeltaNet |
|----------|-----|----------|
| Update scope | Global (all entries) | Local (only the $k_t$ slot) |
| Decay of old info | Exponential, uniform | None (exactly preserved) |
| Redundant writes | Always writes (even if redundant) | No-op if already correct |
| Memory precision | Blurry (smooth blending) | Sharp (exact correction) |
| Stability | Very stable (bounded by interpolation) | Can be unstable (unbounded updates) |

**Thesis connection (SYNTHESIS.md):** The synthesis direction identifies combining $SO(K)$ orthogonal transport (geometric, from SGST) with the delta rule (sharp, content-addressable updates) as the key unexplored gap. The delta rule provides precise memory management that the SGST's smooth spectral transport lacks, while $SO(K)$ transport provides geometric structure (norm preservation, curvature) that the delta rule's flat Euclidean updates lack.

---

### Problem 10: Reaction-Diffusion Decomposition

**Problem:** The "reaction-diffusion" decomposition (Shi et al. 2025, thesis Sec. 2.7.3) says transformer layers = diffusion (attention) + reaction (FFN). Explain each component and why both are needed.

**Solution:**

**Diffusion component (Attention):**

Attention mixes information across positions. Each token receives a weighted sum of all other tokens:

$$h_i = \sum_j \alpha_{ij} v_j$$

where $\alpha_{ij} = \operatorname{softmax}(q_i \cdot k_j / \sqrt{d})$.

This is mathematically analogous to heat diffusion on a graph: information "flows" from each token to every other token, with flow rate proportional to the attention weight. Properties:
- Global communication: every position can access every other position
- Low-pass filtering tendency: averaging many signals smooths out high-frequency variation
- No per-position computation: only mixing, no transformation

Without diffusion (attention), tokens cannot communicate. Each position would process its own content in isolation, with no sequence understanding.

**Reaction component (FFN):**

The feed-forward network transforms each token independently:

$$h_i = W_2 \cdot \operatorname{activation}(W_1 h_i + b_1) + b_2$$

This is analogous to a chemical reaction at each spatial point: local transformation without spatial mixing. Properties:
- Position-independent: the same function applied to every token
- Nonlinear: the activation function enables complex per-token computation
- Dimension expansion: $W_1$ projects to $4d$, enabling richer representation before $W_2$ compresses back to $d$

Without reaction (FFN), the network can only compute linear combinations of input tokens. No nonlinear feature extraction, no position-wise reasoning.

**Why BOTH are needed:**

- Diffusion alone leads to rank collapse: repeatedly averaging representations causes all tokens to converge to the same vector. This is the over-smoothing problem well-documented in GNNs (thesis Ch. 3).

- Reaction alone is position-independent: without mixing, the network cannot perform any sequence-level reasoning. Each token is processed identically regardless of context.

- Together: diffusion distributes information (token A learns about token B), reaction processes the combined information (compute a new feature from the mix), then the next layer's diffusion redistributes the processed features. This alternation builds increasingly abstract representations.

**SGST implementation:**

The SGST's forward-reverse loop is a direct implementation of reaction-diffusion:

1. **FFT (forward):** Transform from spatial to spectral domain. This IS diffusion -- each frequency mode combines information from all positions (the Fourier transform is a global mixing operation).

2. **MLP (in spectral domain):** Apply nonlinear transformation to spectral coefficients. This IS reaction -- per-mode computation that creates new features.

3. **IFFT (reverse):** Transform back to spatial domain. The IFFT reconstructs the field, distributing the spectrally-processed information back to all positions.

The SGST makes the reaction-diffusion structure explicit and physically interpretable: FFT/IFFT provides exact diffusion (Fourier modes are eigenfunctions of the diffusion operator), and the spectral MLP provides frequency-domain reaction. This contrasts with the transformer where the decomposition is implicit in the attention/FFN alternation.

---

## Comprehension Questions

1. Write out the full attention equation $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}(QK^T / \sqrt{d_k}) \, V$. Explain the role of each component: $Q$, $K$, $V$, the softmax, and the $\sqrt{d_k}$ scaling factor.

2. Why does standard attention cost $O(T^2 d)$? At what sequence length does this become prohibitive, and what alternatives exist?

3. Explain the parallel scan algorithm step by step. Why is associativity the key property that enables parallelism? What would go wrong if the operation were not associative?

4. Compare the state size of a Transformer (KV cache) and an SSM (hidden state) as a function of sequence length. Why does the SGST claim approximately 1000x compression, and what is sacrificed to achieve it?

5. What is the reaction-diffusion decomposition of transformer layers? Identify the "diffusion" and "reaction" components, explain why both are necessary, and describe how the SGST implements each one in the spectral domain.

---

## Bridge to Thesis

This unit covers the three main architectural families that the SGST synthesizes and aims to improve upon:

- **Transformers** provide the gold standard for sequence modeling via attention's $O(T^2)$ pairwise computation. The SGST's goal is to match this quality with $O(T)$ cost. Unit 09 showed that attention IS Hopfield retrieval; this unit shows the full architecture surrounding it.

- **GNNs** (thesis Ch. 2.1, Ch. 3) were the original approach in the SGST project. The thesis documents why spatial graph-based methods failed: over-smoothing, limited expressivity (1-WL bound), and inability to capture long-range dependencies without deep stacking. This motivated the shift to spectral methods.

- **SSMs/Mamba** represent the closest existing approach to the SGST's design philosophy: $O(T)$ recurrence with input-dependent transitions. The V12 ablation showed Mamba-style SSM+MLP slightly outperforming the full SGST spectral machinery, establishing the performance bar the geometric approach must clear.

- **Linear attention** connects SSMs to attention through the kernel trick (Key Concept 12). The synthesis direction (holonomic linear attention) aims to equip linear attention with geometric structure ($SO(K)$ transport) and sharp memory (delta rule), potentially achieving the best of both worlds: attention-quality representations with SSM-efficiency computation.

- **The reaction-diffusion framework** (Problem 10) provides the conceptual bridge: every effective sequence model must implement both global mixing (diffusion) and local transformation (reaction). The SGST's FFT-MLP-IFFT loop makes this explicit, with the additional benefit that spectral sparsity (few active frequency modes) reduces the cost of both components.
