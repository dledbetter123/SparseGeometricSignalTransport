# Unit 04: Graphs, Laplacians, and Spectral Graph Theory

## Learning Objectives

By the end of this unit, you will be able to:

1. Represent graphs as adjacency and Laplacian matrices
2. Understand the graph Laplacian's eigenvalues and eigenvectors
3. Connect graph spectral gap to mixing rate and oversmoothing
4. Understand message passing as matrix multiplication
5. See GNN message passing as a low-pass graph filter

## Prerequisites

- Unit 01 (eigendecomposition)
- Unit 03 (Fourier concepts)

## Readings

- Spielman, *Spectral Graph Theory* lecture notes (Yale, free online)
- Thesis Ch. 2.1 (GNNs and Message Passing), Sec. 2.1.1-2.1.3
- Thesis Ch. 3 (Failure of Spatial Approaches), especially Sec. 3.3
- Thesis Sec. 2.2 (Noise and Information Bottleneck in Message Passing)
- Paper: Topping et al. 2022 "Understanding over-squashing via curvature" (referenced in thesis)

---

## Key Concepts

### 1. Graphs: Vertices $V$, Edges $E$, Adjacency Matrix $A$

A graph $G = (V, E)$ consists of a set of vertices (nodes) $V$ and edges $E$ connecting them. The adjacency matrix $A$ is an $n \times n$ matrix where $A_{ij} = 1$ if there is an edge between vertices $i$ and $j$, and $A_{ij} = 0$ otherwise. For undirected graphs, $A$ is symmetric.

### 2. Degree Matrix $D$

The degree matrix $D$ is diagonal, with $D_{ii} = d_i = \sum_j A_{ij}$ (the number of edges connected to vertex $i$). The degree tells you how "connected" each node is. For a regular graph, all degrees are equal.

### 3. Graph Laplacian: $L = D - A$

The unnormalized graph Laplacian is $L = D - A$. It is symmetric and positive semidefinite. Key property: for any signal $x$ on the graph,

$$x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2$$

This measures the "smoothness" of $x$ on the graph -- how much $x$ varies across edges.

The symmetric normalized Laplacian is $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. Its eigenvalues lie in $[0, 2]$.

### 4. Spectral Decomposition: $L = U \Lambda U^T$

Since $L$ is real symmetric, it has an orthogonal eigendecomposition $L = U \Lambda U^T$ where $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ with eigenvalues sorted as $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$. The columns of $U$ are the eigenvectors, which form the "Fourier modes" of the graph.

### 5. Smallest Eigenvalue: $\lambda_1 = 0$ Always

The constant vector (all ones) is always an eigenvector of $L$ with eigenvalue $0$, because $L \cdot \mathbf{1} = (D - A) \cdot \mathbf{1} = d - d = 0$. This is the "DC component" or zero-frequency mode of the graph -- a signal that is constant everywhere.

### 6. Spectral Gap: $\lambda_2$ (Algebraic Connectivity / Fiedler Value)

The second smallest eigenvalue $\lambda_2$ is called the spectral gap or algebraic connectivity (Fiedler value). It measures how well-connected the graph is:

- $\lambda_2 = 0$ means the graph is disconnected
- Small $\lambda_2$ means there is a near-bottleneck (the graph can almost be split in two)
- Large $\lambda_2$ means the graph is well-connected and mixes quickly

The corresponding eigenvector (Fiedler vector) provides the best partition of the graph into two halves.

### 7. Message Passing as Matrix Multiplication

One step of GNN message passing updates node features as:

$$X^{(k+1)} = \bar{A} X^{(k)}$$

where $\bar{A}$ is a normalized adjacency matrix (e.g., $D^{-1}A$ or $D^{-1/2}AD^{-1/2}$). Each node's new feature is a weighted average of its neighbors' features. After $k$ steps, $X^{(k)} = \bar{A}^k X^{(0)}$ -- a matrix power.

### 8. Graph Fourier Transform

The Graph Fourier Transform (GFT) decomposes a signal $x$ on the graph into the eigenbasis of the Laplacian:

$$\hat{x} = U^T x \quad \text{(analysis)}$$
$$x = U \hat{x} \quad \text{(synthesis)}$$

This is the graph-analogue of the DFT from Unit 03. Low-eigenvalue components are "smooth" on the graph (vary slowly across edges); high-eigenvalue components are "rough" (change rapidly across edges).

### 9. Low-Pass Filter: Message Passing Suppresses High-Frequency Components

In the graph spectral domain, $k$ steps of message passing act as:

$$\hat{x}^{(k)}[i] = \lambda_i^k \cdot \hat{x}^{(0)}[i]$$

Since $|\lambda_i| < 1$ for non-leading eigenvalues of the normalized adjacency, higher frequencies decay faster. After many steps, only the lowest-frequency component (the constant/DC mode) survives. This is why GNNs are inherently low-pass filters and why deep GNNs suffer from over-smoothing.

### 10. Over-Squashing: Exponential Attenuation of Long-Range Information

For nodes $u$ and $v$ at graph distance $k$, the sensitivity of node $v$'s representation to node $u$'s input decays exponentially:

$$\left|\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}}\right| \leq C \cdot r^k$$

where $r < 1$ depends on the graph topology (specifically, the curvature along the path). Information from distant nodes arrives exponentially faint -- this is over-squashing, a fundamental limitation of message-passing architectures.

---

## Worked Problems

### Problem 1

**For the path graph $P_4$ (4 nodes in a line: 1-2-3-4), write the adjacency matrix $A$, degree matrix $D$, and Laplacian $L$. Compute eigenvalues of $L$.**

**Solution:**

The path graph $P_4$ has edges: $(1,2), (2,3), (3,4)$.

**Adjacency matrix:**

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

**Degree matrix:**

Node 1: degree 1, Node 2: degree 2, Node 3: degree 2, Node 4: degree 1.

$$D = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 2 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**Laplacian $L = D - A$:**

$$L = \begin{bmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{bmatrix}$$

**Eigenvalues:** The characteristic polynomial of $L$ for the path graph $P_n$ has known eigenvalues:

$$\lambda_k = 2 - 2 \cos\left(\frac{\pi (k-1)}{n}\right), \quad k = 1, 2, \ldots, n$$

For $n = 4$:
- $\lambda_1 = 2 - 2 \cos(0) = 0$
- $\lambda_2 = 2 - 2 \cos(\pi/4) = 2 - \sqrt{2} \approx 0.586$
- $\lambda_3 = 2 - 2 \cos(\pi/2) = 2$
- $\lambda_4 = 2 - 2 \cos(3\pi/4) = 2 + \sqrt{2} \approx 3.414$

**Eigenvalues: $\{0,\; 2 - \sqrt{2},\; 2,\; 2 + \sqrt{2}\} = \{0,\; 0.586,\; 2,\; 3.414\}$**

The spectral gap is $\lambda_2 = 2 - \sqrt{2} \approx 0.586$, indicating moderate connectivity for this small chain.

---

### Problem 2

**For the complete graph $K_4$ (all pairs connected), compute the spectral gap. Compare to $P_4$. What does this tell you about mixing speed?**

**Solution:**

$K_4$ has 4 nodes, each connected to every other node.

**Adjacency matrix:** $A = J - I$ where $J$ is the $4 \times 4$ all-ones matrix and $I$ is identity.

$$A = \begin{bmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix}$$

**Degree matrix:** Every node has degree 3, so $D = 3I$.

**Laplacian:** $L = D - A = 3I - (J - I) = 4I - J$.

**Eigenvalues of $L = 4I - J$:**
- $J$ has eigenvalues: 4 (once, eigenvector $[1,1,1,1]^T/2$) and 0 (three times)
- Therefore $L = 4I - J$ has eigenvalues: $4 - 4 = 0$ (once) and $4 - 0 = 4$ (three times)

**Eigenvalues: $\{0, 4, 4, 4\}$**

**Spectral gap: $\lambda_2 = 4$**

**Comparison:**

| Graph | Spectral gap $\lambda_2$ |
|-------|----------------------|
| $P_4$   | 0.586                |
| $K_4$   | 4.0                  |

$K_4$ has a spectral gap nearly 7 times larger than $P_4$.

**Interpretation:** Larger spectral gap means faster mixing, which means faster convergence to the steady state (uniform/constant signal). For the normalized adjacency, eigenvalues of $\bar{A}$ for $K_4$ are 1 (once) and $-1/3$ (three times). After $k$ steps, non-DC components decay as $(1/3)^k$. **$K_4$ essentially reaches over-smoothing in 1-2 steps.** $P_4$ takes many more steps because its spectral gap is small -- information propagates slowly along the chain.

---

### Problem 3

**Show that one step of message passing on the complete graph $K_n$ produces the average: if $X^{(1)} = \frac{1}{n}J X^{(0)}$, then all rows of $X^{(1)}$ are identical. This is rank collapse in one step.**

**Solution:**

For $K_n$, the normalized adjacency matrix is:

$$\bar{A} = \frac{1}{n-1}(J - I)$$

because each node has degree $n-1$, so the row-normalized adjacency divides each row by $n-1$.

**One step of message passing:**

$$X^{(1)} = \bar{A} X^{(0)} = \frac{1}{n-1}(J - I) X^{(0)}$$

For each node $i$, the new feature vector is:

$$h_i^{(1)} = \frac{1}{n-1} \sum_{j \neq i} h_j^{(0)}$$

This is the average of ALL other nodes' features. For large $n$, this is close to the global mean.

**Two steps:**

$$X^{(2)} = \bar{A} X^{(1)} = \bar{A}^2 X^{(0)}$$

Since every row of $X^{(1)}$ is nearly identical (each is the average of all others), $X^{(2)}$ will be EXACTLY the global average repeated $n$ times.

More precisely, $\bar{A}$ has eigenvalues 1 (for the constant eigenvector) and $-\frac{1}{n-1}$ (for all others). After $k$ steps, non-constant components scale as $\left(-\frac{1}{n-1}\right)^k$, which goes to 0 rapidly.

**For $n = 100$:** After just 1 step, non-constant components are scaled by $\frac{1}{99} \approx 0.01$. After 2 steps: $\approx 0.0001$. **The rank has collapsed from $\operatorname{rank}(X^{(0)})$ to effectively 1.**

This is the most extreme form of over-smoothing: the complete graph destroys all node-level information in a single message-passing step, replacing every node's features with the global average.

---

### Problem 4

**Compute $\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}}$ for a $k$-layer GNN on a path graph, where $v$ is at distance $k$ from $u$. Use the bound from thesis Eq. 2.3.**

**Solution:**

Consider a path graph where node $u$ is at one end and node $v$ is $k$ edges away. There is exactly ONE path of length $k$ from $u$ to $v$ (the graph is a chain).

**The Jacobian chain rule:**

$$\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}} = \text{product of message-passing weights along the path}$$

For each edge $(i, i+1)$ in the path, the message-passing step contributes a factor related to the normalized adjacency weight.

**For the path graph interior**, each node has degree 2. The normalized adjacency weight for each edge is approximately $1/d = 1/2$ (row normalization) or $1/\sqrt{d_i \cdot d_j} = 1/2$ (symmetric normalization).

**Therefore:**

$$\left|\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}}\right| \leq \left(\frac{1}{2}\right)^k$$

**Concrete values:**

| Hops $k$ | Gradient bound |
|--------|---------------|
| 1      | 0.5           |
| 5      | 0.031         |
| 10     | 0.00098       |
| 20     | $9.5 \times 10^{-7}$ |

**For $k = 10$:** The gradient is attenuated by a factor of $1/1024$, meaning information from 10 hops away arrives with less than 0.1% of its original magnitude.

**This is over-squashing.** The thesis (Sec. 2.2) argues this makes long-range dependencies impossible in GNNs. No matter how you train the weights, the gradient signal from distant nodes is exponentially faint. The network literally cannot learn to use long-range information because the learning signal is too weak.

**Key insight:** This is a TOPOLOGICAL limitation, not a capacity limitation. Adding more parameters does not help -- the bottleneck is the graph structure itself.

---

### Problem 5

**The Graph Fourier Transform of signal $x$ is $\hat{x} = U^T x$ where $U$ are eigenvectors of $L$. For the path graph $P_4$ with signal $x = [1, 0, 0, 0]$ (one-hot at node 1), compute the GFT and identify which frequencies dominate.**

**Solution:**

The eigenvectors of the path graph Laplacian are cosine-like functions. For $P_4$, the eigenvectors (columns of $U$) are:

$u_1 = [1, 1, 1, 1]^T / 2$ (constant, $\lambda_1 = 0$)

For the higher modes, the eigenvectors are discrete cosines evaluated at the node positions. The normalized eigenvectors for $P_n$ are:

$$u_k[j] = \sqrt{\frac{2}{n}} \cos\left(\frac{\pi(k-1)(2j-1)}{2n}\right), \quad k = 2, \ldots, n$$

**Computing the GFT:**

$\hat{x}[k] = u_k^T x = u_k[1]$ (since $x$ is one-hot at node 1)

- $\hat{x}[1] = u_1[1] = 1/2$
- $\hat{x}[2] = u_2[1] = \sqrt{2/4} \cos(\pi \cdot 1/(2 \cdot 4)) = \frac{1}{\sqrt{2}} \cos(\pi/8) \approx 0.653$
- $\hat{x}[3] = u_3[1] = \frac{1}{\sqrt{2}} \cos(3\pi/8) \approx 0.271$
- $\hat{x}[4] = u_4[1] = \frac{1}{\sqrt{2}} \cos(5\pi/8) \approx -0.271$ (sign depends on convention)

**Observation:** The GFT coefficients are spread across ALL frequency modes. No single mode dominates -- the energy is distributed across the entire spectrum.

**This is the discrete uncertainty principle on graphs.** A spatially localized signal (one-hot = maximally localized) has maximum spectral spread. Conversely, a spectrally localized signal (single eigenvector) is maximally spread across all nodes. You cannot be localized in both domains simultaneously.

**Implication for GNNs:** A one-hot input (which is how tokens/nodes are initially represented) contains energy at ALL graph frequencies. Message passing acts as a low-pass filter that progressively removes the high-frequency content, eventually leaving only the DC component. This is why all node embeddings converge.

---

### Problem 6

**Show that $k$ steps of message passing act as a polynomial filter on the graph Laplacian: $X^{(k)} = p_k(\bar{A}) X^{(0)}$ where $p_k$ is a degree-$k$ polynomial.**

**Solution:**

**Setup:** Let $\bar{A} = D^{-1/2} A D^{-1/2}$ be the symmetric normalized adjacency with spectral decomposition $\bar{A} = U \Lambda U^T$, where $\Lambda = \operatorname{diag}(\mu_1, \ldots, \mu_n)$ and $\mu_i$ are the eigenvalues.

**$k$ steps of message passing:**

$$X^{(k)} = \bar{A}^k X^{(0)}$$

Using the spectral decomposition:

$$\bar{A}^k = (U \Lambda U^T)^k = U \Lambda^k U^T$$

**In the graph spectral domain:**

$$\hat{X}^{(k)} = U^T X^{(k)} = U^T \bar{A}^k X^{(0)} = \Lambda^k U^T X^{(0)} = \Lambda^k \hat{X}^{(0)}$$

So for each graph frequency $i$:

$$\hat{X}^{(k)}[i] = \mu_i^k \cdot \hat{X}^{(0)}[i]$$

The filter response function is $h(\mu) = \mu^k$ -- a degree-$k$ polynomial.

**Analysis of the filter:**

For the normalized adjacency, eigenvalues satisfy $|\mu_i| \leq 1$, with $\mu_1 = 1$ (corresponding to the constant eigenvector).

- For the leading eigenvalue $\mu_1 = 1$: $h(1) = 1^k = 1$ (DC component preserved)
- For $|\mu_i| < 1$: $|\mu_i|^k \to 0$ as $k \to \infty$ (all other components decay)
- The rate of decay depends on $|\mu_i|$: larger $|\mu_i|$ (closer to 1) decays slower

**This proves GNNs are low-pass filters.** The filter $h(\mu) = \mu^k$ strongly suppresses everything except the component at $\mu = 1$. After $k$ steps:

- Smooth signals (low graph frequency, $\mu$ close to 1): slightly attenuated
- Rough signals (high graph frequency, $\mu$ close to $-1$ or $0$): exponentially suppressed
- DC component ($\mu = 1$): perfectly preserved

**With learnable weights,** the GNN can implement $h(\mu) = \sum_{j=0}^{k} c_j \mu^j$ -- a general degree-$k$ polynomial filter. But the key constraint remains: a $k$-layer GNN can only implement degree-$k$ polynomial filters on the graph spectrum. This fundamentally limits its expressive power.

---

### Problem 7

**The "over-squashing" Jacobian bound (thesis Eq. 2.3) states $\left|\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}}\right| \leq C \cdot r^k$ where $r$ depends on graph curvature. For a graph with negative Ollivier-Ricci curvature (bottleneck), $r < 1$. Explain in your own words what this means for information flow.**

**Solution:**

**What is Ollivier-Ricci curvature?**

The Ollivier-Ricci curvature of an edge $(u, v)$ measures how much the neighborhoods of $u$ and $v$ overlap. Formally, it compares the optimal transport distance between the uniform distributions on the neighborhoods of $u$ and $v$ to the graph distance between $u$ and $v$.

- **Positive curvature:** Neighborhoods overlap significantly (like a sphere -- triangles close up). Information has multiple redundant paths.
- **Zero curvature:** Neighborhoods are "parallel" (like flat space). Information flows without amplification or attenuation.
- **Negative curvature:** Neighborhoods have little overlap (like a saddle or bottleneck). The edge is a "bridge" that information must squeeze through.

**The over-squashing bound:**

$$\left|\frac{\partial h_v^{(k)}}{\partial h_u^{(0)}}\right| \leq C \cdot r^k, \quad \text{where } r < 1 \text{ for negative curvature}$$

This says: the sensitivity of node $v$'s representation (at layer $k$) to node $u$'s input (at layer 0) decays EXPONENTIALLY with the number of hops $k$.

**In plain language:**

Imagine you whisper a message at node $u$ and it has to travel $k$ hops through bottleneck edges to reach node $v$. At each bottleneck edge, the message loses a fixed fraction of its volume (multiplied by $r < 1$). After $k$ hops, the message arrives at $v$ with volume proportional to $r^k$ -- exponentially faint.

**Why it is unfixable in GNNs:**

- The bound is TOPOLOGICAL: it depends on the graph structure, not the learned weights
- Making the GNN deeper (more layers) does NOT help -- each additional layer makes the bound worse (another factor of $r$)
- Making the GNN wider (more features) does NOT help -- the Jacobian norm still decays exponentially
- The only "fix" would be to change the graph topology itself (add edges to remove bottlenecks)

**Connection to SGST:** The SGST approach avoids this entirely by not using message passing at all. Instead of sending information hop-by-hop through the graph, SGST operates in the spectral domain where every frequency mode has global support. There is no "distance" to traverse and no bottleneck to squeeze through.

---

### Problem 8

**Compare the information capacity of message passing (GNN) vs. attention (Transformer) for 100 nodes with features in $\mathbb{R}^{128}$. How many pairwise connections does each have? What is the state size?**

**Solution:**

**GNN (Message Passing):**

Depends on the graph structure. For a typical sparse graph with average degree $d_{\text{avg}} = 4$:

- **Edges:** $|E| = (n \cdot d_{\text{avg}})/2 = (100 \cdot 4)/2 = 200$
- **Messages per layer:** 200 (one per directed edge, or 400 for bidirectional)
- **Each message:** $\mathbb{R}^{128}$ (feature dimension)
- **Node state:** $100 \times 128 = 12{,}800$ values
- **Communication diameter:** $k$ hops requires $k$ layers. For graph diameter $D$, need at least $D$ layers for full communication
- **Gradient attenuation:** Information from $k$ hops away attenuated by $r^k$

**Attention (Transformer):**

Every pair of nodes can communicate directly.

- **Attention pairs:** $100 \times 100 = 10{,}000$ per head
- **Per head state:** $Q, K, V$ matrices each $100 \times d_{\text{head}}$. For $d_{\text{head}} = 64$, 8 heads: $3 \times 100 \times 64 \times 8 = 153{,}600$ values
- **KV cache:** $2 \times 100 \times 128 = 25{,}600$ per layer
- **Communication diameter:** 1 (every pair communicates in a single layer)
- **Gradient attenuation:** Direct gradient path between any two nodes

**Comparison Table:**

| Property | GNN (sparse) | Attention |
|----------|-------------|-----------|
| Pairwise connections | 200 (sparse) | 10,000 (complete) |
| Layers for full comm. | $D$ (diameter) | 1 |
| Gradient path length | $k$ hops | 1 hop |
| Over-squashing? | Yes (exponential) | No |
| Cost per layer | $O(|E| \cdot d) = O(200 \cdot 128)$ | $O(n^2 \cdot d) = O(10000 \cdot 128)$ |
| Cost ratio | 1x | 50x |

**The fundamental trade-off:** GNNs are cheap but lossy -- they exploit graph sparsity but suffer from exponential information attenuation. Attention is expensive but complete -- every pair communicates directly but at $O(n^2)$ cost.

**SGST's proposition:** Get attention-like connectivity (every token can influence every other) at GNN-like cost ($O(N \log N)$ via FFT), by working in the spectral domain where all modes have global support.

---

### Problem 9

**For the bipartite graph $K_{2,3}$ (2 nodes on left, 3 on right, all cross-edges), compute the Laplacian and explain why it has exactly 2 zero eigenvalues if disconnected, or 1 if connected. Is $K_{2,3}$ connected?**

**Solution:**

$K_{2,3}$ has 5 nodes: $L = \{1, 2\}$ (left) and $R = \{3, 4, 5\}$ (right). Every left node connects to every right node. There are no edges within $L$ or within $R$.

**Adjacency matrix (nodes ordered 1, 2, 3, 4, 5):**

$$A = \begin{bmatrix} 0 & 0 & 1 & 1 & 1 \\ 0 & 0 & 1 & 1 & 1 \\ 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 \end{bmatrix}$$

**Degree matrix:**

Nodes 1, 2 have degree 3 (connected to all 3 right nodes).
Nodes 3, 4, 5 have degree 2 (connected to both left nodes).

$$D = \operatorname{diag}(3, 3, 2, 2, 2)$$

**Laplacian $L = D - A$:**

$$L = \begin{bmatrix} 3 & 0 & -1 & -1 & -1 \\ 0 & 3 & -1 & -1 & -1 \\ -1 & -1 & 2 & 0 & 0 \\ -1 & -1 & 0 & 2 & 0 \\ -1 & -1 & 0 & 0 & 2 \end{bmatrix}$$

**Is $K_{2,3}$ connected?**

Yes. Any two nodes are reachable: left nodes connect through any right node (distance 2), right nodes connect through any left node (distance 2), and left-right pairs are directly connected (distance 1). The graph has diameter 2.

**Number of zero eigenvalues:**

A fundamental theorem of spectral graph theory states:

**The multiplicity of 0 as an eigenvalue of $L$ equals the number of connected components of the graph.**

Since $K_{2,3}$ is connected, it has **exactly 1 connected component**, so $L$ has **exactly 1 zero eigenvalue** ($\lambda_1 = 0$). The spectral gap $\lambda_2 > 0$.

**Why this theorem holds:** $Lx = 0$ means $x^T L x = 0$, which means $\sum_{(i,j) \in E} (x_i - x_j)^2 = 0$. This forces $x_i = x_j$ for all adjacent pairs, so $x$ must be constant on each connected component. The number of linearly independent such vectors equals the number of connected components.

**If $K_{2,3}$ were disconnected** (e.g., if we removed all edges), we would have 5 connected components and 5 zero eigenvalues. If we split it into $\{1,3,4\}$ and $\{2,5\}$, we would have 2 components and 2 zero eigenvalues.

---

### Problem 10

**Thesis Table 3.3 summarizes the "fundamental incompatibility" between GNN message passing and language modeling. List at least 4 properties where they conflict, and explain each in terms of the graph-spectral concepts from this unit.**

**Solution:**

**Conflict 1: Spatial Smoothing vs. Sharp Distinctions**

Message passing computes weighted averages of neighbor features. In spectral terms, this is a low-pass filter (Problem 6) that suppresses high-frequency components. After several layers, all node features converge toward the leading eigenvector (constant signal).

But language requires tokens to maintain DISTINCT identities. The word "bank" next to "river" must remain different from "bank" next to "money." Low-pass filtering destroys these distinctions -- it smooths the very differences that carry semantic meaning.

**Conflict 2: Isotropic Aggregation vs. Directional Information Flow**

Standard message passing uses the symmetric normalized adjacency $D^{-1/2}AD^{-1/2}$, which treats all neighbors equally and symmetrically. The graph Laplacian is a symmetric operator.

But language has inherent DIRECTIONALITY: words earlier in a sentence inform the meaning of later words (causal structure). Autoregressive language modeling requires that information flows in one direction only. The symmetric graph Laplacian has no notion of directionality -- its eigenvectors spread symmetrically across the graph.

**Conflict 3: Local Connectivity vs. Long-Range Dependencies**

Message passing reaches $k$ hops in $k$ layers. The Jacobian bound (Problem 4) shows that information from distant nodes is exponentially attenuated. The spectral gap determines the mixing rate: sparse graphs with small spectral gaps transmit information slowly.

But language requires INSTANT long-range access. The subject at position 1 must agree with the verb at position 50. Over-squashing (exponential gradient decay) makes this impossible in message-passing architectures. Even for complete graphs (Problem 2), mixing is too fast and causes over-smoothing instead.

**Conflict 4: Fixed Topology vs. Dynamic Relationships**

The graph structure (adjacency matrix) is fixed at input time. The Laplacian's spectrum is determined before any computation begins. Spectral filters operate on this fixed eigenbasis.

But linguistic relationships are DYNAMIC and context-dependent. "It" might refer to different antecedents depending on context. The relevant "edges" between tokens change with meaning. A fixed graph topology cannot capture this -- the spectral basis would need to change with the content, which is impossible when the graph is fixed a priori.

**Conflict 5: Lossy Aggregation vs. Lossless Routing**

Each message-passing step aggregates (sums/averages) neighbor features, irreversibly mixing them. Once information from multiple neighbors is combined, the individual contributions cannot be recovered. This is entropy increase: the aggregation is a many-to-one mapping.

But language processing needs to ROUTE specific information to specific places. Coreference resolution requires identifying which noun a pronoun refers to -- this needs precise, reversible information pathways, not lossy aggregation. The spectral view shows that aggregation projects onto the low-frequency subspace, permanently discarding the high-frequency components that carry fine-grained distinctions.

**Summary:** Each conflict arises from the fundamental spectral properties of the graph Laplacian: symmetry, locality, fixed structure, and low-pass filtering. These are not bugs that can be patched -- they are intrinsic to the mathematical framework of message passing on graphs. This is why the thesis argues for abandoning graph-based approaches entirely in favor of spectral methods that operate directly in the frequency domain.

---

## Comprehension Questions

1. What does the spectral gap of a graph tell you about how quickly message passing reaches steady state?

2. Why is the graph Laplacian's spectrum analogous to the Fourier transform? What are "low frequency" signals on a graph?

3. Explain over-squashing in terms of the Jacobian bound. Why can't deeper GNNs fix it?

4. The thesis argues message passing is fundamentally incompatible with language. Summarize the geometric mismatch in 2-3 sentences.

5. How does attention solve the limitations of message passing? What does it cost?

---

## Bridge to Thesis

This unit exposes the fundamental limitations of graph-based neural architectures that motivated the SGST project. The key findings:

**Message passing is a low-pass graph filter** (Problem 6). This means GNNs inherently smooth node features toward uniformity, destroying the fine-grained distinctions needed for language understanding. The rate of smoothing is controlled by the spectral gap (Problems 1-2).

**Over-squashing is topological** (Problems 4, 7). Information from distant nodes is exponentially attenuated regardless of model capacity. This cannot be fixed by deeper or wider networks -- the bottleneck is in the graph structure itself.

**The fundamental incompatibility** (Problem 10) between message passing and language modeling is not an engineering problem to be solved with better architectures. It is a mathematical consequence of the spectral properties of the graph Laplacian.

These findings (thesis Ch. 2-3) led to the central insight of SGST: rather than trying to fix message passing, work directly in the spectral domain where every frequency mode has global support (Unit 03) and transport is $O(N \log N)$ via FFT. The next units will show how SGST implements this vision through sparse spectral representations, Parseval attention, and geometric transport.
