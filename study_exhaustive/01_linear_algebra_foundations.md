# Unit 01: Linear Algebra for Geometric Language Modeling

## Learning Objectives

By the end of this unit, you should be able to:

1. Perform and interpret eigendecomposition and singular value decomposition (SVD)
2. Understand rank, nullspace, and their role in information content
3. Work with complex vectors: Hermitian conjugate, unitary matrices
4. Connect rank collapse to GNN oversmoothing (Oono & Suzuki 2020)
5. Understand orthogonal projections and their role in smoothing

## Prerequisites

- Basic matrix multiplication
- Systems of linear equations (row reduction, solving $Ax = b$)

## Readings

- **Primary:** Strang, *Introduction to Linear Algebra*, Ch. 6 (Eigenvalues and Eigenvectors), Ch. 7 (Singular Value Decomposition) -- or equivalently, MIT OCW 18.06 lectures
- **Visual:** 3Blue1Brown, "Essence of Linear Algebra" (YouTube, all episodes). Watch before or alongside Strang.
- **Thesis:** Ch. 2.1 (Graph Neural Networks, rank collapse), Ch. 3.3.1 (Formal rank collapse analysis)
- **Paper:** Gao et al. 2019, "Representation Degeneration Problem in Training Natural Language Generation Models" (referenced in thesis Sec. 2.3.1)

---

## Key Concepts

### 1. Vector Spaces and Bases

A **vector space** is a collection of vectors that is closed under addition and scalar multiplication. A **basis** is a minimal set of vectors that spans the entire space -- every vector in the space can be written as a unique linear combination of basis vectors.

If you have a basis $\{v_1, \ldots, v_n\}$ for $\mathbb{R}^n$, then any vector $x \in \mathbb{R}^n$ can be written as $x = c_1 v_1 + \cdots + c_n v_n$ for unique scalars $c_i$. The number of vectors in a basis is the **dimension** of the space.

Why this matters: In language modeling, token embeddings live in $\mathbb{R}^d$ (typically $d = 512$ or $768$). The question "how many independent directions do these embeddings actually use?" is a question about the dimension of the subspace they span. If 1000 token embeddings in $\mathbb{R}^{512}$ only span a 20-dimensional subspace, most of the representational capacity is wasted.

### 2. Eigenvalues and Eigenvectors

For a square matrix $A$, a nonzero vector $v$ is an **eigenvector** with **eigenvalue** $\lambda$ if:

$$Av = \lambda v$$

Geometrically, $A$ acts on $v$ by simply scaling it (stretching if $|\lambda| > 1$, compressing if $|\lambda| < 1$, flipping if $\lambda < 0$). The eigenvectors are the "natural axes" of the transformation -- the directions that $A$ does not rotate.

To find eigenvalues, solve the **characteristic polynomial**: $\det(A - \lambda I) = 0$.

### 3. Eigendecomposition $A = Q \Lambda Q^{-1}$

If $A$ has $n$ linearly independent eigenvectors (always true for symmetric matrices), we can write:

$$A = Q \Lambda Q^{-1}$$

where $Q = [v_1 \mid v_2 \mid \cdots \mid v_n]$ has eigenvectors as columns, and $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$.

For symmetric/Hermitian matrices, $Q$ is orthogonal/unitary ($Q^{-1} = Q^T$ or $Q^*$), giving:

$$A = Q \Lambda Q^T$$

This decomposition reveals the "DNA" of a linear transformation: what directions it acts on ($Q$), and how much it stretches each ($\Lambda$).

**When it exists:** Any matrix with $n$ distinct eigenvalues, or any symmetric/Hermitian matrix. Fails for defective matrices (e.g., $\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$).

### 4. Singular Value Decomposition $A = U \Sigma V^T$

The SVD always exists, for ANY matrix (even rectangular). For $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^T$$

where $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors), $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with nonneg entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$, and $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors).

Geometric interpretation: Every linear transformation is a rotation ($V^T$), then a scaling along coordinate axes ($\Sigma$), then another rotation ($U$). The singular values $\sigma_i$ tell you how much stretching happens in each direction.

The SVD reveals:
- **Rank**: number of nonzero singular values
- **Best rank-$k$ approximation**: keep only the top $k$ singular values (Eckart-Young theorem)
- **Condition number**: $\sigma_{\max} / \sigma_{\min}$, measures numerical stability

### 5. Matrix Rank

The **rank** of a matrix is the dimension of its column space -- the number of linearly independent columns. Equivalently:
- $\operatorname{rank}(A) =$ number of nonzero singular values
- $\operatorname{rank}(A) =$ number of nonzero eigenvalues (for square matrices, counting correctly)
- $\operatorname{rank}(A) = n - \dim(\operatorname{null}(A))$

For a data matrix $X \in \mathbb{R}^{n \times d}$ where rows are data points, $\operatorname{rank}(X)$ tells you the intrinsic dimensionality of the data. If $\operatorname{rank}(X) \ll \min(n, d)$, the data lives in a low-dimensional subspace.

### 6. Rank-1 Matrices and Outer Products

A **rank-1 matrix** can be written as an outer product:

$$A = u v^T$$

where $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$. Every column of $A$ is a scalar multiple of $u$, and every row is a scalar multiple of $v^T$.

Rank-1 matrices are the "destination" of rank collapse. When repeated averaging sends all information to a single direction, the resulting matrix is rank 1. In the GNN context, this means every node has (essentially) the same representation.

### 7. Complex Vectors and Inner Products

For complex vectors $u, v \in \mathbb{C}^n$, the inner product is:

$$\langle u, v \rangle = u^* v = \overline{u}^T v = \sum_i \overline{u_i} \cdot v_i$$

Note the conjugation on the first argument. This ensures $\langle v, v \rangle = \sum |v_i|^2 \geq 0$ (real and nonneg).

The **Hermitian conjugate** (or conjugate transpose) of a matrix $A$ is:

$$A^* = \overline{A}^T$$

A matrix is **Hermitian** if $A^* = A$ (the complex analog of symmetric). Hermitian matrices have real eigenvalues and orthogonal eigenvectors.

### 8. Unitary Matrices

A matrix $U \in \mathbb{C}^{n \times n}$ is **unitary** if:

$$U^* U = U U^* = I$$

The real version ($U^T U = I$) is called **orthogonal**.

Key properties:
- Columns of $U$ form an orthonormal basis
- $U$ preserves inner products: $\langle Ux, Uy \rangle = \langle x, y \rangle$
- $U$ preserves norms: $\|Ux\| = \|x\|$
- Eigenvalues of $U$ lie on the unit circle: $|\lambda| = 1$

Preserving norms is critical for neural networks: if a layer's weight matrix is unitary/orthogonal, gradients neither explode nor vanish during backpropagation. This is the motivation behind unitary RNNs and the orthogonal gauge group in fiber bundle models.

### 9. Orthogonal Projections

The **orthogonal projection** onto the column space of $U$ (where $U$ has orthonormal columns) is:

$$P = U U^T$$

This is the closest-point map: $Px$ is the point in $\operatorname{col}(U)$ closest to $x$. Properties:
- $P^2 = P$ (idempotent -- projecting twice is the same as projecting once)
- $P^T = P$ (symmetric)
- Eigenvalues are $0$ or $1$
- $\operatorname{rank}(P) =$ number of columns of $U$

In the oversmoothing context, repeated message passing effectively projects token representations onto the leading eigenvectors of the graph Laplacian. Once projected, information in the orthogonal complement is permanently lost.

### 10. Spectral Norm and Condition Number

The **spectral norm** of a matrix $A$ is its largest singular value:

$$\|A\|_2 = \sigma_{\max}(A)$$

The **condition number** is:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

A large condition number means the matrix is nearly singular -- small perturbations in input cause large changes in output. For optimization, ill-conditioned Hessians make gradient descent zigzag inefficiently.

For neural network weight matrices, the ratio $\sigma_{\max}/\sigma_{\min}$ controls whether signals are amplified unevenly across different directions. Orthogonal/unitary matrices have condition number exactly $1$ -- perfectly conditioned.

---

## Worked Problems

### Problem 1: Eigendecomposition of a Symmetric Matrix

**Problem:** Given $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$, find the eigenvalues, eigenvectors, and eigendecomposition. Verify by multiplying $Q \Lambda Q^T$.

**Solution:**

**Step 1: Characteristic polynomial.**

$$\det(A - \lambda I) = \det\begin{pmatrix} 3 - \lambda & 1 \\ 1 & 3 - \lambda \end{pmatrix} = (3 - \lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 4)(\lambda - 2)$$

Eigenvalues: $\lambda_1 = 4$, $\lambda_2 = 2$.

**Step 2: Eigenvectors.**

For $\lambda_1 = 4$:

$$(A - 4I)v = 0 \implies \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} v = 0 \implies v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

Normalized: $v_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$.

For $\lambda_2 = 2$:

$$(A - 2I)v = 0 \implies \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} v = 0 \implies v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

Normalized: $v_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}$.

**Step 3: Form the decomposition.**

$$Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}$$

Since $A$ is symmetric, $Q$ is orthogonal: $Q^{-1} = Q^T$.

**Step 4: Verify $Q \Lambda Q^T = A$.**

$$Q \Lambda Q^T = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix} \begin{bmatrix}4 & 0\\0 & 2\end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}$$

Compute $Q \Lambda$ first:

$$Q \Lambda = \frac{1}{\sqrt{2}} \begin{bmatrix} 4 & 2 \\ 4 & -2 \end{bmatrix}$$

Then $(Q \Lambda) Q^T$:

$$= \frac{1}{\sqrt{2}}\begin{bmatrix}4 & 2\\4 & -2\end{bmatrix} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix} = \frac{1}{2} \begin{bmatrix} 4+2 & 4-2 \\ 4-2 & 4+2 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 6 & 2 \\ 2 & 6 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = A \checkmark$$

**Geometric interpretation:** $A$ stretches vectors along the $[1,1]$ direction by factor $4$ and along the $[1,-1]$ direction by factor $2$. It is an anisotropic scaling aligned with the 45-degree axes.

---

### Problem 2: SVD of a Rectangular Matrix

**Problem:** Compute the SVD of $A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$. What is the rank? What are the principal directions?

**Solution:**

**Step 1: Compute $A^T A$ and $A A^T$.**

$$A^T A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}$$

$$A A^T = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

**Step 2: Singular values.**

Eigenvalues of $A^T A$: $4$ and $1$. So $\sigma_1 = 2$, $\sigma_2 = 1$.

**Step 3: Right singular vectors $V$ (from $A^T A$).**

$A^T A$ is already diagonal, so $V = I_2$:

$$v_1 = [0, 1]^T \quad (\text{for eigenvalue } 4)$$
$$v_2 = [1, 0]^T \quad (\text{for eigenvalue } 1)$$

Wait -- let us be careful. $A^T A = \operatorname{diag}(1, 4)$. The eigenvalues are $1$ and $4$. To match convention $\sigma_1 \geq \sigma_2$:
- Eigenvalue $4$ has eigenvector $[0, 1]^T \Rightarrow v_1 = [0, 1]^T$, $\sigma_1 = 2$
- Eigenvalue $1$ has eigenvector $[1, 0]^T \Rightarrow v_2 = [1, 0]^T$, $\sigma_2 = 1$

So $V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ (swap columns of identity).

**Step 4: Left singular vectors $U$ (from $u_i = Av_i / \sigma_i$).**

$$u_1 = \frac{A v_1}{\sigma_1} = \frac{A [0, 1]^T}{2} = \frac{[0, 2, 0]^T}{2} = [0, 1, 0]^T$$

$$u_2 = \frac{A v_2}{\sigma_2} = \frac{A [1, 0]^T}{1} = \frac{[1, 0, 0]^T}{1} = [1, 0, 0]^T$$

Complete $U$ by adding $u_3 = [0, 0, 1]^T$ (orthogonal to both).

$$U = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Step 5: Form the SVD.**

$$\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$$

$$A = U \Sigma V^T$$

**Rank** $= 2$ (two nonzero singular values).

**Principal directions:** In the domain ($\mathbb{R}^2$), the principal directions are $v_1 = [0,1]$ and $v_2 = [1,0]$ -- the coordinate axes. In the codomain ($\mathbb{R}^3$), they map to $u_1 = [0,1,0]$ and $u_2 = [1,0,0]$. The third dimension of $\mathbb{R}^3$ is entirely in the nullspace of $A^T$.

---

### Problem 3: Repeated Averaging and Rank Collapse

**Problem:** Show that if $A = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ is applied $k$ times, $A^k$ converges to a rank-1 matrix. What does this illustrate about repeated averaging?

**Solution:**

**Step 1: Find eigenvalues of $A$.**

$$\det(A - \lambda I) = \det\begin{pmatrix} \frac{1}{2} - \lambda & \frac{1}{2} \\ \frac{1}{2} & \frac{1}{2} - \lambda \end{pmatrix} = \left(\frac{1}{2} - \lambda\right)^2 - \frac{1}{4} = \lambda^2 - \lambda = \lambda(\lambda - 1)$$

Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = 0$.

**Step 2: Find eigenvectors.**

For $\lambda_1 = 1$: $(A - I)v = 0 \implies \begin{bmatrix} -1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix} v = 0 \implies v_1 = [1, 1]^T / \sqrt{2}$.

For $\lambda_2 = 0$: $Av = 0 \implies \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{bmatrix} v = 0 \implies v_2 = [1, -1]^T / \sqrt{2}$.

**Step 3: Compute $A^k$ using eigendecomposition.**

$$A = Q \Lambda Q^T \text{ where } Q = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}, \quad \Lambda = \operatorname{diag}(1, 0)$$

$$A^k = Q \Lambda^k Q^T = Q \operatorname{diag}(1^k, 0^k) Q^T = Q \operatorname{diag}(1, 0) Q^T$$

For any $k \geq 1$:

$$A^k = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix} \begin{bmatrix}1 & 0\\0 & 0\end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix} = \frac{1}{2} \begin{bmatrix}1\\1\end{bmatrix} \begin{bmatrix}1 & 1\end{bmatrix} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = A$$

So $A^k = A$ for all $k \geq 1$. The matrix is **idempotent**: one application of averaging already projects onto the rank-1 subspace spanned by $[1, 1]^T$.

**Step 4: Interpret.**

After ONE step of averaging, both components become identical: if $x = [a, b]^T$, then $Ax = [(a+b)/2, (a+b)/2]^T$. The difference $(a - b)$ is destroyed. This is the **oversmoothing phenomenon**: repeated averaging on a graph projects all node features onto the leading eigenvector of the transition matrix, collapsing to rank 1. The information in the orthogonal complement (the $[1, -1]$ direction, which encodes the *difference* between nodes) is annihilated.

---

### Problem 4: Rank Deficiency in Token Embeddings

**Problem:** Given $X \in \mathbb{R}^{100 \times 512}$ (100 tokens, 512 dims), $X$ has rank 50. What does this mean about the token representations? How would you detect this with SVD?

**Solution:**

**Step 1: Interpret rank 50.**

Rank 50 means the 100 token vectors, each living in $\mathbb{R}^{512}$, collectively span only a 50-dimensional subspace. Out of 512 available dimensions, only 50 carry independent information. The remaining 462 dimensions are either unused or are linear combinations of the active 50.

Put differently: there exists a 50-dimensional subspace $S$ of $\mathbb{R}^{512}$ such that every token embedding lies in $S$. You could represent all 100 tokens using only 50 coordinates instead of 512, with zero information loss.

**Step 2: Detection via SVD.**

Compute the SVD: $X = U \Sigma V^T$.

The singular values are $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{100}$ (since $\min(100, 512) = 100$).

Rank 50 means:

$$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{50} > 0 = \sigma_{51} = \sigma_{52} = \cdots = \sigma_{100}$$

In practice, exact zeros are rare due to floating-point noise. Instead, look for a sharp drop: if $\sigma_{50}$ is much larger than $\sigma_{51}$, the effective rank is approximately 50.

**Step 3: Quantify the waste.**

Effective dimensionality used: 50.
Available dimensionality: 512.
Wasted capacity: $512 - 50 = 462$ dimensions, or about 90.2% of the embedding space.

This is the **representation degeneration problem** (Gao et al. 2019): token embeddings in language models tend to occupy a narrow cone in embedding space, wasting most of the available dimensions. The singular value spectrum reveals this -- a handful of large singular values dominate, with the rest near zero.

---

### Problem 5: Rank Equality for $A$, $A^T A$, and $A A^T$

**Problem:** Prove that for any matrix $A$, $\operatorname{rank}(A) = \operatorname{rank}(A^T A) = \operatorname{rank}(A A^T)$.

**Solution:**

We will show $\operatorname{Null}(A) = \operatorname{Null}(A^T A)$. Since rank $=$ (number of columns) $-$ dim(nullspace), equal nullspaces imply equal ranks.

**Direction 1: $\operatorname{Null}(A) \subseteq \operatorname{Null}(A^T A)$.**

Suppose $Ax = 0$. Then $A^T A x = A^T (Ax) = A^T (0) = 0$. So $x \in \operatorname{Null}(A^T A)$.

**Direction 2: $\operatorname{Null}(A^T A) \subseteq \operatorname{Null}(A)$.**

Suppose $A^T A x = 0$. Then:

$$x^T A^T A x = 0$$
$$(Ax)^T (Ax) = 0$$
$$\|Ax\|^2 = 0$$

Since the squared norm is zero, $Ax = 0$. So $x \in \operatorname{Null}(A)$.

**Conclusion:** $\operatorname{Null}(A) = \operatorname{Null}(A^T A)$.

If $A$ is $m \times n$, both $A$ and $A^T A$ map from $\mathbb{R}^n$. By the rank-nullity theorem:

$$\operatorname{rank}(A) = n - \dim(\operatorname{Null}(A)) = n - \dim(\operatorname{Null}(A^T A)) = \operatorname{rank}(A^T A)$$

The same argument applies to $\operatorname{rank}(A A^T) = \operatorname{rank}(A^T)$: show $\operatorname{Null}(A^T) = \operatorname{Null}(A A^T)$. And $\operatorname{rank}(A^T) = \operatorname{rank}(A)$ since they have the same singular values.

Therefore: $\operatorname{rank}(A) = \operatorname{rank}(A^T A) = \operatorname{rank}(A A^T)$.

---

### Problem 6: Unitary Matrices Preserve Norms

**Problem:** Let $U$ be a $3 \times 3$ unitary matrix. Prove that $\|Ux\| = \|x\|$ for all $x \in \mathbb{C}^3$. Why does this matter for gradient flow in neural networks?

**Solution:**

**Proof:**

$$\|Ux\|^2 = (Ux)^* (Ux) \quad [\text{definition of norm squared}]$$
$$= x^* U^* U x \quad [\text{property of conjugate transpose: } (AB)^* = B^* A^*]$$
$$= x^* I x \quad [U \text{ is unitary: } U^* U = I]$$
$$= x^* x$$
$$= \|x\|^2$$

Taking square roots: $\|Ux\| = \|x\|$.

**Why this matters for neural networks:**

In a deep network, the forward pass through layer $l$ computes $h^{(l)} = f(W^{(l)} h^{(l-1)})$. During backpropagation, gradients flow backward through the chain rule. The gradient through a linear layer is multiplied by the weight matrix (or its transpose).

If $W$ is unitary/orthogonal:
- **Forward pass:** $\|W h\| = \|h\|$, so signal magnitude is preserved through layers.
- **Backward pass:** The gradient is multiplied by $W^T$ (or $W^*$). Since $W^T$ is also unitary/orthogonal, $\|W^T g\| = \|g\|$, so gradient magnitude is preserved.

Without this property:
- If $\|W\| > 1$ (largest singular value $> 1$): signals/gradients **explode** exponentially with depth.
- If $\|W\| < 1$ (largest singular value $< 1$): signals/gradients **vanish** exponentially with depth.

This is the fundamental reason why unitary/orthogonal RNNs (Arjovsky et al. 2016) and orthogonal initialization (Saxe et al. 2014) improve training of deep networks. In the SGST architecture, the orthogonal gauge group $SO(K)$ for fiber bundle transport is motivated precisely by this norm-preservation property.

---

### Problem 7: Graph Diffusion on the Complete Graph

**Problem:** Compute the rank of $A^k$ where $A = \frac{1}{d} \mathbf{1} \mathbf{1}^T$ (the normalized all-ones matrix, $d = 3$). Show this is the steady-state of graph diffusion on a complete graph.

**Solution:**

**Step 1: Understand $A$.**

For $d = 3$:

$$A = \frac{1}{3} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

This is a rank-1 matrix: $A = \frac{1}{3} [1, 1, 1]^T [1, 1, 1] = v v^T$ where $v = [1, 1, 1]^T / \sqrt{3}$, scaled by 1.

More precisely, $A = \frac{1}{d} \mathbf{1} \mathbf{1}^T$ where $\mathbf{1} = [1, 1, 1]^T$.

**Step 2: Compute $A^2$.**

$$A^2 = \left(\frac{1}{3}\right)^2 (\mathbf{1} \mathbf{1}^T)(\mathbf{1} \mathbf{1}^T) = \frac{1}{9} \mathbf{1} (\mathbf{1}^T \mathbf{1}) \mathbf{1}^T = \frac{1}{9}(3) \mathbf{1} \mathbf{1}^T = \frac{1}{3} \mathbf{1} \mathbf{1}^T = A$$

$A$ is **idempotent**: $A^2 = A$. Therefore $A^k = A$ for all $k \geq 1$.

**Step 3: Rank.**

$\operatorname{rank}(A^k) = \operatorname{rank}(A) = 1$ for all $k \geq 1$.

**Step 4: Connection to the complete graph $K_3$.**

The adjacency matrix of $K_3$ (with self-loops) is the $3 \times 3$ all-ones matrix. The degree of each node is 3. The row-normalized adjacency (transition matrix) is:

$$T = D^{-1} A_{\text{adj}} = \frac{1}{3} \begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix} = A$$

One step of diffusion: $X^{(1)} = T X^{(0)} = A X^{(0)}$.

For any initial feature matrix $X^{(0)} \in \mathbb{R}^{3 \times f}$:

$$X^{(1)} = A X^{(0)} = \frac{1}{3} \mathbf{1} \mathbf{1}^T X^{(0)}$$

Each row of $X^{(1)}$ is the same: $\frac{1}{3}(\text{sum of all rows of } X^{(0)})$. After a single diffusion step, all nodes have the global average as their feature. $\operatorname{rank}(X^{(1)}) = 1$.

This is the most extreme case of rank collapse: the complete graph collapses in ONE step. Sparser graphs take more steps but the same eigenvalue decay mechanism applies.

---

### Problem 8: Message Passing as Matrix Multiplication

**Problem:** Given the message passing rule $h_v^{(l+1)} = \frac{1}{|N(v)|} \sum_{u \in N(v)} h_u^{(l)}$, show this can be written as $X^{(l+1)} = \bar{A} X^{(l)}$ where $\bar{A} = D^{-1} A$. Then argue why eigenvalues of $\bar{A}$ control the convergence rate.

**Solution:**

**Step 1: Matrix form.**

Let $A$ be the adjacency matrix: $A_{vu} = 1$ if $u \in N(v)$, else $0$.
Let $D$ be the diagonal degree matrix: $D_{vv} = |N(v)|$.
Let $X^{(l)} \in \mathbb{R}^{n \times d}$ have node features as rows.

The message passing rule for node $v$ is:

$$h_v^{(l+1)} = \frac{1}{|N(v)|} \sum_{u \in N(v)} h_u^{(l)} = \frac{1}{D_{vv}} \sum_u A_{vu} h_u^{(l)} = [D^{-1} A]_v X^{(l)}$$

In matrix form: $X^{(l+1)} = D^{-1} A X^{(l)} = \bar{A} X^{(l)}$.

**Step 2: Eigendecomposition of $\bar{A}$.**

Suppose $\bar{A}$ has eigendecomposition $\bar{A} = Q \Lambda Q^{-1}$ with eigenvalues $\lambda_1, \ldots, \lambda_n$.

For a connected graph with self-loops:
- $\lambda_1 = 1$ (the largest eigenvalue, with eigenvector related to the degree distribution)
- $|\lambda_i| < 1$ for all $i \geq 2$

**Step 3: $k$-step behavior.**

$$X^{(k)} = \bar{A}^k X^{(0)} = Q \Lambda^k Q^{-1} X^{(0)}$$

As $k$ grows:

$$\Lambda^k = \operatorname{diag}(1^k, \lambda_2^k, \ldots, \lambda_n^k) = \operatorname{diag}(1, \lambda_2^k, \ldots, \lambda_n^k)$$

Since $|\lambda_i| < 1$ for $i \geq 2$, we have $\lambda_i^k \to 0$ as $k \to \infty$.

Therefore:

$$\bar{A}^k \to Q \operatorname{diag}(1, 0, \ldots, 0) Q^{-1} = q_1 w_1^T$$

where $q_1$ is the first column of $Q$ and $w_1^T$ is the first row of $Q^{-1}$. This is a rank-1 matrix.

**Step 4: Convergence rate.**

The rate of collapse is controlled by $|\lambda_2|$ (the second-largest eigenvalue magnitude, called the **spectral gap**):

$$\|X^{(k)} - X^{(\infty)}\| = O(|\lambda_2|^k)$$

This is precisely the thesis result (Eq. 2.2): the manifold diameter $d_M(X^l)$ decays as $O((s \lambda)^l)$ where $s$ involves the spectral gap. A larger spectral gap (smaller $|\lambda_2|$) means faster collapse -- more connected graphs oversmooth faster.

---

### Problem 9: Complex Vector Arithmetic

**Problem:** For complex vector $z = [2+i, 1-3i, 4]^T$, compute: (a) $\|z\|$, (b) $z^*$ (conjugate transpose), (c) $z^*z$, (d) the outer product $zz^*$.

**Solution:**

**(a) Norm $\|z\|$:**

$$\|z\| = \sqrt{|2+i|^2 + |1-3i|^2 + |4|^2} = \sqrt{(2^2 + 1^2) + (1^2 + 3^2) + (4^2)} = \sqrt{5 + 10 + 16} = \sqrt{31} \approx 5.568$$

**(b) Conjugate transpose $z^*$:**

$$z^* = \overline{z}^T = [2-i, \; 1+3i, \; 4] \quad \text{(a row vector)}$$

**(c) Inner product $z^* z$:**

$$z^* z = (2-i)(2+i) + (1+3i)(1-3i) + (4)(4)$$
$$= (4 - i^2) + (1 - 9i^2) + 16$$
$$= (4 + 1) + (1 + 9) + 16$$
$$= 5 + 10 + 16 = 31 = \|z\|^2 \checkmark$$

**(d) Outer product $zz^*$:**

$zz^*$ is a $3 \times 3$ matrix where $(zz^*)_{jk} = z_j \cdot \overline{z_k}$.

Row 1: $(2+i)(2-i) = 5$, $(2+i)(1+3i) = 2+6i+i+3i^2 = 2+7i-3 = -1+7i$, $(2+i)(4) = 8+4i$

Row 2: $(1-3i)(2-i) = 2-i-6i+3i^2 = 2-7i-3 = -1-7i$, $(1-3i)(1+3i) = 1+9 = 10$, $(1-3i)(4) = 4-12i$

Row 3: $(4)(2-i) = 8-4i$, $(4)(1+3i) = 4+12i$, $(4)(4) = 16$

$$zz^* = \begin{bmatrix} 5 & -1+7i & 8+4i \\ -1-7i & 10 & 4-12i \\ 8-4i & 4+12i & 16 \end{bmatrix}$$

Note that $zz^*$ is **Hermitian**: $(zz^*)^* = (z^*)^* z^* = zz^*$. Also, entry $(j,k)$ is the conjugate of entry $(k,j)$, as expected.

---

### Problem 10: Anisotropy and Representation Degeneration

**Problem:** The "anisotropy" of an embedding matrix $X \in \mathbb{R}^{n \times d}$ is measured by the average cosine similarity between all pairs of rows. If every row of $X$ is identical, what is the anisotropy? If rows are orthogonal? Why is high anisotropy (near 1) a sign of representation degeneration?

**Solution:**

**Case 1: All rows identical.**

Let $x_i = x$ for all $i$. Then for any pair:

$$\cos(x_i, x_j) = \frac{x^T x}{\|x\| \|x\|} = \frac{\|x\|^2}{\|x\|^2} = 1$$

Average cosine similarity $= 1$. **Anisotropy is maximal.**

**Case 2: Rows are orthogonal.**

For $i \neq j$:

$$\cos(x_i, x_j) = \frac{x_i^T x_j}{\|x_i\| \|x_j\|} = \frac{0}{\|x_i\| \|x_j\|} = 0$$

Average cosine similarity (over distinct pairs) $= 0$. **Anisotropy is minimal.**

(Note: this requires $n \leq d$, since you cannot have more than $d$ mutually orthogonal vectors in $\mathbb{R}^d$.)

**Why high anisotropy signals degeneration:**

If the average cosine similarity is near 1, it means almost all token embeddings point in nearly the same direction in $\mathbb{R}^d$. This has severe consequences:

1. **Indistinguishability:** Tokens that should have different meanings have nearly identical representations. The model cannot discriminate between "cat" and "democracy."

2. **Wasted capacity:** The embeddings occupy a narrow cone rather than utilizing the full $d$-dimensional sphere. If all tokens lie in a 1D subspace, only $1/d$ of the representational capacity is used.

3. **Downstream failures:** Softmax attention computes dot products between queries and keys. If all keys point the same direction, attention weights become uniform -- the model cannot attend selectively.

This is the **representation degeneration problem** described in thesis Sec. 2.3.1. Gao et al. (2019) showed that standard language model training objectives push embeddings toward high anisotropy. The SGST architecture counters this by enforcing spectral sparsity, which forces tokens to occupy distinct spectral modes rather than collapsing to a shared direction.

---

## Comprehension Questions

Answer these after completing the readings and problems. Aim for precise, mathematical answers.

1. In your own words, explain why repeated message passing on a graph causes rank collapse. What is the mathematical mechanism?

2. What is the geometric interpretation of the SVD? (Hint: rotation, scaling, rotation.)

3. Why do unitary matrices preserve norms? Write the proof from memory.

4. If a 512-dimensional embedding matrix has effective rank 20, what percentage of the representational capacity is being wasted?

5. The thesis claims dense embeddings "intrinsically degenerate." What mathematical property causes this, and how does spectral sparsity avoid it?

---

## Bridge to Thesis

The concepts in this unit connect to the thesis as follows:

- **Rank collapse** (Problems 3, 7, 8) is the central failure mode of GNNs that the thesis diagnoses in Chapters 2-3. The eigenvalue decay of the graph transition matrix causes exponential convergence to rank 1, making deep GNNs useless. This motivates the entire geometric transport approach.

- **Representation degeneration** (Problems 4, 10) is why dense embeddings fail, as analyzed in thesis Sec. 2.3. The singular value spectrum of embedding matrices reveals the low effective rank that spectral sparsity explicitly controls.

- **Unitary/orthogonal matrices** (Problem 6) motivate the gauge group structure in fiber bundles (thesis Sec. 2.5). The $SO(K)$ orthogonal group used for parallel transport in SGST is chosen precisely because it preserves norms during message transport between nodes.

- **SVD** reveals the low effective rank of neural representations, which spectral sparsity (thesis Sec. 2.3.3) directly addresses by representing tokens as sparse combinations of frequency modes rather than dense vectors.
