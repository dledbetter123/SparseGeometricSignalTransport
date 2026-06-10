# Research: Efficient State Computation for Matrix-Valued Fibers

**Date**: 2026-04-01
**Context**: V16d/e explored matrix-valued fibers (linear attention in spectral space). The bottleneck is computing S[t] = gamma[t] * S[t-1] + k[t] v[t]^T across T positions.

---

## The Problem

The matrix fiber stores a dxd state matrix per head. At each position:
1. **Read**: output = q @ S (query the accumulated state)
2. **Deposit**: S = gamma * S + k outer v (decay + add new key-value pair)

This is inherently sequential: S[t] depends on S[t-1]. For T=256 positions, a naive Python loop is 256 sequential steps.

## Approaches Tested

### 1. Sequential Loop (V16d initial)
```python
for t in range(T):
    out[t] = q[t] @ S
    S = gamma[t] * S + k[t] @ v[t].T
```
- **Speed**: ~3,700ms/step (32x32 matrices, 8 subbundles)
- **Why slow**: Python loop over 256 steps, each with einsum ops. No parallelism.
- **State capacity**: 8 x 32x32 = 8,192 values

### 2. Chunked Sequential (V16d optimized)
Split sequence into chunks of C=64. Within each chunk, sequential loop. Between chunks, carry state.
```
for chunk in range(n_chunks):     # 4 iterations
    for t in range(chunk_size):   # 64 iterations
        read and deposit
    carry S to next chunk
```
- **Speed**: ~1,500ms/step (32x32), ~500ms/step (17x17)
- **Why faster**: Chunks can be batched in the outer loop. But inner loop still sequential.
- **Limitation**: Still O(T) sequential steps (just organized in chunks)

### 3. Parallel Scan with Scalar Decay (V16e)
Key insight: the decay gamma is a **scalar**, not a matrix. The recurrence S[t] = gamma[t] * S[t-1] + B[t] can be solved via parallel associative scan where the composition is:
```
(a2, B2) * (a1, B1) = (a2*a1, a2*B1 + B2)
```
This is the same scalar scan we use for the Wilson fiber, but B is now a dxd matrix instead of a scalar. The scalar multiply `a2*B1` broadcasts across all d^2 elements.
```python
def matrix_parallel_scan(A_diag, B_mat):
    # A_diag: (N, T) scalars, B_mat: (N, T, d, d) matrices
    for step in log2(T) iterations:
        B_new = A_right * B_left + B_right   # scalar * matrix + matrix
        A_new = A_right * A_left             # scalar * scalar
```
- **Speed**: 475ms/step (16x 8x8), 1,400ms/step (16x 32x32)
- **Why faster**: O(T log T) parallel steps instead of O(T) sequential. Each step is fully vectorized.
- **Tradeoff**: More total FLOPs (T log T vs T), but parallel.

## Speed Summary

| Config | State values | ms/step | Method |
|---|---|---|---|
| V16 scalar fiber (136 complex) | 272 | 390 | Parallel scan (scalar) |
| V16d (8 x 17x17 chunked) | 2,312 | 500 | Chunked sequential |
| V16e (16 x 8x8 parallel) | **1,024** | **475** | **Parallel scan (matrix)** |
| V16e (16 x 32x32 parallel) | 16,384 | 1,400 | Parallel scan (matrix) |
| V16e (16 x 64x64 chunked) | was tested | 3,700 | Chunked sequential |
| GPT-224d attention | 114,688 | 137 | Batched matmul (highly optimized) |

## Why GPT Is Still Faster

GPT's attention does two matmuls: `Q @ K^T` and `softmax @ V`. These are large, dense, embarrassingly parallel matrix multiplications that map perfectly to GPU/MPS hardware. The entire computation is one kernel launch.

Our parallel scan does `T log T = 256 * 8 = 2048` sequential stages, each touching `B*H` matrices. Even though each stage is vectorized, the sequential dependency between stages prevents full parallelization. The scan creates a chain of data dependencies that hardware can't pipeline as efficiently as a single large matmul.

## Future Optimization Paths

### Path A: Custom CUDA/Metal Kernels (Mamba's approach)
Mamba achieves near-attention speed by fusing the entire scan into a single kernel that keeps the state in registers/shared memory. No Python overhead, no intermediate tensor allocation. This is engineering work — same algorithm, 10-100x faster implementation.

**Estimated speedup**: 5-10x (from ~475ms to ~50-100ms)
**Effort**: Weeks of kernel development. Out of scope for thesis.

### Path B: Larger d with Structured Matrices
Instead of dense dxd state, use structured matrices:
- **Diagonal + low-rank**: S = diag(d) + UV^T where U, V are (d, r). State: d + 2dr values. Update: O(d + dr).
- **Block-diagonal**: Multiple small blocks instead of one big block. Already what we do with heads.
- **Toeplitz**: Banded structure. State: 2d-1 values. Equivalent to a learned convolution kernel.

These reduce the per-step cost from O(d^2) to O(d) while retaining more capacity than scalar.

### Path C: Chunked Parallel Hybrid (what we partially implemented)
Within chunks: use the parallel scan (or even attention, since chunks are small).
Between chunks: carry state.

If chunk_size=32 and d=8:
- Within-chunk: parallel scan on 32 steps, very fast
- Between-chunk: 8 sequential state carries
- Total: ~40 parallel steps instead of 256

This is how FlashLinearAttention works. Optimal chunk size depends on hardware.

### Path D: Trade State Size for Speed
Use many scalar fibers (d=1) with diverse decay rates instead of matrix fibers:
- 1024 scalar fibers with different gammas: 1,024 state values
- Fully parallel via existing scalar parallel scan
- Same state capacity as 16 x 8x8 but simpler
- No associative memory (can't store key-value pairs), just decayed averages
- Speed: same as V16's scalar fiber (~390ms)

This is RetNet's approach: multi-scale exponential decay replaces associative memory.

### Path E: Approximate Parallel Scan via Taylor Expansion
Approximate the matrix recurrence with Taylor series:
```
S[t] = gamma^t * S[0] + sum_{i=0}^{t-1} gamma^{t-1-i} * B[i]
     ≈ polynomial in gamma applied to cumulative sums of B
```
For near-1 gamma, the geometric series converges slowly and needs many terms. For smaller gamma, fewer terms suffice. Content-dependent gamma makes this harder.

Unlikely to be practical for language modeling where gamma varies per position.

## Recommendation for Thesis

Use the 16 x 8x8 parallel scan configuration (V16e, 475ms/step). It provides:
- 1,024 state values (vs 272 for scalar fiber — 3.8x richer)
- Associative memory (q @ S retrieval, not just decayed average)
- Fully parallel (O(T log T))
- Reasonable speed for 10K-20K step experiments

The speed gap to attention (475ms vs 137ms = 3.5x) is primarily implementation overhead (Python parallel scan vs optimized matmul kernels), not algorithmic. A custom kernel would close most of this gap.

For the thesis argument: report the algorithmic complexity (O(n log n) vs O(n^2)) and note that the constant-factor gap is implementation, not fundamental. SPECTRE demonstrated 7x speedup over FlashAttention at 128K context with optimized FFT kernels — similar engineering applied to our matrix scan would yield comparable gains.

## State Capacity Comparison

| Method | State per layer | Retrieval type | Grows with n? |
|---|---|---|---|
| Scalar fiber (V16) | 272 | Decayed average | No (O(1)) |
| Matrix fiber 16x8x8 (V16e) | 1,024 | Associative (q@S) | No (O(1)) |
| Matrix fiber 16x32x32 | 16,384 | Associative (q@S) | No (O(1)) |
| Attention KV cache (GPT) | n x 2d | Exact pairwise | **Yes (O(n))** |
| Attention at n=256, d=224 | 114,688 | Exact pairwise | Yes |
| Attention at n=4096, d=224 | 1,835,008 | Exact pairwise | Yes |

The fundamental advantage of fixed-size state: at n=4096, our 1,024-value state is 1,800x smaller than attention's KV cache. At n=128K (SPECTRE's regime), the ratio is 57,000x. This is where the architecture's scaling advantage lives — not at n=256 where attention is cheap.
