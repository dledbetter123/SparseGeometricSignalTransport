# Finding: Learning Rate Scales with State Capacity

**Date**: 2026-04-01

## Observation

Attention models (GPT-224d) train stably at higher learning rates (3e-3, 6e-4) on WikiText-103. Our matrix fiber architecture (V16e) requires lower learning rates (1e-4) to train without instability.

## Analysis

Attention's KV cache stores 114,688 values per layer — the full history at full precision. Each gradient step adjusts how the model routes through this high-dimensional state space. Because the state is large and the routing is via softmax (which distributes gradient smoothly across all stored values), each parameter update makes small, distributed changes across many memories. The large state absorbs perturbations. Higher learning rates work because the system has high effective dimensionality — each step moves through a large space where individual directions matter less.

Our matrix fiber stores 1,024 values per layer (16 heads x 8x8). This is 112x smaller than attention's state. Every gradient step adjusts the same compact state that ALL tokens read from and write to. A large learning rate causes large changes to this shared state, which cascades — every position's output changes when S changes. The small state amplifies perturbations. Lower learning rates are necessary because each parameter update has outsized effect on the shared, compact representation.

Put differently: attention's gradients are diluted across 114K state values. Our gradients are concentrated into 1K state values. The same learning rate produces 112x more impact per state element in our architecture.

## The Constraint

This is a compute bottleneck, not an architectural limitation. The matrix fiber's state capacity is limited to 1,024 values by the parallel scan's O(T log T x d^2) cost on MPS hardware. At d=32 (16,384 state values), step time increases from 475ms to 1,400ms — impractical for 20K step experiments. At d=64, it's 3,700ms.

With custom CUDA/Metal kernels (Mamba-style), the scan would be 5-10x faster, allowing d=32 or d=64 at practical speeds. This would both increase state capacity and allow higher learning rates — the two are linked.

## Implication

When comparing our architecture to attention:
- The learning rate difference (1e-4 vs 3e-3) is not a hyperparameter tuning failure
- It reflects the fundamental relationship between state capacity and gradient sensitivity
- At matched state capacity (requiring optimized kernels or larger hardware), the learning rate gap would narrow
- For thesis: report this as an expected consequence of compact state, not a weakness

## Recommendation

For current experiments (MPS, Python parallel scan):
- V16e: lr=1e-4, warmup=1000, hold=3000
- GPT-224d: lr=1e-4 (same schedule for fair comparison, even though GPT could go higher)

For future experiments (with kernel optimization):
- Scale fiber_key_dim from 8 to 32
- Test lr=3e-4 or 6e-4 with larger state
- The hypothesis: lr scales roughly as sqrt(state_capacity / baseline_capacity)
