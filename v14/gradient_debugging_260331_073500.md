# Gradient Debugging Log — 260331 073500

## The Bug

After implementing V14 (Wilson fiber + Langevin settler + proximal sparsity), the fiber gradient norm was **exactly 0.0000** across all 8 blocks. The settler had healthy gradients (0.55 total), but the fiber — the entire gauge transport mechanism — was receiving no learning signal.

## Discovery

First forward+backward pass on the full model:

```
Fiber grad norm:   0.0000
Settler grad norm: 0.5209
Fiber/Settler ratio: 0.0000
```

The Wilson fiber has ~840K params (47% of model) contributing nothing to learning.

## Diagnosis Step 1: Is the computation graph connected?

Tested `messages.requires_grad` and `messages.grad_fn` after running the fiber:

```
messages.requires_grad: True
messages.grad_fn: <AddBackward0 object at 0x11eac3dc0>
```

The messages tensor IS in the autograd graph. The graph isn't disconnected.

## Diagnosis Step 2: Direct loss from fiber output

Fed `ctx = msg_proj(messages)` directly into a test loss (`ctx.sum().backward()`):

```
fiber.base_decay: grad norm = 557.008362
fiber.wilson_proj.2.weight: grad norm = 6355889.500000
fiber.wilson_proj.2.bias: grad norm = 3437.449951
msg_proj.weight: grad norm = 124781.921875
```

**Gradients flow perfectly when the loss depends directly on the fiber output.** The bug is downstream — somewhere between `ctx` and the final loss, the gradient dies.

## Diagnosis Step 3: Single block + decoder (full loss path)

Ran one block manually, decoded to logits, computed CE loss, backpropagated:

```
fiber.base_decay: grad = 0.000000
fiber.wilson_proj.0.weight: grad = 0.000000
settler.memory: grad = 0.000014
settler.gate: grad = 0.123631
settler.msg_proj.weight: grad = 0.000000
```

The gate gets gradients (0.12), the memory gets a tiny gradient (1.4e-5), but **msg_proj.weight is exactly 0**. Since msg_proj is the only path from messages to the settler output, zero msg_proj gradient means zero fiber gradient.

## Root Cause Analysis

The gradient path from loss to fiber goes through:

```
loss → decoder → constellation → d_mag/d_phase → gate × delta → (x - x0) → Langevin loop → ctx → msg_proj → messages → fiber
```

Inside the Langevin loop, the path from `ctx` to `x` passes through:

```
ctx → (x + ctx) → F.normalize → q → scores = β·(q @ m_norm.T) → softmax → weights → (weights @ memory) → attractor → (attractor - x) → η·grad → x
```

### The compounding attenuation

Each link in this chain has a small multiplicative factor at initialization:

| Factor | Value | Source |
|---|---|---|
| `msg_proj` init | std=0.01 | `nn.init.normal_(weight, std=0.01)` |
| `F.normalize` Jacobian | O(1/‖v‖) | Projection onto tangent plane |
| Softmax Jacobian | ~1/256 | Near-uniform over 256 atoms (random memory) |
| Memory values | std=0.02 | `randn * 0.02` initialization |
| `langevin_eta` | 0.3 | Step size |
| Gate | 0.12 | `sigmoid(-2.0)` |

**Combined attenuation**: 0.01 × 1 × (1/256) × 0.02 × 0.3 × 0.12 ≈ **2.8 × 10⁻⁹**

This is below float32 precision (~1.2 × 10⁻⁷). The gradient is mathematically nonzero but numerically zero.

### Why the gate gets gradients but msg_proj doesn't

The gate's gradient path is: `loss → d_mag → gate × delta → gate`. This only passes through the gate itself — one multiplication. The msg_proj path passes through the entire Langevin chain.

### Why the direct `ctx.sum()` test worked

That test bypassed the Langevin chain entirely. The gradient went `loss → ctx → msg_proj → messages → fiber` with no attenuation.

## First Fix Attempt: Non-zero msg_proj init

Changed from `nn.init.zeros_(self.msg_proj.weight)` to `nn.init.normal_(self.msg_proj.weight, std=0.01)`.

**Result**: Still 0.0000. The zero-init was one factor, but removing it only changed one of the six compounding factors. The product was still below float32.

## Solution: Direct ctx skip connection

Added `ctx` directly to the output delta, bypassing the Langevin chain entirely:

```python
# Before (broken):
delta = x - x0

# After (fixed):
delta = (x - x0) + ctx
```

The `ctx = msg_proj(messages)` now enters the output **both** through the Langevin query (nonlinear path) **and** through a direct additive skip (linear path).

New gradient path for the skip: `loss → d_mag → gate × delta → ctx → msg_proj → messages → fiber`

Attenuation: `msg_proj_std × gate` = 0.01 × 0.12 ≈ 1.2 × 10⁻³. Well above float32 precision.

**Result**:

```
Fiber grad norm:   1.0438
Settler grad norm: 0.5525
Fiber/Settler ratio: 1.8892

Per-block:
  Block 1: fiber=0.1585 settler=0.0837
  Block 2: fiber=0.1428 settler=0.0788
  Block 3: fiber=0.1408 settler=0.0743
  Block 4: fiber=0.1435 settler=0.0701
  Block 5: fiber=0.1164 settler=0.0663
  Block 6: fiber=0.1232 settler=0.0628
  Block 7: fiber=0.1027 settler=0.0597
  Block 8: fiber=0.1160 settler=0.0568
```

Healthy gradients across all 8 blocks, fiber/settler ratio ~1.9.

## Architectural Consequence

The `ctx` skip serves double duty:
1. **Biases the Langevin query** — modulates which memory attractors are contextually relevant (nonlinear path)
2. **Provides direct fiber signal** — linear projection of messages feeds directly into the block output (linear path)

As training progresses:
- The Langevin path strengthens (memory atoms learn meaningful patterns, softmax becomes peaked)
- The direct path provides immediate gradient flow from step 0
- Both contribute to the final constellation update

This is analogous to residual connections in transformers — the skip connection ensures gradient flow even when the main path (Langevin settling) has high attenuation.

## Lesson

When chaining multiple attenuating operations (normalization → softmax over many atoms → small-init parameters → gating), check the **product** of all multiplicative factors against float32 precision. Each factor may seem reasonable alone, but six factors of 0.01-0.3 compound to ~10⁻⁹. The fix is a direct skip connection that bypasses the attenuating chain.
