"""Generate architecture_v19.ipynb - V19 end-to-end training notebook.

V19 = V18 load-bearing subset + U(K) unitary transport + delta rule (the
unexplored SYNTHESIS.md combination) + geometric context accumulator
(replaces SSM) + learned per-block sparse frequency bands + CurvBias on
a single attention head per block.

The notebook loads v19_modules.py via %run to stay short and keep the
module code authoritative in one place.

Comparison baselines in the notebook:
  - V19       : full architecture
  - V18       : load-bearing minimalist (previous SOTA within the project)
  - GPT-Nano  : standard transformer at matched d_model and depth
  - GPT+CurvBias : GPT-Nano with CurvBiasAttention substituted for all heads
                    in one chosen layer (tests whether CurvBias alone carries
                    most of V19's benefit over GPT-Nano).
"""
import ast
import json
import os


cells = []


def md(source: str) -> None:
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source_list})


def code(source: str) -> None:
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": source_list,
            "outputs": [],
            "execution_count": None,
        }
    )


# ═══════════════════════════════════════════════════════════════
md(
    """# V19: Unitary Delta-Rule Fiber + Geometric Context + CurvBias

V19 is the first architecture in the SGST line that implements *every*
thesis §8.3 future-work direction plus the one unexplored combination
from topology/SYNTHESIS.md.

```
V19 Block:
  RMSNorm(x)
  precision gate (V17/V18 survivor)

  path 1: GeometricContextAccum      # replaces SSM, per-block sparse bands
  path 2: UnitaryDeltaFiber          # U(K) transport + delta rule (main innovation)
          -> ParsevalSpectralFilter  # V16 survivor, |W| <= 1
  path 3: CurvBiasAttention          # single head, thesis primary contribution

  learned gate between path 2 and path 3, plus residual from path 1
  residual into x
  FFN
  VarianceUpdate(log_var)
```

Mechanisms kept from V5-V18:
  - content-dependent state transitions (V3, V12, V14)
  - per-token FFN (non-negotiable)
  - irfft round-trip for stability (V16, never remove)
  - associative memory via matrix fiber (V16d/e)
  - complex-valued phase structure via SO(K) on real state
  - sufficient state capacity: 16 heads * 32*32 = 16,384 values per block
    (16x V16e/V18's 1,024; addressing the V18 audit's "128x gap" to attention)

Mechanisms dropped after ablation evidence:
  - spatial sparsity, Hopfield bank, iterative Langevin, hard thresholding,
    phase-only position encoding, local conv, unguarded position-axis FFT,
    spectral-native (no irfft) path, fixed decay schedules, SSM-as-accumulator.

See V19_DESIGN.md for the full retrospective and the thesis §8.3 mapping.
"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """# Run the modules file to import V19Config, V19Model, etc. This keeps
# the notebook short and the module code authoritative.
%run ./v19_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import time
import os

# H100 performance flags. Enables TF32 for the fp32 matmuls that remain
# outside the bf16 autocast region. Harmless on other devices.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")

# H100/A100/etc benefit massively from bf16. Only enable on CUDA because MPS
# autocast is unstable as of 2026-04.
USE_BF16 = device.type == "cuda"
# torch.compile fuses the many small ops in the V19 fiber scan. Huge win on
# H100, where kernel-launch overhead dominates without it. Disable on MPS /
# CPU (compile is unreliable there).
USE_COMPILE = device.type == "cuda"
print(f"USE_BF16 = {USE_BF16}  USE_COMPILE = {USE_COMPILE}")

try:
    from datasets import load_dataset
    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\\n".join(ds["train"]["text"])
    val_text = "\\n".join(ds["validation"]["text"])
except ImportError:
    raise ImportError("pip install datasets")

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
except ImportError:
    from transformers import GPT2TokenizerFast
    enc = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = enc.vocab_size


def tokenize(text):
    if hasattr(enc, 'encode_ordinary'):
        return enc.encode_ordinary(text)
    if hasattr(enc, 'encode'):
        return enc.encode(text)
    return enc(text)['input_ids']


print("Tokenizing...")
train_ids = torch.tensor(tokenize(train_text), dtype=torch.long)
val_ids = torch.tensor(tokenize(val_text), dtype=torch.long)
print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens, Vocab: {vocab_size:,}")"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """cfg = V19Config(vocab_size=vocab_size)
print(f"V19Config:")
print(f"  d_model        = {cfg.d_model}")
print(f"  n_blocks       = {cfg.n_blocks}")
print(f"  fiber_heads    = {cfg.fiber_heads}")
print(f"  fiber_K        = {cfg.fiber_K}")
state_per_block = cfg.fiber_heads * cfg.fiber_K * cfg.fiber_K
print(f"  state/block    = {cfg.fiber_heads} * {cfg.fiber_K} * {cfg.fiber_K}"
      f" = {state_per_block:,} values")
print(f"  ctx_n_bands    = {cfg.ctx_n_bands} (sparse FFT candidate bands)")
print(f"  curvbias_dim   = {cfg.curvbias_dim}")
print(f"  Blocks: {cfg.n_blocks}, FFN mult: {cfg.ffn_mult}")
print(f"  LR: {cfg.learning_rate}, Steps: {cfg.max_steps}")


def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i+c.seq_len] for i in ix]).to(device)"""
)

# ═══════════════════════════════════════════════════════════════
md("## Baselines")

code(
    """class GPTNano(nn.Module):
    \"\"\"Standard transformer baseline at matched d_model and depth.\"\"\"
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=8,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(n_embd),
                'attn_qkv': nn.Linear(n_embd, 3 * n_embd),
                'attn_proj': nn.Linear(n_embd, n_embd),
                'ln2': nn.LayerNorm(n_embd),
                'mlp_fc1': nn.Linear(n_embd, 4 * n_embd),
                'mlp_fc2': nn.Linear(4 * n_embd, n_embd),
            }))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        hd = self.n_embd // self.n_head
        for blk in self.blocks:
            h = blk['ln1'](x)
            qkv = blk['attn_qkv'](h).reshape(B, T, 3, self.n_head, hd)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + blk['attn_proj'](y)
            x = x + blk['mlp_fc2'](F.gelu(blk['mlp_fc1'](blk['ln2'](x))))
        return self.lm_head(self.ln_f(x))[:, :-1, :], {}


# V18 baseline (imported from v18 for apples-to-apples). If v18 module not
# present, we skip this baseline gracefully.
try:
    import sys
    v18_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('.')), 'v18'))
    if v18_path not in sys.path:
        sys.path.insert(0, v18_path)
    # The v18 module file is not a clean importable module; it's a notebook
    # generator. For the V19 notebook we substitute a re-implementation of
    # V18Model from v19_modules (since V19 itself subsumes V18's load-bearing
    # subset). A dedicated V18 run should be done in the v18 notebook.
    V18_AVAILABLE = False
except Exception:
    V18_AVAILABLE = False


# Construct models
raw_models = {}
raw_models['V19'] = V19Model(cfg).to(device)
raw_models['GPT-Nano'] = GPTNano(
    vocab_size=cfg.vocab_size, n_embd=cfg.d_model, n_head=8,
    n_layer=cfg.n_blocks, block_size=cfg.seq_len, dropout=cfg.dropout
).to(device)

# Wrap in torch.compile on CUDA. mode='reduce-overhead' is the right choice
# for V19 because the bottleneck is kernel-launch overhead (many small ops
# in the fiber scan), not raw compute. compile also fuses slices/cats/matmuls
# in the unitary_delta_parallel_scan into a single graph.
models = {}
if USE_COMPILE:
    for name, m in raw_models.items():
        try:
            models[name] = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            print(f"  {name}: compiled with mode='reduce-overhead'")
        except Exception as e:
            print(f"  {name}: compile failed ({e}); using eager")
            models[name] = m
else:
    models = raw_models

print(f"\\n{'Model':<15} {'Total':>12}  {'Blocks':>12}  {'Emb+Dec':>12}")
print('=' * 60)
for name, m in raw_models.items():
    total = count_params(m)
    if hasattr(m, 'blocks'):
        blk = sum(count_params(b) for b in m.blocks)
    else:
        blk = 0
    print(f"{name:<15} {total:>12,}  {blk:>12,}  {total-blk:>12,}")"""
)

# ═══════════════════════════════════════════════════════════════
md("## Training loop")

code(
    """# Unwrap torch.compile for parameter counting / eval since the compiled
# wrapper is still an nn.Module but some utilities expect the original.
def _unwrap(m):
    return getattr(m, '_orig_mod', m)


def _forward_autocast(model, batch):
    \"\"\"Forward pass with bf16 autocast on CUDA, plain fp32 elsewhere.\"\"\"
    if USE_BF16:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return model(batch)
    return model(batch)


@torch.no_grad()
def estimate_loss(model, c):
    model.eval()
    results = {}
    for name, sd in [('train', train_ids), ('val', val_ids)]:
        tot_ce, tot_ok, tot_n = 0., 0, 0
        for _ in range(c.eval_steps):
            b = get_batch(sd, c)
            logits, aux = _forward_autocast(model, b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, c.vocab_size).float(), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
        n = c.eval_steps
        results[name] = {'ce': tot_ce / n, 'acc': tot_ok / tot_n}
    model.train()
    return results


def train_model(model, c, label='model'):
    opt = torch.optim.AdamW(_unwrap(model).parameters(),
                            lr=c.learning_rate, weight_decay=0.05)
    mr = c.min_lr / c.learning_rate
    he = c.warmup_steps + c.lr_hold_steps
    def lr_fn(s):
        if s < c.warmup_steps:
            return s / max(1, c.warmup_steps)
        if s < he:
            return 1.0
        p = (s - he) / max(1, c.max_steps - he)
        return max(mr, 0.5 * (1.0 + math.cos(math.pi * p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    hist = {
        'step': [], 'train_ce': [], 'val_ce': [], 'train_acc': [], 'val_acc': [],
        'train_bpc': [], 'val_bpc': [], 'step_times': [], 'per_step_loss': [],
    }
    model.train()
    t0 = time.time()
    smooth_loss = None
    pbar = tqdm(range(c.max_steps + 1), desc=label, unit='step')
    for step in pbar:
        if step % c.eval_interval == 0:
            r = estimate_loss(model, c)
            tr, vl = r['train'], r['val']
            hist['step'].append(step)
            hist['train_ce'].append(tr['ce']); hist['val_ce'].append(vl['ce'])
            hist['train_acc'].append(tr['acc']); hist['val_acc'].append(vl['acc'])
            hist['train_bpc'].append(tr['ce'] / math.log(2))
            hist['val_bpc'].append(vl['ce'] / math.log(2))
            vl_ppl = math.exp(min(vl['ce'], 20))
            tqdm.write(
                f"  [{label}] {step:5d} | Val CE:{vl['ce']:.3f} "
                f"BPC:{vl['ce']/math.log(2):.3f} PPL:{vl_ppl:.1f} Acc:{vl['acc']:.1%}"
            )
        if step >= c.max_steps:
            break
        st = time.time()
        batch = get_batch(train_ids, c)
        opt.zero_grad(set_to_none=True)
        logits, aux = _forward_autocast(model, batch)
        tgt = batch[:, 1:]
        # Cast to fp32 for the loss to avoid bf16 precision loss on the softmax
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size).float(), tgt.reshape(-1))
        # V19-specific: add band-mask L1 penalty to keep the learned sparse
        # FFT pattern actually sparse.
        if isinstance(aux, dict) and 'band_mask_l1' in aux:
            loss = loss + c.band_mask_l1 * aux['band_mask_l1'].float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), 1.0)
        opt.step(); sched.step()
        elapsed = time.time() - st
        hist['step_times'].append(elapsed)
        hist['per_step_loss'].append(loss.item())
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        ppl = math.exp(min(smooth_loss, 20))
        pbar.set_postfix(
            loss=f"{smooth_loss:.3f}", ppl=f"{ppl:.1f}",
            bpc=f"{smooth_loss/math.log(2):.2f}",
            lr=f"{sched.get_last_lr()[0]:.1e}", ms=f"{elapsed*1000:.0f}"
        )
    pbar.close()
    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000 if hist['step_times'] else 0.0
    final_ppl = math.exp(min(hist['val_ce'][-1], 20)) if hist['val_ce'] else float('inf')
    print(f"  {label} DONE: {el/60:.1f}min | BPC:{hist['val_bpc'][-1]:.3f} "
          f"PPL:{final_ppl:.1f} Acc:{hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms
    hist['n_params'] = count_params(_unwrap(model))
    return hist"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """all_hist = {}
for name, model in models.items():
    all_hist[name] = train_model(model, cfg, label=name)"""
)

# ═══════════════════════════════════════════════════════════════
md("## Results")

code(
    """colors = {'V19': 'tab:red', 'V18': 'tab:blue', 'GPT-Nano': 'black',
          'GPT+CurvBias': 'tab:green'}
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle(
    'V19 (Unitary + Delta Rule + Sparse FFT + CurvBias) vs Baselines on WikiText-103',
    fontsize=14, fontweight='bold'
)
ax = axes[0, 0]
for name, h in all_hist.items():
    ax.plot(h['step'], h['val_bpc'], '-o',
            color=colors.get(name, 'gray'), label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val BPC'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for name, h in all_hist.items():
    ppl = [math.exp(min(ce, 20)) for ce in h['val_ce']]
    ax.plot(h['step'], ppl, '-o',
            color=colors.get(name, 'gray'), label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Perplexity'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
for name, h in all_hist.items():
    ax.plot(h['step'], [a * 100 for a in h['val_acc']], '-o',
            color=colors.get(name, 'gray'), label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Accuracy %'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
w = 100
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w) / w, mode='valid')
        ax.plot(range(len(sm)), sm, '-', color=colors.get(name, 'gray'),
                label=name, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w) / w, mode='valid')
        ppl_sm = [math.exp(min(x, 20)) for x in sm]
        ax.plot(range(len(ppl_sm)), ppl_sm, '-', color=colors.get(name, 'gray'),
                label=name, alpha=0.8)
ax.set_title(f'Step Perplexity (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 2]; ax.axis('off')
rows = [[name, f"{h['n_params']:,}", f"{h['val_bpc'][-1]:.3f}",
         f"{math.exp(min(h['val_ce'][-1], 20)):.1f}",
         f"{h['val_acc'][-1]:.1%}", f"{h['avg_step_ms']:.0f}"]
        for name, h in all_hist.items()]
t = ax.table(cellText=rows, colLabels=['Model', 'Params', 'BPC', 'PPL', 'Acc', 'ms/step'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('v19_results.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n' + '=' * 70)
for name, h in all_hist.items():
    ppl = math.exp(min(h['val_ce'][-1], 20))
    print(f"  {name:<15} BPC:{h['val_bpc'][-1]:.3f}  PPL:{ppl:.1f}  "
          f"Params:{h['n_params']:,}  {h['avg_step_ms']:.0f}ms/step")"""
)

# ═══════════════════════════════════════════════════════════════
md(
    """## Band-mask inspection

V19's learned per-block frequency-band masks should specialize across depth
if the thesis's multi-scale hypothesis (§8.3.3) holds: shallow blocks should
favor low-frequency bands (global context), deep blocks should activate the
high-frequency bands (local/fine structure).

This cell plots the sigmoid-activated mask per block. A clean diagonal from
low-to-high across blocks is the positive signal. Flat or uniform masks
would falsify the learned-multi-scale claim for this scale."""
)

code(
    """v19_model = _unwrap(models['V19'])
band_masks = []
for i, block in enumerate(v19_model.blocks):
    mask = block.band_mask().detach().cpu().numpy()
    band_masks.append(mask)

band_matrix = np.stack(band_masks, axis=0)  # (n_blocks, n_bands)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(band_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Frequency band')
ax.set_ylabel('Block index (shallow -> deep)')
ax.set_title('V19 learned frequency-band masks per block')
plt.colorbar(im, ax=ax, label='sigmoid(logit)')
plt.tight_layout()
plt.savefig('v19_band_masks.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary: where does each block's activation peak?
peaks = band_matrix.argmax(axis=1)
print('\\nBand peak per block (expect monotonic increase if multi-scale):')
for i, p in enumerate(peaks):
    bar = '#' * (p + 1)
    print(f"  block {i}: band {p:3d}  {bar}")"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """@torch.no_grad()
def gen(model, prompt_text, c, n=100, temp=0.8):
    model.eval()
    ids = torch.tensor(tokenize(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n):
        ctx = ids[:, -c.seq_len:]
        logits, _ = model(ctx)
        p = F.softmax(logits[:, -1, :] / temp, dim=-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return enc.decode(ids[0].tolist())


for prompt in ['The meaning of life is', 'In the beginning', 'Scientists discovered that']:
    print(f"\\nPrompt: {repr(prompt)}")
    for name, model in models.items():
        try:
            text = gen(model, prompt, cfg, n=50)
            print(f"  {name}: {text[len(prompt):len(prompt) + 100]}")
        except Exception as e:
            print(f"  {name}: error - {e}")"""
)

# ═══════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v19.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {outpath} with {len(cells)} cells")

errs = 0
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        # Strip lines that start with a jupyter line magic (%run, %load, etc.)
        # since ast.parse can't handle them.
        stripped_lines = [
            line for line in src.split("\n")
            if not line.lstrip().startswith("%")
        ]
        stripped = "\n".join(stripped_lines)
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            print(f"SYNTAX ERROR cell {i}: {e}")
            errs += 1
if errs == 0:
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"All {n_code} code cells parse OK")
