"""Generate architecture_v20.ipynb — the V20 A8-vs-A10 comparison.

Two conditions only:
    A8  V20 full                   — shared SpatialMLP (matches V12.1)
    A10 V20 strict-subsp           — PerSubbundleMLP (strict orthogonal subbundles)

Both use bf16 autocast on CUDA for speed. torch.compile is deliberately
not used — in earlier runs the compile path produced failures and did
not provide a measurable speedup over eager + bf16 autocast on H100 for
this architecture. TF32 matmul flags stay on.
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
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source_list,
        "outputs": [],
        "execution_count": None,
    })


# ═══════════════════════════════════════════════════════════════
md(
    """# V20: A8 vs A10 — Shared vs Strict-Orthogonal FFN

Two conditions, both V20 full (sparse-spectral constellation + non-abelian
$SO(K)$ transport + forward-reverse FFT loop + fiber + SpatialMLP):

| | FFN | Cross-subbundle mixing |
|---|---|---|
| **A8 V20 full** | `SpatialMLP` — single $\\text{fiber\\_dim} \\to 4\\cdot\\text{fiber\\_dim} \\to \\text{fiber\\_dim}$ | yes (matches V12.1) |
| **A10 V20 strict-subsp** | `PerSubbundleMLP` — $n_\\text{sub}$ independent $K \\to 4K \\to K$ | no (strict orthogonal) |

A8 has ~8× more FFN parameters per block but allows information to flow
across subbundles at the FFN step. A10 enforces the thesis's "orthogonal
subbundle decomposition" claim strictly — subbundles only interact
through the spectral transport kernel's $q$ dependence.

Both paths use `bf16` autocast on CUDA. `torch.compile` is not used.
"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """%run ./v20_modules.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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

USE_BF16 = device.type == "cuda"
print(f"USE_BF16 = {USE_BF16}")"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """try:
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
    """cfg = V20Config(vocab_size=vocab_size)
print(f"V20Config defaults:")
print(f"  n_subbundles       = {cfg.n_subbundles}")
print(f"  subbundle_dim      = {cfg.subbundle_dim}")
print(f"  spectral_half_dim  = {cfg.spectral_half_dim}")
print(f"  n_modes            = {cfg.n_modes}")
print(f"  fiber_dim          = {cfg.fiber_dim}")
print(f"  fiber_K            = {cfg.fiber_K}")
print(f"  active_modes/sub   = {cfg.active_modes_per_sub}")
print(f"  n_blocks           = {cfg.n_blocks}")
print(f"  state per block    = {cfg.n_subbundles} * {cfg.fiber_K}^2"
      f" = {cfg.n_subbundles * cfg.fiber_K ** 2:,} values")


def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i + c.seq_len] for i in ix]).to(device)"""
)

# ═══════════════════════════════════════════════════════════════
md("## Training loop with bf16 autocast")

code(
    """def _forward_autocast(model, batch):
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
    opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.05)
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
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size).float(), tgt.reshape(-1))
        if isinstance(aux, dict) and 'band_mask_l1' in aux:
            loss = loss + c.band_mask_l1 * aux['band_mask_l1'].float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    hist['n_params'] = sum(p.numel() for p in model.parameters())
    return hist"""
)

# ═══════════════════════════════════════════════════════════════
md("## Build A8 and A10")

code(
    """def build_a8():
    \"\"\"V20 full — shared SpatialMLP (matches V12.1).\"\"\"
    return V20Model(cfg).to(device)


def build_a10():
    \"\"\"V20 with PerSubbundleMLP swapped in for SpatialMLP.
    Strict orthogonal subbundle decomposition end-to-end: the FFN no
    longer mixes information across subbundles.\"\"\"
    model = V20Model(cfg).to(device)
    for blk in model.blocks:
        blk.spatial_mlp = PerSubbundleMLP(cfg).to(device)
    return model


BUILDERS = {
    'A8 V20 full':          build_a8,
    'A10 V20 strict-subsp': build_a10,
}

models = {}
for label, builder in BUILDERS.items():
    print(f"Building {label} ...")
    m = builder()
    models[label] = m
    n = sum(p.numel() for p in m.parameters())
    blk = sum(sum(p.numel() for p in b.parameters()) for b in m.blocks)
    print(f"  {label}: total {n:,}  blocks {blk:,}  emb+head {n - blk:,}")"""
)

# ═══════════════════════════════════════════════════════════════
code(
    """all_hist = {}
for name, model in models.items():
    all_hist[name] = train_model(model, cfg, label=name)"""
)

# ═══════════════════════════════════════════════════════════════
md("## Plot results")

code(
    """colors = {
    'A8 V20 full':          'tab:red',
    'A10 V20 strict-subsp': 'tab:olive',
}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V20: A8 (shared SpatialMLP) vs A10 (PerSubbundleMLP) on WikiText-103',
             fontsize=14, fontweight='bold')

ax = axes[0, 0]
for name, h in all_hist.items():
    ax.plot(h['step'], h['val_bpc'], '-o', color=colors.get(name, 'gray'),
            label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val BPC'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for name, h in all_hist.items():
    ppl = [math.exp(min(ce, 20)) for ce in h['val_ce']]
    ax.plot(h['step'], ppl, '-o', color=colors.get(name, 'gray'),
            label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Perplexity'); ax.set_yscale('log')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
for name, h in all_hist.items():
    ax.plot(h['step'], [a * 100 for a in h['val_acc']], '-o',
            color=colors.get(name, 'gray'), label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Accuracy %')
ax.legend(); ax.grid(True, alpha=0.3)

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
ax.set_title(f'Step PPL (smooth {w})')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 2]; ax.axis('off')
rows = [[name, f"{h['n_params']:,}", f"{h['val_bpc'][-1]:.3f}",
         f"{math.exp(min(h['val_ce'][-1], 20)):.1f}",
         f"{h['val_acc'][-1]:.1%}", f"{h['avg_step_ms']:.0f}"]
        for name, h in all_hist.items()]
t = ax.table(cellText=rows,
             colLabels=['Condition', 'Params', 'BPC', 'PPL', 'Acc', 'ms/step'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('v20_results.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n' + '=' * 70)
for name, h in all_hist.items():
    ppl = math.exp(min(h['val_ce'][-1], 20))
    print(f"  {name:<22} BPC:{h['val_bpc'][-1]:.3f}  PPL:{ppl:7.1f}"
          f"  Params:{h['n_params']:>12,}  {h['avg_step_ms']:.0f}ms/step")"""
)

# ═══════════════════════════════════════════════════════════════
md("## V20 band-mask inspection")

code(
    """# Prefer A8 if it was trained; fall back to A10.
_inspect_label = 'A8 V20 full' if 'A8 V20 full' in models else 'A10 V20 strict-subsp'
v20 = models[_inspect_label]
band_masks = [blk.band_mask().detach().cpu().numpy() for blk in v20.blocks]
band_matrix = np.stack(band_masks, axis=0)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(band_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Frequency mode')
ax.set_ylabel('Block index (shallow -> deep)')
ax.set_title(f'{_inspect_label} learned frequency-mask per block')
plt.colorbar(im, ax=ax, label='sigmoid(logit)')
plt.tight_layout()
plt.savefig('v20_band_masks.png', dpi=150, bbox_inches='tight')
plt.show()

peaks = band_matrix.argmax(axis=1)
print(f'\\n{_inspect_label} band peak per block (expect monotonic if multi-scale):')
for i, p in enumerate(peaks):
    bar = '#' * (p + 1)
    print(f'  block {i}: band {p:3d}  {bar}')"""
)

# ═══════════════════════════════════════════════════════════════
md(
    """## What this comparison tells us

- **If A8 ≫ A10**: cross-subbundle mixing at the FFN step is load-bearing,
  and V12.1's original convention is correct. Subbundles are *not* fully
  independent feature channels.
- **If A10 ≫ A8**: strict orthogonal subbundle decomposition is genuinely
  better, and the 8× extra parameters in A8's shared `SpatialMLP` are
  wasted or actively harmful (noise across subbundles).
- **If A8 ≈ A10**: the FFN variant doesn't matter at this scale. The
  bottleneck is elsewhere (fiber state capacity, transport expressivity,
  or the softplus/gate activation pathway)."""
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
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v20.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {outpath} with {len(cells)} cells")

errs = 0
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        stripped_lines = [
            line for line in src.split("\n") if not line.lstrip().startswith("%")
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
