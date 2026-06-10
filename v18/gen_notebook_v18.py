"""Generate architecture_v18.ipynb - Precision-Gated Linear Attention.

V18: Clean-sheet architecture keeping only what earned its keep across V1-V17.
Dense embedding. Precision-weighted matrix fiber. FFN. Learned variance evolution.
No constellation, no irfft, no Parseval filter, no local conv, no Wilson line."""
import json
import os

cells = []

def md(source):
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source_list})

def code(source):
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": source_list,
                  "outputs": [], "execution_count": None})


# ═══════════════════════════════════════════════════════════════
md("""# V18: Precision-Gated Linear Attention

Everything that survived V1-V17, nothing that didn't.

```
Embedding:
  x = token_emb + precision routing (position + content -> which dims are active)

Block (x N):
  LayerNorm(x)
  precision = exp(-log_var)
  effective = x * sigmoid(precision)          # only confident dims contribute
  q, k, v, gamma = project(effective, precision)
  S[t] = gamma * S[t-1] + k v^T              # parallel matrix scan (causal)
  context = q @ S[t-1]                        # associative retrieval
  x = x + context
  x = x + FFN(LayerNorm(x))
  log_var = update(log_var, context)          # learned variance evolution

Decoder:
  logits = Linear(x, vocab)
```

No constellation. No irfft. No Parseval filter. No Wilson line.
No local conv. No subbundles. Dense embedding, single table.
The precision IS the routing mechanism.""")

# ═══════════════════════════════════════════════════════════════
code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm.auto import tqdm
import math
import time
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

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
print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens, Vocab: {vocab_size:,}")""")

# ═══════════════════════════════════════════════════════════════
code("""@dataclass
class V18Config:
    # Model dimensions
    d_model: int = 256                # token embedding dimension
    n_heads: int = 16                 # matrix fiber heads
    head_dim: int = 8                 # state matrix dxd per head

    # FFN
    ffn_mult: int = 4

    # Model
    n_blocks: int = 8
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    lr_hold_steps: int = 3000
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 20000
    eval_interval: int = 500
    eval_steps: int = 10

cfg = V18Config(vocab_size=vocab_size)
print(f"V18: d_model={cfg.d_model} heads={cfg.n_heads} head_dim={cfg.head_dim}")
print(f"State: {cfg.n_heads} x {cfg.head_dim}x{cfg.head_dim} = {cfg.n_heads * cfg.head_dim**2} values")
print(f"Blocks: {cfg.n_blocks}, FFN mult: {cfg.ffn_mult}")
print(f"LR: {cfg.learning_rate}, Steps: {cfg.max_steps}")

def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i+c.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
md("## Architecture")

code("""def matrix_parallel_scan(A_diag, B_mat):
    \"\"\"Parallel scan: S[t] = a[t] * S[t-1] + B[t]. O(T log T).
    A_diag: (N, T) scalar decays. B_mat: (N, T, d, d) deposits.\"\"\"
    N, T, d, _ = B_mat.shape
    a = A_diag
    b = B_mat
    for s in range(int(math.ceil(math.log2(T)))):
        step = 2 ** s
        if step >= T: break
        a_r = a[:, step:].unsqueeze(-1).unsqueeze(-1)
        b = torch.cat([b[:, :step], a_r * b[:, :-step] + b[:, step:]], dim=1)
        a = torch.cat([a[:, :step], a[:, step:] * a[:, :-step]], dim=1)
    return b


class PrecisionEmbedding(nn.Module):
    \"\"\"Dense token embedding + learned positional precision.
    Position is encoded as WHICH DIMS ARE ACTIVE, not as an additive vector.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, D)   # standard pos embedding
        self.pos_prec = nn.Embedding(cfg.max_seq_len, D)   # positional precision template
        self.prec_mix = nn.Linear(2 * D, D)                # content + position -> precision
        nn.init.zeros_(self.pos_prec.weight)
        nn.init.zeros_(self.prec_mix.weight)
        nn.init.zeros_(self.prec_mix.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.drop(x)
        # Precision: which dims are active at this (content, position)
        pp = self.pos_prec(pos).unsqueeze(0).expand(B, -1, -1)
        log_var = self.prec_mix(torch.cat([self.tok_emb(token_ids), pp], dim=-1))
        return x, log_var


class PrecisionFiber(nn.Module):
    \"\"\"Precision-gated matrix-valued linear attention.
    High-precision dims deposit strongly and query precisely.
    Low-precision dims are silent. Routing emerges from precision overlap.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        d = cfg.head_dim
        H = cfg.n_heads
        # Input: effective_x (D) + precision (D) = 2D
        self.proj = nn.Linear(2 * D, H * (3 * d + 1))
        self.out_proj = nn.Linear(H * d, D)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.H = H
        self.d = d

    def forward(self, x, log_var):
        B, T, D = x.shape
        d = self.d
        H = self.H

        # Precision gates the input
        precision = torch.exp(-log_var)
        prec_gate = torch.sigmoid(precision - 1.0)
        effective = x * prec_gate

        # Project to q, k, v, decay
        inp = torch.cat([effective, prec_gate], dim=-1)
        proj = self.proj(inp).reshape(B, T, H, 3 * d + 1)

        q = F.elu(proj[..., :d]) + 1
        k = F.elu(proj[..., d:2*d]) + 1
        v = proj[..., 2*d:3*d]
        gamma = torch.sigmoid(proj[..., 3*d:]).squeeze(-1)

        # Deposit: k outer v
        kv = torch.einsum('bthf,bthe->bthfe', k, v)

        # Parallel matrix scan
        kv_flat = kv.permute(0, 2, 1, 3, 4).reshape(B * H, T, d, d)
        gamma_flat = gamma.permute(0, 2, 1).reshape(B * H, T)
        S_all = matrix_parallel_scan(gamma_flat, kv_flat)

        # Causal shift
        S_all = F.pad(S_all[:, :-1], (0, 0, 0, 0, 1, 0))
        S_all = S_all.reshape(B, H, T, d, d)

        # Query
        q_perm = q.permute(0, 2, 1, 3)
        output = torch.einsum('bhtd,bhtde->bhte', q_perm, S_all)
        output = output.permute(0, 2, 1, 3).reshape(B, T, H * d)

        return torch.sigmoid(self.gate) * self.out_proj(output)


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(D, D * cfg.ffn_mult),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D * cfg.ffn_mult, D),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)


class VarianceUpdate(nn.Module):
    \"\"\"Learned variance evolution based on retrieved context.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Tanh(),
        )
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, log_var, context):
        delta = self.net(torch.cat([context, log_var], dim=-1))
        return (log_var + 0.1 * delta).clamp(min=-6, max=2)


class V18Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.norm1 = nn.LayerNorm(D)
        self.fiber = PrecisionFiber(cfg)
        self.norm2 = nn.LayerNorm(D)
        self.ffn = FFN(cfg)
        self.var_update = VarianceUpdate(cfg)

    def forward(self, x, log_var):
        # Sub-layer 1: precision-gated linear attention
        normed = self.norm1(x)
        context = self.fiber(normed, log_var)
        x = x + context

        # Sub-layer 2: FFN
        x = x + self.ffn(self.norm2(x))

        # Variance evolution
        log_var = self.var_update(log_var, context)

        return x, log_var


class V18Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = PrecisionEmbedding(cfg)
        self.blocks = nn.ModuleList([V18Block(cfg) for _ in range(cfg.n_blocks)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, token_ids):
        x, log_var = self.embedding(token_ids)
        for block in self.blocks:
            x, log_var = block(x, log_var)
        logits = self.head(self.norm_f(x))[:, :-1, :]
        return logits, {}


def count_params(m):
    return sum(p.numel() for p in m.parameters())

_b = V18Block(cfg)
print(f"V18Block: {count_params(_b):,} params")
print(f"  Fiber:  {count_params(_b.fiber):,}")
print(f"  FFN:    {count_params(_b.ffn):,}")
print(f"  VarUpd: {count_params(_b.var_update):,}")""")

# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
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
            torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        hd = self.n_embd // self.n_head
        for blk in self.blocks:
            h = blk['ln1'](x)
            qkv = blk['attn_qkv'](h).reshape(B, T, 3, self.n_head, hd)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            att = (q @ k.transpose(-2,-1)) * (hd**-0.5)
            att = att.masked_fill(self.causal_mask[:,:,:T,:T]==0, float('-inf'))
            y = (F.softmax(att, dim=-1) @ v).transpose(1,2).reshape(B, T, self.n_embd)
            x = x + blk['attn_proj'](y)
            x = x + blk['mlp_fc2'](F.gelu(blk['mlp_fc1'](blk['ln2'](x))))
        return self.lm_head(self.ln_f(x))[:, :-1, :], {}


# Match d_model=256, n_layers=8 for both
models = {}
models['V18'] = V18Model(cfg).to(device)
models['GPT-256d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=256, n_head=8,
                              n_layer=8, block_size=cfg.seq_len).to(device)

print(f"\\n{'Model':<15} {'Total':>10}  {'Blocks':>10}  {'Emb+Dec':>10}")
print('=' * 50)
for name, m in models.items():
    total = count_params(m)
    blk = sum(count_params(b) for b in m.blocks)
    print(f"{name:<15} {total:>10,}  {blk:>10,}  {total-blk:>10,}")""")

# ═══════════════════════════════════════════════════════════════
md("## Training")

code("""@torch.no_grad()
def estimate_loss(model, c):
    model.eval()
    results = {}
    for name, sd in [('train', train_ids), ('val', val_ids)]:
        tot_ce, tot_ok, tot_n = 0., 0, 0
        for _ in range(c.eval_steps):
            b = get_batch(sd, c)
            logits, _ = model(b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
        n = c.eval_steps
        results[name] = {'ce': tot_ce/n, 'acc': tot_ok/tot_n}
    model.train()
    return results

def train_model(model, c, label='model'):
    opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.05)
    mr = c.min_lr / c.learning_rate
    he = c.warmup_steps + c.lr_hold_steps
    def lr_fn(s):
        if s < c.warmup_steps: return s / max(1, c.warmup_steps)
        if s < he: return 1.0
        p = (s - he) / max(1, c.max_steps - he)
        return max(mr, 0.5 * (1.0 + math.cos(math.pi * p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    hist = {'step':[], 'train_ce':[], 'val_ce':[], 'train_acc':[], 'val_acc':[],
            'train_bpc':[], 'val_bpc':[], 'step_times':[], 'per_step_loss':[]}
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
            hist['train_bpc'].append(tr['ce']/math.log(2))
            hist['val_bpc'].append(vl['ce']/math.log(2))
            vl_ppl = math.exp(min(vl['ce'], 20))
            tqdm.write(f"  [{label}] {step:5d} | Val CE:{vl['ce']:.3f} "
                       f"BPC:{vl['ce']/math.log(2):.3f} PPL:{vl_ppl:.1f} Acc:{vl['acc']:.1%}")
        if step >= c.max_steps: break
        st = time.time()
        batch = get_batch(train_ids, c)
        opt.zero_grad()
        logits, _ = model(batch)
        tgt = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        elapsed = time.time() - st
        hist['step_times'].append(elapsed)
        hist['per_step_loss'].append(loss.item())
        if smooth_loss is None: smooth_loss = loss.item()
        else: smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        ppl = math.exp(min(smooth_loss, 20))
        pbar.set_postfix(loss=f"{smooth_loss:.3f}", ppl=f"{ppl:.1f}",
                         bpc=f"{smooth_loss/math.log(2):.2f}",
                         lr=f"{sched.get_last_lr()[0]:.1e}", ms=f"{elapsed*1000:.0f}")
    pbar.close()
    el = time.time() - t0
    ms = np.mean(hist['step_times']) * 1000
    final_ppl = math.exp(min(hist['val_ce'][-1], 20))
    print(f"  {label} DONE: {el/60:.1f}min | BPC:{hist['val_bpc'][-1]:.3f} "
          f"PPL:{final_ppl:.1f} Acc:{hist['val_acc'][-1]:.1%} | {ms:.0f}ms/step")
    hist['avg_step_ms'] = ms; hist['n_params'] = count_params(model)
    return hist""")

# ═══════════════════════════════════════════════════════════════
code("""all_hist = {}
for name, model in models.items():
    all_hist[name] = train_model(model, cfg, label=name)""")

# ═══════════════════════════════════════════════════════════════
md("## Results")

code("""colors = {'V18': 'tab:blue', 'GPT-256d': 'black'}
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V18 (Precision-Gated Linear Attention) vs GPT on WikiText-103',
             fontsize=14, fontweight='bold')
ax = axes[0, 0]
for name, h in all_hist.items():
    ax.plot(h['step'], h['val_bpc'], '-o', color=colors[name], label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val BPC'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[0, 1]
for name, h in all_hist.items():
    ppl = [math.exp(min(ce, 20)) for ce in h['val_ce']]
    ax.plot(h['step'], ppl, '-o', color=colors[name], label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Perplexity'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[0, 2]
for name, h in all_hist.items():
    ax.plot(h['step'], [a*100 for a in h['val_acc']], '-o', color=colors[name],
            label=name, markersize=3)
ax.set_xlabel('Step'); ax.set_title('Val Accuracy %'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1, 0]
w = 100
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ax.plot(range(len(sm)), sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Loss (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1, 1]
for name, h in all_hist.items():
    if len(h['per_step_loss']) > w:
        sm = np.convolve(h['per_step_loss'], np.ones(w)/w, mode='valid')
        ppl_sm = [math.exp(min(x, 20)) for x in sm]
        ax.plot(range(len(ppl_sm)), ppl_sm, '-', color=colors[name], label=name, alpha=0.8)
ax.set_title(f'Step Perplexity (smooth {w})'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1, 2]; ax.axis('off')
rows = [[name, f"{h['n_params']:,}", f"{h['val_bpc'][-1]:.3f}",
         f"{math.exp(min(h['val_ce'][-1],20)):.1f}",
         f"{h['val_acc'][-1]:.1%}", f"{h['avg_step_ms']:.0f}"]
        for name, h in all_hist.items()]
t = ax.table(cellText=rows, colLabels=['Model','Params','BPC','PPL','Acc','ms/step'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.8)
ax.set_title('Final Results', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('v18_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('\\n' + '='*70)
for name, h in all_hist.items():
    ppl = math.exp(min(h['val_ce'][-1], 20))
    print(f"  {name:<15} BPC:{h['val_bpc'][-1]:.3f}  PPL:{ppl:.1f}  "
          f"Params:{h['n_params']:,}  {h['avg_step_ms']:.0f}ms/step")""")

# ═══════════════════════════════════════════════════════════════
code("""@torch.no_grad()
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
            print(f"  {name}: {text[len(prompt):len(prompt)+100]}")
        except Exception as e:
            print(f"  {name}: error - {e}")""")

# ═══════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells,
}
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v18.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {outpath} with {len(cells)} cells")
import ast
errs = 0
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        try: ast.parse(''.join(c['source']))
        except SyntaxError as e:
            print(f"SYNTAX ERROR cell {i}: {e}"); errs += 1
if errs == 0:
    print(f"All {sum(1 for c in cells if c['cell_type']=='code')} code cells parse OK")
