"""Generate architecture_v15.ipynb — Spectral Attention Transformer.

V15: Standard causal self-attention operating on the constellation representation.
The spectral structure is in the representation (mag, phase embedding, irfft decoder),
not in the mixing mechanism. Attention does what attention does best.

Tests: does the spectral representation itself add value, with a proven mixing mechanism?"""
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
md("""# V15: Spectral Attention Transformer

Stop fighting attention. Use it.

The spectral structure lives in the **representation**, not the mixing:
- **Embedding**: token → (mag, phase) across 136 modes, positional encoding via phase shift
- **Mixing**: standard multi-head causal self-attention on the (mag, phase) flat vector
- **Processing**: standard FFN
- **Decoding**: (mag, phase) → complex → irfft → spatial → vocab logits

This tests the simplest possible question: **does the spectral constellation
representation help, when paired with a mixing mechanism that actually works?**

If V15 matches GPT-224d: the constellation is neutral overhead — no benefit, no harm.
If V15 beats GPT-224d: the spectral representation provides useful inductive bias.
If V15 loses to GPT-224d: the constellation representation actively hurts.""")

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
class V15Config:
    # Spectral
    n_modes: int = 136               # 2M = 272 = attention dim
    n_subbundles: int = 8
    fiber_dim: int = 256

    # Attention
    n_heads: int = 8                 # 272 / 8 = 34 per head
    attn_dim: int = 272              # = 2 * n_modes (mag + phase)

    # FFN
    ffn_mult: int = 4

    # Model
    n_blocks: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    lr_hold_steps: int = 500
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 10000
    eval_interval: int = 500
    eval_steps: int = 10

    @property
    def subbundle_dim(self):
        return self.fiber_dim // self.n_subbundles

    @property
    def spectral_half_dim(self):
        return self.subbundle_dim // 2 + 1

cfg = V15Config(vocab_size=vocab_size)
print(f"V15: vocab={cfg.vocab_size:,} seq={cfg.seq_len} batch={cfg.batch_size}")
print(f"Modes: {cfg.n_modes}, Attn dim: {cfg.attn_dim}, Heads: {cfg.n_heads}")
print(f"Blocks: {cfg.n_blocks}, FFN mult: {cfg.ffn_mult}, LR: {cfg.learning_rate}")

def get_batch(data, c):
    ix = torch.randint(0, len(data) - c.seq_len - 1, (c.batch_size,))
    return torch.stack([data[i:i+c.seq_len] for i in ix]).to(device)""")

# ═══════════════════════════════════════════════════════════════
md("## Components")

code("""class Constellation:
    def __init__(self, mag, phase):
        self.mag = mag
        self.phase = phase
    def to_complex(self):
        return self.mag * torch.exp(1j * self.phase)
    def to_flat(self):
        return torch.cat([self.mag, self.phase], dim=-1)


class MagPhaseNorm(nn.Module):
    \"\"\"RMSNorm on magnitudes, phases untouched.\"\"\"
    def __init__(self, n_modes):
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(n_modes))
    def forward(self, c):
        mag_rms = (c.mag ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return Constellation(c.mag / mag_rms * self.mag_scale, c.phase)


class ConstellationEmbedding(nn.Module):
    \"\"\"Token → (mag, phase). Positional encoding via phase shift.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.mag_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        self.phase_emb = nn.Embedding(cfg.vocab_size, cfg.n_modes)
        nn.init.uniform_(self.phase_emb.weight, -math.pi, math.pi)
        freqs = torch.zeros(cfg.n_modes)
        for k in range(cfg.n_subbundles):
            off = k * cfg.spectral_half_dim
            freqs[off:off+cfg.spectral_half_dim] = (
                2 * math.pi * torch.fft.rfftfreq(cfg.subbundle_dim, d=1.0))
        self.register_buffer('freqs', freqs)
    def forward(self, token_ids):
        B, T = token_ids.shape
        mag = self.mag_emb(token_ids)
        phase = self.phase_emb(token_ids)
        pos = torch.arange(T, device=token_ids.device).float()
        phase = phase + (pos.unsqueeze(-1) * self.freqs).unsqueeze(0)
        return Constellation(mag, phase)


class ConstellationDecoder(nn.Module):
    \"\"\"(mag, phase) → complex → irfft → spatial → vocab logits.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.fiber_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )
    def forward(self, constellation):
        spectral = constellation.to_complex()
        shd = self.cfg.spectral_half_dim
        subs = spectral.reshape(*spectral.shape[:-1], self.cfg.n_subbundles, shd)
        spatial = torch.fft.irfft(subs, n=self.cfg.subbundle_dim, dim=-1)
        spatial = spatial.reshape(*spectral.shape[:-1], self.cfg.fiber_dim)
        return self.head(self.norm(spatial))


def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("Components loaded.")""")

# ═══════════════════════════════════════════════════════════════
md("""## V15 Block: Attention + FFN on Constellation Space

Standard transformer block, but operating on (mag, phase) representation.
The attention QKV are computed from the flat constellation vector.
The output is interpreted as (delta_mag, delta_phase) and added as a residual.""")

code("""class SpectralAttention(nn.Module):
    \"\"\"Multi-head causal self-attention on the constellation flat vector.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        D = cfg.attn_dim  # 272
        self.n_heads = cfg.n_heads
        self.head_dim = D // cfg.n_heads
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        self.register_buffer('causal_mask',
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
                  .view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, H, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(y))


class ConstellationFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.attn_dim
        self.net = nn.Sequential(
            nn.Linear(D, D * cfg.ffn_mult),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D * cfg.ffn_mult, D),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)


class V15Block(nn.Module):
    \"\"\"Pre-norm: attention + FFN on constellation flat vector.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        M = cfg.n_modes
        self.norm1 = MagPhaseNorm(M)
        self.attn = SpectralAttention(cfg)
        self.norm2 = MagPhaseNorm(M)
        self.ffn = ConstellationFFN(cfg)

    def forward(self, constellation):
        M = constellation.mag.shape[-1]
        # Sub-layer 1: attention
        normed = self.norm1(constellation)
        x = normed.to_flat()          # (B, T, 2M)
        delta = self.attn(x)          # (B, T, 2M)
        constellation = Constellation(
            constellation.mag + delta[..., :M],
            constellation.phase + delta[..., M:])
        # Sub-layer 2: FFN
        normed2 = self.norm2(constellation)
        x2 = normed2.to_flat()
        delta2 = self.ffn(x2)
        return Constellation(
            constellation.mag + delta2[..., :M],
            constellation.phase + delta2[..., M:])


class V15Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = ConstellationEmbedding(cfg)
        self.blocks = nn.ModuleList([V15Block(cfg) for _ in range(cfg.n_blocks)])
        self.decoder = ConstellationDecoder(cfg)

    def forward(self, token_ids):
        c = self.embedding(token_ids)
        for block in self.blocks:
            c = block(c)
        logits = self.decoder(c)[:, :-1, :]
        return logits, {}


_b = V15Block(cfg)
print(f"V15Block: {count_params(_b):,} params")
print(f"  Attention: {count_params(_b.attn):,}")
print(f"  FFN:       {count_params(_b.ffn):,}")
print(f"  Norms:     {sum(count_params(n) for n in [_b.norm1, _b.norm2]):,}")""")

# ═══════════════════════════════════════════════════════════════
code("""class GPTNano(nn.Module):
    \"\"\"Standard transformer baseline. Same hyperparams, dense embedding.\"\"\"
    def __init__(self, vocab_size, n_embd=224, n_head=8, n_layer=12,
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


models = {}
models['V15'] = V15Model(cfg).to(device)
models['GPT-224d'] = GPTNano(vocab_size=cfg.vocab_size, n_embd=224, n_head=8,
                              n_layer=12, block_size=cfg.seq_len).to(device)

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

        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        ppl = math.exp(min(smooth_loss, 20))
        pbar.set_postfix(loss=f"{smooth_loss:.3f}", ppl=f"{ppl:.1f}",
                         bpc=f"{smooth_loss/math.log(2):.2f}",
                         lr=f"{sched.get_last_lr()[0]:.1e}",
                         ms=f"{elapsed*1000:.0f}")

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

code("""colors = {'V15': 'tab:blue', 'GPT-224d': 'black'}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('V15 (Spectral Attention) vs GPT-224d on WikiText-103', fontsize=14, fontweight='bold')

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
plt.savefig('v15_results.png', dpi=150, bbox_inches='tight')
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

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_v15.ipynb")
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
