"""
V12.1 Training Script — Character-Level Tiny Shakespeare

Trains V12.1 and GPT-Nano baseline, compares BPC / Accuracy / Speed.
Usage: python v12/train_v12_1.py
"""

import math
import time
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Add parent dir to path so we can import v12_1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v12_1 import V12_1Config, V12_1Model, GPTNano

# ── Device ───────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_shakespeare.txt")

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

print(f"Tiny Shakespeare: {len(data):,} chars, vocab {vocab_size}")
print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")


def get_batch(split_data, cfg):
    max_start = len(split_data) - cfg.seq_len - 1
    starts = torch.randint(0, max_start, (cfg.batch_size,))
    return torch.stack([split_data[s:s + cfg.seq_len] for s in starts]).to(device)


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, cfg, is_gpt=False):
    model.eval()
    results = {}
    for name, sd in [("train", train_data), ("val", val_data)]:
        tot_ce, tot_ok, tot_n, tot_sp = 0., 0, 0, 0.
        for _ in range(cfg.eval_steps):
            b = get_batch(sd, cfg)
            logits, info = model(b)
            tgt = b[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            tot_ce += ce.item()
            tot_ok += (logits.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
            if not is_gpt:
                tot_sp += info.get("spectral_sparsity", 0.0)
        n = cfg.eval_steps
        results[name] = {
            "ce": tot_ce / n, "acc": tot_ok / tot_n,
            "sparsity": tot_sp / n if not is_gpt else 0.0,
        }
    model.train()
    return results


# ── Training Loop ────────────────────────────────────────────────────

def train_model(model, cfg, label="V12.1", is_gpt=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.05)

    min_ratio = getattr(cfg, 'min_lr', 0) / cfg.learning_rate
    hold_end = cfg.warmup_steps + getattr(cfg, 'lr_hold_steps', 0)

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        if step < hold_end:
            return 1.0  # hold at peak
        progress = (step - hold_end) / max(1, cfg.max_steps - hold_end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "step": [], "train_ce": [], "val_ce": [],
        "train_acc": [], "val_acc": [],
        "train_bpc": [], "val_bpc": [],
        "sparsity": [], "lr": [],
        "step_times": [], "per_step_loss": [],
    }

    model.train()
    total_start = time.time()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nTraining {label}: {n_params:,} params")
    print(f"Steps: {cfg.max_steps}, Batch: {cfg.batch_size}, Seq: {cfg.seq_len}")
    print("=" * 70)

    for step in range(cfg.max_steps + 1):
        # Eval
        if step % cfg.eval_interval == 0:
            res = estimate_loss(model, cfg, is_gpt=is_gpt)
            tr, vl = res["train"], res["val"]
            history["step"].append(step)
            history["train_ce"].append(tr["ce"])
            history["val_ce"].append(vl["ce"])
            history["train_acc"].append(tr["acc"])
            history["val_acc"].append(vl["acc"])
            history["train_bpc"].append(tr["ce"] / math.log(2))
            history["val_bpc"].append(vl["ce"] / math.log(2))
            history["sparsity"].append(vl["sparsity"])
            history["lr"].append(scheduler.get_last_lr()[0])
            sp_str = f" | Sp: {vl['sparsity']:.1%}" if not is_gpt else ""
            print(f"[{label}] Step {step:5d} | Train CE: {tr['ce']:.3f} | "
                  f"Val CE: {vl['ce']:.3f} | Val BPC: {vl['ce'] / math.log(2):.2f} | "
                  f"Val Acc: {vl['acc']:.1%}{sp_str}")

        if step >= cfg.max_steps:
            break

        # Train step
        step_start = time.time()
        batch = get_batch(train_data, cfg)
        optimizer.zero_grad()

        logits, info = model(batch)
        targets = batch[:, 1:]

        if is_gpt:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        else:
            # Deep supervision: weighted sum of per-block losses
            ce_loss = 0.
            total_weight = 0.
            for block_logits, weight in info["intermediate_logits"]:
                ce_loss += weight * F.cross_entropy(
                    block_logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
                )
                total_weight += weight
            ce_loss /= total_weight

            # Dictionary coherence regularization
            dcl, nd = 0., 0
            for blk in model.blocks:
                for dr, di in zip(blk.memory.dict_real, blk.memory.dict_imag):
                    atoms = torch.fft.ifft(torch.complex(dr, di), dim=-1).real
                    An = F.normalize(atoms, dim=-1)
                    g = An @ An.T
                    dcl += (g - torch.eye(g.size(0), device=g.device)).pow(2).mean()
                    nd += 1
            dcl /= max(nd, 1)
            loss = ce_loss + 0.1 * dcl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history["per_step_loss"].append(loss.item())
        history["step_times"].append(time.time() - step_start)

        if step % 100 == 0 and step > 0:
            avg_ms = np.mean(history["step_times"][-100:]) * 1000
            print(f"  [step {step}] loss={loss.item():.4f} avg_step={avg_ms:.1f}ms", end="\r")

    total_time = time.time() - total_start
    print()
    print("=" * 70)
    vl = history["val_ce"][-1]
    va = history["val_acc"][-1]
    print(f"[{label}] FINAL | Val CE: {vl:.3f} | Val BPC: {vl / math.log(2):.2f} | "
          f"Val Acc: {va:.1%}")
    print(f"[{label}] Total time: {total_time:.1f}s | "
          f"Avg step: {np.mean(history['step_times']) * 1000:.1f}ms")

    history["total_time"] = total_time
    history["avg_step_ms"] = np.mean(history["step_times"]) * 1000
    return history


# ── Text Generation ──────────────────────────────────────────────────

@torch.no_grad()
def generate_text(model, prompt, cfg, max_new=200, temperature=0.8, is_gpt=False):
    model.eval()
    ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        ctx = ids[:, -cfg.seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return "".join(itos[i.item()] for i in ids[0])


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # V12.1 Config
    cfg = V12_1Config(vocab_size=vocab_size)
    print(f"\nV12.1 Config:")
    print(f"  Fiber: {cfg.fiber_dim} = {cfg.n_subbundles} x {cfg.subbundle_dim}")
    print(f"  Spectral sparsity: {cfg.spectral_sparsity}/{cfg.subbundle_dim} = "
          f"{100 * cfg.spectral_sparsity / cfg.subbundle_dim:.0f}% active")
    print(f"  Blocks: {cfg.n_blocks}, Context dim: {cfg.context_dim}")
    print(f"  MLP hidden: {cfg.mlp_hidden}")
    print(f"  Langevin steps: {cfg.langevin_steps}, beta: {cfg.beta_init}→{cfg.beta_final}")
    print(f"  Atoms: {cfg.atoms_per_subbundle} per subbundle")

    # V12.1 Model
    v12_model = V12_1Model(cfg).to(device)
    v12_params = sum(p.numel() for p in v12_model.parameters())

    # Parameter breakdown
    n_embed = sum(p.numel() for p in v12_model.embedding.parameters())
    n_ctx = sum(sum(p.numel() for p in blk.context_acc.parameters()) for blk in v12_model.blocks)
    n_transport = sum(sum(p.numel() for p in blk.transport.parameters()) for blk in v12_model.blocks)
    n_memory = sum(sum(p.numel() for p in blk.memory.parameters()) for blk in v12_model.blocks)
    n_settler = sum(sum(p.numel() for p in blk.settler.parameters()) for blk in v12_model.blocks)
    n_mlp = sum(sum(p.numel() for p in blk.mlp.parameters()) for blk in v12_model.blocks)
    n_decoder = sum(p.numel() for p in v12_model.decoder.parameters()) + \
                sum(p.numel() for p in v12_model.final_norm.parameters())
    n_other = v12_params - n_embed - n_ctx - n_transport - n_memory - n_settler - n_mlp - n_decoder

    print(f"\nV12.1 parameters: {v12_params:,}")
    print(f"  Embedding:             {n_embed:,} ({100 * n_embed / v12_params:.1f}%)")
    print(f"  Context Accum ({cfg.n_blocks}):    {n_ctx:,} ({100 * n_ctx / v12_params:.1f}%)")
    print(f"  Transport ({cfg.n_blocks}):        {n_transport:,} ({100 * n_transport / v12_params:.1f}%)")
    print(f"  Memory ({cfg.n_blocks}):           {n_memory:,} ({100 * n_memory / v12_params:.1f}%)")
    print(f"  Settler ({cfg.n_blocks}):          {n_settler:,} ({100 * n_settler / v12_params:.1f}%)")
    print(f"  SpatialMLP ({cfg.n_blocks}):       {n_mlp:,} ({100 * n_mlp / v12_params:.1f}%)")
    print(f"  Decoder + norms:       {n_decoder:,} ({100 * n_decoder / v12_params:.1f}%)")
    print(f"  Block norms/gates:     {n_other:,} ({100 * n_other / v12_params:.1f}%)")

    # GPT-Nano Baseline (12 layers, matched to user's config)
    gpt_model = GPTNano(
        vocab_size=vocab_size, n_embd=128, n_head=4, n_layer=12,
        block_size=cfg.seq_len, dropout=cfg.dropout
    ).to(device)
    gpt_params = sum(p.numel() for p in gpt_model.parameters())

    print(f"\nGPT-Nano parameters: {gpt_params:,}")
    print(f"V12.1 parameters:    {v12_params:,}")
    print(f"Ratio: GPT/V12.1 = {gpt_params / v12_params:.2f}x")

    # Train V12.1
    print("\n" + "=" * 70)
    print("TRAINING V12.1")
    print("=" * 70)
    v12_history = train_model(v12_model, cfg, label="V12.1", is_gpt=False)

    # Train GPT-Nano with same config
    print("\n" + "=" * 70)
    print("TRAINING GPT-NANO")
    print("=" * 70)
    gpt_history = train_model(gpt_model, cfg, label="GPT", is_gpt=True)

    # ── Comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    v12_final_bpc = v12_history["val_bpc"][-1]
    v12_final_acc = v12_history["val_acc"][-1]
    gpt_final_bpc = gpt_history["val_bpc"][-1]
    gpt_final_acc = gpt_history["val_acc"][-1]

    print(f"{'':20s} {'V12.1':>12s} {'GPT-Nano':>12s} {'Delta':>12s}")
    print(f"{'-' * 56}")
    print(f"{'Parameters':20s} {v12_params:>12,} {gpt_params:>12,} {v12_params - gpt_params:>+12,}")
    print(f"{'Val BPC':20s} {v12_final_bpc:>12.3f} {gpt_final_bpc:>12.3f} {v12_final_bpc - gpt_final_bpc:>+12.3f}")
    print(f"{'Val Accuracy':20s} {v12_final_acc:>11.1%} {gpt_final_acc:>11.1%} {v12_final_acc - gpt_final_acc:>+11.1%}")
    print(f"{'Avg ms/step':20s} {v12_history['avg_step_ms']:>12.1f} {gpt_history['avg_step_ms']:>12.1f} {v12_history['avg_step_ms'] - gpt_history['avg_step_ms']:>+12.1f}")
    print(f"{'Total time (s)':20s} {v12_history['total_time']:>12.1f} {gpt_history['total_time']:>12.1f}")

    if v12_final_bpc <= gpt_final_bpc:
        print(f"\n*** V12.1 WINS on BPC: {v12_final_bpc:.3f} <= {gpt_final_bpc:.3f} ***")
    else:
        print(f"\n    V12.1 gap: +{v12_final_bpc - gpt_final_bpc:.3f} BPC")

    if v12_final_acc >= gpt_final_acc:
        print(f"*** V12.1 WINS on Accuracy: {v12_final_acc:.1%} >= {gpt_final_acc:.1%} ***")
    else:
        print(f"    V12.1 gap: {v12_final_acc - gpt_final_acc:+.1%} accuracy")

    # Text generation samples
    print("\n" + "=" * 70)
    print("TEXT GENERATION (temperature=0.8)")
    print("=" * 70)
    prompts = ["ROMEO:\n", "To be or not to ", "The king ", "O, "]
    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        print(f"  V12.1: {repr(generate_text(v12_model, prompt, cfg)[len(prompt):len(prompt) + 100])}")
        print(f"  GPT:   {repr(generate_text(gpt_model, prompt, cfg, is_gpt=True)[len(prompt):len(prompt) + 100])}")


if __name__ == "__main__":
    main()
