"""
Ablation Study: Is the SSM Doing All the Heavy Lifting?

4-way ablation to isolate contributions of each V12.1 component:

  A) Full V12.1          — baseline (SSM + spectral transport + Hopfield + MLP)
  B) Zero SSM context    — SSM runs but output zeroed → transport gets constant D,A
  C) SSM + MLP only      — no spectral domain at all, pure spatial SSM + MLP
  D) No spectral sparsity — all modes active (no top-k), everything else intact

If (B) drops hard:  SSM context is critical for transport modulation
If (C) matches (A): SSM+MLP alone suffices, spectral components are dead weight
If (C) << (A):      spectral components contribute beyond what SSM provides → thesis holds
If (D) drops:       sparsity itself matters, not just FFT/IFFT as a basis change

Usage: python v12/ablation_ssm.py
"""

import math
import time
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v12_1 import (
    V12_1Config, V12_1Model, V12_1Block,
    SpectralTokenEmbedding, ContextAccumulator, SpectralTransport,
    SimplifiedMemoryBank, SpectralHopfieldSettler, SpatialMLP,
    spectral_sparsify, spectral_to_spatial, spatial_to_spectral,
    spectral_proximal, parallel_associative_scan,
)

# ── Device ───────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_shakespeare.txt")
if not os.path.exists(DATA_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH,
    )

with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

print(f"Tiny Shakespeare: {len(data):,} chars, vocab {vocab_size}")


def get_batch(split_data, cfg):
    max_start = len(split_data) - cfg.seq_len - 1
    starts = torch.randint(0, max_start, (cfg.batch_size,))
    return torch.stack([split_data[s:s + cfg.seq_len] for s in starts]).to(device)


# ═════════════════════════════════════════════════════════════════════
# ABLATION B: Zero SSM Context
# ═════════════════════════════════════════════════════════════════════
# SSM still exists (params counted) but its output is zeroed.
# Transport kernel gets D=constant, A=0 → fixed spectral filter.
# Tests: does context-dependent transport matter?

class ZeroContextBlock(V12_1Block):
    """V12.1 block where SSM context is zeroed before reaching transport."""

    def forward(self, spectral_x):
        cfg = self.cfg

        # 1. Spatial projection
        x_spatial_in = spectral_to_spatial(spectral_x, cfg)

        # 2. SSM runs (gradients flow, params exist) but output is zeroed
        q_t = self.context_acc(x_spatial_in)
        q_t = torch.zeros_like(q_t)  # <-- THE ABLATION

        # 3-8: everything else identical
        transported = self.transport(spectral_x, q_t)
        dense_field = spectral_to_spatial(transported, cfg)
        dense_field = self.norm1(dense_field)
        spatial_atoms = self.memory.get_spatial_atoms()
        settled = self.settler(dense_field, spatial_atoms)
        mlp_out = self.mlp(self.norm2(settled))
        gate = torch.sigmoid(self.res_gate)
        x_spatial_out = x_spatial_in + gate * self.dropout(settled + mlp_out)
        spectral_out = spatial_to_spectral(x_spatial_out, cfg)
        spectral_out = spectral_sparsify(spectral_out, cfg)
        return spectral_out


class ZeroContextModel(V12_1Model):
    """V12.1 with SSM context zeroed → context-independent transport."""

    def __init__(self, cfg):
        super(V12_1Model, self).__init__()
        self.cfg = cfg
        self.embedding = SpectralTokenEmbedding(cfg)
        self.blocks = nn.ModuleList([ZeroContextBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.fiber_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )
        weights = torch.zeros(cfg.n_blocks)
        for i in range(cfg.n_blocks):
            if (i + 1) % 2 == 0:
                weights[i] = (i + 1) / cfg.n_blocks
        weights[-1] = 1.0
        self.register_buffer("block_loss_weights", weights)


# ═════════════════════════════════════════════════════════════════════
# ABLATION C: SSM + MLP Only (No Spectral Domain)
# ═════════════════════════════════════════════════════════════════════
# Pure spatial architecture: embedding → [SSM → MLP → residual] x N → decoder
# No FFT, no IFFT, no spectral sparsity, no Hopfield.
# This is a simplified Mamba-like model.

class SSMOnlyBlock(nn.Module):
    """Pure spatial: SSM context → MLP → gated residual. No spectral domain."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_acc = ContextAccumulator(cfg)
        # Project context back to fiber_dim for residual mixing
        self.ctx_proj = nn.Linear(cfg.context_dim, cfg.fiber_dim)
        self.mlp = SpatialMLP(cfg)
        self.norm1 = nn.LayerNorm(cfg.fiber_dim)
        self.norm2 = nn.LayerNorm(cfg.fiber_dim)
        self.res_gate = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # x: (B, T, D) real spatial
        q_t = self.context_acc(x)                     # SSM
        ctx_out = self.ctx_proj(q_t)                   # project to fiber_dim
        h = self.norm1(x + ctx_out)                    # residual + norm
        mlp_out = self.mlp(self.norm2(h))              # MLP
        gate = torch.sigmoid(self.res_gate)
        return x + gate * self.dropout(ctx_out + mlp_out)


class SSMOnlyModel(nn.Module):
    """SSM + MLP only — no spectral domain, no Hopfield. Dense spatial throughout."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Standard dense embedding (no spectral)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.fiber_dim)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.fiber_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SSMOnlyBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.fiber_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.drop(self.embedding(token_ids) + self.pos_embedding(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.decoder(self.final_norm(x))[:, :-1, :]
        info = {"spectral_sparsity": 0.0, "intermediate_logits": [(logits, 1.0)]}
        return logits, info


# ═════════════════════════════════════════════════════════════════════
# ABLATION D: No Spectral Sparsity (All Modes Active)
# ═════════════════════════════════════════════════════════════════════
# Replace top-k sparsification with identity. FFT/IFFT still runs.
# Tests: does enforced sparsity matter, or is FFT just a basis change?

def no_sparsify(x_complex, cfg):
    """Identity — keep all spectral modes."""
    return x_complex

def no_spectral_proximal(x_spatial, cfg):
    """No-op proximal — skip projection."""
    return x_spatial


class NoSparsitySettler(SpectralHopfieldSettler):
    """Settler with no spectral proximal projection."""

    def forward(self, dense_field, spatial_atoms):
        cfg = self.cfg
        B, T, D = dense_field.shape
        sd = cfg.subbundle_dim
        K = cfg.n_subbundles
        BT = B * T

        x = dense_field
        betas = torch.linspace(cfg.beta_init, cfg.beta_final, cfg.langevin_steps,
                               device=dense_field.device)
        M_all = spatial_atoms.unsqueeze(0).expand(BT, -1, -1, -1)

        for step in range(cfg.langevin_steps):
            beta = betas[step].item()
            x_subs = x.reshape(BT, K, sd)
            sim = torch.einsum('bks,bkas->bka', x_subs, M_all)
            w = F.softmax(beta * sim, dim=-1)
            grad_E = -torch.einsum('bka,bkas->bks', w, M_all)
            inhib = self.W_inh * x
            x = x - cfg.langevin_lr * (grad_E.reshape(B, T, D) + inhib)
            if not self.training:
                x = x + math.sqrt(2.0 * cfg.langevin_lr / beta) * torch.randn_like(x)
            # NO spectral proximal — this is the ablation
        return x


class NoSparsityBlock(V12_1Block):
    """V12.1 block with spectral sparsity removed (all modes active)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.settler = NoSparsitySettler(cfg)  # no proximal

    def forward(self, spectral_x):
        cfg = self.cfg
        x_spatial_in = spectral_to_spatial(spectral_x, cfg)
        q_t = self.context_acc(x_spatial_in)
        transported = self.transport(spectral_x, q_t)
        dense_field = spectral_to_spatial(transported, cfg)
        dense_field = self.norm1(dense_field)
        spatial_atoms = self.memory.get_spatial_atoms()
        settled = self.settler(dense_field, spatial_atoms)
        mlp_out = self.mlp(self.norm2(settled))
        gate = torch.sigmoid(self.res_gate)
        x_spatial_out = x_spatial_in + gate * self.dropout(settled + mlp_out)
        # FFT but NO sparsify
        spectral_out = spatial_to_spectral(x_spatial_out, cfg)
        # skip sparsify — all modes kept
        return spectral_out


class NoSparsityModel(V12_1Model):
    """V12.1 with spectral sparsity disabled — all modes active."""

    def __init__(self, cfg):
        super(V12_1Model, self).__init__()
        self.cfg = cfg
        # Use standard spectral embedding but don't sparsify output
        self.embedding = SpectralTokenEmbedding(cfg)
        self.blocks = nn.ModuleList([NoSparsityBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.fiber_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.fiber_dim, cfg.fiber_dim), nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fiber_dim, cfg.vocab_size),
        )
        weights = torch.zeros(cfg.n_blocks)
        for i in range(cfg.n_blocks):
            if (i + 1) % 2 == 0:
                weights[i] = (i + 1) / cfg.n_blocks
        weights[-1] = 1.0
        self.register_buffer("block_loss_weights", weights)


# ═════════════════════════════════════════════════════════════════════
# Training Infrastructure
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, cfg, is_ssm_only=False):
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
            tot_sp += info.get("spectral_sparsity", 0.0)
        n = cfg.eval_steps
        results[name] = {
            "ce": tot_ce / n, "acc": tot_ok / tot_n,
            "sparsity": tot_sp / n,
        }
    model.train()
    return results


def train_ablation(model, cfg, label, steps=3000):
    """Shorter training for ablation — 3K steps to establish trends."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.05)

    min_ratio = cfg.min_lr / cfg.learning_rate
    hold_end = cfg.warmup_steps + cfg.lr_hold_steps

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        if step < hold_end:
            return 1.0
        progress = (step - hold_end) / max(1, steps - hold_end)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"step": [], "val_bpc": [], "val_acc": [], "sparsity": [], "step_times": []}
    n_params = sum(p.numel() for p in model.parameters())
    is_ssm_only = isinstance(model, SSMOnlyModel)

    model.train()
    print(f"\n{'=' * 60}")
    print(f"Training [{label}]: {n_params:,} params for {steps} steps")
    print(f"{'=' * 60}")

    eval_interval = 500
    for step in range(steps + 1):
        # Eval
        if step % eval_interval == 0:
            res = estimate_loss(model, cfg, is_ssm_only=is_ssm_only)
            vl = res["val"]
            bpc = vl["ce"] / math.log(2)
            history["step"].append(step)
            history["val_bpc"].append(bpc)
            history["val_acc"].append(vl["acc"])
            history["sparsity"].append(vl["sparsity"])
            sp_str = f" | Sp: {vl['sparsity']:.1%}" if vl['sparsity'] > 0 else ""
            print(f"  [{label}] Step {step:5d} | Val BPC: {bpc:.3f} | "
                  f"Val Acc: {vl['acc']:.1%}{sp_str}")

        if step >= steps:
            break

        # Train step
        step_start = time.time()
        batch = get_batch(train_data, cfg)
        optimizer.zero_grad()

        logits, info = model(batch)
        targets = batch[:, 1:]

        if is_ssm_only:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        else:
            # Deep supervision
            ce_loss, total_weight = 0., 0.
            for block_logits, weight in info["intermediate_logits"]:
                ce_loss += weight * F.cross_entropy(
                    block_logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                total_weight += weight
            ce_loss /= total_weight
            # DCL
            dcl, nd = 0., 0
            if hasattr(model, 'blocks'):
                for blk in model.blocks:
                    if hasattr(blk, 'memory'):
                        for dr, di in zip(blk.memory.dict_real, blk.memory.dict_imag):
                            atoms = torch.fft.irfft(torch.complex(dr, di), n=cfg.subbundle_dim, dim=-1)
                            An = F.normalize(atoms, dim=-1)
                            g = An @ An.T
                            dcl += (g - torch.eye(g.size(0), device=g.device)).pow(2).mean()
                            nd += 1
            dcl = dcl / max(nd, 1)
            loss = ce_loss + 0.1 * dcl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history["step_times"].append(time.time() - step_start)

        if step % 100 == 0 and step > 0:
            avg_ms = np.mean(history["step_times"][-100:]) * 1000
            print(f"    [step {step}] loss={loss.item():.4f} avg={avg_ms:.0f}ms", end="\r")

    avg_ms = np.mean(history["step_times"]) * 1000
    print(f"\n  [{label}] DONE | Final BPC: {history['val_bpc'][-1]:.3f} | "
          f"Acc: {history['val_acc'][-1]:.1%} | Avg: {avg_ms:.0f}ms/step")

    history["avg_step_ms"] = avg_ms
    history["n_params"] = n_params
    return history


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    cfg = V12_1Config(vocab_size=vocab_size)
    ABLATION_STEPS = 3000  # enough to establish trends, not full 10K

    print("\n" + "=" * 60)
    print("ABLATION STUDY: Is the SSM Doing All the Heavy Lifting?")
    print("=" * 60)
    print(f"Steps per ablation: {ABLATION_STEPS}")
    print(f"Config: {cfg.n_blocks} blocks, fiber={cfg.fiber_dim}, "
          f"context={cfg.context_dim}, sparsity={cfg.spectral_sparsity}/{cfg.subbundle_dim}")

    results = {}

    # ── A: Full V12.1 (baseline) ────────────────────────────────────
    print("\n[A] Full V12.1 — baseline")
    model_a = V12_1Model(cfg).to(device)
    results["A: Full V12.1"] = train_ablation(model_a, cfg, "A: Full V12.1", ABLATION_STEPS)
    del model_a
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── B: Zero SSM Context ─────────────────────────────────────────
    print("\n[B] Zero SSM Context — transport gets constant D,A")
    model_b = ZeroContextModel(cfg).to(device)
    results["B: Zero Context"] = train_ablation(model_b, cfg, "B: Zero Context", ABLATION_STEPS)
    del model_b
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── C: SSM + MLP Only ───────────────────────────────────────────
    print("\n[C] SSM + MLP Only — no spectral domain, no Hopfield")
    model_c = SSMOnlyModel(cfg).to(device)
    results["C: SSM+MLP Only"] = train_ablation(model_c, cfg, "C: SSM+MLP Only", ABLATION_STEPS)
    del model_c
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── D: No Spectral Sparsity ─────────────────────────────────────
    print("\n[D] No Spectral Sparsity — all modes active")
    model_d = NoSparsityModel(cfg).to(device)
    results["D: No Sparsity"] = train_ablation(model_d, cfg, "D: No Sparsity", ABLATION_STEPS)
    del model_d

    # ═════════════════════════════════════════════════════════════════
    # Results Summary
    # ═════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25s} {'Params':>10s} {'Val BPC':>10s} {'Val Acc':>10s} {'ms/step':>10s}")
    print("-" * 65)

    baseline_bpc = results["A: Full V12.1"]["val_bpc"][-1]
    baseline_acc = results["A: Full V12.1"]["val_acc"][-1]

    for name, h in results.items():
        bpc = h["val_bpc"][-1]
        acc = h["val_acc"][-1]
        delta_bpc = bpc - baseline_bpc
        delta_str = f"(+{delta_bpc:.3f})" if delta_bpc > 0.01 else f"({delta_bpc:+.3f})" if abs(delta_bpc) > 0.001 else "(baseline)"
        print(f"{name:<25s} {h['n_params']:>10,} {bpc:>10.3f} {acc:>9.1%} {h['avg_step_ms']:>9.0f}  {delta_str}")

    # ── Interpretation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    bpc_a = results["A: Full V12.1"]["val_bpc"][-1]
    bpc_b = results["B: Zero Context"]["val_bpc"][-1]
    bpc_c = results["C: SSM+MLP Only"]["val_bpc"][-1]
    bpc_d = results["D: No Sparsity"]["val_bpc"][-1]

    print(f"\nSSM context contribution:    {bpc_b - bpc_a:+.3f} BPC (B vs A)")
    print(f"  → {'SSM context IS critical' if (bpc_b - bpc_a) > 0.05 else 'SSM context has marginal effect'}")

    print(f"\nSpectral components value:   {bpc_c - bpc_a:+.3f} BPC (C vs A)")
    if bpc_c - bpc_a > 0.1:
        print(f"  → SSM+MLP alone is {bpc_c - bpc_a:.3f} BPC worse. SPECTRAL COMPONENTS CONTRIBUTE.")
        print(f"  → The SSM is NOT doing all the heavy lifting.")
    elif bpc_c - bpc_a < 0.03:
        print(f"  → SSM+MLP matches full model. Spectral components may be redundant.")
        print(f"  → WARNING: the SSM may be doing most of the work.")
    else:
        print(f"  → Moderate contribution from spectral components.")

    print(f"\nSparsity contribution:       {bpc_d - bpc_a:+.3f} BPC (D vs A)")
    print(f"  → {'Sparsity matters' if (bpc_d - bpc_a) > 0.03 else 'Sparsity has marginal effect (FFT/IFFT is the value)'}")

    # Save summary for thesis
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_ssm_results.md")
    with open(summary_path, "w") as f:
        f.write("# SSM Ablation Study Results\n\n")
        f.write(f"Steps per ablation: {ABLATION_STEPS}\n\n")
        f.write(f"| Model | Params | Val BPC | Val Acc | ms/step |\n")
        f.write(f"|-------|--------|---------|---------|----------|\n")
        for name, h in results.items():
            f.write(f"| {name} | {h['n_params']:,} | {h['val_bpc'][-1]:.3f} | "
                    f"{h['val_acc'][-1]:.1%} | {h['avg_step_ms']:.0f} |\n")
        f.write(f"\n## Deltas from baseline\n\n")
        f.write(f"- SSM context: {bpc_b - bpc_a:+.3f} BPC\n")
        f.write(f"- Spectral components: {bpc_c - bpc_a:+.3f} BPC\n")
        f.write(f"- Sparsity: {bpc_d - bpc_a:+.3f} BPC\n")
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
