"""Benchmark V19 components to diagnose wall-clock breakdown.

Run on H100 (or any GPU) to see where time is going. Also works on MPS and CPU.

Usage:
    /path/to/python benchmark_v19.py
    /path/to/python benchmark_v19.py --seq-len 512 --batch-size 16
    /path/to/python benchmark_v19.py --compile
    /path/to/python benchmark_v19.py --amp  # bfloat16 autocast (CUDA only)

What it measures:
    1. End-to-end V19 forward step (incl. backward) in ms
    2. Per-component breakdown: embedding, context accum, unitary fiber,
       channel gate, CurvBias attention, FFN
    3. Compared to GPT-Nano at the same d_model / n_blocks
    4. Peak memory usage
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from v19_modules import (
    V19Config,
    V19Model,
    V19Block,
    PrecisionEmbedding,
    GeometricContextAccum,
    UnitaryDeltaFiber,
    ChannelGate,
    CurvBiasAttention,
    FFN,
    count_params,
)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline GPT-Nano (matched d_model / n_blocks for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

class GPTNano(nn.Module):
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
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

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


# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _device_from_arg(dev_arg: str) -> torch.device:
    if dev_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev_arg)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_callable(fn, device, n_warmup=3, n_iters=10):
    """Return (avg_ms, std_ms) of calling fn() on the given device."""
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Component-level timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_components(cfg: V19Config, device: torch.device, amp: bool):
    B = cfg.batch_size
    T = cfg.seq_len
    D = cfg.d_model
    dtype = torch.float32
    amp_dtype = torch.bfloat16 if amp and device.type == "cuda" else None

    def run_under_amp(fn):
        if amp_dtype is None:
            return fn()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            return fn()

    x = torch.randn(B, T, D, device=device)
    log_var = torch.zeros(B, T, D, device=device)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    results = {}

    # 1. PrecisionEmbedding
    emb = PrecisionEmbedding(cfg).to(device)
    def f_emb():
        with torch.no_grad():
            run_under_amp(lambda: emb(ids))
    results["PrecisionEmbedding"] = time_callable(f_emb, device)

    # 2. GeometricContextAccum (with a full band mask so it does real work)
    ctx = GeometricContextAccum(cfg).to(device)
    band = torch.ones(cfg.ctx_n_bands, device=device)
    def f_ctx():
        with torch.no_grad():
            run_under_amp(lambda: ctx(x, band))
    results["GeometricContextAccum"] = time_callable(f_ctx, device)

    # 3. UnitaryDeltaFiber — the main suspect at K=32
    fiber = UnitaryDeltaFiber(cfg).to(device)
    def f_fiber():
        with torch.no_grad():
            run_under_amp(lambda: fiber(x))
    results["UnitaryDeltaFiber"] = time_callable(f_fiber, device)

    # 4. ChannelGate
    cg = ChannelGate(cfg).to(device)
    def f_cg():
        with torch.no_grad():
            run_under_amp(lambda: cg(x, x))
    results["ChannelGate"] = time_callable(f_cg, device)

    # 5. CurvBiasAttention (single head)
    attn = CurvBiasAttention(cfg).to(device)
    def f_attn():
        with torch.no_grad():
            run_under_amp(lambda: attn(x))
    results["CurvBiasAttention"] = time_callable(f_attn, device)

    # 6. FFN
    ffn = FFN(cfg).to(device)
    def f_ffn():
        with torch.no_grad():
            run_under_amp(lambda: ffn(x))
    results["FFN"] = time_callable(f_ffn, device)

    # 7. Single V19Block (includes everything plus the residuals)
    block = V19Block(cfg, block_idx=0).to(device)
    def f_block():
        with torch.no_grad():
            run_under_amp(lambda: block(x, log_var))
    results["V19Block (single)"] = time_callable(f_block, device)

    return results


def benchmark_full_model(cfg: V19Config, device: torch.device, amp: bool,
                         compile_mode: bool):
    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    tgt = ids[:, 1:]

    results = {}

    # --- V19 ---
    v19 = V19Model(cfg).to(device)
    if compile_mode:
        v19 = torch.compile(v19)  # type: ignore
    opt19 = torch.optim.AdamW(v19.parameters(), lr=1e-4)

    def step_v19():
        opt19.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, aux = v19(ids)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
                if aux.get("band_mask_l1") is not None:
                    loss = loss + cfg.band_mask_l1 * aux["band_mask_l1"]
        else:
            logits, aux = v19(ids)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            if aux.get("band_mask_l1") is not None:
                loss = loss + cfg.band_mask_l1 * aux["band_mask_l1"]
        loss.backward()
        opt19.step()

    results["V19"] = {
        "params": count_params(v19),
        "time": time_callable(step_v19, device, n_warmup=5, n_iters=10),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        step_v19()
        results["V19"]["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- GPT-Nano at matched d_model/n_blocks ---
    gpt = GPTNano(
        vocab_size=cfg.vocab_size, n_embd=cfg.d_model, n_head=8,
        n_layer=cfg.n_blocks, block_size=cfg.seq_len, dropout=cfg.dropout,
    ).to(device)
    if compile_mode:
        gpt = torch.compile(gpt)  # type: ignore
    opt_g = torch.optim.AdamW(gpt.parameters(), lr=1e-4)

    def step_gpt():
        opt_g.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = gpt(ids)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
        else:
            logits, _ = gpt(ids)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
        loss.backward()
        opt_g.step()

    results["GPT-Nano"] = {
        "params": count_params(gpt),
        "time": time_callable(step_gpt, device, n_warmup=5, n_iters=10),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        step_gpt()
        results["GPT-Nano"]["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="V19 benchmark.")
    p.add_argument("--device", default="auto", help='"auto" | "cuda" | "mps" | "cpu"')
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--fiber-heads", type=int, default=16)
    p.add_argument("--fiber-K", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    p.add_argument("--compile", action="store_true", dest="compile_mode",
                   help="Wrap models in torch.compile")
    p.add_argument("--skip-components", action="store_true",
                   help="Skip per-component breakdown (only run full-model)")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    print(
        f"Config: d_model={args.d_model} n_blocks={args.n_blocks} "
        f"fiber_heads={args.fiber_heads} fiber_K={args.fiber_K} "
        f"seq_len={args.seq_len} batch_size={args.batch_size}"
    )
    print(f"AMP: {args.amp}  Compile: {args.compile_mode}")

    cfg = V19Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        max_seq_len=args.seq_len,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        fiber_heads=args.fiber_heads,
        fiber_K=args.fiber_K,
    )

    if not args.skip_components:
        print("\n=== Component breakdown (forward only, 1 block worth) ===")
        comp = benchmark_components(cfg, device, args.amp)
        print(f"{'Component':<28} {'ms':>10} {'±stdev':>10}")
        print("-" * 52)
        for name, (mean, std) in comp.items():
            print(f"{name:<28} {mean:>10.2f} {std:>10.2f}")
        per_block = comp.get("V19Block (single)", (0.0, 0.0))[0]
        print(f"\nEstimated full-fwd (V19Block * {args.n_blocks}) = {per_block * args.n_blocks:.1f} ms")

    print("\n=== Full model forward + backward + step ===")
    full = benchmark_full_model(cfg, device, args.amp, args.compile_mode)
    print(f"{'Model':<12} {'Params':>14} {'ms/step':>12} {'±stdev':>10} {'Mem (MiB)':>12}")
    print("-" * 62)
    for name, info in full.items():
        mean, std = info["time"]
        mem = info.get("peak_mem_mib", float("nan"))
        print(f"{name:<12} {info['params']:>14,} {mean:>12.2f} {std:>10.2f} {mem:>12.1f}")

    if "V19" in full and "GPT-Nano" in full:
        v19_ms = full["V19"]["time"][0]
        gpt_ms = full["GPT-Nano"]["time"][0]
        ratio = v19_ms / max(gpt_ms, 1e-6)
        print(f"\nV19 / GPT-Nano ratio: {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
