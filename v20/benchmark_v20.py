"""Benchmark V20 components and full model.

Measures:
  1. Per-component ms: CloudNorm, ConstellationEmbedding, SpectralTransportKernel,
     SparseIFFT, PerSubbundleUnitaryDeltaFiber, SpatialMLP, SparseFFT, ProximalTopK,
     V20Block (single)
  2. Full V20 vs GPT-Nano forward + backward + step, at matched d_model and depth
  3. FFT cost fraction — the key number for deciding Level 0 vs Level 1 sparse FFT
     from V20_DESIGN.md §IV

Usage:
    /path/to/python benchmark_v20.py                          # defaults
    /path/to/python benchmark_v20.py --amp                    # bf16 autocast (CUDA)
    /path/to/python benchmark_v20.py --compile                # torch.compile
    /path/to/python benchmark_v20.py --amp --compile          # both, the H100 target
    /path/to/python benchmark_v20.py --seq-len 512 --batch-size 16

Flags:
    --skip-components      only run the full-model block
    --skip-fft-fraction    don't bother isolating FFT cost
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from v20_modules import (
    V20Config,
    V20Model,
    V20Block,
    Constellation,
    CloudNorm,
    ConstellationEmbedding,
    SpectralTransportKernel,
    SparseIFFT,
    SparseFFT,
    PerSubbundleUnitaryDeltaFiber,
    SpatialMLP,
    ProximalTopK,
    count_params,
)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-Nano baseline (matched d_model=fiber_dim, matched n_blocks)
# ─────────────────────────────────────────────────────────────────────────────

class GPTNano(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(n_embd),
                "attn_qkv": nn.Linear(n_embd, 3 * n_embd),
                "attn_proj": nn.Linear(n_embd, n_embd),
                "ln2": nn.LayerNorm(n_embd),
                "mlp_fc1": nn.Linear(n_embd, 4 * n_embd),
                "mlp_fc2": nn.Linear(4 * n_embd, n_embd),
            }))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        hd = self.n_embd // self.n_head
        for blk in self.blocks:
            h = blk["ln1"](x)
            qkv = blk["attn_qkv"](h).reshape(B, T, 3, self.n_head, hd)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + blk["attn_proj"](y)
            x = x + blk["mlp_fc2"](F.gelu(blk["mlp_fc1"](blk["ln2"](x))))
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


def time_callable(fn, device, n_warmup: int = 3, n_iters: int = 10):
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Component breakdown
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_components(cfg: V20Config, device: torch.device, amp: bool):
    B = cfg.batch_size
    T = cfg.seq_len
    M = cfg.n_modes
    D = cfg.fiber_dim

    amp_dtype = torch.bfloat16 if amp and device.type == "cuda" else None

    def run_under_amp(fn):
        if amp_dtype is None:
            return fn()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            return fn()

    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    c = Constellation(
        mag=torch.randn(B, T, M, device=device).abs(),
        phase=torch.randn(B, T, M, device=device),
        log_var=torch.zeros(B, T, M, device=device),
    )
    spatial = torch.randn(B, T, D, device=device)
    band = torch.ones(M, device=device)

    results = {}

    cn = CloudNorm(M).to(device)
    def f_cn():
        with torch.no_grad():
            run_under_amp(lambda: cn(c))
    results["CloudNorm"] = time_callable(f_cn, device)

    emb = ConstellationEmbedding(cfg).to(device)
    def f_emb():
        with torch.no_grad():
            run_under_amp(lambda: emb(ids))
    results["ConstellationEmbedding"] = time_callable(f_emb, device)

    tk = SpectralTransportKernel(cfg).to(device)
    def f_tk():
        with torch.no_grad():
            run_under_amp(lambda: tk(c))
    results["SpectralTransportKernel"] = time_callable(f_tk, device)

    ifft = SparseIFFT(cfg).to(device)
    def f_ifft():
        with torch.no_grad():
            run_under_amp(lambda: ifft(c, band))
    results["SparseIFFT"] = time_callable(f_ifft, device)

    fiber = PerSubbundleUnitaryDeltaFiber(cfg).to(device)
    def f_fiber():
        with torch.no_grad():
            run_under_amp(lambda: fiber(spatial))
    results["PerSubbundleUnitaryDeltaFiber"] = time_callable(f_fiber, device)

    mlp = SpatialMLP(cfg).to(device)
    def f_mlp():
        with torch.no_grad():
            run_under_amp(lambda: mlp(spatial))
    results["SpatialMLP"] = time_callable(f_mlp, device)

    fft = SparseFFT(cfg).to(device)
    def f_fft():
        with torch.no_grad():
            run_under_amp(lambda: fft(spatial, band, c.log_var))
    results["SparseFFT"] = time_callable(f_fft, device)

    prox = ProximalTopK(cfg).to(device)
    def f_prox():
        with torch.no_grad():
            run_under_amp(lambda: prox(c))
    results["ProximalTopK"] = time_callable(f_prox, device)

    block = V20Block(cfg, block_idx=0).to(device)
    def f_block():
        with torch.no_grad():
            run_under_amp(lambda: block(c))
    results["V20Block (single)"] = time_callable(f_block, device)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full-model bench
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_full_model(cfg: V20Config, device: torch.device, amp: bool,
                         compile_mode: bool):
    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    tgt = ids[:, 1:]

    results = {}

    # --- V20 ---
    v20 = V20Model(cfg).to(device)
    if compile_mode:
        try:
            v20 = torch.compile(v20, mode="reduce-overhead", fullgraph=False)
            print("  V20: compiled")
        except Exception as e:
            print(f"  V20: compile failed ({e}); using eager")
    opt20 = torch.optim.AdamW(
        (v20._orig_mod if hasattr(v20, "_orig_mod") else v20).parameters(),
        lr=1e-4,
    )

    def step_v20():
        opt20.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, aux = v20(ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size).float(), tgt.reshape(-1)
                )
                if aux.get("band_mask_l1") is not None:
                    loss = loss + cfg.band_mask_l1 * aux["band_mask_l1"].float()
        else:
            logits, aux = v20(ids)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
            )
            if aux.get("band_mask_l1") is not None:
                loss = loss + cfg.band_mask_l1 * aux["band_mask_l1"]
        loss.backward()
        opt20.step()

    results["V20"] = {
        "params": count_params(v20._orig_mod if hasattr(v20, "_orig_mod") else v20),
        "time": time_callable(step_v20, device, n_warmup=5, n_iters=10),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        step_v20()
        results["V20"]["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- GPT-Nano at matched d_model=fiber_dim and matched n_blocks ---
    gpt = GPTNano(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.fiber_dim,
        n_head=8,
        n_layer=cfg.n_blocks,
        block_size=cfg.seq_len,
        dropout=cfg.dropout,
    ).to(device)
    if compile_mode:
        try:
            gpt = torch.compile(gpt, mode="reduce-overhead", fullgraph=False)
            print("  GPT-Nano: compiled")
        except Exception as e:
            print(f"  GPT-Nano: compile failed ({e}); using eager")
    opt_g = torch.optim.AdamW(
        (gpt._orig_mod if hasattr(gpt, "_orig_mod") else gpt).parameters(),
        lr=1e-4,
    )

    def step_gpt():
        opt_g.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = gpt(ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size).float(), tgt.reshape(-1)
                )
        else:
            logits, _ = gpt(ids)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
            )
        loss.backward()
        opt_g.step()

    results["GPT-Nano"] = {
        "params": count_params(gpt._orig_mod if hasattr(gpt, "_orig_mod") else gpt),
        "time": time_callable(step_gpt, device, n_warmup=5, n_iters=10),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        step_gpt()
        results["GPT-Nano"]["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FFT cost fraction
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fft_cost_fraction(cfg: V20Config, device: torch.device, amp: bool):
    """Isolate the FFT/IFFT cost as a fraction of the full block forward.

    If this fraction is large (say, > 30%), Level 1 sparse FFT is worth
    implementing. If small (< 10%), Level 0 is fine and the Level 1 / 2
    optimizations are not on the critical path.
    """
    B = cfg.batch_size
    T = cfg.seq_len
    M = cfg.n_modes
    D = cfg.fiber_dim

    amp_dtype = torch.bfloat16 if amp and device.type == "cuda" else None

    def run_under_amp(fn):
        if amp_dtype is None:
            return fn()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            return fn()

    c = Constellation(
        mag=torch.randn(B, T, M, device=device).abs(),
        phase=torch.randn(B, T, M, device=device),
        log_var=torch.zeros(B, T, M, device=device),
    )
    spatial = torch.randn(B, T, D, device=device)
    band = torch.ones(M, device=device)

    ifft = SparseIFFT(cfg).to(device)
    fft = SparseFFT(cfg).to(device)
    block = V20Block(cfg, block_idx=0).to(device)

    def fft_pair():
        with torch.no_grad():
            run_under_amp(lambda: (ifft(c, band), fft(spatial, band, c.log_var)))

    def full_block():
        with torch.no_grad():
            run_under_amp(lambda: block(c))

    fft_mean, _ = time_callable(fft_pair, device, n_warmup=5, n_iters=20)
    blk_mean, _ = time_callable(full_block, device, n_warmup=5, n_iters=20)
    return fft_mean, blk_mean, fft_mean / max(blk_mean, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="V20 benchmark")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-subbundles", type=int, default=8)
    p.add_argument("--subbundle-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true", dest="compile_mode")
    p.add_argument("--skip-components", action="store_true")
    p.add_argument("--skip-fft-fraction", action="store_true")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")

    cfg = V20Config(
        vocab_size=args.vocab_size,
        n_subbundles=args.n_subbundles,
        subbundle_dim=args.subbundle_dim,
        n_blocks=args.n_blocks,
        max_seq_len=args.seq_len,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print(
        f"Config: n_sub={cfg.n_subbundles} sub_dim={cfg.subbundle_dim}"
        f" shd={cfg.spectral_half_dim} n_modes={cfg.n_modes}"
        f" fiber_dim={cfg.fiber_dim} K={cfg.fiber_K}"
        f" active_modes={cfg.active_modes_per_sub}/sub n_blocks={cfg.n_blocks}"
        f" seq_len={cfg.seq_len} batch={cfg.batch_size}"
    )
    state_per_block = cfg.n_subbundles * cfg.fiber_K * cfg.fiber_K
    print(f"State per block: {state_per_block:,} values")
    print(f"AMP: {args.amp}  Compile: {args.compile_mode}")

    if not args.skip_components:
        print("\n=== Component breakdown (forward only) ===")
        comp = benchmark_components(cfg, device, args.amp)
        print(f"{'Component':<34} {'ms':>10} {'±stdev':>10}")
        print("-" * 58)
        for name, (mean, std) in comp.items():
            print(f"{name:<34} {mean:>10.3f} {std:>10.3f}")
        per_block = comp.get("V20Block (single)", (0.0, 0.0))[0]
        print(f"\nV20Block × {cfg.n_blocks} = {per_block * cfg.n_blocks:.1f} ms (fwd only estimate)")

    if not args.skip_fft_fraction:
        print("\n=== FFT cost fraction (V20_DESIGN.md §IV Level 0 vs 1 decision) ===")
        fft_ms, blk_ms, frac = benchmark_fft_cost_fraction(cfg, device, args.amp)
        print(f"  FFT+IFFT pair : {fft_ms:.3f} ms")
        print(f"  Full V20Block : {blk_ms:.3f} ms")
        print(f"  FFT fraction  : {frac * 100:.1f}%")
        if frac > 0.3:
            print("  >>> Level 1 sparse FFT (batched per-subbundle) is worth implementing.")
        elif frac > 0.1:
            print("  >>> Level 0 is fine; Level 1 is a minor optimization.")
        else:
            print("  >>> FFT is negligible; focus optimization elsewhere.")

    print("\n=== Full model forward + backward + step ===")
    full = benchmark_full_model(cfg, device, args.amp, args.compile_mode)
    print(f"{'Model':<12} {'Params':>14} {'ms/step':>12} {'±stdev':>10} {'Mem (MiB)':>12}")
    print("-" * 62)
    for name, info in full.items():
        mean, std = info["time"]
        mem = info.get("peak_mem_mib", float("nan"))
        mem_str = f"{mem:.1f}" if mem == mem else "n/a"
        print(f"{name:<12} {info['params']:>14,} {mean:>12.2f} {std:>10.2f} {mem_str:>12}")

    if "V20" in full and "GPT-Nano" in full:
        v20_ms = full["V20"]["time"][0]
        gpt_ms = full["GPT-Nano"]["time"][0]
        ratio = v20_ms / max(gpt_ms, 1e-6)
        print(f"\nV20 / GPT-Nano ratio: {ratio:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
