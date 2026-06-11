#!/usr/bin/env python
"""Benchmark torch.compile on the DensityViTMAE pretraining step.

This is intentionally data-free: it isolates model forward/backward/optimizer
time from voxelization and DataLoader effects. Shapes match the 40M atomblob
pretraining runs by default: B x 11 x 64 x 64 x 64.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxbind.models.density_vit import DensityViTMAE


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(args: argparse.Namespace) -> DensityViTMAE:
    return DensityViTMAE(
        grid_dim=args.grid_dim,
        patch_size=args.patch_size,
        n_in_channels=args.n_in_channels,
        n_recon_channels=args.n_in_channels,
        n_channels=32,
        dim=args.dim,
        depth=args.depth,
        n_heads=args.heads,
        mlp_ratio=4,
        dropout=0.1,
        n_struct_channels=0,
        pretext_style="mae",
        dual_head=False,
        head_hidden_dim=67,
        head_depth=2,
        head_style="patch_mlp",
        pos_encoding=args.pos_encoding,
    )


def summarize(times: list[float]) -> str:
    vals = sorted(t * 1000.0 for t in times)
    return (
        f"mean={statistics.mean(vals):.2f}ms "
        f"p50={vals[int(0.50 * (len(vals) - 1))]:.2f}ms "
        f"p90={vals[int(0.90 * (len(vals) - 1))]:.2f}ms "
        f"min={vals[0]:.2f}ms max={vals[-1]:.2f}ms"
    )


def run_steps(model: torch.nn.Module, x: torch.Tensor, *, steps: int, device: torch.device) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    times: list[float] = []
    for _ in range(steps):
        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out_pretext, a_hat = model(x)
        loss = out_pretext.square().mean()
        if a_hat is not None:
            loss = loss + a_hat.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sync(device)
        times.append(time.perf_counter() - t0)
    return times


def bench_one(label: str, model: torch.nn.Module, x: torch.Tensor, args: argparse.Namespace,
              device: torch.device) -> dict:
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    warm = run_steps(model, x, steps=args.warmup, device=device)
    measured = run_steps(model, x, steps=args.steps, device=device)
    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return {
        "label": label,
        "warmup_s": sum(warm),
        "times": measured,
        "peak_gb": peak_gb,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--backend", default="inductor")
    p.add_argument("--fullgraph", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--grid-dim", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--n-in-channels", type=int, default=11)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pos-encoding", choices=["learnable", "rope3d"], default="learnable")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    x = torch.randn(
        args.batch_size, args.n_in_channels, args.grid_dim, args.grid_dim, args.grid_dim,
        device=device,
    )

    eager = make_model(args).to(device)
    eager_result = bench_one("eager", eager, x, args, device)
    del eager
    if device.type == "cuda":
        torch.cuda.empty_cache()

    compiled = make_model(args).to(device)
    compile_kwargs = {
        "backend": args.backend,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
    }
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    compiled = torch.compile(compiled, **compile_kwargs)
    compiled_result = bench_one("compiled", compiled, x, args, device)

    eager_mean = statistics.mean(eager_result["times"])
    comp_mean = statistics.mean(compiled_result["times"])
    speedup = eager_mean / comp_mean if comp_mean > 0 else float("nan")

    print(f"shape: B={args.batch_size}, C={args.n_in_channels}, G={args.grid_dim}, "
          f"dim={args.dim}, depth={args.depth}, heads={args.heads}")
    print(f"compile: backend={args.backend}, mode={args.compile_mode}, "
          f"fullgraph={args.fullgraph}, dynamic={args.dynamic}")
    for result in (eager_result, compiled_result):
        print(f"{result['label']:<8} warmup_total={result['warmup_s']:.2f}s "
              f"peak={result['peak_gb']:.2f}GB {summarize(result['times'])}")
    print(f"steady_state_speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
