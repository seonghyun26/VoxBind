import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxbind.dataset import create_dataloaders
from voxbind.metrics import create_metrics_for_training
from voxbind.models import create_model
from voxbind.models.adamw import AdamW
from voxbind.models.ema import ModelEma
from voxbind.voxelizer import Voxelizer


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma > 0:
        return voxels + voxels.clone().normal_(0, sigma)
    return voxels


def summarize(values: list[float]) -> dict:
    values_sorted = sorted(values)
    n = len(values_sorted)
    p50_idx = int(0.50 * (n - 1))
    p90_idx = int(0.90 * (n - 1))
    return {
        "mean_ms": statistics.mean(values_sorted) * 1000.0,
        "p50_ms": values_sorted[p50_idx] * 1000.0,
        "p90_ms": values_sorted[p90_idx] * 1000.0,
        "min_ms": values_sorted[0] * 1000.0,
        "max_ms": values_sorted[-1] * 1000.0,
    }


def format_summary(name: str, values: list[float]) -> str:
    s = summarize(values)
    return (
        f"{name:<12} mean={s['mean_ms']:.2f}ms "
        f"p50={s['p50_ms']:.2f}ms p90={s['p90_ms']:.2f}ms "
        f"min={s['min_ms']:.2f}ms max={s['max_ms']:.2f}ms"
    )


def build_cfg(args: argparse.Namespace):
    config_dir = str(REPO_ROOT / "voxbind" / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config_train")

    cfg.wandb = False
    cfg.debug = args.debug
    cfg.num_workers = args.num_workers
    cfg.bsz = args.batch_size
    cfg.aug = args.aug
    cfg.smooth_sigma = args.smooth_sigma
    if args.prefetch_factor is not None:
        cfg.prefetch_factor = args.prefetch_factor
    if args.data_dir is not None:
        cfg.dset.data_dir = args.data_dir
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short VoxBind train-step benchmark on a few batches."
    )
    parser.add_argument("--batches", type=int, default=20, help="Measured batches.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup batches.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch factor.")
    parser.add_argument("--smooth-sigma", type=float, default=0.9, help="Noise sigma.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Loader split.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device.")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory.")
    parser.add_argument("--train-miou-interval", type=int, default=8, help="mIoU update interval.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Use debug dataset subset.")
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True, help="Use data augmentation.")
    parser.add_argument(
        "--use-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap model with DataParallel when multiple GPUs are visible.",
    )
    parser.add_argument(
        "--include-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include metric updates in the benchmarked step.",
    )
    parser.add_argument(
        "--include-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include EMA update in the benchmarked step.",
    )
    parser.add_argument(
        "--backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run backward + optimizer step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    loader_train, loader_val, _ = create_dataloaders(cfg)
    loader = loader_train if args.split == "train" else loader_val

    model = create_model(cfg, device=str(device)).to(device)
    if args.use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.train()

    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad(set_to_none=True)

    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    model_ema = ModelEma(model, decay=0.999) if args.include_ema else None
    metrics = None
    if args.include_metrics:
        metrics, _ = create_metrics_for_training(device=str(device))
        metrics.reset()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    total_steps = args.warmup + args.batches
    loader_it = iter(loader)

    data_wait_times = []
    voxel_times = []
    model_times = []
    total_step_times = []

    for step in range(total_steps):
        t_step = time.perf_counter()

        t0 = time.perf_counter()
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)
        t_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            voxels_lig = voxelizer.forward(batch["ligand"], num_channels=7)
            smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
            voxels_poc = voxelizer.forward(batch["pocket"], num_channels=4)
            if args.split == "train":
                voxels_poc[: int((0.2 * voxels_poc.shape[0]) + 0.9999)].zero_()
        sync_if_cuda(device)
        t_vox = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model(smooth_voxels_lig, voxels_poc)
        loss = criterion(pred, voxels_lig)
        if args.backward:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if model_ema is not None:
            model_ema.update(model)
        if metrics is not None:
            metrics.update(
                loss.detach(),
                pred.detach(),
                voxels_lig,
                update_miou=(step % max(1, args.train_miou_interval) == 0),
            )
        sync_if_cuda(device)
        t_model = time.perf_counter() - t0

        t_total = time.perf_counter() - t_step

        if step >= args.warmup:
            data_wait_times.append(t_data)
            voxel_times.append(t_vox)
            model_times.append(t_model)
            total_step_times.append(t_total)

    mean_step_s = statistics.mean(total_step_times)
    samples_per_sec = args.batch_size / mean_step_s

    print("==== VoxBind Short Benchmark ====")
    print(f"device={device}  split={args.split}  batches={args.batches}  warmup={args.warmup}")
    print(
        f"batch_size={args.batch_size}  num_workers={args.num_workers}  "
        f"aug={args.aug}  debug={args.debug}"
    )
    print(f"include_metrics={args.include_metrics}  include_ema={args.include_ema}  backward={args.backward}")
    print(format_summary("data_wait", data_wait_times))
    print(format_summary("voxelize", voxel_times))
    print(format_summary("model_step", model_times))
    print(format_summary("step_total", total_step_times))
    print(f"throughput  {samples_per_sec:.2f} samples/s")

    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
        print(f"max_memory_allocated  {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
