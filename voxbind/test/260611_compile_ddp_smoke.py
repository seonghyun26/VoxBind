#!/usr/bin/env python
"""Synthetic-data DDP smoke for the amp+compile default (run on GPU 0-3).

The real dataloader is blocked on this box by a PRE-EXISTING crops/preprocessing
alignment check (`crossdocked_xray.py:421`, data-vintage mismatch — fires in dataset
__init__, before the model is built, so it is unrelated to the throughput knobs). This
drives the EXACT at-risk wiring instead — bf16 autocast -> torch.compile -> DDP-wrap ->
full train step — with synthetic batches, to confirm the new default (amp+compile ON in
the v5 configs) actually compiles and trains under 4-way DDP and that grads sync.

    CUDA_VISIBLE_DEVICES=0,1,2,3 /home/shpark/.conda/envs/voxbind/bin/torchrun \
        --standalone --nproc_per_node=4 voxbind/test/260611_compile_ddp_smoke.py
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from voxbind.train_density_vit_mae import maybe_compile_model   # the exact compile path
from voxbind.models.density_vit import DensityViTMAE
from voxbind.models.ema import ModelEma

import os as _os
B = int(_os.environ.get("SMOKE_B", "8"))   # per-GPU batch (prod config uses 32)
G, NIN = 64, 13                            # gradmag v5 = 13 input channels
NSTEPS = 8


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", local_rank)
    torch.cuda.set_device(dev)
    torch.distributed.init_process_group("nccl", device_id=dev)
    is_main = rank == 0
    torch.set_float32_matmul_precision("high")

    # gradmag v5 model dims (config_train_atomblob_density_gradmag_..._v5.yaml)
    torch.manual_seed(0)
    model = DensityViTMAE(
        grid_dim=G, patch_size=8, n_in_channels=NIN, n_recon_channels=NIN,
        n_channels=32, dim=512, depth=12, n_heads=8, mlp_ratio=4, dropout=0.1,
        n_struct_channels=0, pretext_style="mae", head_style="patch_mlp",
        head_hidden_dim=67, head_depth=2, pos_encoding="learnable",
    ).to(dev)

    # new default: compile.enabled=true (matches the edited v5 configs).
    # SMOKE_COMPILE=0 runs the eager DDP baseline for an apples-to-apples speedup check.
    compile_on = os.environ.get("SMOKE_COMPILE", "1") == "1"
    cfg = OmegaConf.create({"compile": {"enabled": compile_on, "backend": "inductor",
                                        "mode": "default", "fullgraph": False, "dynamic": False}})
    if is_main:
        print(f"[smoke] amp=on  compile={'on' if compile_on else 'OFF (eager baseline)'}  "
              f"world={world}  per-GPU B={B}", flush=True)
    # EXACT train-script ordering: compile THEN DDP, same DDP kwargs as main()
    model_train = maybe_compile_model(model, cfg, is_main=is_main)
    model_train = torch.nn.parallel.DistributedDataParallel(
        model_train, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False, gradient_as_bucket_view=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    ema = ModelEma(model, decay=0.999, foreach=False)

    g = torch.Generator(device=dev).manual_seed(1000 + rank)   # different data per rank
    step_ms, losses = [], []
    for i in range(NSTEPS):
        x = torch.randn(B, NIN, G, G, G, device=dev, generator=g)
        target = torch.randn(B, NIN, G, G, G, device=dev, generator=g)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):        # amp default
            out, _ = model_train(x)
        loss = F.mse_loss(out.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        opt.step()
        ema.update(model)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
        step_ms.append(dt); losses.append(loss.item())
        if is_main:
            print(f"  step {i}: loss={loss.item():.4f}  {dt:9.1f} ms"
                  f"{'   <- compile warmup' if i == 0 else ''}", flush=True)

    # DDP actually communicated? post-backward .grad is the all-reduced average, so it
    # must be bit-identical on every rank — check via MAX vs MIN reduction.
    p = next(model.parameters())
    gmax = p.grad.detach().clone(); gmin = p.grad.detach().clone()
    torch.distributed.all_reduce(gmax, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(gmin, op=torch.distributed.ReduceOp.MIN)
    synced = torch.allclose(gmax, gmin)
    ema_p = next(iter(ema.module.state_dict().values()))
    finite = (all(l == l and abs(l) != float("inf") for l in losses)
              and torch.isfinite(ema_p).all().item())

    torch.distributed.barrier()
    if is_main:
        steady = sum(step_ms[1:]) / max(1, len(step_ms) - 1)
        print(f"\n  compile warmup (step0) = {step_ms[0]:.0f} ms ; "
              f"steady-state = {steady:.1f} ms/step (per-GPU B={B})")
        print(f"  losses finite={finite}  grads synced across {world} ranks={synced}")
        ok = finite and synced
        print(f"\n{'PASS' if ok else 'FAIL'}: amp+compile under {world}-way DDP — "
              f"{'compiles, trains, grads sync' if ok else 'SOMETHING BROKE'}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
