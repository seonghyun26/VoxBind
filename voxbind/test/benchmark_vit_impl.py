#!/usr/bin/env python
"""Autoresearch-style bench for DensityViTMAE *implementation-level* space/time levers.

Distinct from voxbind/test/benchmark_vit_opts.py, which tunes SYSTEMS knobs (amp /
channels_last / fused AdamW / foreach EMA / compile). This script probes the model
CODE itself for redundant work and memory, one component per research thread:

  profile      : module-level fwd time + activation-memory breakdown (where the
                 time/memory goes -> motivates the rest).
  roundtrip    : patch_mlp head re-derives tokens via `_patchify(_tokens_to_voxels(.))`,
                 which is the identity. Skip both 8D permute+contiguous copies of the
                 (B, c_half, G,G,G) voxel tensor. Expected: time + space win, bit-exact.
  checkpoint   : activation checkpointing on the transformer blocks. Expected: large
                 peak-mem cut for a ~25-35% step-time cost; unlocks bigger batch.
  attn         : attention q/k/v tensor layout (permute vs unbind vs chunk) at the real
                 (B,N,C,H) shape. Pure reorder -> bit-exact; pick the fastest.
  patch_embed  : non-overlapping Conv3d(k=s=p) patch embed == one gemm on unfolded
                 patches. Conv may hit a slow cuDNN path; Linear(unfold) is the gemm.

Each component is gated for correctness against a shared-init reference with a tiered
tolerance (bitwise for pure reorders; a small band where the kernel/reduction order
changes). Data-free (fixed-seed random input) to isolate model fwd/bwd/opt from the
DataLoader + voxelizer. Pin to one idle GPU, e.g.

    CUDA_VISIBLE_DEVICES=4 python voxbind/test/benchmark_vit_impl.py --component profile,roundtrip
"""

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxbind.models.density_vit import (
    DensityViTMAE, RoPE3D, _patchify, _unpatchify,
)


# ── production model shape (config_train_..._v5 / atomblob_density_gradmag) ──────
def build_cfg(args):
    return dict(
        grid_dim=args.grid_dim, patch_size=args.patch_size,
        n_in_channels=args.n_in, n_recon_channels=args.n_in,
        n_channels=32, dim=args.dim, depth=args.depth, n_heads=args.heads,
        mlp_ratio=4, dropout=0.1, n_struct_channels=0, pretext_style="mae",
        dual_head=False, head_hidden_dim=67, head_depth=2, head_style="patch_mlp",
    )


def make_model(cfg, device, *, dropout=None, channels_last=False, init_sd=None):
    torch.manual_seed(0)
    cfg2 = cfg if dropout is None else {**cfg, "dropout": dropout}
    m = DensityViTMAE(**cfg2).to(device)
    if init_sd is not None:
        m.load_state_dict(init_sd)
    if channels_last:
        m = m.to(memory_format=torch.channels_last_3d)
    return m


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def amp_ctx(enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


# ── flexible MAE forward: same math as DensityViTMAE.forward, with two levers ────
# `skip_roundtrip`: in patch_mlp mode feed decoder_proj tokens straight to recon_mlp
#   instead of _patchify(_tokens_to_voxels(.)) (which is the identity on those tokens).
# `ckpt`: wrap each transformer block in activation checkpointing.
def mae_forward(mae, density, *, skip_roundtrip=False, ckpt=False):
    enc = mae.encoder
    z = enc.patch_embed(density).flatten(2).transpose(1, 2)       # (B, N, D)
    if enc.pos_embed is not None:
        z = z + enc.pos_embed
    for blk in enc.blocks:
        if ckpt:
            z = torch.utils.checkpoint.checkpoint(blk, z, enc.rope, use_reentrant=False)
        else:
            z = blk(z, rope=enc.rope)
    z = enc.norm(z)
    z = enc.decoder_proj(z)                                       # (B, N, c_half*p^3)

    if mae.head_style == "patch_mlp":
        if skip_roundtrip:
            tok = z                                               # identity shortcut
        else:
            zv = enc._tokens_to_voxels(z)                         # (B, c_half, G,G,G)
            tok = _patchify(zv, mae.g_p, mae.patch_size)          # back to (B, N, c_half*p^3)
        tok = mae.recon_mlp(tok)
        out_pretext = _unpatchify(tok, mae.g_p, mae.patch_size, mae.n_recon)
    else:
        zv = enc._tokens_to_voxels(z)
        out_pretext = mae.head_density(zv)

    out_structure = None
    if mae.head_structure is not None:
        zv = enc._tokens_to_voxels(z)
        out_structure = mae.head_structure(zv)
    return out_pretext, out_structure


def full_step(model, opt, x, target, *, amp, channels_last, fwd_kwargs, capture=False):
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    opt.zero_grad(set_to_none=True)
    with amp_ctx(amp):
        out_pretext, _ = mae_forward(model, x, **fwd_kwargs)
    out = out_pretext.float()
    loss = F.mse_loss(out, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return (out.detach().clone() if capture else None)


def measure_full(model, x, target, *, amp=False, channels_last=False, fwd_kwargs,
                 warmup, iters):
    """Full train-step timing (fwd+loss+bwd+clip+opt.step) + peak mem for one model+config."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        full_step(model, opt, x, target, amp=amp, channels_last=channels_last, fwd_kwargs=fwd_kwargs)
    sync()
    step_ms, fwd_ms = [], []
    for _ in range(iters):
        sync(); t = time.perf_counter()
        full_step(model, opt, x, target, amp=amp, channels_last=channels_last, fwd_kwargs=fwd_kwargs)
        sync(); step_ms.append((time.perf_counter() - t) * 1e3)
        sync(); t = time.perf_counter()
        with torch.no_grad(), amp_ctx(amp):
            xc = x.to(memory_format=torch.channels_last_3d) if channels_last else x
            mae_forward(model, xc, **fwd_kwargs)
        sync(); fwd_ms.append((time.perf_counter() - t) * 1e3)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del opt; torch.cuda.empty_cache()
    return dict(step_ms_mean=statistics.mean(step_ms), step_p50=_p(step_ms, .5),
                step_p90=_p(step_ms, .9), fwd_ms_mean=statistics.mean(fwd_ms),
                peak_gb=peak_gb)


def grads_of(model):
    return {n: (p.grad.detach().float().cpu() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def correctness_full(model, x, target, *, base_kwargs, var_kwargs):
    """Bitwise check: same model+weights, two forward variants. eval() -> dropout off.
    Compare forward output AND per-parameter gradient after one backward."""
    model.eval()
    def fwd_back(kwargs):
        model.zero_grad(set_to_none=True)
        out, _ = mae_forward(model, x, **kwargs)
        F.mse_loss(out.float(), target).backward()
        return out.detach().float().cpu(), grads_of(model)
    ob, gb = fwd_back(base_kwargs)
    ov, gv = fwd_back(var_kwargs)
    out_max = (ob - ov).abs().max().item()
    g_max = max((gb[n] - gv[n]).abs().max().item()
                for n in gb if gb[n] is not None and gv[n] is not None)
    model.zero_grad(set_to_none=True); model.train()
    return dict(out_maxabs=out_max, grad_maxabs=g_max)


def _p(vals, q):
    v = sorted(vals); return v[int(q * (len(v) - 1))]


def _oom(fn):
    """Run fn(); return (result, None) or (None, 'OOM') on CUDA OOM."""
    try:
        return fn(), None
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "out of memory" in str(e).lower():
            return None, "OOM"
        raise


# ════════════════════════════════════════════════════════════════════════════════
# Component: profile — module-level fwd time + activation-memory breakdown
# ════════════════════════════════════════════════════════════════════════════════
def comp_profile(cfg, x, target, device, args, emit):
    emit("==== component: profile (module-level fwd time + activation memory) ====")
    model = make_model(cfg, device, dropout=0.0); model.train()
    enc = model.encoder
    B = x.shape[0]

    def ev():
        return torch.cuda.Event(enable_timing=True)

    # stage-timed forward (CUDA events), averaged
    stages = ["patch_embed", "blocks", "attn(sum)", "mlp(sum)", "norm",
              "decoder_proj", "tokens->vox", "patchify", "recon_mlp", "unpatchify"]
    acc = {s: 0.0 for s in stages}
    n = args.iters
    for _ in range(args.warmup):
        mae_forward(model, x)
    sync()
    for _ in range(n):
        e = {k: (ev(), ev()) for k in stages}
        with torch.no_grad():
            e["patch_embed"][0].record()
            z = enc.patch_embed(x).flatten(2).transpose(1, 2)
            if enc.pos_embed is not None:
                z = z + enc.pos_embed
            e["patch_embed"][1].record()
            # blocks, with attn/mlp split
            e["blocks"][0].record()
            attn_ms = mlp_ms = 0.0
            for blk in enc.blocks:
                a0, a1, m0, m1 = ev(), ev(), ev(), ev()
                a0.record(); h = blk.attn(blk.norm1(z), rope=enc.rope); a1.record()
                z = z + h
                m0.record(); h = blk.mlp(blk.norm2(z)); m1.record()
                z = z + h
                sync(); attn_ms += a0.elapsed_time(a1); mlp_ms += m0.elapsed_time(m1)
            e["blocks"][1].record()
            e["norm"][0].record(); z = enc.norm(z); e["norm"][1].record()
            e["decoder_proj"][0].record(); z = enc.decoder_proj(z); e["decoder_proj"][1].record()
            e["tokens->vox"][0].record(); zv = enc._tokens_to_voxels(z); e["tokens->vox"][1].record()
            e["patchify"][0].record(); tok = _patchify(zv, model.g_p, model.patch_size); e["patchify"][1].record()
            e["recon_mlp"][0].record(); tok = model.recon_mlp(tok); e["recon_mlp"][1].record()
            e["unpatchify"][0].record(); _unpatchify(tok, model.g_p, model.patch_size, model.n_recon); e["unpatchify"][1].record()
        sync()
        for k in ["patch_embed", "blocks", "norm", "decoder_proj", "tokens->vox",
                  "patchify", "recon_mlp", "unpatchify"]:
            if k in e:
                acc[k] += e[k][0].elapsed_time(e[k][1])
        acc["attn(sum)"] += attn_ms
        acc["mlp(sum)"] += mlp_ms
    for k in acc:
        acc[k] /= n
    fwd_total = acc["patch_embed"] + acc["blocks"] + acc["decoder_proj"] + acc["tokens->vox"] \
        + acc["patchify"] + acc["recon_mlp"] + acc["unpatchify"]

    # activation memory of the key intermediates (fp32 bytes)
    def mb(t):
        return t.numel() * 4 / 1e6
    z0 = enc.patch_embed(x).flatten(2).transpose(1, 2)
    zdp = enc.decoder_proj(enc.norm(z0))               # (B,N,c_half*p^3)
    zv = enc._tokens_to_voxels(zdp)
    act = {"tokens (B,N,D)": mb(z0), "decoder_proj out (B,N,c_half*p^3)": mb(zdp),
           "voxel map (B,c_half,G^3)": mb(zv)}

    # param counts per module
    def nparams(m):
        return sum(p.numel() for p in m.parameters())
    params = {
        "patch_embed": nparams(enc.patch_embed),
        "blocks(all)": nparams(enc.blocks),
        "per-block": nparams(enc.blocks[0]),
        "decoder_proj": nparams(enc.decoder_proj),
        "recon_mlp(head)": nparams(model.recon_mlp),
        "total": nparams(model),
    }

    emit(f"  forward stage time (ms, mean over {n}; B={B}):")
    for k in stages:
        emit(f"    {k:14s} {acc[k]:7.3f} ms  ({100*acc[k]/fwd_total:5.1f}% of fwd)")
    emit(f"    {'FWD TOTAL':14s} {fwd_total:7.3f} ms")
    emit("  key activation tensors (fp32):")
    for k, v in act.items():
        emit(f"    {k:34s} {v:8.1f} MB")
    emit("  params:")
    for k, v in params.items():
        emit(f"    {k:16s} {v/1e6:6.2f} M")
    rt = acc["tokens->vox"] + acc["patchify"]
    emit(f"  >> roundtrip tax (tokens->vox + patchify) = {rt:.3f} ms/fwd "
         f"({100*rt/fwd_total:.1f}% of fwd) + {act['voxel map (B,c_half,G^3)']:.0f} MB activation")
    return dict(stage_ms=acc, fwd_total_ms=fwd_total, activations_mb=act, params=params,
                roundtrip_tax_ms=rt)


# ════════════════════════════════════════════════════════════════════════════════
# Component: roundtrip — skip _tokens_to_voxels + _patchify in the patch_mlp head
# ════════════════════════════════════════════════════════════════════════════════
def comp_roundtrip(cfg, x, target, device, args, emit):
    emit("==== component: roundtrip (skip identity _patchify(_tokens_to_voxels(.))) ====")
    init = {k: v.detach().cpu().clone()
            for k, v in make_model(cfg, device).state_dict().items()}
    model = make_model(cfg, device, dropout=0.1, init_sd=init)

    base = dict(skip_roundtrip=False)
    var = dict(skip_roundtrip=True)
    res = {}
    for amp in (False, True):
        mb = measure_full(model, x, target, amp=amp, fwd_kwargs=base,
                          warmup=args.warmup, iters=args.iters)
        mv = measure_full(model, x, target, amp=amp, fwd_kwargs=var,
                          warmup=args.warmup, iters=args.iters)
        tag = "amp" if amp else "fp32"
        spd = (1 - mv["step_ms_mean"] / mb["step_ms_mean"]) * 100
        dmem = mb["peak_gb"] - mv["peak_gb"]
        emit(f"  [{tag}] step {mb['step_ms_mean']:.2f} -> {mv['step_ms_mean']:.2f} ms "
             f"({spd:+.1f}%)  peak {mb['peak_gb']:.2f} -> {mv['peak_gb']:.2f} GB ({dmem:+.2f})")
        res[tag] = dict(baseline=mb, variant=mv, speedup_pct=spd, dmem_gb=dmem)
    corr = correctness_full(model, x, target, base_kwargs=base, var_kwargs=var)
    ok = corr["out_maxabs"] <= 1e-5 and corr["grad_maxabs"] <= 1e-4
    emit(f"  correctness: out_max|Δ|={corr['out_maxabs']:.1e} grad_max|Δ|={corr['grad_maxabs']:.1e} "
         f"-> {'SAME-RESULTS (bit-exact)' if ok else 'CHANGES-RESULTS'}")
    res["correctness"] = corr
    res["same_results"] = ok
    return res


# ════════════════════════════════════════════════════════════════════════════════
# Component: checkpoint — activation checkpointing on transformer blocks
# ════════════════════════════════════════════════════════════════════════════════
def comp_checkpoint(cfg, x, target, device, args, emit):
    emit("==== component: checkpoint (activation checkpointing on the 12 blocks) ====")
    init = {k: v.detach().cpu().clone()
            for k, v in make_model(cfg, device).state_dict().items()}
    model = make_model(cfg, device, dropout=0.1, init_sd=init)
    base = dict(ckpt=False)
    var = dict(ckpt=True)
    res = {}
    for amp in (False, True):
        mb = measure_full(model, x, target, amp=amp, fwd_kwargs=base,
                          warmup=args.warmup, iters=args.iters)
        mv = measure_full(model, x, target, amp=amp, fwd_kwargs=var,
                          warmup=args.warmup, iters=args.iters)
        tag = "amp" if amp else "fp32"
        cost = (mv["step_ms_mean"] / mb["step_ms_mean"] - 1) * 100
        memcut = (1 - mv["peak_gb"] / mb["peak_gb"]) * 100
        emit(f"  [{tag}] step {mb['step_ms_mean']:.2f} -> {mv['step_ms_mean']:.2f} ms "
             f"({cost:+.1f}% time)  peak {mb['peak_gb']:.2f} -> {mv['peak_gb']:.2f} GB (-{memcut:.0f}%)")
        res[tag] = dict(baseline=mb, variant=mv, time_cost_pct=cost, mem_cut_pct=memcut)
    corr = correctness_full(model, x, target, base_kwargs=base, var_kwargs=var)
    ok = corr["out_maxabs"] <= 1e-5 and corr["grad_maxabs"] <= 1e-4
    emit(f"  correctness: out_max|Δ|={corr['out_maxabs']:.1e} grad_max|Δ|={corr['grad_maxabs']:.1e} "
         f"-> {'SAME-RESULTS (bit-exact)' if ok else 'CHANGES-RESULTS'}")
    res["correctness"] = corr; res["same_results"] = ok

    # batch-scaling: max batch that fits, baseline vs checkpoint (fp32, the tight case)
    emit("  batch-size scaling (fp32; peak GB, OOM = did not fit):")
    G = cfg["grid_dim"]; Nin = cfg["n_in_channels"]
    scaling = {}
    for B in args.scan_batches:
        row = {}
        for tag, kw in (("baseline", base), ("ckpt", var)):
            def run():
                xx = torch.randn(B, Nin, G, G, G, device=device)
                tt = torch.randn(B, Nin, G, G, G, device=device)
                m = measure_full(model, xx, tt, amp=False, fwd_kwargs=kw,
                                 warmup=2, iters=4)
                del xx, tt
                return m
            out, err = _oom(run)
            torch.cuda.empty_cache()
            row[tag] = "OOM" if err else round(out["peak_gb"], 2)
        emit(f"    B={B:4d}  baseline={row['baseline']}  ckpt={row['ckpt']}")
        scaling[str(B)] = row
    res["scaling"] = scaling
    return res


# ════════════════════════════════════════════════════════════════════════════════
# Component: attn — q/k/v tensor layout micro-bench at the real attention shape
# ════════════════════════════════════════════════════════════════════════════════
def _attn_layout(qkv_lin, proj, x, H, layout):
    B, N, C = x.shape
    hd = C // H
    qkv = qkv_lin(x)
    if layout == "permute":                 # current impl
        q, k, v = qkv.reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4).unbind(0)
    elif layout == "unbind":
        q, k, v = qkv.reshape(B, N, 3, H, hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    elif layout == "chunk":
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, H, hd).transpose(1, 2) for t in (q, k, v))
    else:
        raise ValueError(layout)
    o = F.scaled_dot_product_attention(q, k, v)
    return proj(o.transpose(1, 2).reshape(B, N, C))


def comp_attn(cfg, x, target, device, args, emit):
    emit("==== component: attn (q/k/v tensor layout: permute vs unbind vs chunk) ====")
    B = x.shape[0]; N = (cfg["grid_dim"] // cfg["patch_size"]) ** 3
    C = cfg["dim"]; H = cfg["n_heads"]
    torch.manual_seed(0)
    qkv = nn.Linear(C, 3 * C).to(device)
    proj = nn.Linear(C, C).to(device)
    layouts = ["permute", "unbind", "chunk"]
    res = {}
    for amp in (False, True):
        tag = "amp" if amp else "fp32"
        out_ref = None
        torch.manual_seed(0)
        xb = torch.randn(B, N, C, device=device, requires_grad=True)   # shared across layouts
        for lay in layouts:
            fn = (lambda z, _l=lay: _attn_layout(qkv, proj, z, H, _l))
            if args.compile:
                fn = torch.compile(fn)
            wu = args.warmup_compile if args.compile else args.warmup
            for _ in range(wu):
                with amp_ctx(amp):
                    o = fn(xb)
                o.sum().backward(); xb.grad = None
            sync()
            step_ms = []
            captured = None
            for i in range(args.iters):
                sync(); t = time.perf_counter()
                with amp_ctx(amp):
                    o = fn(xb)
                loss = o.sum(); loss.backward()
                sync(); step_ms.append((time.perf_counter() - t) * 1e3)
                if i == 0:
                    captured = o.detach().float().cpu()
                xb.grad = None
            torch.cuda.reset_peak_memory_stats()
            with amp_ctx(amp):
                o = fn(xb); o.sum().backward()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            ms = statistics.mean(step_ms)
            if out_ref is None:
                out_ref = captured
            dmax = (captured - out_ref).abs().max().item()
            emit(f"  [{tag}] {lay:8s} fwd+bwd {ms:7.3f} ms  peak {peak_gb:.2f} GB  "
                 f"out_max|Δ| vs permute={dmax:.1e}")
            res[f"{tag}_{lay}"] = dict(ms=ms, peak_gb=peak_gb, out_maxabs_vs_permute=dmax)
    return res


# ════════════════════════════════════════════════════════════════════════════════
# Component: patch_embed — Conv3d(k=s=p) vs Linear on unfolded patches (equivalent)
# ════════════════════════════════════════════════════════════════════════════════
def comp_patch_embed(cfg, x, target, device, args, emit):
    emit("==== component: patch_embed (Conv3d k=s=p vs equivalent Linear(unfold)) ====")
    Cin = cfg["n_in_channels"]; D = cfg["dim"]; p = cfg["patch_size"]; gp = cfg["grid_dim"] // p
    B = x.shape[0]
    torch.manual_seed(0)
    conv = nn.Conv3d(Cin, D, kernel_size=p, stride=p).to(device)
    # equivalent Linear: weight = conv.weight.reshape(D, Cin*p^3); same bias
    lin = nn.Linear(Cin * p ** 3, D).to(device)
    with torch.no_grad():
        lin.weight.copy_(conv.weight.reshape(D, Cin * p ** 3))
        lin.bias.copy_(conv.bias)

    def f_conv(xx):
        return conv(xx).flatten(2).transpose(1, 2)                  # (B, N, D)

    def f_lin(xx):
        tok = _patchify(xx, gp, p)                                  # (B, N, Cin*p^3)
        return lin(tok)

    def f_lin_cl(xx):                                               # conv on channels_last_3d
        return None

    variants = [("conv", f_conv, False), ("conv_cl", f_conv, True), ("linear", f_lin, False)]
    res = {}
    ref = None
    for amp in (False, True):
        tag = "amp" if amp else "fp32"
        ref = None
        for name, fn, cl in variants:
            xx = x.to(memory_format=torch.channels_last_3d) if cl else x
            xx = xx.clone().requires_grad_(True)
            fnc = torch.compile(fn) if args.compile else fn
            wu = args.warmup_compile if args.compile else args.warmup
            for _ in range(wu):
                with amp_ctx(amp):
                    o = fnc(xx)
                o.sum().backward(); xx.grad = None
            sync()
            step_ms, captured = [], None
            for i in range(args.iters):
                sync(); t = time.perf_counter()
                with amp_ctx(amp):
                    o = fnc(xx)
                o.sum().backward()
                sync(); step_ms.append((time.perf_counter() - t) * 1e3)
                if i == 0:
                    captured = o.detach().float().cpu()
                xx.grad = None
            ms = statistics.mean(step_ms)
            if ref is None:
                ref = captured
            dmax = (captured - ref).abs().max().item()
            emit(f"  [{tag}] {name:8s} fwd+bwd {ms:7.3f} ms  out_max|Δ| vs conv={dmax:.1e}")
            res[f"{tag}_{name}"] = dict(ms=ms, out_maxabs_vs_conv=dmax)
    return res


COMPONENTS = {
    "profile": comp_profile, "roundtrip": comp_roundtrip, "checkpoint": comp_checkpoint,
    "attn": comp_attn, "patch_embed": comp_patch_embed,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", required=True,
                    help="comma-separated: " + ",".join(COMPONENTS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-in", type=int, default=13)
    ap.add_argument("--grid-dim", type=int, default=64)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--warmup-compile", type=int, default=15)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the timed callable (attn/patch_embed micro-benches)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--scan-batches", type=int, nargs="*", default=[16, 32, 48, 64, 96, 128])
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "voxbind/log"))
    return ap.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    cfg = build_cfg(args)
    gpu = torch.cuda.get_device_name(0)
    B = args.batch_size
    x = torch.randn(B, args.n_in, args.grid_dim, args.grid_dim, args.grid_dim, device=device)
    target = torch.randn(B, args.n_in, args.grid_dim, args.grid_dim, args.grid_dim, device=device)

    requested = [c.strip() for c in args.component.split(",") if c.strip()]
    for c in requested:
        assert c in COMPONENTS, f"unknown component {c!r}; pick from {list(COMPONENTS)}"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for comp in requested:
        log = []
        def emit(line, _log=log):
            print(line, flush=True); _log.append(line)
        emit(f"# gpu={gpu}  B={B} dim={args.dim} depth={args.depth} heads={args.heads} "
             f"n_in={args.n_in} patch={args.patch_size} grid={args.grid_dim}")
        result = COMPONENTS[comp](cfg, x, target, device, args, emit)
        out_path = out_dir / f"260611_vit_impl_{comp}.json"
        with open(out_path, "w") as f:
            json.dump({"meta": dict(gpu=gpu, batch_size=B, cfg=cfg, component=comp),
                       "result": result, "report": log}, f, indent=2)
        emit(f"[saved] {out_path}")
        emit("")


if __name__ == "__main__":
    main()
