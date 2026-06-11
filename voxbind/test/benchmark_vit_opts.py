#!/usr/bin/env python
"""Autoresearch-style smoke bench for DensityViTMAE training-step optimizations.

Runs an iterative research loop — baseline -> hypothesis -> stack-a-knob ->
measure -> keep-or-reject — over the speed/memory levers identified for the ViT:

    amp (bf16 autocast) -> channels_last_3d -> fused AdamW -> foreach EMA ->
    torch.compile           (main line, learnable PE)
    + rope_fast             (side experiment, rope3d PE only)

Each knob is timed as a FULL train step (fwd + loss + backward + grad-clip +
optimizer.step + EMA.update) and gated for correctness against the fp32
reference with a TIERED tolerance:

  near-bitwise knobs (channels_last / fused / foreach_ema / rope_fast):
      forward output max|Δ| small AND post-K-step weight max|Δ| ~ float noise.
  precision-changing knobs (amp / compile):
      output cosine >= 0.999 and per-step loss relative error <= 1%.

Data-free (fixed-seed random input) to isolate model fwd/bwd/opt/ema from the
DataLoader + voxelizer. Intended to run pinned to one idle GPU, e.g.
    CUDA_VISIBLE_DEVICES=3 python voxbind/test/benchmark_vit_opts.py
"""

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxbind.models.density_vit import DensityViTMAE
from voxbind.models.ema import ModelEma
from voxbind.models.adamw import AdamW as CustomAdamW


# ── knob set: each is a delta applied on top of the incumbent ─────────────────
DEFAULT_STACK = ["amp", "channels_last", "fused_adamw", "foreach_ema", "compile"]

HYP = {
    "amp":           "bf16 autocast halves matmul cost + unlocks flash SDPA",
    "channels_last": "channels_last_3d gives cuDNN better Conv3d kernels (patch-embed/heads)",
    "fused_adamw":   "fused AdamW collapses the 45M-param update into one kernel",
    "foreach_ema":   "foreach EMA replaces the per-tensor Python lerp with 2 multi-tensor kernels",
    "compile":       "torch.compile fuses the transformer + elementwise ops",
    "rope_fast":     "fused q/k rotate + fewer temporaries in RoPE.rotate()",
}
# correctness tier per knob:
#   'bitwise' — pure reordering of identical math (fp32) → expect ~float noise (<=1e-4)
#   'band'    — changes the numeric kernel/precision (bf16 / TF32 conv-algo / compile
#               fp-reassoc) → judge by cosine>=0.999 + loss rel-err, NOT bitwise
TIER = {
    "amp": "band", "compile": "band", "channels_last": "band",
    "fused_adamw": "bitwise", "foreach_ema": "bitwise", "rope_fast": "bitwise",
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clean(name):
    """Strip compile/DDP wrapper prefixes so weight keys line up across variants."""
    return name.replace("_orig_mod.", "").replace("module.", "")


def _float_weights(model):
    return {_clean(n): p_.detach().float().cpu()
            for n, p_ in model.state_dict().items() if p_.is_floating_point()}


def build_cfg(args):
    return dict(
        grid_dim=args.grid_dim, patch_size=args.patch_size, n_in_channels=args.n_in,
        n_recon_channels=args.n_in, n_channels=32, dim=args.dim, depth=args.depth,
        n_heads=args.heads, mlp_ratio=4, dropout=0.1, n_struct_channels=0,
        pretext_style="mae", dual_head=False, head_hidden_dim=67, head_depth=2,
        head_style="patch_mlp",
    )


def make_model(cfg, *, pos_encoding, rope_fast, channels_last, device, init_sd=None, dropout=None):
    torch.manual_seed(0)
    cfg2 = cfg if dropout is None else {**cfg, "dropout": dropout}
    m = DensityViTMAE(**cfg2, pos_encoding=pos_encoding, rope_fast=rope_fast).to(device)
    if init_sd is not None:
        m.load_state_dict(init_sd)
    if channels_last:
        m = m.to(memory_format=torch.channels_last_3d)
    m.train()
    return m


def make_opt(model, fused):
    if fused:
        # vendored-matching betas=(0.99,0.999); eps=1e-8 (not the vendored eps=0 — torch's
        # fused kernel nans at eps=0; v_hat>>1e-8 so the change is numerically negligible).
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2,
                                 betas=(0.99, 0.999), eps=1e-8, fused=True)
    return CustomAdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)


def amp_ctx(enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled \
        else contextlib.nullcontext()


def train_step(model, opt, ema, x, target, *, amp, channels_last, capture_out=False):
    """One full training step; returns (loss_value, optional step-0 forward output)."""
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    opt.zero_grad(set_to_none=True)
    with amp_ctx(amp):
        out_pretext, _ = model(x)
    out = out_pretext.float()
    loss = F.mse_loss(out, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema.update(model)
    captured = out.detach().clone() if capture_out else None
    return loss.detach(), captured


def knobs_for(stack):
    s = set(stack)
    return dict(amp="amp" in s, channels_last="channels_last" in s,
                fused="fused_adamw" in s, foreach_ema="foreach_ema" in s,
                compile="compile" in s)


# ── timing ────────────────────────────────────────────────────────────────────
def measure(cfg, stack, *, pos_encoding, rope_fast, x, target, device, args):
    k = knobs_for(stack)
    model = make_model(cfg, pos_encoding=pos_encoding, rope_fast=rope_fast,
                       channels_last=k["channels_last"], device=device)
    if k["compile"]:
        model = torch.compile(model, mode=args.compile_mode)
    opt = make_opt(model, k["fused"])
    ema = ModelEma(model, decay=0.999, foreach=k["foreach_ema"])

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.warmup_compile if k["compile"] else args.warmup):
        train_step(model, opt, ema, x, target, amp=k["amp"], channels_last=k["channels_last"])
    sync()
    warmup_s = time.perf_counter() - t0

    step_ms, fwd_ms = [], []
    for _ in range(args.iters):
        sync(); t = time.perf_counter()
        train_step(model, opt, ema, x, target, amp=k["amp"], channels_last=k["channels_last"])
        sync(); step_ms.append((time.perf_counter() - t) * 1e3)
        # fwd-only
        sync(); t = time.perf_counter()
        with torch.no_grad(), amp_ctx(k["amp"]):
            xc = x.to(memory_format=torch.channels_last_3d) if k["channels_last"] else x
            model(xc)
        sync(); fwd_ms.append((time.perf_counter() - t) * 1e3)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res = dict(step_ms=step_ms, fwd_ms=fwd_ms, peak_gb=peak_gb, warmup_s=warmup_s)
    del model, opt, ema
    torch.cuda.empty_cache()
    return res


def p(vals, q):
    v = sorted(vals)
    return v[int(q * (len(v) - 1))]


# ── correctness vs fp32 reference ───────────────────────────────────────────────
def run_ref_and_capture(cfg, *, pos_encoding, x, target, device, init_sd, k_steps):
    """fp32 reference: fixed init+input, K steps. Capture step-0 fwd out + final weights.

    dropout=0 so the only source of divergence vs a candidate is the knob's numerics
    (otherwise the per-forward dropout/SDPA-dropout mask would swamp the comparison)."""
    model = make_model(cfg, pos_encoding=pos_encoding, rope_fast=False,
                       channels_last=False, device=device, init_sd=init_sd, dropout=0.0)
    opt = make_opt(model, fused=False)
    ema = ModelEma(model, decay=0.999, foreach=False)
    out0 = losses = None
    losses = []
    for i in range(k_steps):
        loss, cap = train_step(model, opt, ema, x, target, amp=False,
                               channels_last=False, capture_out=(i == 0))
        if i == 0:
            out0 = cap
        losses.append(loss.item())
    final = _float_weights(model)
    del model, opt, ema; torch.cuda.empty_cache()
    return dict(out0=out0.cpu(), losses=losses, final=final)


def check(cfg, stack, *, pos_encoding, rope_fast, x, target, device, init_sd, ref, args):
    k = knobs_for(stack)
    model = make_model(cfg, pos_encoding=pos_encoding, rope_fast=rope_fast,
                       channels_last=k["channels_last"], device=device, init_sd=init_sd,
                       dropout=0.0)
    if k["compile"]:
        model = torch.compile(model, mode=args.compile_mode)
    opt = make_opt(model, k["fused"])
    ema = ModelEma(model, decay=0.999, foreach=k["foreach_ema"])
    out0 = None
    losses = []
    for i in range(args.corr_steps):
        loss, cap = train_step(model, opt, ema, x, target, amp=k["amp"],
                               channels_last=k["channels_last"], capture_out=(i == 0))
        if i == 0:
            out0 = cap
        losses.append(loss.item())
    final = _float_weights(model)

    # float64 reduction: a plain fp32 cosine over ~100M elements accumulates enough
    # rounding to read >1; double() makes it reliable. Also report relative-L2.
    a, b = out0.cpu().double().flatten(), ref["out0"].double().flatten()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    out_cos = min(1.0, ((a * b).sum() / denom).item())
    out_rel_l2 = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    out_maxabs = (a - b).abs().max().item()
    # final-weight max|Δ| over shared float tensors
    w_max = max(((final[n] - ref["final"][n]).abs().max().item())
                for n in ref["final"] if n in final)
    loss_relerr = max(abs(l - r) / (abs(r) + 1e-12)
                      for l, r in zip(losses, ref["losses"]))
    del model, opt, ema; torch.cuda.empty_cache()
    return dict(out_cos=out_cos, out_rel_l2=out_rel_l2, out_maxabs=out_maxabs,
                w_maxabs=w_max, loss_relerr=loss_relerr)


def gate(knob, corr, args, floor):
    """Tiered accept on correctness. Returns (ok, reason).

    `floor` = run-to-run nondeterminism of the fp32 baseline (ref vs ref): 3D-conv /
    attention backward use atomic adds, so two identical fp32 runs already differ by
    ~floor after K steps. A 'bitwise' knob is judged same-results if its divergence is
    within that floor (i.e. indistinguishable from rerunning the baseline)."""
    if TIER[knob] == "band":
        ok = corr["out_cos"] >= args.cos_tol and corr["loss_relerr"] <= args.relerr_tol
        return ok, (f"cos={corr['out_cos']:.5f} relL2={corr['out_rel_l2']:.2e} "
                    f"loss_relerr={corr['loss_relerr']:.2e}")
    w_tol = max(args.bitwise_tol, 3.0 * floor["w"])
    o_tol = max(args.bitwise_tol, 3.0 * floor["out"])
    ok = corr["w_maxabs"] <= w_tol and corr["out_maxabs"] <= o_tol
    return ok, (f"out_max|Δ|={corr['out_maxabs']:.1e} w_max|Δ|={corr['w_maxabs']:.1e} "
                f"(nondet floor={floor['w']:.1e})")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-in", type=int, default=12)
    ap.add_argument("--grid-dim", type=int, default=64)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--warmup-compile", type=int, default=15)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--corr-steps", type=int, default=5)
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--cos-tol", type=float, default=0.999)
    ap.add_argument("--relerr-tol", type=float, default=0.01)
    ap.add_argument("--bitwise-tol", type=float, default=1e-4)
    ap.add_argument("--stack", nargs="*", default=DEFAULT_STACK)
    ap.add_argument("--out", default=str(REPO_ROOT / "voxbind/log/260611_vit_opts_bench.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    torch.set_float32_matmul_precision("high")   # TF32 on — matches training
    torch.manual_seed(0)
    cfg = build_cfg(args)
    gpu = torch.cuda.get_device_name(0)

    B = args.batch_size
    x = torch.randn(B, args.n_in, args.grid_dim, args.grid_dim, args.grid_dim, device=device)
    # forward output is (B, n_recon, G,G,G) — fixed target for a realistic MSE loss
    target = torch.randn(B, args.n_in, args.grid_dim, args.grid_dim, args.grid_dim, device=device)

    log = []

    def emit(line):
        print(line, flush=True)
        log.append(line)

    emit("==== ViT pretraining optimization — autoresearch ====")
    emit(f"gpu={gpu}  B={B} dim={args.dim} depth={args.depth} heads={args.heads} "
         f"n_in={args.n_in} patch={args.patch_size} grid={args.grid_dim} head=patch_mlp")
    emit(f"metric: full train-step ms/iter (fwd+bwd+clip+opt+ema); compile_mode={args.compile_mode}")
    emit(f"gate(bitwise) max|Δ|<= {args.bitwise_tol:g} ; gate(band) cos>= {args.cos_tol} & "
         f"loss_relerr<= {args.relerr_tol}")
    emit("")

    # reference init weights (fixed) for correctness comparisons
    ref_model = make_model(cfg, pos_encoding="learnable", rope_fast=False,
                           channels_last=False, device=device)
    init_sd = {k_: v.detach().cpu().clone() for k_, v in ref_model.state_dict().items()}
    del ref_model; torch.cuda.empty_cache()

    # ---- baseline (fp32, learnable, custom AdamW loop, loop EMA) ----
    base = measure(cfg, [], pos_encoding="learnable", rope_fast=False,
                   x=x, target=target, device=device, args=args)
    ref = run_ref_and_capture(cfg, pos_encoding="learnable", x=x, target=target,
                              device=device, init_sd=init_sd, k_steps=args.corr_steps)
    # second identical fp32 run → nondeterminism floor (conv/attn backward atomic adds)
    ref2 = run_ref_and_capture(cfg, pos_encoding="learnable", x=x, target=target,
                               device=device, init_sd=init_sd, k_steps=args.corr_steps)
    floor = dict(out=(ref["out0"] - ref2["out0"]).abs().max().item(),
                 w=max((ref["final"][n] - ref2["final"][n]).abs().max().item() for n in ref["final"]))
    base_step = statistics.mean(base["step_ms"])
    emit(f"baseline [fp32, learnable, customAdamW-loop, loop-EMA]")
    emit(f"  step={base_step:7.2f} ms/iter  p50={p(base['step_ms'],.5):.2f} "
         f"p90={p(base['step_ms'],.9):.2f}  fwd={statistics.mean(base['fwd_ms']):.2f} ms  "
         f"peak={base['peak_gb']:.2f} GB  -> reference captured")
    emit(f"  nondeterminism floor (fp32 ref vs ref): out={floor['out']:.1e} "
         f"weights={floor['w']:.1e} after {args.corr_steps} steps")
    emit("")

    results = {"baseline": dict(step_ms_mean=base_step, peak_gb=base["peak_gb"],
                                fwd_ms_mean=statistics.mean(base["fwd_ms"]))}

    # ---- Phase A: per-knob ISOLATION — each knob alone vs the fp32 baseline. ----
    # Correctness is the marginal effect of THIS knob (no other knob's error mixed in);
    # speed is this knob alone vs baseline. This is the "does each component keep the
    # same results" answer.
    emit("Phase A — per-knob isolation (each alone vs fp32 baseline)")
    corr_ok = {}
    exp = 0
    for knob in args.stack:
        exp += 1
        m = measure(cfg, [knob], pos_encoding="learnable", rope_fast=False,
                    x=x, target=target, device=device, args=args)
        corr = check(cfg, [knob], pos_encoding="learnable", rope_fast=False, x=x, target=target,
                     device=device, init_sd=init_sd, ref=ref, args=args)
        cand_step = statistics.mean(m["step_ms"])
        ratio = cand_step / base_step
        ok, why = gate(knob, corr, args, floor)
        corr_ok[knob] = ok
        spd = f"{(1 - ratio) * 100:+.0f}% vs base" if ratio <= 1 else f"{(ratio - 1) * 100:+.0f}% slower"
        verdict = "same-results" if ok else "CHANGES-RESULTS"
        emit(f"exp {exp} [{knob:13s}] {HYP[knob]}")
        emit(f"        step={cand_step:7.2f} ms ({spd})  peak={m['peak_gb']:.2f}GB | "
             f"corr[{TIER[knob]}]: {why} -> {verdict}")
        results[f"isolated_{knob}"] = dict(step_ms_mean=cand_step, ratio_vs_base=ratio,
                                           peak_gb=m["peak_gb"], same_results=ok, **corr)
    emit("")

    # ---- Phase B: cumulative STACK — add correctness-passing knobs in order. ----
    emit("Phase B — cumulative stack (marginal speed as each knob is added)")
    incumbent, inc_step = [], base_step
    for knob in args.stack:
        if not corr_ok[knob]:
            emit(f"   skip {knob} (changes results)")
            continue
        cand = incumbent + [knob]
        m = measure(cfg, cand, pos_encoding="learnable", rope_fast=False,
                    x=x, target=target, device=device, args=args)
        cand_step = statistics.mean(m["step_ms"])
        ratio = cand_step / inc_step
        spd = f"{(1 - ratio) * 100:+.0f}%" if ratio <= 1 else f"{(ratio - 1) * 100:+.0f}% slower"
        emit(f"   +{knob:13s} -> {cand_step:7.2f} ms ({spd} vs inc)  peak={m['peak_gb']:.2f}GB")
        results[f"stack_{knob}"] = dict(step_ms_mean=cand_step, ratio_vs_inc=ratio,
                                        peak_gb=m["peak_gb"], stack=list(cand))
        incumbent, inc_step, inc_peak = cand, cand_step, m["peak_gb"]
    # overall correctness of the full optimized stack vs fp32 baseline (band: amp dominates)
    full_corr = check(cfg, incumbent, pos_encoding="learnable", rope_fast=False, x=x, target=target,
                      device=device, init_sd=init_sd, ref=ref, args=args)
    cum = base_step / inc_step
    emit("")
    emit(f"  -> cumulative: {base_step:.2f} ms -> {inc_step:.2f} ms  ({cum:.2f}x faster)  "
         f"peak {base['peak_gb']:.2f} -> {inc_peak:.2f} GB")
    emit(f"     full-stack {incumbent} vs fp32: cos={full_corr['out_cos']:.5f} "
         f"loss_relerr={full_corr['loss_relerr']:.2e}")
    results["cumulative"] = dict(base_ms=base_step, final_ms=inc_step, speedup=cum,
                                 base_peak_gb=base["peak_gb"], final_peak_gb=inc_peak,
                                 stack=list(incumbent), full_corr=full_corr)
    emit("")

    # ---- side experiment: rope3d PE, rope_fast on top of the accepted stack ----
    emit("side experiment [rope3d PE] — rope_fast vs original rotate()")
    emit("  (fp32 rows isolate rope_fast's effect; +stack rows show it under amp+channels_last)")
    stack_no_compile = [s for s in incumbent if s != "compile"]   # accepted knobs sans compile
    # rope3d has no learnable pos_embed → needs its own reference init weights.
    r_model = make_model(cfg, pos_encoding="rope3d", rope_fast=False,
                         channels_last=False, device=device)
    rope_init_sd = {k_: v.detach().cpu().clone() for k_, v in r_model.state_dict().items()}
    del r_model; torch.cuda.empty_cache()
    r_ref = run_ref_and_capture(cfg, pos_encoding="rope3d", x=x, target=target,
                                device=device, init_sd=rope_init_sd, k_steps=args.corr_steps)
    rope_variants = [
        ("rope3d/fp32",                False, []),
        ("rope3d+fast/fp32",           True,  []),
        ("rope3d+fast/+stack",         True,  stack_no_compile),
        ("rope3d+fast/+stack+compile", True,  stack_no_compile + ["compile"]),
    ]
    for tag, rf, st in rope_variants:
        m = measure(cfg, st, pos_encoding="rope3d", rope_fast=rf,
                    x=x, target=target, device=device, args=args)
        corr = check(cfg, st, pos_encoding="rope3d", rope_fast=rf, x=x, target=target,
                     device=device, init_sd=rope_init_sd, ref=r_ref, args=args)
        ms = statistics.mean(m["step_ms"])
        emit(f"  {tag:28s} step={ms:7.2f} ms  peak={m['peak_gb']:.2f}GB  "
             f"corr vs fp32-rope: out_max|Δ|={corr['out_maxabs']:.1e} w_max|Δ|={corr['w_maxabs']:.1e}")
        results[f"side_{tag.strip().replace('+','_').replace(' ','')}"] = dict(
            step_ms_mean=ms, peak_gb=m["peak_gb"], **corr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": dict(gpu=gpu, cfg=cfg, batch_size=B,
                                compile_mode=args.compile_mode), "results": results,
                   "report": log}, f, indent=2)
    emit("")
    emit(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
