"""Parse smoke-test trial logs and emit results.json + a self-contained HTML.

Reads each voxbind/log/smoke_ddp/<trial>/{run.log,meta.json,gpu_{pre,post}.txt}
and produces:
  * results.json  — structured per-trial parse
  * HTML report   — single file, no external CDNs (inline CSS/SVG)

Only depends on stdlib; safe to run inside the conda env without extra deps.
"""
import argparse
import glob
import html
import json
import os
import re
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# log parsing
EPOCH_LINE = re.compile(r"epoch:\s+(\d+)\s+\(([\d.]+)s\)")
TRAIN_LINE = re.compile(r"\[train\]\s+\|\s+loss:\s+([-\d.eE+]+)(?:\s+\|\s+miou:\s+([-\d.eE+]+))?")
VAL_LINE = re.compile(r"\[val\]\s+\|\s+loss:\s+([-\d.eE+]+)(?:\s+\|\s+miou:\s+([-\d.eE+]+))?")
DDP_LINE = re.compile(r"DDP:\s+world_size=(\d+)\s+rank=(\d+)\s+local_rank=(\d+)")
PARAM_LINE = re.compile(r"model has ([\d.]+)M parameters")
TRAIN_BATCHES_LINE = re.compile(r"train batches per rank:\s+(\d+).*effective batch\s+=\s+(\d+)")
TRAIN_VAL_SIZE_LINE = re.compile(r"training/val set size:\s+(\d+)/(\d+)")
RESUMING_LINE = re.compile(r"resuming from:")
RESUMED_EPOCHS_LINE = re.compile(r"model trained for (\d+) epochs?")


def parse_log(log_path: str) -> dict:
    """Parse a single trial's run.log."""
    if not os.path.isfile(log_path):
        return {"parse_error": f"log file not found: {log_path}"}

    epochs = []  # [{epoch, time_s, train_loss, train_miou, val_loss, val_miou}]
    ddp_ranks = set()
    n_params_m = None
    train_batches_per_rank = None
    effective_batch = None
    n_train = n_val = None
    error_lines = []
    resume_observed = False
    resumed_from_epoch = None

    cur = None  # accumulator for the current epoch block
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = DDP_LINE.search(line)
            if m:
                ddp_ranks.add(int(m.group(2)))
                continue

            m = PARAM_LINE.search(line)
            if m:
                n_params_m = float(m.group(1))
                continue

            m = TRAIN_BATCHES_LINE.search(line)
            if m:
                train_batches_per_rank = int(m.group(1))
                effective_batch = int(m.group(2))
                continue

            m = TRAIN_VAL_SIZE_LINE.search(line)
            if m:
                n_train = int(m.group(1))
                n_val = int(m.group(2))
                continue

            if RESUMING_LINE.search(line):
                resume_observed = True
                continue

            m = RESUMED_EPOCHS_LINE.search(line)
            if m:
                resumed_from_epoch = int(m.group(1))
                continue

            m = EPOCH_LINE.search(line)
            if m:
                if cur is not None:
                    epochs.append(cur)
                cur = {
                    "epoch": int(m.group(1)),
                    "time_s": float(m.group(2)),
                    "train_loss": None,
                    "train_miou": None,
                    "val_loss": None,
                    "val_miou": None,
                }
                continue

            m = TRAIN_LINE.search(line)
            if m and cur is not None:
                cur["train_loss"] = float(m.group(1))
                cur["train_miou"] = float(m.group(2)) if m.group(2) else None
                continue

            m = VAL_LINE.search(line)
            if m and cur is not None:
                cur["val_loss"] = float(m.group(1))
                cur["val_miou"] = float(m.group(2)) if m.group(2) else None
                continue

            # Surface common DDP/CUDA errors — but skip benign warnings.
            # PyTorch's distributed logging tags lines as [W...] (warning),
            # [E...] (error), [I...] (info). We only want E-level + Python
            # tracebacks. Also skip lines self-identifying as warnings.
            low = line.lower()
            is_warning = (
                "[w" in line[:20].lower()         # [W514 ...] format
                or "userwarning" in low
                or "not an error" in low
                or "futurewarning" in low
                or "deprecationwarning" in low
                or "warning:" in low
            )
            if not is_warning and any(t in low for t in [
                "traceback", "error", "cuda out of memory", "runtimeerror",
                "ncclerror", "deadlock", " hang ", "killed", "[e",
            ]):
                if len(error_lines) < 40:
                    error_lines.append(line)

    if cur is not None:
        epochs.append(cur)

    return {
        "epochs": epochs,
        "ranks_seen": sorted(ddp_ranks),
        "n_params_m": n_params_m,
        "train_batches_per_rank": train_batches_per_rank,
        "effective_batch": effective_batch,
        "n_train": n_train,
        "n_val": n_val,
        "error_lines": error_lines,
        "resume_observed": resume_observed,
        "resumed_from_epoch": resumed_from_epoch,
    }


def parse_gpu_snapshot(path: str) -> list:
    """Parse nvidia-smi CSV snapshot file (prefixed with [pre]/[post])."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[pre]") and "index" in line.lower():
                continue
            # Strip the "[pre] " / "[post] " prefix
            line = re.sub(r"^\[(pre|post)\]\s*", "", line)
            if line.startswith("index") or not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    out.append({
                        "index": int(parts[0]),
                        "mem_used_mib": int(parts[1].replace("MiB", "").strip()),
                        "mem_total_mib": int(parts[2].replace("MiB", "").strip()),
                        "util_gpu_pct": int(parts[3].replace("%", "").strip()),
                    })
                except ValueError:
                    pass
    return out


def collect_trials(log_root: str) -> list:
    trials = []
    for trial_dir in sorted(glob.glob(os.path.join(log_root, "*"))):
        if not os.path.isdir(trial_dir):
            continue
        trial = os.path.basename(trial_dir)
        meta_path = os.path.join(trial_dir, "meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as e:
                meta = {"_meta_parse_error": str(e)}

        log_path = os.path.join(trial_dir, "run.log")
        parsed = parse_log(log_path)
        gpu_pre = parse_gpu_snapshot(os.path.join(trial_dir, "gpu_pre.txt"))
        gpu_post = parse_gpu_snapshot(os.path.join(trial_dir, "gpu_post.txt"))

        # Validation checks
        checks = run_validation_checks(meta, parsed)

        trials.append({
            "trial": trial,
            "meta": meta,
            "parsed": parsed,
            "gpu_pre": gpu_pre,
            "gpu_post": gpu_post,
            "checks": checks,
        })
    return trials


def run_validation_checks(meta: dict, parsed: dict) -> list:
    """Run autoresearch-style PASS/FAIL checks for this trial.

    Returns a list of {name, status, detail}.
    """
    checks = []
    mode = meta.get("mode", "?")
    nproc = int(meta.get("nproc", 1))

    # 1. Exit code 0
    exit_code = meta.get("exit_code", -1)
    checks.append({
        "name": "process_exit_clean",
        "status": "PASS" if exit_code == 0 else "FAIL",
        "detail": f"exit_code={exit_code}",
    })

    # 2. At least one epoch finished
    n_epochs = len(parsed.get("epochs", []))
    checks.append({
        "name": "at_least_one_epoch",
        "status": "PASS" if n_epochs >= 1 else "FAIL",
        "detail": f"completed {n_epochs} epoch(s)",
    })

    # 3. No NaN/Inf losses
    bad_loss = False
    for ep in parsed.get("epochs", []):
        for k in ("train_loss", "val_loss"):
            v = ep.get(k)
            if v is None:
                continue
            if v != v or v == float("inf") or v == float("-inf"):
                bad_loss = True
    checks.append({
        "name": "losses_finite",
        "status": "PASS" if not bad_loss else "FAIL",
        "detail": "no NaN/Inf detected" if not bad_loss else "NaN or Inf observed",
    })

    # 4. DDP-only: all expected ranks appeared in the log
    if mode == "ddp":
        seen = parsed.get("ranks_seen", [])
        ok = len(seen) == nproc and seen == list(range(nproc))
        checks.append({
            "name": "all_ddp_ranks_initialized",
            "status": "PASS" if ok else "FAIL",
            "detail": f"expected ranks 0..{nproc-1}, saw {seen}",
        })

    # 5. No error keywords surfaced
    err = parsed.get("error_lines", [])
    checks.append({
        "name": "no_error_keywords",
        "status": "PASS" if not err else "FAIL",
        "detail": "clean" if not err else f"{len(err)} suspicious line(s)",
    })

    # 6. Checkpoint file got written
    exp_dir = meta.get("exp_dir", "")
    ckpt = os.path.join(exp_dir, "checkpoint.pth.tar")
    checks.append({
        "name": "checkpoint_written",
        "status": "PASS" if os.path.isfile(ckpt) else "FAIL",
        "detail": ckpt,
    })

    # 7. Convergence: if we ran >= 2 epochs, the last epoch's train_loss
    # should be lower than the first epoch's. 5% margin to avoid noise.
    epochs = parsed.get("epochs", [])
    train_losses = [e.get("train_loss") for e in epochs if e.get("train_loss") is not None]
    if len(train_losses) >= 2:
        first, last = train_losses[0], train_losses[-1]
        # Allow slight noise: pass if last < first * 0.99
        improved = last < first * 0.99
        checks.append({
            "name": "loss_decreased",
            "status": "PASS" if improved else "FAIL",
            "detail": f"epoch[0].train_loss={first:.2f} -> epoch[-1].train_loss={last:.2f}",
        })

    # 8. Resume-only: did train_ddp.py log a resume-from line and a non-zero
    # epoch count from the saved checkpoint?
    trial_name = meta.get("trial") or ""
    is_resume_trial = "resume" in trial_name.lower() or "resume" in str(meta.get("overrides", ""))
    if is_resume_trial:
        rfe = parsed.get("resumed_from_epoch")
        checks.append({
            "name": "resume_detected",
            "status": "PASS" if (parsed.get("resume_observed") and rfe is not None and rfe >= 1) else "FAIL",
            "detail": (
                f"resume_observed={parsed.get('resume_observed')} resumed_from_epoch={rfe}"
            ),
        })

    return checks


# ---------------------------------------------------------------------------
# HTML rendering (single self-contained file, no CDNs)

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
       Arial, sans-serif; margin: 0; padding: 24px; background: #fafafa;
       color: #222; max-width: 1100px; margin-left: auto; margin-right: auto; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
h3 { margin-top: 24px; }
.subtitle { color: #666; margin-bottom: 24px; }
.hypothesis { background: #f0f7ff; border-left: 4px solid #3b82f6;
              padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
        background: white; font-size: 13px; }
th, td { border: 1px solid #e5e5e5; padding: 8px 10px; text-align: left;
         vertical-align: top; }
th { background: #f5f5f5; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
.pass { color: #15803d; font-weight: 600; }
.fail { color: #b91c1c; font-weight: 600; }
.warn { color: #b45309; font-weight: 600; }
.kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background: #f1f3f5; padding: 2px 5px; border-radius: 3px;
       font-size: 12px; }
.bar-row { display: flex; align-items: center; margin: 4px 0; }
.bar-label { width: 110px; font-size: 13px; }
.bar-track { flex: 1; background: #eef; height: 18px; border-radius: 3px;
             position: relative; overflow: hidden; }
.bar-fill { background: #3b82f6; height: 100%; }
.bar-value { width: 90px; padding-left: 8px; font-size: 12px; color: #555; }
details { background: white; border: 1px solid #ddd; border-radius: 4px;
          padding: 10px 14px; margin: 8px 0; }
details summary { cursor: pointer; font-weight: 600; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 4px;
      overflow-x: auto; font-size: 12px; line-height: 1.45; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.footer { color: #888; font-size: 12px; margin-top: 48px; border-top: 1px solid #ddd;
          padding-top: 12px; }
.svg-plot { background: white; border: 1px solid #e5e5e5; border-radius: 4px;
            padding: 8px; }
"""


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else "—"


def status_span(s: str) -> str:
    cls = "pass" if s == "PASS" else "fail"
    return f'<span class="{cls}">{s}</span>'


def render_bar(label: str, value: float, max_value: float, value_str: str) -> str:
    pct = (value / max_value * 100.0) if max_value > 0 else 0
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{esc(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="bar-value">{esc(value_str)}</div>'
        f'</div>'
    )


def render_loss_svg(trials: list) -> str:
    """Tiny inline SVG: train_loss by epoch, one line per trial."""
    # Collect (trial, [(epoch, train_loss)])
    series = []
    for t in trials:
        pts = [(e["epoch"], e["train_loss"]) for e in t["parsed"].get("epochs", [])
               if e.get("train_loss") is not None]
        if pts:
            series.append((t["trial"], pts))
    if not series:
        return "<p><em>(no training loss recorded — runs may have failed before completing an epoch)</em></p>"

    all_x = [p[0] for _, pts in series for p in pts]
    all_y = [p[1] for _, pts in series for p in pts]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1e-9

    W, H = 700, 280
    pad_l, pad_r, pad_t, pad_b = 60, 140, 20, 36

    def sx(x): return pad_l + (x - x_min) / (x_max - x_min) * (W - pad_l - pad_r)
    def sy(y): return pad_t + (1 - (y - y_min) / (y_max - y_min)) * (H - pad_t - pad_b)

    colors = ["#3b82f6", "#ef4444", "#10b981", "#a855f7", "#f59e0b"]
    svg = [f'<svg class="svg-plot" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    # axes
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H-pad_b}" stroke="#999"/>')
    svg.append(f'<line x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" y2="{H-pad_b}" stroke="#999"/>')
    # y ticks
    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        py = sy(y)
        svg.append(f'<line x1="{pad_l-3}" y1="{py}" x2="{pad_l}" y2="{py}" stroke="#999"/>')
        svg.append(f'<text x="{pad_l-6}" y="{py+4}" font-size="10" text-anchor="end" fill="#555">{y:.2g}</text>')
    # x ticks
    for x in range(int(x_min), int(x_max)+1):
        px = sx(x)
        svg.append(f'<line x1="{px}" y1="{H-pad_b}" x2="{px}" y2="{H-pad_b+3}" stroke="#999"/>')
        svg.append(f'<text x="{px}" y="{H-pad_b+16}" font-size="10" text-anchor="middle" fill="#555">{x}</text>')
    # axis labels
    svg.append(f'<text x="{(pad_l+W-pad_r)/2}" y="{H-6}" font-size="11" text-anchor="middle" fill="#333">epoch</text>')
    svg.append(f'<text x="14" y="{(pad_t+H-pad_b)/2}" font-size="11" text-anchor="middle" fill="#333" transform="rotate(-90 14,{(pad_t+H-pad_b)/2})">train loss</text>')
    # series
    for i, (name, pts) in enumerate(series):
        c = colors[i % len(colors)]
        path = " ".join([f"{('M' if j==0 else 'L')} {sx(x):.1f} {sy(y):.1f}" for j,(x,y) in enumerate(pts)])
        svg.append(f'<path d="{path}" stroke="{c}" stroke-width="2" fill="none"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}"/>')
        # legend
        ly = pad_t + 14 + i*18
        svg.append(f'<line x1="{W-pad_r+14}" y1="{ly}" x2="{W-pad_r+34}" y2="{ly}" stroke="{c}" stroke-width="2"/>')
        svg.append(f'<text x="{W-pad_r+38}" y="{ly+4}" font-size="11" fill="#333">{esc(name)}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def render_html(trials: list, out_path: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build trial table
    rows = []
    rows.append(
        "<tr>"
        "<th>trial</th><th>mode</th><th>gpus</th><th>nproc</th><th>bsz</th>"
        "<th>eff. batch</th><th>n_params</th><th>wall_s</th><th>epoch[0] s</th>"
        "<th>train_loss[last]</th><th>val_loss[last]</th><th>checks</th>"
        "</tr>"
    )
    for t in trials:
        meta = t["meta"]
        p = t["parsed"]
        epochs = p.get("epochs", [])
        last = epochs[-1] if epochs else {}
        first = epochs[0] if epochs else {}
        passed = sum(1 for c in t["checks"] if c["status"] == "PASS")
        total = len(t["checks"])
        check_cls = "pass" if passed == total else "fail"
        rows.append(
            "<tr>"
            f"<td><b>{esc(t['trial'])}</b></td>"
            f"<td>{esc(meta.get('mode'))}</td>"
            f"<td><span class='kbd'>{esc(meta.get('gpus'))}</span></td>"
            f"<td>{esc(meta.get('nproc'))}</td>"
            f"<td>{esc(meta.get('bsz'))}</td>"
            f"<td>{esc(p.get('effective_batch'))}</td>"
            f"<td>{esc(p.get('n_params_m'))}M</td>"
            f"<td>{esc(meta.get('wall_clock_s'))}</td>"
            f"<td>{esc(first.get('time_s'))}</td>"
            f"<td>{esc(last.get('train_loss'))}</td>"
            f"<td>{esc(last.get('val_loss'))}</td>"
            f"<td class='{check_cls}'>{passed}/{total}</td>"
            "</tr>"
        )
    trial_table = '<table>' + "".join(rows) + '</table>'

    # Throughput bar chart (epoch[0] seconds, lower is better)
    bars = []
    times = []
    for t in trials:
        eps = t["parsed"].get("epochs", [])
        if eps and eps[0].get("time_s") is not None:
            times.append((t["trial"], eps[0]["time_s"]))
    if times:
        max_t = max(v for _, v in times)
        for name, v in times:
            bars.append(render_bar(name, v, max_t, f"{v:.2f}s"))
    bar_block = "\n".join(bars) if bars else "<p><em>(no per-epoch timing available)</em></p>"

    # Per-trial deep-dive sections
    deep_sections = []
    for t in trials:
        meta = t["meta"]
        p = t["parsed"]
        # Checks list
        check_html = '<table><tr><th>check</th><th>status</th><th>detail</th></tr>'
        for c in t["checks"]:
            check_html += f"<tr><td>{esc(c['name'])}</td><td>{status_span(c['status'])}</td><td>{esc(c['detail'])}</td></tr>"
        check_html += "</table>"
        # Epochs table
        ep_html = '<table><tr><th>epoch</th><th>time_s</th><th>train_loss</th><th>train_miou</th><th>val_loss</th><th>val_miou</th></tr>'
        for e in p.get("epochs", []):
            ep_html += (
                f"<tr><td>{esc(e.get('epoch'))}</td>"
                f"<td>{esc(e.get('time_s'))}</td>"
                f"<td>{esc(e.get('train_loss'))}</td>"
                f"<td>{esc(e.get('train_miou'))}</td>"
                f"<td>{esc(e.get('val_loss'))}</td>"
                f"<td>{esc(e.get('val_miou'))}</td></tr>"
            )
        ep_html += "</table>"
        # GPU snapshots
        gpu_html = ""
        for label, snap in (("pre-launch", t["gpu_pre"]), ("post-finish", t["gpu_post"])):
            if not snap:
                continue
            gpu_html += f"<h4>{label}</h4><table><tr><th>idx</th><th>mem_used</th><th>util</th></tr>"
            for g in snap:
                gpu_html += (
                    f"<tr><td>{g['index']}</td>"
                    f"<td>{g['mem_used_mib']}/{g['mem_total_mib']} MiB</td>"
                    f"<td>{g['util_gpu_pct']}%</td></tr>"
                )
            gpu_html += "</table>"
        # Errors
        err_block = ""
        errs = p.get("error_lines", [])
        if errs:
            err_block = (
                "<h4>Suspicious log lines</h4><pre>"
                + esc("\n".join(errs))
                + "</pre>"
            )

        deep_sections.append(
            f"<details><summary>{esc(t['trial'])}  "
            f"(mode={esc(meta.get('mode'))}, gpus={esc(meta.get('gpus'))})</summary>"
            f"<h4>Validation checks</h4>{check_html}"
            f"<h4>Per-epoch metrics</h4>{ep_html}"
            f"{gpu_html}{err_block}"
            f"</details>"
        )

    overall_pass = all(
        all(c["status"] == "PASS" for c in t["checks"]) for t in trials
    )

    body = f"""
<h1>DDP smoke test — VoxBind</h1>
<div class="subtitle">Generated {now} · trials live in <code>voxbind/log/smoke_ddp/</code></div>

<div class="hypothesis">
  <b>Hypothesis.</b> Replacing <code>DataParallel</code> with
  <code>DistributedDataParallel</code> in the training loop (via <code>voxbind/train_ddp.py</code>)
  preserves end-to-end functionality: training launches, all ranks contribute, the
  forward/backward all-reduce produces finite losses, and a checkpoint is written.
  This is a <em>functional</em> smoke test only — not a convergence test.
</div>

<h2>Overall</h2>
<p>Status: {'<span class="pass">PASS</span>' if overall_pass else '<span class="fail">FAIL</span>'}
   &nbsp;&nbsp; (all checks PASS across every trial)</p>

<h2>Trial summary</h2>
{trial_table}

<h2>Epoch-0 wall-clock (lower is better)</h2>
<p>First-epoch time on the same effective batch (lets you eyeball DP vs DDP overhead under contention).</p>
{bar_block}

<h2>Train loss per epoch</h2>
{render_loss_svg(trials)}

<h2>Per-trial detail</h2>
{''.join(deep_sections)}

<div class="footer">
Trial configs: dset=crossdocked, with_density=false, subset_n=8, num_epochs=2.
Generated by <code>voxbind/scripts/04a_test_ddp_to_html.py</code>.
</div>
"""
    out = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>VoxBind DDP smoke test</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--out-html", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    trials = collect_trials(args.log_root)

    # serialize JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "trials": trials}, f, indent=2)
    print(f"wrote {args.out_json}")

    render_html(trials, args.out_html)


if __name__ == "__main__":
    main()
