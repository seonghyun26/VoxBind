#!/usr/bin/env python
"""Live experiment-status dashboard for the VoxBind density/affinity line.

Reads the REAL state on every request — running torchrun jobs (+ epoch/GPUs from
/proc), armed watch_n_launch chains (+ their gate file), background precomputes,
and finished probe CSVs — so the page always reflects what is actually happening.
No hard-coded "running/done"; refresh (or wait 30 s) to update.

Run:
    python notebook/html/experiment_status_app.py --port 8732 --host 0.0.0.0
Then browse  http://<host>:8732/
"""
import argparse, os, re, csv, html, datetime, statistics
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOX = Path(__file__).resolve().parents[2] / "voxbind"   # notebook/html/ -> VoxBind/ -> voxbind
PDB = VOX / "dataset" / "data" / "pdbbind"
LOG = VOX / "log"


def esc(x): return html.escape(str(x))


# ── live process state (via /proc) ────────────────────────────────────────────
def _proc_iter():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "ignore")
        except Exception:
            continue
        # Skip Claude-Code/IDE transient shell wrappers (zsh -c that `source`s a shell-snapshot
        # then `eval`s a command): their cmdline echoes whatever job string they launched, which
        # otherwise reads as a false-positive running job long after the real (detached) child exits.
        if "shell-snapshots" in cmd:
            continue
        yield pid, cmd


def _environ(pid, key):
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().decode("utf8", "ignore")
        m = re.search(rf"{key}=([^\x00]*)", env)
        return m.group(1) if m else None
    except Exception:
        return None


def _epoch_of(exp):
    p = LOG / f"{exp}.log"
    if not p.exists():
        return None
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        return None
    eps = re.findall(r">> epoch:\s+(\d+)\s*\(", txt)
    if eps:
        return int(eps[-1])
    if re.search(r"start density-ViT-MAE", txt):
        return 0
    return None


_TRAIN_SCRIPTS = ("train_density_vit_mae.py", "train_density_cha_mae.py")


def running_trainings():
    seen = {}
    for pid, cmd in _proc_iter():
        if not any(s in cmd for s in _TRAIN_SCRIPTS) or "exp_name=" not in cmd:
            continue
        m = re.search(r"exp_name=(\S+)", cmd)
        if not m:
            continue
        exp = m.group(1)
        if exp in seen:
            continue
        gpus = _environ(pid, "CUDA_VISIBLE_DEVICES") or "?"
        seen[exp] = dict(exp=exp, gpus=gpus, epoch=_epoch_of(exp))
    return list(seen.values())


def active_watchers():
    out, seen = [], set()
    for pid, cmd in _proc_iter():
        # skip interactive-shell wrappers / our own grep cmdlines that merely *mention* the
        # watcher string (e.g. leftover `zsh -c ... eval '...99_watch...'` snapshot shells) —
        # a real armed watcher is invoked as `bash .../99_watch_n_launch.sh`.
        if "shell-snapshots" in cmd or "zsh -c" in cmd or "grep" in cmd:
            continue
        if not re.search(r"(?:^|/|\s)(?:ba)?sh\s+\S*99_watch_n_launch\.sh", cmd) or "--wait" not in cmd:
            continue
        w = re.search(r"--wait\s+(\S+)", cmd)
        gate = os.path.basename(w.group(1)) if w else "?"
        # label: the script/condition after the trailing "--"
        lbl = "—"
        sm = re.search(r"run_gradmagonly_chain", cmd)
        if sm:
            lbl = "gradmag-only (train → probe)"
        elif "01c_pdbbind_probe.py features" in cmd:
            tag = re.search(r"--tag (\S+)", cmd)
            lbl = f"auto-probe ({tag.group(1) if tag else '?'})"
        key = (gate, lbl)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(gate=gate, label=lbl, gate_ready=(PDB / gate).exists()))
    return out


# Chained auto-probes that live in their OWN process (run_*_chain.sh train-then-probe scripts,
# or *_probe_watch.sh checkpoint-gated probe watchers). active_watchers() only sees the generic
# 99_watch_n_launch.sh; these bespoke scripts would otherwise be invisible. Surface each as a
# queued step that fires when its target training reaches e99.
CHAIN_SCRIPTS = [
    dict(script="run_gradmagonly_chain.sh",
         exp="260612_gradmagonly_density_xray_vit_mae_40m_v5_pretrain",
         out_csv="probe_results_e99_v5_filtered_gradmagonly.csv",
         label="auto-probe — gradmag-only (frozen features → MLP, 839 split)"),
    dict(script="260613_cha_mae_probe_watch.sh",
         exp="260613_cha_mae_gradmag_v5_pretrain",
         out_csv="probe_results_e99_v5_cha_mae.csv",
         label="auto-probe — ChA-MAEViT (frozen features → MLP affinity probe, 839 split)"),
    dict(script="run_v6_chain.sh",
         exp="260615_atomblob_density_gradmag_vit_mae_40m_invfreq_v6_pretrain",
         out_csv="probe_results_e99_v5_v6_ligandmatched.csv",
         label="auto-probe — v6 ligand-matched density (C+D+G pretrain → affinity probe, 839 split)"),
    dict(script="run_v7_chain.sh",
         exp="260615_atomblob_density_gradmag_vit_mae_40m_invfreq_v7_pretrain",
         out_csv="probe_results_e99_v5_v7_combined.csv",
         label="auto-probe — v7 combined CrossDocked∪PDBbind-train (C+D+G pretrain → affinity probe, 839 split)"),
]


def chained_probes():
    # which chain/watch scripts are actually running (the bash process, not a wrapper/grep)?
    live = set()
    for pid, cmd in _proc_iter():
        c = cmd.strip()
        for cs in CHAIN_SCRIPTS:
            if re.match(rf"bash \S*{re.escape(cs['script'])}\s*$", c):
                live.add(cs["script"])
    training = {r["exp"]: r["epoch"] for r in running_trainings()}
    out = []
    for cs in CHAIN_SCRIPTS:
        if cs["script"] not in live:
            continue
        ep = training.get(cs["exp"])
        if cs["exp"] in training:                 # still in training phase
            fires = f"when training reaches e99 (now e{ep if ep is not None else '?'}/100)"
            ready = False
        else:                                      # training done → probe is running now
            fires = "training done → probing now"
            ready = True
        out.append(dict(label=cs["label"], fires=fires, gate_ready=ready, out_csv=cs["out_csv"]))
    return out


def active_precomputes():
    out, seen = [], set()
    for pid, cmd in _proc_iter():
        if "python" not in cmd:
            continue
        m = re.search(r"(00[a-z]_[a-z_]+\.py)\s+(\w+)", cmd)        # 00x_*.py <subcmd>
        if m:
            label = f"{m.group(1)} {m.group(2)}"
        else:
            m2 = re.search(r"(voxelize_[a-z_]+\.py)", cmd)          # voxelize_*.py precompute
            if not m2:
                continue
            label = m2.group(1)
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


# ── completed / pending results (curated manifest, read live) ──────────────────
# Grouped by experiment (not per-run): what each did + its conclusion. The status_csv
# is only used to mark done vs pending (file exists → done); no metrics are shown.
COMPLETED = [
    dict(label="Coords + density + gradmag",
         status_csv="probe_results_e99_v5_filtered_rope3d.csv", cond="atomblob_density_gradmag",
         did="The full multimodal encoder — protein/ligand atom coords + X-ray density + gradient field ‖∇ρ‖ — "
             "MAE-pretrained, frozen, then probed for binding affinity. Swept positional encoding "
             "(learnable vs 3D-RoPE) and the density/gradmag reconstruction weights (invfreq · w1 · balanced).",
         concl="<b>Best affinity encoder of all.</b> 3D-RoPE positional encoding beats learnable; the loss-weight "
               "variants all tie — <i>having</i> density+gradmag is what matters, not how you weight them."),
    dict(label="Coords only (atoms)",
         status_csv="probe_results_e99_v5_filtered_atomblob_ligvdw.csv", cond="atomblob_ligvdw",
         did="The same encoder with density+gradmag <b>dropped</b> — atoms only (element-wise vdW radii). The "
             "matched control for “does density actually help?”. Also ran a RoPE variant.",
         concl="Adding density on top lifts affinity ~+0.05 ρ (0.595 vs 0.544) — but the <b>noise control shows "
               "this is capacity, not density signal</b> (noise reproduces the +0.05). RoPE helps the atoms-only "
               "encoder."),
    dict(label="Density-field only (no atoms)",
         status_csv="probe_results_e99_v5_filtered_densityonly.csv", cond="density_gradmag",
         did="Encoders with <b>no atom channels</b> — only the X-ray density (and a pure 1-channel ‖∇ρ‖-vs-density "
             "check). Tests how much the electron-density field alone carries.",
         concl="Density alone is weaker than atoms for affinity but well above chance. Pure density ≈ "
               "density+gradmag → the <b>gradmag channel is redundant</b> on top of density."),
    dict(label="HBGSA — external supervised baseline",
         status_csv="probe_results_e99_v5_filtered_hbgsa_no_cl1.csv", cond="hbgsa_supervised",
         did="Reimplemented an external <b>fully-supervised</b> affinity model (H-bond graph + sequence + pocket + "
             "SMILES), trained end-to-end on the same split.",
         concl="Loses to our <b>frozen self-supervised</b> probe → the SSL representation beats "
               "supervised-from-scratch on this task."),
    dict(label="Noise control", status_csv="probe_results_e99_v5_filtered_noisecontrol.csv",
         cond="atomblob_density_gradmag",
         did="Negative control: density+gradmag replaced by <b>matched random noise</b> (same value distribution, "
             "no spatial signal); everything else identical to the reference encoder.",
         concl="<b>Overturns &ldquo;density helps&rdquo;.</b> Noise density+gradmag (test ρ <b>0.609</b>) "
               "<b>matches/exceeds</b> real density+gradmag (0.595), both ≫ coords-only (0.544). The +0.05 "
               "&ldquo;density gain&rdquo; is <b>added-channel capacity, not signal</b> — reproduced by pure noise."),
    dict(label="Zero control", status_csv="probe_results_e99_v5_filtered_zerocontrol.csv",
         cond="atomblob_density_gradmag",
         did="Negative control #2: density+gradmag channels <b>zero-filled</b> (no information AND no variance) at "
             "both pretrain and probe; real coords kept. Separates raw parameter-capacity from noise-as-regularizer.",
         concl="<b>Clinches the capacity verdict.</b> Zero channels (test ρ <b>0.599</b>) ≈ noise (0.609) ≈ real "
               "(0.595), all ≫ coords-only (0.544). Even <i>zero-information, zero-variance</i> channels give the "
               "+0.05 → <b>pure patch-embed capacity</b>, ruling out even noise-as-regularizer."),
    dict(label="Gradmag only", status_csv="probe_results_e99_v5_filtered_gradmagonly.csv", cond="density_gradmag",
         did="Encoder whose <b>only</b> input is the gradient magnitude ‖∇ρ‖ — an edge/surface map of the density, "
             "with no density values and no atoms.",
         concl="Gradmag-only test ρ <b>0.551</b> — <b>beats density-only (0.505)</b>, ~ties coords-only (0.544): "
               "the ‖∇ρ‖ field is the <b>strongest single channel</b> and carries real standalone signal — but "
               "redundant with coords (per the noise control)."),
    dict(label="v6 ligand-matched density", status_csv="probe_results_e99_v5_v6_ligandmatched.csv",
         cond="atomblob_density_gradmag",
         did="C+D+G encoder pretrained on <b>v6</b> — the tt_min subset (5,270) where density matches BOTH pocket and "
             "ligand (vs v5, ~86% cross-docked → ligand-region density mismatched). Tests whether <i>genuinely matched</i> "
             "density carries affinity signal the capacity controls can't explain.",
         concl="Pending — C+D+G pretrain + probe on GPU 0-3."),
    dict(label="v7 combined matched corpus", status_csv="probe_results_e99_v5_v7_combined.csv",
         cond="atomblob_density_gradmag",
         did="C+D+G encoder pretrained on <b>v7</b> = v6 ∪ PDBbind-2020-train matched complexes (7,873) — a larger "
             "ligand-matched density corpus (density ↔ pocket+ligand for every sample). Tests whether more matched "
             "density beats v6 and the contaminated v5.",
         concl="Pending — queued after v6 frees GPU 0-3."),
    dict(label="Uniform-loss coords-only", status_csv="probe_results_e99_v5_uniform_coords.csv",
         cond="atomblob_ligvdw",
         did="Coords-only encoder re-pretrained with a <b>fully-uniform</b> MAE loss (no inv_freq, all channel/pos "
             "weights = 1) — a second pretraining draw of the 0.544 baseline.",
         concl="<b>The +0.05 gap was an invfreq artifact.</b> Coords-only <b>jumps 0.544 → 0.601</b> (+0.057) just "
               "by dropping invfreq — the inv_freq/pos weighting was specifically <b>handicapping the coords "
               "baseline</b>. Real coords-only performance is ~0.60, same cluster as everything else."),
    dict(label="Uniform-loss coords+density+gradmag", status_csv="probe_results_e99_v5_uniform_dg.csv",
         cond="atomblob_density_gradmag",
         did="Full 13-ch encoder re-pretrained with a <b>fully-uniform</b> MAE loss (no inv_freq, all weights = 1), "
             "real density+gradmag — second draw of the 0.595 baseline under matched-to-coords weighting.",
         concl="test ρ <b>0.595</b> — <b>unchanged</b> from invfreq (0.595), and now <b>≤ uniform coords-only "
               "(0.601)</b>. Under matched loss <b>density+gradmag adds nothing</b> over coordinates. Settles the "
               "arc: the &ldquo;density helps&rdquo; story was a low coords baseline, not a real signal."),
    dict(label="Uniform real d+g (seed 43, variance draw)", status_csv="probe_results_e99_v5_uniform_dg2.csv",
         cond="atomblob_density_gradmag",
         did="Real coords+density+gradmag, fully-uniform loss, learnable PE, <b>fresh seed 43</b> — real partner of "
             "the uniform-noise run, and a 2nd independent draw of uniform-d+g (seed42 = 0.595).",
         concl="<i>Result pending</i> — quantifies pretraining-run variance (does the 0.595 hold under a new seed?) "
               "and gives the real side of the real-vs-noise pair under matched uniform loss."),
    dict(label="Uniform-loss noise control", status_csv="probe_results_e99_v5_uniform_noise.csv",
         cond="atomblob_density_gradmag",
         did="Coords + <b>matched-noise</b> density+gradmag re-pretrained with the fully-uniform MAE loss — the "
             "uniform-weights analog of the invfreq noise control (0.609). Noise at both pretrain and probe.",
         concl="<i>Result pending</i> — if it lands ~0.60 like the rest, it confirms the whole {real, noise, zero, "
               "coords-only} set collapses to one cluster under matched loss; only the invfreq coords baseline (0.544) "
               "was ever an outlier."),
    dict(label="ChA-MAEViT (channel-grouped MAE)", status_csv="probe_results_e99_v5_cha_mae.csv",
         cond="atomblob_density_gradmag",
         did="<b>Channel-grouped</b> masked-autoencoder pretraining (ChA-MAEViT — per-channel-group tokens + "
             "attention + memory tokens) on the full 13-ch coords+density+gradmag stack, frozen, then probed "
             "for affinity on the canonical 839 split. Tests whether channel-aware fusion beats the standard "
             "fused ViT.",
         concl="ChA-MAEViT test ρ <b>0.613</b> — marginally the <b>best affinity number</b>, but only ~ties the "
               "fused rope3d ViT (0.606; +0.018 vs matched learnable 0.595, ~2.7σ). Channel-aware fusion gives at "
               "most a <b>small</b> affinity gain. (Survived the e3 OOM after the expandable_segments fix → full 100ep.)"),
]


def read_metrics(csv_name, cond):
    p = PDB / csv_name
    if not p.exists():
        return None
    try:
        rows = [r for r in csv.DictReader(open(p)) if r.get("condition") == cond]
    except Exception:
        return None
    if not rows:
        return None
    out = {}
    for col in ("best_val_spearman", "test_spearman", "test_pearson", "test_rmse"):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, "")))
            except (TypeError, ValueError):
                pass
        if vals:
            out[col] = (sum(vals) / len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return out or None


# ── render ─────────────────────────────────────────────────────────────────────
CSS = """
:root{--ink:#1c2433;--soft:#5b6678;--line:#e3e7ee;--line2:#aeb6c4;--bg:#f5f6f8;--card:#fff;
--run:#1d6fd0;--run-bg:#e7f0fc;--queue:#b07a17;--queue-bg:#fcf3e0;--done:#2f6f4f;--done-bg:#eaf5ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.page{max-width:1280px;margin:0 auto;padding:40px 24px 90px}
header.doc{border-bottom:2px solid var(--ink);padding-bottom:15px;margin-bottom:30px}
header.doc .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin-bottom:6px}
header.doc h1{font-size:26px;font-weight:650;margin:0}
header.doc .sub{font-size:13px;color:var(--soft);margin-top:6px}
.sec{margin-bottom:38px}
.sec-head{font-size:18px;font-weight:640;margin:0 0 13px;display:flex;align-items:center;gap:10px}
.dot{width:11px;height:11px;border-radius:50%}
.dot.run{background:var(--run)}.dot.queue{background:var(--queue)}.dot.done{background:var(--done)}
.sec-head .count{font-size:12.5px;font-weight:500;color:var(--soft)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px 13px;text-align:left;vertical-align:top}
thead th{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--soft);background:#fafbfc;border-bottom:1px solid var(--line);white-space:nowrap}
tbody tr{border-top:1px solid var(--line)}tbody tr:hover{background:#fbfcfd}
.exp{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:#1d4ed8;word-break:break-word}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.run{background:var(--run-bg);color:var(--run)}.pill.queue{background:var(--queue-bg);color:var(--queue)}
.pill.done{background:var(--done-bg);color:var(--done)}.pill.ext{background:#eef1f6;color:var(--soft)}
.pe{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px}
.pe.rope{background:var(--done-bg);color:#1d5a3a}.pe.learn{background:#eef1f6;color:var(--soft)}.pe.ext{background:#eef1f6;color:var(--soft)}
.num{font-weight:660}.best{color:#1d5a3a;font-weight:720}.mut{color:var(--soft)}.small{font-size:12.5px;color:var(--soft)}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{height:7px;border-radius:4px;background:#eef1f5;overflow:hidden;margin-top:4px;max-width:150px}
.bar>i{display:block;height:100%;background:var(--run);border-radius:4px}
.empty{padding:18px;color:var(--soft);font-size:13.5px;text-align:center}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px;font-size:12px;color:var(--soft)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.note{margin-top:11px;font-size:12.5px;color:var(--soft);line-height:1.55}
"""


def render():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    runs = running_trainings()
    watchers = active_watchers()
    chained = chained_probes()
    preps = active_precomputes()

    # one-line "what it tests" for the in-flight (running / queued) experiments
    # NB: order matters — cha_mae exp names also contain "gradmag", so match it first.
    def _desc(name):
        if "uniform" in name:
            if "noise" in name:
                return ("Uniform-loss noise control — coords + <b>NOISE</b> density+gradmag under the fully-uniform "
                        "loss (the invfreq noise control was 0.609). Checks whether it too collapses into the "
                        "~0.60 cluster once invfreq is removed.")
            if "seed43" in name or "variance" in name:
                return ("Real coords+density+gradmag, uniform loss, <b>fresh seed 43</b> — the real partner of the "
                        "uniform-noise run (GPU 4-7), and a 2nd independent pretraining draw of uniform-d+g "
                        "(seed42 was 0.595). Tests real-vs-noise under matched loss + pretraining-run variance.")
            which = "coords+density+gradmag (13ch)" if "density_gradmag" in name else "coords-only (11ch)"
            return (f"No-invfreq variance/confound check — {which} re-pretrained with a <b>fully-uniform</b> MAE "
                    "loss (no inv_freq, no channel/pos weights; all=1). Tests whether the +0.05 &ldquo;capacity&rdquo; "
                    "gap is an invfreq artifact or just pretraining-run variance.")
        if "cha_mae" in name or "channelvit" in name or "channel_vit" in name:
            return ("ChA-MAEViT — channel-grouped masked-autoencoder pretraining (per-channel-group tokens + "
                    "attention) on coords+density+gradmag; tests whether channel-aware fusion beats the fused ViT.")
        if "noisecontrol" in name:
            return ("Density-ablation control — density+gradmag replaced by matched noise; tests whether "
                    "density's +0.05 ρ gain is real signal or just model capacity.")
        if "zerocontrol" in name:
            return ("Density-ablation control — density+gradmag channels <b>zeroed</b> (vs the noise control). "
                    "Isolates raw parameter-capacity (if zero ≈ noise ≈ 0.609) from noise-as-regularizer "
                    "(if zero falls to coords-only 0.544).")
        if "gradmag" in name:
            return ("Gradmag-only — encoder sees only ‖∇ρ‖ (no density values, no atoms); tests whether the "
                    "gradient field alone carries affinity signal.")
        if "voxelize_misato" in name:
            return ("MISATO QM-property voxelization — precomputing density/gradmag voxels for the MISATO "
                    "partial-charge / interaction-energy regression targets.")
        return ""

    # RUNNING
    if runs or preps:
        rows = ""
        for r in runs:
            ep = r["epoch"]
            pct = (ep / 100 * 100) if ep is not None else 0
            prog = (f'<span class="num">e{ep} / 100</span><div class="bar"><i style="width:{pct:.0f}%"></i></div>'
                    if ep is not None else '<span class="small">starting…</span>')
            rows += (f'<tr><td><span class="pill run">training</span><br><span class="exp">{esc(r["exp"])}</span></td>'
                     f'<td class="small" style="white-space:normal">{_desc(r["exp"])}</td>'
                     f'<td>GPU {esc(r["gpus"])}</td><td>{prog}</td></tr>')
        for p in preps:
            rows += (f'<tr><td><span class="pill run">precompute</span><br><span class="exp">{esc(p)}</span></td>'
                     f'<td class="small" style="white-space:normal">prep for: {_desc(p)}</td>'
                     f'<td class="mut">CPU</td><td class="small">materializing…</td></tr>')
        running_html = (f'<div class="wrap"><table><thead><tr><th style="width:22%">Experiment</th>'
                        f'<th>What it tests</th><th>Resource</th><th>Progress</th></tr></thead>'
                        f'<tbody>{rows}</tbody></table></div>')
    else:
        running_html = '<div class="wrap"><div class="empty">Nothing training right now.</div></div>'

    # QUEUED — gate-watchers (99_watch_n_launch.sh) + in-script chained probes (run_*_chain.sh)
    if watchers or chained:
        rows = ""
        for w in watchers:
            state = ('<span class="small" style="color:var(--done)">gate ready → firing</span>'
                     if w["gate_ready"] else f'<span class="small">waiting for <code>{esc(w["gate"])}</code></span>')
            d = _desc(w["label"])
            d_html = f'<div class="small" style="white-space:normal;margin-top:3px">{d}</div>' if d else ""
            rows += (f'<tr><td style="white-space:normal"><span class="pill queue">chained</span> '
                     f'{esc(w["label"])}{d_html}</td><td>{state}</td></tr>')
        for c in chained:
            state = (f'<span class="small" style="color:var(--done)">{esc(c["fires"])}</span>'
                     if c["gate_ready"] else f'<span class="small">{esc(c["fires"])}</span>')
            rows += (f'<tr><td style="white-space:normal"><span class="pill queue">chained</span> '
                     f'{esc(c["label"])}'
                     f'<div class="small" style="white-space:normal;margin-top:3px">→ writes '
                     f'<code>{esc(c["out_csv"])}</code></div></td><td>{state}</td></tr>')
        queued_html = (f'<div class="wrap"><table><thead><tr><th style="width:62%">Step</th>'
                       f'<th>Fires when</th></tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        queued_html = '<div class="wrap"><div class="empty">No chained steps armed.</div></div>'

    # CONCLUDED experiments only — what each did + its conclusion. Pending experiments
    # live in Running / Queued above (with their "what it tests") until their CSV lands.
    rows = ""
    n_done = 0
    for r in COMPLETED:
        if read_metrics(r["status_csv"], r["cond"]) is None:
            continue
        n_done += 1
        rows += (f'<tr><td style="white-space:normal"><b>{esc(r["label"])}</b></td>'
                 f'<td class="small" style="white-space:normal">{r["did"]}</td>'
                 f'<td style="white-space:normal">{r["concl"]}</td></tr>')
    completed_html = (f'<div class="wrap"><table><thead><tr>'
                      f'<th style="width:19%">Experiment</th><th style="width:43%">What it did</th>'
                      f'<th style="width:38%">Conclusion</th></tr></thead>'
                      f'<tbody>{rows}</tbody></table></div>')
    n_pend = len(COMPLETED) - n_done

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Experiment Status — VoxBind</title><style>{CSS}</style></head><body><div class="page">
<header class="doc"><div class="date">VoxBind · density &amp; affinity line</div>
<h1>Experiment Status</h1>
<div class="sub">Live — read from running processes + probe CSVs · loaded {now} · auto-refresh 30 s ·
affinity, LP-PDBBind <code>new_split</code> n 2172/480/839</div></header>

<section class="sec"><h2 class="sec-head"><span class="dot run"></span> Running
<span class="count">— {len(runs)} training{'' if len(runs)==1 else 's'}{', '+str(len(preps))+' precompute' if preps else ''}</span></h2>{running_html}</section>

<section class="sec"><h2 class="sec-head"><span class="dot queue"></span> Queued / chained
<span class="count">— {len(watchers)+len(chained)} armed</span></h2>{queued_html}</section>

<section class="sec"><h2 class="sec-head"><span class="dot done"></span> Concluded experiments
<span class="count">— {n_done} concluded{(' · '+str(n_pend)+' still in flight (above)') if n_pend else ''}</span></h2>{completed_html}
<div class="note"><b>Big picture:</b> the X-ray density carries <b>real</b> affinity signal on top of atom coordinates
(the noise control, running now, is the decisive test of that); 3D-RoPE is the best positional encoding; the
gradient channel is redundant given density; and the frozen self-supervised representation beats an external
supervised baseline. <span class="mut">(Numbers live on the per-encoder table at :8731.)</span></div></section>

</div></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html", "/experiment_status.html"):
            self.send_error(404); return
        try:
            body = render().encode("utf-8")
        except Exception as e:
            body = f"<pre>render error: {html.escape(repr(e))}</pre>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"serving experiment status on http://{a.host}:{a.port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
