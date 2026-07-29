"""make_plinder_big_report.py — self-contained HTML report for the big-PLINDER
ChannelViT density-MAE pretrain (process + results).

Reads whatever artifacts exist (selection funnels, coverage stats, build stats,
training log, GPU-util log, probe CSVs) and emits report_plinder_big.html with
inline CSS + base64 PNG plots. Re-run any time — sections render only if their
inputs are present, so it works incrementally (dataset first, training later).

    python dataset/make_plinder_big_report.py
"""
import base64
import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOX = Path(__file__).resolve().parents[1]
DATA = VOX / "dataset" / "data"
PLINDER = DATA / "plinder"
BIG = DATA / "xray_crops_aligned_plinder_big"
EXP = VOX / "exps" / "260620_plinder_big_channelvit_cdg_40m_pretrain"
OUT = VOX / "report_plinder_big.html"

C = {"bg": "#0f1419", "card": "#1a212b", "ink": "#e6edf3", "mut": "#9aa7b4",
     "acc": "#4fb3ff", "good": "#39d98a", "warn": "#ffb454", "bad": "#ff6b6b",
     "line": "#2b3440"}


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=C["card"])
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style_ax(ax):
    ax.set_facecolor(C["card"])
    for s in ax.spines.values():
        s.set_color(C["line"])
    ax.tick_params(colors=C["mut"])
    ax.xaxis.label.set_color(C["mut"]); ax.yaxis.label.set_color(C["mut"])
    ax.title.set_color(C["ink"])
    ax.grid(True, color=C["line"], lw=0.5, alpha=0.6)


# ── sections ────────────────────────────────────────────────────────────────────

def sec_header():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""
    <h1>Big-PLINDER ChannelViT density-MAE pretrain</h1>
    <p class="sub">Pretraining a Channel-ViT encoder on a maximally-covered, density-quality-gated
    PLINDER corpus &mdash; ligand <i>and</i> protein density verified per complex,
    atom types restricted to the VoxBind channels.</p>
    <p class="mut">Generated {now}</p>
    """


def sec_motivation():
    # ChannelViT vs fused on CrossDocked v5 (from the existing probe CSV)
    rows = ""
    p = DATA / "pdbbind" / "probe_results_e99_v5_channelvit.csv"
    if p.exists():
        import statistics as st
        by = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                by.setdefault(r["condition"], []).append(r)
        def agg(cond):
            rs = by.get(cond, [])
            if not rs:
                return None
            ts = [float(x["test_spearman"]) for x in rs]
            vs = [float(x["best_val_spearman"]) for x in rs]
            return st.mean(vs), st.mean(ts)
        for cond, label in [("atomblob_density_gradmag", "Fused ViT-MAE (1 Conv3d)"),
                            ("atomblob_density_gradmag_channelvit", "ChannelViT (channel-group)")]:
            a = agg(cond)
            if a:
                rows += (f"<tr><td>{label}</td><td>{a[0]:.3f}</td>"
                         f"<td class='{'bad' if 'channelvit' in cond else ''}'>{a[1]:.3f}</td></tr>")
    table = (f"<table><thead><tr><th>encoder (CrossDocked v5, 78k)</th>"
             f"<th>val &rho;</th><th>test &rho;</th></tr></thead><tbody>{rows}</tbody></table>"
             if rows else "<p class='mut'>(probe CSV not found)</p>")
    return f"""
    <h2>1 &middot; Why bigger, cleaner data</h2>
    <p>The Channel-ViT encoder (per-channel-group patch embed) was already trained on the 78k
    CrossDocked-v5 set. It reaches a <b>higher validation</b> &rho; than the fused baseline but a
    <b>lower test</b> &rho; &mdash; it <b>overfits</b> the smaller set:</p>
    {table}
    <p>Its extra patch-embed capacity should benefit from more &mdash; and cleaner &mdash; data.
    PLINDER provides per-system real-space density validation, so we can build a large corpus where
    the density genuinely matches the modeled atoms. This run tests that hypothesis.</p>
    """


def _funnel_table(funnel, title):
    if not funnel:
        return ""
    head = "<tr><th>stage</th><th>ligand instances</th><th>unique PDB</th></tr>"
    body = ""
    for s in funnel:
        extra = ""
        if "dropped_out_of_vocab" in s:
            extra = f" <span class='mut'>(&minus;{s['dropped_out_of_vocab']} out-of-vocab)</span>"
        body += (f"<tr><td>{s['stage']}{extra}</td>"
                 f"<td>{s['ligand_instances']:,}</td><td>{s['unique_pdb']:,}</td></tr>")
    return f"<h4>{title}</h4><table><thead>{head}</thead><tbody>{body}</tbody></table>"


def sec_selection():
    big = _read_json(PLINDER / "plinder_funnel.json")
    # the funnel.json reflects the LAST 03a run (big). Old build = the documented 20,498.
    fn = big["funnel"] if big else None
    n_sel = big["n_selected"] if big else "?"
    n_pdb = big["n_unique_pdb"] if big else "?"
    args = big.get("args", {}) if big else {}
    return f"""
    <h2>2 &middot; Selection &mdash; maximum coverage</h2>
    <p>The old PLINDER build kept <b>20,498</b> complexes: one ligand per PDB
    (<code>dedup=pdb</code>), single-ligand systems only, and restricted to PLINDER's own
    <code>train</code> split. None of our downstream evals (PDBbind / MISATO / CrossDocked) use
    PLINDER's benchmark, so that split is irrelevant to <i>our</i> leakage &mdash; we keep our own
    held-out test exclusion and drop the rest. New 03a settings:</p>
    <ul>
      <li><b>all PLINDER splits</b> (taps the 790k <code>none</code>-split systems) &mdash; our own
          PDBbind/MISATO/CrossDocked test ids are still excluded;</li>
      <li><b>no dedup</b> + <b>multi-ligand systems</b> &mdash; every distinct ligand instance kept;</li>
      <li><b>ligand atom-type filter</b>: drop any ligand whose SMILES has a heavy atom outside the 7
          VoxBind ligand channels (C/O/N/S/F/Cl/P). This removes e.g. a Re-carbonyl complex with
          RSCC=1.0 that passes the density filter but is not representable in the channels.</li>
    </ul>
    <p>Quality gates unchanged: X-ray, resolution &le; {args.get('max_res','?')} &Aring;,
    proper non-ion/artifact/cofactor ligand, RDKit-loadable, 6&ndash;50 heavy atoms,
    0 unresolved heavy atoms, ligand RSCC &ge; {args.get('min_rscc','?')}, non-covalent.</p>
    <p class="big">{n_sel:,} ligand instances &nbsp;/&nbsp; {n_pdb:,} unique PDB
    &nbsp;<span class="mut">(vs 20,498 before &mdash; {n_sel/20498:.2f}&times;)</span></p>
    {_funnel_table(fn, "Selection funnel (new, maximum-coverage)")}
    """ if big else "<h2>2 &middot; Selection</h2><p class='mut'>(funnel not found)</p>"


def sec_coverage():
    cov = _read_json(BIG / "coverage_stats.json")
    if not cov:
        return ("<h2>3 &middot; Protein + ligand density coverage</h2>"
                "<p class='mut'>(coverage stats not built yet)</p>")
    img = ""
    npz = BIG / "coverage_records.npz"
    if npz.exists():
        d = np.load(npz)
        lc, pc = d["lig_cov"], d["poc_cov"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, a, name, col in [(axes[0], lc, "ligand", C["acc"]), (axes[1], pc, "pocket", C["good"])]:
            ax.hist(a, bins=40, range=(0, 1), color=col, alpha=0.85)
            ax.axvline(cov["lig_cov_frac"] if name == "ligand" else cov["poc_cov_frac"],
                       color=C["warn"], lw=1.6, ls="--", label="gate")
            ax.set_title(f"{name} coverage fraction"); ax.set_xlabel(f"frac atoms ≥ {cov['cov_sigma']}σ")
            ax.set_ylabel("complexes"); _style_ax(ax); ax.legend(facecolor=C["card"], labelcolor=C["mut"])
        fig.patch.set_facecolor(C["card"])
        img = f"<img src='{_png(fig)}' alt='coverage histograms'/>"
    la, pa = cov.get("ligand_all", {}).get("cov_frac", {}), cov.get("pocket_all", {}).get("cov_frac", {})
    lz, pz = cov.get("ligand_all", {}).get("median_z", {}), cov.get("pocket_all", {}).get("median_z", {})
    return f"""
    <h2>3 &middot; Protein + ligand density coverage check</h2>
    <p>03a's RSCC gate validates only the <i>ligand</i>'s real-space fit. The user asked to check the
    <b>protein too</b>. So in 03c, for every complex we sample the deposited 2Fo-Fc map at <b>both</b>
    the ligand heavy atoms <b>and</b> the pocket protein heavy atoms (trilinear, same convention as the
    training crop), z-score by the map RMS (the crystallographic &sigma;), and keep the complex only if
    a minimum fraction of <i>each</i> sits in real density.</p>
    <table><thead><tr><th>gate</th><th>value</th></tr></thead><tbody>
      <tr><td>coverage threshold</td><td>atom value &ge; {cov['cov_sigma']} &sigma;</td></tr>
      <tr><td>ligand: keep if covered fraction &ge;</td><td>{cov['lig_cov_frac']}</td></tr>
      <tr><td>pocket: keep if covered fraction &ge;</td><td>{cov['poc_cov_frac']}</td></tr>
      <tr><td>complexes with a usable map</td><td>{cov['n_with_map']:,}</td></tr>
      <tr><td>passed BOTH gates</td><td class="good">{cov['n_passed']:,} ({100*cov['frac_passed']:.1f}%)</td></tr>
    </tbody></table>
    <p class="mut">ligand coverage: median {la.get('median','?')} (mean {la.get('mean','?')}),
    median z &asymp; {lz.get('median','?')}&sigma; &nbsp;|&nbsp;
    pocket coverage: median {pa.get('median','?')} (mean {pa.get('mean','?')}),
    median z &asymp; {pz.get('median','?')}&sigma;. The clean PLINDER selection already has high
    coverage on both partners; the gate removes the genuinely poorly-resolved tail.</p>
    {img}
    """


def sec_dataset():
    st = _read_json(BIG / "stats.json")
    if not st:
        return ""
    return f"""
    <h2>4 &middot; Final pretraining set</h2>
    <table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>
      <tr><td>train crops (frozen, 64&sup3; float16)</td><td class="good">{st.get('n_train',0):,}</td></tr>
      <tr><td>complexes built</td><td>{st.get('n_built',0):,}</td></tr>
      <tr><td>skipped</td><td>{st.get('n_skipped',0):,}</td></tr>
      <tr><td>pocket radius</td><td>{st.get('pocket_radius','?')} &Aring;</td></tr>
      <tr><td>density normalization</td><td>arcsinh(x/{st.get('arcsinh_scale','?')}) z-score (v5 stats)</td></tr>
      <tr><td>channels</td><td>13 = 7 lig + 4 poc atoms + density + gradmag</td></tr>
    </tbody></table>
    """


def _parse_train_log():
    """Pull (epoch, train_loss, val_loss, sec) from the newest train log."""
    logs = sorted(EXP.glob("train_density_vit_mae.log")) + \
        sorted((VOX / "log").glob("260620_plinder_big_channelvit_cdg_40m_pretrain_*.log"))
    epochs = []
    for lg in logs:
        try:
            txt = lg.read_text(errors="ignore")
        except Exception:
            continue
        cur = {}
        for line in txt.splitlines():
            m = re.search(r"epoch: (\d+) \(([\d.]+)s\)", line)
            if m:
                cur = {"epoch": int(m.group(1)), "sec": float(m.group(2))}
                continue
            m = re.search(r">> \[train\].*loss: ([\d.]+)", line)
            if m and cur:
                cur["train"] = float(m.group(1))
            m = re.search(r">> \[val\].*loss: ([\d.]+)", line)
            if m and cur and "train" in cur:
                cur["val"] = float(m.group(1)); epochs.append(cur); cur = {}
    # dedup by epoch (keep last)
    seen = {e["epoch"]: e for e in epochs}
    return [seen[k] for k in sorted(seen)]


def _parse_util():
    f = VOX / "log" / "gpu_util_plinder_big.csv"
    if not f.exists():
        return None
    vals = []
    with open(f) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(vals) if vals else None


def sec_training():
    ep = _parse_train_log()
    util = _parse_util()
    if not ep and util is None:
        return ("<h2>5 &middot; Training</h2><p class='mut'>(not started / no log yet)</p>"
                + _config_block())
    img = ""
    if ep:
        xs = [e["epoch"] for e in ep]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(xs, [e.get("train", np.nan) for e in ep], color=C["acc"], label="train")
        ax.plot(xs, [e.get("val", np.nan) for e in ep], color=C["good"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("MAE loss"); ax.set_title("Reconstruction loss")
        _style_ax(ax); ax.legend(facecolor=C["card"], labelcolor=C["mut"])
        fig.patch.set_facecolor(C["card"])
        img = f"<img src='{_png(fig)}' alt='loss curve'/>"
    util_line = ""
    if util is not None and len(util):
        warm = util[len(util) // 5:]  # drop cold start
        col = C["good"] if warm.mean() >= 80 else C["warn"]
        util_line = (f"<p class='big'>GPU util (warm): "
                     f"<span style='color:{col}'>{warm.mean():.0f}%</span> "
                     f"<span class='mut'>mean over {len(warm)} samples, target &ge; 80%</span></p>")
    last = ep[-1] if ep else None
    spe = f"{last['sec']:.0f}s/epoch" if last else "?"
    prog = f"epoch {last['epoch']+1}/100" if last else "starting"
    return f"""
    <h2>5 &middot; Training</h2>
    <p>{prog} &middot; {spe}</p>
    {util_line}
    {img}
    {_config_block()}
    """


def _config_block():
    return """
    <h4>Recipe</h4>
    <table><thead><tr><th>setting</th><th>value</th></tr></thead><tbody>
      <tr><td>encoder</td><td>ChannelViT (patch_embed=channel_group, groups [7,4,1,1]), dim512/depth12/heads8</td></tr>
      <tr><td>input</td><td>13ch atomblob + v5 density + gradmag, ligand vdW radii</td></tr>
      <tr><td>objective</td><td>MAE, mask 0.60 atom-biased, inv-freq channel weighting</td></tr>
      <tr><td>throughput</td><td>bf16 amp ON, torch.compile OFF (no C compiler), 4&times;H200 eff-batch 128</td></tr>
      <tr><td>loader</td><td>frozen 64&sup3; crops (not on-the-fly resample), 16 workers / prefetch 8</td></tr>
      <tr><td>schedule</td><td>100 epochs, lr 1e-4, wd 5e-2, seed 42</td></tr>
    </tbody></table>
    <p class="mut">GPU protocol: the sig=1.0 denoiser is stopped for this run and auto-resumed (+400 ep)
    on completion via scripts/archive/chains/chain_plinder_big_then_resume.sh.</p>
    """


def main():
    parts = [sec_header(), sec_motivation(), sec_selection(), sec_coverage(),
             sec_dataset(), sec_training()]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Big-PLINDER ChannelViT pretrain</title><style>
    body{{background:{C['bg']};color:{C['ink']};font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
        max-width:920px;margin:0 auto;padding:32px 22px}}
    h1{{font-size:27px;margin:0 0 4px}} h2{{font-size:21px;margin:34px 0 10px;border-bottom:1px solid {C['line']};padding-bottom:6px}}
    h4{{color:{C['mut']};margin:18px 0 6px;font-size:14px;text-transform:uppercase;letter-spacing:.04em}}
    .sub{{color:{C['ink']};font-size:16px}} .mut{{color:{C['mut']}}} .big{{font-size:18px;margin:14px 0}}
    .good{{color:{C['good']}}} .warn{{color:{C['warn']}}} .bad{{color:{C['bad']}}}
    code{{background:{C['card']};padding:1px 5px;border-radius:4px;color:{C['acc']}}}
    table{{border-collapse:collapse;width:100%;margin:10px 0;background:{C['card']};border-radius:8px;overflow:hidden}}
    th,td{{padding:7px 12px;text-align:left;border-bottom:1px solid {C['line']}}}
    th{{color:{C['mut']};font-weight:600;font-size:13px}} tbody tr:last-child td{{border-bottom:none}}
    img{{max-width:100%;border-radius:8px;margin:12px 0;border:1px solid {C['line']}}}
    ul{{color:{C['ink']}}} li{{margin:3px 0}}
    </style></head><body>{''.join(parts)}
    <p class="mut" style="margin-top:40px;border-top:1px solid {C['line']};padding-top:12px">
    VoxBind &middot; report regenerated by dataset/make_plinder_big_report.py</p>
    </body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
