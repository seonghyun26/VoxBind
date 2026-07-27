#!/usr/bin/env python
"""Score and plot sampled CrossDocked2020 train/test Vina distributions.

The held-out test split has 100 complexes and is scored in full.  The training
split has 99,881 complexes after VoxBind's fixed 100-complex validation holdout,
so the default report uses a deterministic simple-random sample of 500 training
instances (seed 260730).  Each ligand is evaluated against its paired
CrossDocked pocket10 receptor with the same three-stage AutoDock Vina stack used
by the VoxBind metrics pipeline:

    score_only -> local minimize -> full dock

The full docking search uses exhaustiveness=8, a configurable number of Vina
CPUs, and a fixed Vina seed of 1.  Results checkpoint to JSONL as they finish,
so interrupted runs can resume without repeating successful complexes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# Keep plotting self-contained on compute nodes whose home config is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "crossdocked_vina_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from rdkit import Chem
from scipy.stats import gaussian_kde, ks_2samp


SCRIPT_PATH = Path(__file__).resolve()
REPO = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "voxbind/dataset/data").is_dir()),
    None,
)
if REPO is None:
    raise RuntimeError("could not locate the VoxBind repository from the script path")
VOXBIND = REPO / "voxbind"
DATA = VOXBIND / "dataset/data"
STRUCTURES = DATA / "crossdocked_pocket10"
TARGETDIFF = REPO.parent / "TargetDIff"
WEBAPP = REPO / "notebook/webapp"
METRICS_PY = WEBAPP / "metrics.py"

TRAIN_SAMPLE_SEED = 260730
VINA_SEED = 1
EXHAUSTIVENESS = 8
TRAIN_SHUFFLE_SEED = 1234
VAL_SIZE = 100
METRICS = ("score_only", "minimize", "dock")


def load_metrics_module():
    """Load the local webapp metrics module without colliding with sklearn.metrics."""
    for p in (str(TARGETDIFF), str(WEBAPP), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("voxbind_web_metrics", METRICS_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {METRICS_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VB_METRICS = load_metrics_module()


def _job_id(split: str, ligand_rel: str) -> str:
    """Return an ID stable across sampled and exhaustive runs."""
    return f"{split}:{ligand_rel}"


def split_manifest(train_n: int) -> list[dict]:
    """Reproduce VoxBind's split logic and select the evaluation complexes."""
    train_all = torch.load(DATA / "data_train.pt", map_location="cpu", weights_only=False)
    test_all = torch.load(DATA / "data_test.pt", map_location="cpu", weights_only=False)

    random.Random(TRAIN_SHUFFLE_SEED).shuffle(train_all)
    train = train_all[:-VAL_SIZE]
    test = test_all
    del train_all

    if train_n > len(train):
        raise ValueError(f"train_n={train_n} exceeds train split size {len(train)}")
    if train_n == len(train):
        train_idx = list(range(len(train)))
    else:
        train_idx = sorted(random.Random(TRAIN_SAMPLE_SEED).sample(range(len(train)), train_n))

    jobs: list[dict] = []
    indexed_splits = (
        ("train", ((i, train[i]) for i in train_idx)),
        ("test", enumerate(test)),
    )
    for split, indexed_rows in indexed_splits:
        for split_index, (pocket, ligand) in indexed_rows:
            pocket_rel = str(pocket["id"])
            ligand_rel = str(ligand["id"])
            jobs.append(
                {
                    "job_id": _job_id(split, ligand_rel),
                    "split": split,
                    "split_index": split_index,
                    "family": ligand_rel.split("/", 1)[0],
                    "pocket_rel": pocket_rel,
                    "ligand_rel": ligand_rel,
                    "pocket_path": str(STRUCTURES / pocket_rel),
                    "ligand_path": str(STRUCTURES / ligand_rel),
                }
            )
    return jobs


def _finite_result(result: dict) -> bool:
    return all(
        isinstance(result.get(k), (int, float)) and math.isfinite(float(result[k]))
        for k in METRICS
    )


def score_one(job: dict, cpu: int, tmp_root: str) -> dict:
    """Score one ligand/pocket pair in an isolated temporary directory."""
    started = time.time()
    out = {k: job[k] for k in ("job_id", "split", "split_index", "family",
                                "pocket_rel", "ligand_rel")}
    pocket_src = Path(job["pocket_path"])
    ligand_src = Path(job["ligand_path"])
    try:
        if not pocket_src.exists():
            raise FileNotFoundError(pocket_src)
        if not ligand_src.exists():
            raise FileNotFoundError(ligand_src)

        with tempfile.TemporaryDirectory(prefix="cdx_vina_", dir=tmp_root) as tmp:
            tmp_path = Path(tmp)
            receptor = tmp_path / "receptor.pdb"
            ligand = tmp_path / "ligand.sdf"
            shutil.copy2(pocket_src, receptor)
            shutil.copy2(ligand_src, ligand)

            supplier = Chem.SDMolSupplier(str(ligand), removeHs=False)
            mol = next((m for m in supplier if m is not None), None)
            if mol is None:
                raise ValueError("RDKit could not read ligand")

            # Build the receptor cache once.  VinaDockingTask then sees the
            # adjacent .pqr/.pdbqt and reuses it for all three stages.
            VB_METRICS._prepare_receptor(receptor)

            from utils.evaluation.docking_vina import VinaDockingTask

            task = VinaDockingTask(str(receptor), mol, tmp_dir=str(tmp_path))
            scores: dict[str, float] = {}
            for mode in METRICS:
                result = task.run(
                    mode=mode,
                    exhaustiveness=EXHAUSTIVENESS,
                    cpu=cpu,
                    seed=VINA_SEED,
                )
                scores[mode] = float(result[0]["affinity"])
            if not _finite_result(scores):
                raise ValueError(f"non-finite Vina result: {scores}")
            out.update(scores)
            out["ok"] = True
    except Exception as exc:  # one failed complex must not abort the distribution
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = round(time.time() - started, 3)
    return out


def read_checkpoint(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Checkpoints from the earlier 500-instance audit used a sample-local
        # ordinal in job_id.  Re-key from split + ligand path so all 600 valid
        # results are reused by the exhaustive run.
        if row.get("split") and row.get("ligand_rel"):
            key = _job_id(str(row["split"]), str(row["ligand_rel"]))
        else:
            key = str(row["job_id"])
        rows[key] = row
    return rows


def _write_status(
    path: Path,
    *,
    total: int,
    prior: dict[str, dict],
    cached: int,
    started: float,
    running: bool,
) -> None:
    finished = len(prior)
    successful = sum(bool(row.get("ok")) for row in prior.values())
    failed = finished - successful
    elapsed = max(time.time() - started, 1e-9)
    finished_this_run = max(finished - cached, 0)
    rate = finished_this_run / elapsed
    payload = {
        "running": running,
        "total": total,
        "finished": finished,
        "successful": successful,
        "failed": failed,
        "pending": max(total - finished, 0),
        "elapsed_seconds": elapsed,
        "rate_per_hour": rate * 3600,
        "eta_seconds": (total - finished) / rate if rate > 0 else None,
        "updated_unix": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_scoring(
    jobs: list[dict],
    checkpoint: Path,
    workers: int,
    cpu: int,
    tmp_root: Path,
    status_path: Path,
    max_attempts: int,
) -> list[dict]:
    prior = read_checkpoint(checkpoint)
    job_ids = {j["job_id"] for j in jobs}
    prior = {key: row for key, row in prior.items() if key in job_ids}
    pending = [j for j in jobs if not bool(prior.get(j["job_id"], {}).get("ok"))]
    cached_success = len(jobs) - len(pending)
    print(
        f"jobs={len(jobs)} cached_success={cached_success} "
        f"pending={len(pending)} workers={workers} vina_cpu={cpu}",
        flush=True,
    )
    started = time.time()
    _write_status(
        status_path,
        total=len(jobs),
        prior={k: v for k, v in prior.items() if v.get("ok")},
        cached=cached_success,
        started=started,
        running=bool(pending),
    )
    if pending:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a") as stream:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending_iter = iter(pending)
                in_flight: dict = {}
                attempts = {j["job_id"]: 0 for j in pending}

                def submit(job: dict) -> None:
                    attempts[job["job_id"]] += 1
                    future = pool.submit(score_one, job, cpu, str(tmp_root))
                    in_flight[future] = job

                for _ in range(min(len(pending), workers * 2)):
                    submit(next(pending_iter))

                finished_this_run = 0
                while in_flight:
                    done_set, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done_set:
                        job = in_flight.pop(future)
                        job_id = job["job_id"]
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = {
                                **{k: job[k] for k in (
                                    "job_id", "split", "split_index", "family",
                                    "pocket_rel", "ligand_rel",
                                )},
                                "ok": False,
                                "error": f"worker crashed: {type(exc).__name__}: {exc}",
                            }
                        row["attempt"] = attempts[job_id]
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        prior[job_id] = row

                        if not row.get("ok") and attempts[job_id] < max_attempts:
                            submit(job)
                        else:
                            finished_this_run += 1
                            try:
                                submit(next(pending_iter))
                            except StopIteration:
                                pass

                        if (
                            finished_this_run == 1
                            or finished_this_run % 100 == 0
                            or finished_this_run == len(pending)
                        ):
                            ok = sum(bool(prior.get(j["job_id"], {}).get("ok")) for j in jobs)
                            elapsed = time.time() - started
                            rate = finished_this_run / max(elapsed, 1e-9) * 3600
                            remaining = len(pending) - finished_this_run
                            eta_h = remaining / rate if rate > 0 else float("inf")
                            print(
                                f"[{finished_this_run:6d}/{len(pending)}] "
                                f"total successful={ok} rate={rate:.1f}/h "
                                f"eta={eta_h:.1f}h",
                                flush=True,
                            )
                            _write_status(
                                status_path,
                                total=len(jobs),
                                prior={
                                    k: v for k, v in prior.items()
                                    if k in job_ids and (v.get("ok") or attempts.get(k, 0) >= max_attempts)
                                },
                                cached=cached_success,
                                started=started,
                                running=finished_this_run < len(pending),
                            )

    _write_status(
        status_path,
        total=len(jobs),
        prior={k: v for k, v in prior.items() if k in job_ids},
        cached=cached_success,
        started=started,
        running=False,
    )
    rows = []
    for job in jobs:
        row = dict(prior[job["job_id"]])
        # Always use exhaustive-manifest metadata, including the true split
        # index, even for rows migrated from the earlier sampled checkpoint.
        row.update({k: job[k] for k in (
            "job_id", "split", "split_index", "family", "pocket_rel", "ligand_rel",
        )})
        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame, manifest: dict) -> dict:
    summary: dict = {"manifest": manifest, "splits": {}, "comparisons": {}}
    for split in ("train", "test"):
        sdf = frame[(frame["split"] == split) & frame["ok"]].copy()
        summary["splits"][split] = {
            "requested": int((frame["split"] == split).sum()),
            "successful": int(len(sdf)),
            "failed": int(((frame["split"] == split) & ~frame["ok"]).sum()),
            "families": int(sdf["family"].nunique()),
            "metrics": {},
        }
        for metric in METRICS:
            x = sdf[metric].dropna().to_numpy(float)
            summary["splits"][split]["metrics"][metric] = {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "std": float(np.std(x, ddof=1)),
                "q25": float(np.quantile(x, .25)),
                "q75": float(np.quantile(x, .75)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "positive_n": int(np.sum(x > 0)),
                "positive_pct": float(100 * np.mean(x > 0)),
            }

    for metric in METRICS:
        train = frame[(frame["split"] == "train") & frame["ok"]][metric].dropna().to_numpy(float)
        test = frame[(frame["split"] == "test") & frame["ok"]][metric].dropna().to_numpy(float)
        ks = ks_2samp(train, test, alternative="two-sided", method="auto")
        summary["comparisons"][metric] = {
            "median_test_minus_train": float(np.median(test) - np.median(train)),
            "mean_test_minus_train": float(np.mean(test) - np.mean(train)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
    return summary


def _density(ax, values: np.ndarray, color: str, label: str, xgrid: np.ndarray) -> None:
    """Draw normalized histogram plus KDE within the shared display window."""
    inside = values[(values >= xgrid[0]) & (values <= xgrid[-1])]
    bins = np.linspace(xgrid[0], xgrid[-1], 31)
    ax.hist(
        inside,
        bins=bins,
        density=True,
        color=color,
        alpha=.13,
        edgecolor="none",
    )
    if len(values) > 20_000:
        # Deterministic thinning keeps exhaustive KDE rendering tractable.
        stride = math.ceil(len(values) / 20_000)
        kde_values = values[::stride]
    else:
        kde_values = values
    if len(np.unique(kde_values)) > 1:
        kde = gaussian_kde(kde_values)
        ax.plot(xgrid, kde(xgrid), color=color, lw=2.25, label=label)


def make_figure(frame: pd.DataFrame, summary: dict, png: Path, svg: Path) -> None:
    colors = {"train": "#38559b", "test": "#c66a2b"}
    titles = {
        "score_only": "Vina score-only",
        "minimize": "Vina minimize",
        "dock": "Vina dock",
    }
    valid = frame[frame["ok"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=False)

    for ax, metric in zip(axes, METRICS):
        train = valid[valid["split"] == "train"][metric].dropna().to_numpy(float)
        test = valid[valid["split"] == "test"][metric].dropna().to_numpy(float)
        pooled = np.r_[train, test]

        # A robust central window keeps a handful of clash failures from
        # compressing the useful body of the distribution.  Tail counts are
        # printed explicitly, so the plot does not silently discard them.
        lo, hi = np.quantile(pooled, [.01, .99])
        pad = max(.4, .05 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        xgrid = np.linspace(lo, hi, 500)

        if hi > 0:
            ax.axvspan(max(0, lo), hi, color="#faeceb", alpha=.8, zorder=0)
        if lo < 0 < hi:
            ax.axvline(0, color="#a33f39", lw=1, ls=":", alpha=.9)

        _density(ax, train, colors["train"], "train", xgrid)
        _density(ax, test, colors["test"], "test", xgrid)

        for split, x, y in (("train", train, .97), ("test", test, .89)):
            med = float(np.median(x))
            ax.axvline(med, color=colors[split], lw=1.2, ls="--", alpha=.9)
            ax.text(
                .025, y,
                f"{split} median {med:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                color=colors[split],
                fontsize=8.5, fontweight="semibold",
            )

        tail_lines = []
        for split, x in (("train", train), ("test", test)):
            n_lo = int(np.sum(x < lo))
            n_hi = int(np.sum(x > hi))
            if n_lo or n_hi:
                tail_lines.append(f"{split}: {n_lo} left / {n_hi} right")
        if tail_lines:
            ax.text(
                .98, .04,
                "outside view · " + " · ".join(tail_lines),
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7.1, color="#7a8699",
            )

        positive_train = int(np.sum(train > 0))
        positive_test = int(np.sum(test > 0))
        ax.text(
            .98, .78,
            f"positive pockets · train {positive_train:,} · test {positive_test:,}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.8, color="#a33f39", fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": 1.5},
        )

        ax.set_xlim(lo, hi)
        ax.set_title(titles[metric], fontsize=12, fontweight="semibold", pad=10)
        ax.set_xlabel("Vina energy (kcal/mol) · lower is stronger", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.grid(axis="y", color="#e3e7ee", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)

    n_train = summary["splits"]["train"]["successful"]
    n_test = summary["splits"]["test"]["successful"]
    train_coverage = (
        "complete split"
        if summary["manifest"].get("train_evaluated_in_full")
        else "random instance sample"
    )
    handles = [
        Line2D([0], [0], color=colors["train"], lw=2.5,
               label=f"train · {train_coverage} (n={n_train:,})"),
        Line2D([0], [0], color=colors["test"], lw=2.5,
               label=f"test · complete split (n={n_test:,})"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "CrossDocked2020 reference-ligand Vina distributions",
        y=1.10,
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        .5, -.015,
        "Common scoring pipeline: pocket10 receptor · exhaustiveness 8 · "
        "Vina CPU 4 · fixed seed 1.  Display window = pooled 1st–99th percentile + 5% pad.",
        ha="center", va="top", fontsize=8.2, color="#5b6678",
    )
    fig.tight_layout(rect=[0, .035, 1, .92], w_pad=2.2)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--train-n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--render-only", action="store_true",
                        help="skip scoring and rebuild CSV/summary/figure from checkpoint")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "crossdocked_vina_train_test_checkpoint.jsonl"
    csv_path = out / "crossdocked_vina_train_test.csv"
    summary_path = out / "crossdocked_vina_train_test_summary.json"
    positive_path = out / "crossdocked_vina_positive_scores.csv"
    status_path = out / "crossdocked_vina_full_status.json"
    png_path = out / "fig_crossdocked_vina_train_test.png"
    svg_path = out / "fig_crossdocked_vina_train_test.svg"
    tmp_root = Path("/tmp/crossdocked_vina_train_test")

    jobs = split_manifest(args.train_n)
    if args.render_only:
        rows_by_id = read_checkpoint(checkpoint)
        missing = [j["job_id"] for j in jobs if j["job_id"] not in rows_by_id]
        if missing:
            raise RuntimeError(f"checkpoint missing {len(missing)} jobs")
        rows = []
        for job in jobs:
            row = dict(rows_by_id[job["job_id"]])
            row.update({k: job[k] for k in (
                "job_id", "split", "split_index", "family", "pocket_rel", "ligand_rel",
            )})
            rows.append(row)
    else:
        rows = run_scoring(
            jobs,
            checkpoint,
            args.workers,
            args.cpu,
            tmp_root,
            status_path,
            args.max_attempts,
        )

    frame = pd.DataFrame(rows)
    for metric in METRICS:
        if metric not in frame:
            frame[metric] = np.nan
    if "split" not in frame:
        raise RuntimeError("checkpoint contains no usable split rows")
    frame["ok"] = frame["ok"].fillna(False).astype(bool)
    frame.to_csv(csv_path, index=False)

    positive_rows = []
    id_columns = ["split", "split_index", "family", "pocket_rel", "ligand_rel"]
    for metric in METRICS:
        selected = frame[frame["ok"] & (frame[metric] > 0)][id_columns + [metric]]
        for row in selected.to_dict("records"):
            row["vina_stage"] = metric
            row["vina_energy"] = row.pop(metric)
            positive_rows.append(row)
    pd.DataFrame(
        positive_rows,
        columns=id_columns + ["vina_stage", "vina_energy"],
    ).sort_values(
        ["split", "vina_stage", "vina_energy"],
        ascending=[True, True, False],
    ).to_csv(positive_path, index=False)

    manifest = {
        "dataset": "CrossDocked2020",
        "train_split_instances": 99881,
        "train_instances_evaluated": args.train_n,
        "train_sample_n": args.train_n,
        "train_sample_seed": TRAIN_SAMPLE_SEED,
        "train_evaluated_in_full": args.train_n == 99881,
        "test_split_instances": 100,
        "test_scored_in_full": True,
        "validation_holdout_n": VAL_SIZE,
        "split_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "receptor": "paired pocket10 PDB",
        "vina_exhaustiveness": EXHAUSTIVENESS,
        "vina_cpu": args.cpu,
        "vina_seed": VINA_SEED,
        "workers": args.workers,
    }
    summary = summarize(frame, manifest)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_figure(frame, summary, png_path, svg_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {positive_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
