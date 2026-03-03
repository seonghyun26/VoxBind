"""check_xray_data.py

Check X-ray density availability for every PDB ID in the CrossDocked2020 benchmark.
Uses the PDBe Summary + EDS API endpoints (EBI only, RCSB not reachable).

Strategy (mirrors check_density_availability in test.ipynb)
------------------------------------------------------------
1. PDBe summary → experimental method
2a. cryo-EM  : look for EMDB accession in related_structures
2b. X-ray    : call electron_density_statistics endpoint;
               HTTP 200 → structure factors deposited (EDS map available)

Outputs
-------
- data/check_xray_cache.json   : per-PDB result cache (resumable)
- check_xray_data.png          : summary bar chart
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO / "voxbind" / "dataset" / "data"
CACHE_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "check_xray_cache.json"

# ── API settings ───────────────────────────────────────────────────────────
SUMMARY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
EDS_URL     = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/electron_density_statistics/{pdb_id}"
MAX_WORKERS = 20
RETRY_WAIT  = 2.0


# ── 1. Load benchmark PDB IDs ──────────────────────────────────────────────
def load_benchmark_pdbs(data_dir: Path) -> tuple[list[str], int]:
    """Return (sorted unique PDB IDs, total sample count)."""
    pdb_ids: set[str] = set()
    total_samples = 0
    for pt_path in sorted(data_dir.glob("data_*.pt")):
        samples = torch.load(pt_path, weights_only=False)
        total_samples += len(samples)
        for pocket_dict, _ in samples:
            pdb_id = pocket_dict["id"].split("/")[1].split("_")[0].upper()
            pdb_ids.add(pdb_id)
        print(f"  {pt_path.name}: {len(samples):,} samples, {len(pdb_ids):,} unique PDB IDs so far")
    return sorted(pdb_ids), total_samples


# ── 2. check_density_availability (mirrors test.ipynb) ────────────────────
def check_density_availability(pdb_id: str, session: requests.Session) -> dict:
    pdb_id = pdb_id.lower()
    for attempt in range(3):
        try:
            r = session.get(SUMMARY_URL.format(pdb_id=pdb_id), timeout=15)
            break
        except Exception:
            time.sleep(RETRY_WAIT * (attempt + 1))
    else:
        return {"pdb_id": pdb_id, "has_density": False,
                "source": None, "method": "error"}

    if r.status_code != 200:
        return {"pdb_id": pdb_id, "has_density": False,
                "source": None, "method": "not_found"}

    data    = r.json().get(pdb_id, [{}])[0]
    methods = data.get("experimental_method", [])
    methods_upper = [m.upper() for m in methods]

    result = {
        "pdb_id":      pdb_id,
        "method":      ", ".join(methods),
        "has_density": False,
        "source":      None,
        "source_id":   None,
    }

    # cryo-EM
    if "ELECTRON MICROSCOPY" in methods_upper:
        related = data.get("related_structures", [])
        emdb    = [s for s in related if s.get("resource") == "EMDB"]
        if emdb:
            result["has_density"] = True
            result["source"]      = "EMDB"
            result["source_id"]   = emdb[0].get("accession")

    # X-ray diffraction
    elif "X-RAY DIFFRACTION" in methods_upper:
        for attempt in range(3):
            try:
                eds = session.get(EDS_URL.format(pdb_id=pdb_id), timeout=15)
                break
            except Exception:
                time.sleep(RETRY_WAIT * (attempt + 1))
        else:
            eds = None
        if eds is not None and eds.status_code == 200:
            result["has_density"] = True
            result["source"]      = "PDBe EDS"
            result["source_id"]   = pdb_id

    return result


# ── 3. Run checks with caching ─────────────────────────────────────────────
def run_checks(pdb_ids: list[str]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"Loaded cache: {len(cache):,} entries")

    to_check = [p for p in pdb_ids if p not in cache]
    print(f"Need to query: {len(to_check):,} PDB IDs")

    if to_check:
        session = requests.Session()
        with tqdm(total=len(to_check), unit="PDB", desc="Checking density") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = {pool.submit(check_density_availability, p, session): p
                        for p in to_check}
                for fut in as_completed(futs):
                    result = fut.result()
                    cache[result["pdb_id"].upper()] = result
                    pbar.update(1)

        CACHE_FILE.write_text(json.dumps(cache, indent=2))
        print(f"Cache saved → {CACHE_FILE}")

    return cache


# ── 4. Plot ────────────────────────────────────────────────────────────────
def plot_summary(pdb_ids: list[str], total_samples: int, cache: dict[str, dict]):
    results = [cache.get(p, cache.get(p.lower(), {})) for p in pdb_ids]

    n_total_pdbs   = len(pdb_ids)
    n_xray         = sum(1 for r in results if "X-ray" in r.get("method", "")
                         or "x-ray" in r.get("method", "").lower())
    n_has_density  = sum(1 for r in results if r.get("has_density"))
    n_no_density   = n_total_pdbs - n_has_density

    print("\n" + "─" * 52)
    print(f"  Total benchmark samples          : {total_samples:>8,}")
    print(f"  Total unique PDB IDs             : {n_total_pdbs:>8,}")
    print(f"  X-ray diffraction method         : {n_xray:>8,}  ({100*n_xray/n_total_pdbs:.1f}%)")
    print(f"  Has density (EDS / EMDB)         : {n_has_density:>8,}  ({100*n_has_density/n_total_pdbs:.1f}%)")
    print(f"  No density                       : {n_no_density:>8,}  ({100*n_no_density/n_total_pdbs:.1f}%)")
    print("─" * 52)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: bar chart of key counts ─────────────────────────────────────
    bars = {
        "Total\nsamples":     total_samples,
        "Total\nPDB IDs":     n_total_pdbs,
        "X-ray\ndiffraction": n_xray,
        "Has density\n(EDS)": n_has_density,
    }
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    ax = axes[0]
    rects = ax.bar(list(bars.keys()), list(bars.values()), color=colors, width=0.55)
    ax.bar_label(rects, fmt=lambda v: f"{int(v):,}", padding=4, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("CrossDocked2020 — data overview")
    ax.set_ylim(0, max(bars.values()) * 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Right: pie — density availability ─────────────────────────────────
    method_counts = {}
    for r in results:
        m = r.get("method", "not_found") or "not_found"
        key = (
            "X-ray (no EDS)"   if "x-ray" in m.lower() and not r.get("has_density") else
            "X-ray (has EDS)"  if "x-ray" in m.lower() and r.get("has_density")     else
            "cryo-EM (EMDB)"   if r.get("source") == "EMDB"                         else
            "Not found / other"
        )
        method_counts[key] = method_counts.get(key, 0) + 1

    pie_labels = list(method_counts.keys())
    pie_vals   = list(method_counts.values())
    pie_colors = ["#4C72B0", "#C44E52", "#55A868", "#937860"][:len(pie_labels)]
    axes[1].pie(pie_vals, labels=pie_labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=90, pctdistance=0.75)
    axes[1].set_title("Density availability breakdown")

    plt.suptitle("CrossDocked2020 benchmark — X-ray density coverage", fontsize=13)
    plt.tight_layout()
    out = Path(__file__).parent / "check_xray_data.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== CrossDocked2020 X-ray density check ===\n")

    print("Loading benchmark PDB IDs...")
    pdb_ids, total_samples = load_benchmark_pdbs(DATA_DIR)
    print(f"\nTotal unique PDB IDs : {len(pdb_ids):,}")
    print(f"Total samples        : {total_samples:,}\n")

    cache = run_checks(pdb_ids)

    plot_summary(pdb_ids, total_samples, cache)
