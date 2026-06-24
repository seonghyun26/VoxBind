"""voxbind.splits — frozen, version-controlled train/val/test partitions.

Why this package exists
-----------------------
The PDBbind affinity split used to be *recomputed on every server* as an
intersection of LP-PDBBind's ``new_split`` column with two **server-local,
non-deterministic** artifacts:

  * ``voxels*/availability.csv``  — which density crops actually built here
  * ``ligand_rscc.csv``          — which wwPDB validation reports downloaded here

Both live under the git-ignored ``dataset/data/`` tree, so two machines silently
ended up scoring on slightly different complexes → slightly different metrics.

This package makes a split a **committed artifact**: an explicit ``pid,split``
manifest, frozen once on a complete machine and tracked in git, so ``git pull``
gives every server the *identical* member set. Local data availability becomes a
*separate, loud* concern — never a silent edit to who is in the test set.

Public API
----------
    load_split(scheme)            -> {"train":[pids], "val":[pids], "test":[pids]}
    list_schemes()                -> ["lp_edrscc_v1", "time_v1", "misato_md_v1", ...]
    scheme_info(scheme)           -> dict from MANIFEST.json (counts, sha256, provenance)
    check_local_availability(...) -> loud-warn intersection with what's on disk
    content_hash(pids)            -> sha256 over a pid list (stamp a result's effective set)

The manifests + MANIFEST.json are tracked despite the repo-wide ``*.csv`` /
``*.json`` ignore via this directory's local ``.gitignore`` negation.

Regenerate with ``make_splits.py``; gate a run with ``verify.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLITS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SPLITS_DIR / "MANIFEST.json"

SPLIT_NAMES = ("train", "val", "test")


# ── hashing (canonical, order-independent) ───────────────────────────────────
def _norm(pid: str) -> str:
    return str(pid).strip().lower()


def content_hash(pids) -> str:
    """sha256 over a *set* of pids (sorted, deduped, lowercased).

    Use to stamp the effective set a run actually scored on, so two servers can
    confirm they evaluated the identical complexes even when both are partial.
    """
    uniq = sorted({_norm(p) for p in pids})
    return hashlib.sha256("\n".join(uniq).encode()).hexdigest()


def manifest_hash(pid_split_pairs) -> str:
    """sha256 over (pid, split) membership — the frozen-manifest integrity hash.

    ``pid_split_pairs`` is an iterable of (pid, split). Sorted before hashing so
    row order in the CSV never affects the hash.
    """
    lines = sorted(f"{_norm(p)}\t{str(s).strip().lower()}" for p, s in pid_split_pairs)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ── manifest registry ────────────────────────────────────────────────────────
def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"split manifest not found: {MANIFEST_PATH}\n"
            f"  regenerate it with:  python voxbind/splits/make_splits.py"
        )
    return json.loads(MANIFEST_PATH.read_text())


def list_schemes() -> list[str]:
    return sorted(read_manifest().get("schemes", {}).keys())


def scheme_info(scheme: str) -> dict:
    schemes = read_manifest().get("schemes", {})
    if scheme not in schemes:
        raise KeyError(
            f"unknown split scheme '{scheme}'. known: {sorted(schemes)}\n"
            f"  (add it in make_splits.py, then re-run to refresh MANIFEST.json)"
        )
    return schemes[scheme]


def _read_manifest_csv(path: Path) -> list[tuple[str, str]]:
    """Read a ``pid,split`` CSV → list of (pid, split). Stdlib only (no pandas)."""
    import csv

    pairs: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "pid" not in rdr.fieldnames or "split" not in rdr.fieldnames:
            raise ValueError(f"{path}: expected header 'pid,split', got {rdr.fieldnames}")
        for row in rdr:
            pid = _norm(row["pid"])
            split = str(row["split"]).strip().lower()
            if pid and split in SPLIT_NAMES:
                pairs.append((pid, split))
    return pairs


# ── the one loader everything should use ─────────────────────────────────────
def load_split(scheme: str, *, verify: bool = True) -> dict[str, list[str]]:
    """Return ``{"train":[pids], "val":[pids], "test":[pids]}`` for a frozen scheme.

    Reads the committed manifest CSV named in MANIFEST.json and (by default)
    asserts its content hash + per-split counts still match the pinned values,
    so a corrupted / accidentally-regenerated manifest fails loudly instead of
    silently shifting the split.
    """
    info = scheme_info(scheme)
    csv_path = SPLITS_DIR / info["file"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"manifest CSV for scheme '{scheme}' missing: {csv_path}\n"
            f"  it should be tracked in git; try a fresh checkout or re-run make_splits.py"
        )
    pairs = _read_manifest_csv(csv_path)

    if verify:
        got_hash = manifest_hash(pairs)
        if got_hash != info["sha256"]:
            raise ValueError(
                f"[split:{scheme}] manifest hash MISMATCH — {csv_path} has drifted "
                f"from the pinned definition.\n"
                f"  expected {info['sha256']}\n  got      {got_hash}\n"
                f"  If the change is intentional, re-run make_splits.py to re-pin MANIFEST.json."
            )

    out: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    for pid, split in pairs:
        out[split].append(pid)
    for s in SPLIT_NAMES:
        out[s] = sorted(out[s])

    if verify:
        for s in SPLIT_NAMES:
            exp = int(info["counts"].get(s, -1))
            if exp >= 0 and len(out[s]) != exp:
                raise ValueError(
                    f"[split:{scheme}] {s} count {len(out[s])} != pinned {exp} ({csv_path})"
                )
    return out


# ── availability: loud-warn, never silent-drop ───────────────────────────────
def check_local_availability(expected, present, *, label: str = "", warn=print) -> dict:
    """Intersect a frozen split's pids with what's locally present (loud-warn policy).

    ``expected`` : pids the frozen split says belong here (e.g. ``load_split(s)["test"]``)
    ``present``  : pids this server actually has features/crops for (set / iterable)

    Returns ``{used, missing, n_expected, n_used, frac_present, content_hash}`` and,
    if anything is missing, emits a loud warning naming the count + a sample. The
    ``content_hash`` is over the *used* set so two runs can confirm they scored the
    identical complexes. Membership is NEVER silently shrunk without this record.
    """
    present_set = {_norm(p) for p in present}
    exp = [_norm(p) for p in expected]
    used = [p for p in exp if p in present_set]
    missing = [p for p in exp if p not in present_set]
    n_exp = len(exp)
    ch = content_hash(used)
    if missing:
        warn(
            f"[split:{label or 'split'}] WARNING: {len(missing)}/{n_exp} pids missing "
            f"locally — scoring on {len(used)} (effective N), NOT the full frozen set. "
            f"missing e.g. {missing[:6]}{'…' if len(missing) > 6 else ''}  "
            f"used.content_hash={ch[:12]}  "
            f"(metrics are only cross-server comparable when this hash matches)"
        )
    return {
        "used": used,
        "missing": missing,
        "n_expected": n_exp,
        "n_used": len(used),
        "frac_present": (len(used) / n_exp) if n_exp else 0.0,
        "content_hash": ch,
    }


# ── frozen PLINDER pretraining selection (cross-server stable) ────────────────
PLINDER_DIR = SPLITS_DIR / "plinder"


def frozen_plinder_selection(*, verify: bool = True, warn=print):
    """Path to the committed frozen PLINDER selection CSV, or ``None`` if not present.

    The selection (which ligand-instances make up the density pretraining corpus) is
    pinned in ``splits/plinder/plinder_selected.csv`` with its sha256 in
    ``plinder_inputs.json`` — so every server builds from the *same* list instead of
    re-deriving it from the (mutable) PLINDER annotation table. With ``verify`` the
    sha256 is checked and a drifted/edited CSV fails LOUDLY (mirrors ``load_split``).
    Re-pin deliberately via ``03a_plinder_select.py --rederive --freeze``.
    """
    csv = PLINDER_DIR / "plinder_selected.csv"
    inputs = PLINDER_DIR / "plinder_inputs.json"
    if not csv.exists():
        return None
    if verify and inputs.exists():
        meta = json.loads(inputs.read_text())
        want = meta.get("selection_sha256")
        got = hashlib.sha256(csv.read_bytes()).hexdigest()
        if want and got != want:
            raise ValueError(
                f"[plinder] frozen selection hash MISMATCH — {csv} has drifted from the "
                f"pinned definition ({meta.get('name', '?')}).\n"
                f"  expected {want}\n  got      {got}\n"
                f"  If intentional, re-run 03a_plinder_select.py --rederive --freeze to re-pin."
            )
    return csv


def plinder_inputs() -> dict:
    """The pinned-inputs manifest (bucket / filters / build params / leakage hashes), or {}."""
    p = PLINDER_DIR / "plinder_inputs.json"
    return json.loads(p.read_text()) if p.exists() else {}
