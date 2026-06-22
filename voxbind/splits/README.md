# `voxbind/splits/` — frozen, version-controlled data splits

**Problem this solves.** The PDBbind affinity split used to be *recomputed on every
server* as an intersection of LP-PDBBind's `new_split` column with two
**server-local, non-deterministic** artifacts under the git-ignored `dataset/data/`
tree:

- `voxels*/availability.csv` — which density crops actually built on *this* machine
- `ligand_rscc.csv` — which wwPDB validation reports actually downloaded here

A few failed downloads or partial builds → a slightly different set of complexes →
**slightly different metrics on different servers**. The split was *derived from
filesystem state*, not pinned.

**Fix.** A split is now a **committed artifact**: an explicit `pid,split` manifest,
frozen once on a complete machine and tracked in git, so `git pull` gives every
server the *identical* member set. Local data availability is a **separate, loud**
check — never a silent edit to who is in the test set.

## Layout

```
voxbind/splits/
  __init__.py        load_split() · check_local_availability() · content_hash()
  make_splits.py     the ONLY place a split is computed (reads primary inputs)
  verify.py          network-free hash gate — run before any train/probe
  MANIFEST.json      per-scheme counts + sha256 + provenance (the pin)
  <scheme>.csv       pid,split  (the frozen membership; tracked despite *.csv ignore)
```

`*.csv` / `*.json` are ignored repo-wide; this dir's `.gitignore` re-includes the
manifests so they commit with a plain `git add`.

## Schemes

| scheme | train / val / test | partition | what it measures |
|---|---|---|---|
| `lp_edrscc_v1` | 5817 / 1498 / 2813 | LP-PDBBind `new_split` ∩ ED ∩ lig&poc RSCC≥0.8, non-cov | **novel-target** (sequence-dedup) |
| `time_v1` | 8308 / 934 / 1182 | deposition year: train ≤2016 / val 2017 / test ≥2018 (same RSCC bar) | **temporal** generalization |
| `misato_md_v1` | 13765 / 1595 / 1612 | mirror of MISATO official 8:1:1 MD split | MISATO QM/MD targets |

> **`time_v1` caveat:** a temporal split does **not** control sequence redundancy —
> a recent test target may be near-identical to a training one. It measures
> temporal generalization, *not* novelty. Use `lp_edrscc_v1` for the novel-target claim.

## Usage

```python
from voxbind.splits import load_split, check_local_availability

split = load_split("lp_edrscc_v1")          # {"train":[pids], "val":[...], "test":[...]}
                                            # raises loudly if the manifest drifted

# Decouple membership from local availability (loud-warn policy):
avail = check_local_availability(split["test"], present=have_features,
                                 label="lp_edrscc_v1/test")
# avail["used"] = pids to score; avail["content_hash"] stamps the effective set so
# two servers can confirm they evaluated the IDENTICAL complexes.
```

In the probe: `python dataset/01c_pdbbind_probe.py --split lp_edrscc_v1` (also `time`, `misato`).

## Regenerate / verify / commit

```bash
cd /home/shpark/prj-denovo/VoxBind
python voxbind/splits/verify.py                 # gate: assert manifests intact (exit 1 on drift)

# regenerate ONLY on a machine with complete inputs, then commit the new pins:
python voxbind/splits/make_splits.py
git add voxbind/splits/*.csv voxbind/splits/MANIFEST.json && git commit
```

Daily training/probe code **never** recomputes — it reads the frozen manifest.
`make_splits.py` is the deliberate, audited re-pin step.
