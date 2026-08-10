"""01c_pdbbind_probe.py — Frozen-encoder features + MLP affinity probe for PDBbind.

Consolidated Phase 5 / 6 entry point. Three subcommands:

    cd voxbind
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features --condition atomblob --epoch 99
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py probe --epoch 99 --seeds 3
    CUDA_VISIBLE_DEVICES=6 python dataset/01c_pdbbind_probe.py finetune \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99

──────────────────────────────────────────────────────────────────────────────
features — frozen-encoder features for PDBbind
──────────────────────────────────────────────────────────────────────────────
For one pretraining condition at a time (atomblob OR atomblob_density), load
the EMA encoder weights from a checkpoint, freeze it, run every PDBbind refined
complex through, mean-pool the post-norm patch tokens, and save a 512-D
feature vector per complex.

The pre-norm patch tokens come from the encoder forward up to (but not
including) `decoder_proj`. There are 8³ = 512 patch tokens of dim 512; mean
across the 512 tokens yields a single (512,) vector per complex.

    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features \
        --condition atomblob          --epoch 99
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features \
        --condition atomblob_density  --epoch 99
  → dataset/data/pdbbind/features/{condition}_e{epoch}.pt
        Each is a dict-of-tensors: { pdb_id: torch.float32 [512] }

──────────────────────────────────────────────────────────────────────────────
probe — 2-layer MLP probe on frozen-encoder features
──────────────────────────────────────────────────────────────────────────────
Compares pretrained-encoder representations on PDBbind affinity:
  (i)  atomblob          — 11-ch input  (7 lig + 4 pocket atoms)
  (ii) atomblob_density  — 12-ch input  (11 atoms + 1 2Fo-Fc density)

For each condition, the 512-D mean-pooled patch tokens (from `features`) are fed
through an identical 2-layer MLP head:
        Linear(512 → hidden) → SiLU → Dropout → Linear(hidden → 1)

Both conditions are evaluated on the **same complexes** (intersection of
on-disk feature sets across conditions — limited by the density variant
since EDS coverage is the bottleneck). This ensures the Δ in test ρ comes
only from the encoder, not from a different sample pool.

No ligand fingerprint, no other side information — purely the encoder's
512-D representation → pK. The encoder already sees the ligand as
channels 0–6, so its mean-pooled output already encodes ligand identity
implicitly; we test what this representation is worth.

    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py probe \
        --epoch 99 --seeds 3
  → dataset/data/pdbbind/probe_results_e<N>[_<ver>][_filtered][_<tag>].csv  (this run only)
        version from --voxel_version, _filtered from --cl1_only, _<tag> from --tag
        cols: condition, seed, n_train, n_val, n_test,
              best_val_spearman, val_pearson, test_spearman, test_pearson,
              test_rmse, epoch_stopped
    (Auto-consolidation is OFF — per-run CSVs + the live dashboard are the source of
     truth. To build a combined table on demand, version/variant lifted from each
     filename into columns:  python dataset/consolidate_probe_results.py [--summary])

  Optional probe flags:
    --target T              pK (default) | hbonds | bfactor | bfactor_rel; non-pK adds a _<T> CSV token
    --conditions COND ...   default: atomblob atomblob_density atomblob_weighted atomblob_merged_density
    --no_intersect          let each condition use its own pdb_id pool (NOT recommended)
    --no_covalent_filter    keep covalent complexes (default: drop)
    --cl1_only              restrict to LP_PDBBind CL1=True (cleanest subset); names the CSV *_filtered*
    --tag LABEL             optional run label in the CSV name (after the version/_filtered token)
    --max_epochs N          default: 200
    --patience N            default: 30  (epochs of no val-ρ improvement)

──────────────────────────────────────────────────────────────────────────────
finetune — frozen-encoder control vs end-to-end encoder fine-tuning
──────────────────────────────────────────────────────────────────────────────
Quantifies how much *unfreezing the encoder* buys on PDBbind affinity. Both arms
share the identical pdb_id splits and the identical MLP2 head over the mean-pooled
token rep — the ONLY difference is whether encoder gradients flow:

  • frozen   — EMA encoder fixed; encode each complex once (no_grad) → train the
               head on cached features. Reproduces the `probe` baseline.
  • finetune — encoder + head trained end-to-end (voxels re-encoded every step).
               Default full-encoder unfreeze at a low LR (1e-5) + early stopping;
               bf16/fp16 autocast keeps it within a shared GPU's memory.

    CUDA_VISIBLE_DEVICES=6 python dataset/01c_pdbbind_probe.py finetune \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3
  → dataset/data/pdbbind/finetune_results_e<N>[_<ver>][_filtered][_<tag>].csv
        (same --voxel_version / --cl1_only / --tag naming as `probe`)
        Probe schema + a `mode` (frozen|finetune), `ft_scope`, `encoder_lr`,
        `head_lr` column, plus a printed frozen-vs-finetune Δ summary.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset/ — for sibling import

from voxbind.models.density_vit import DensityViT
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore
# Each probe run self-consolidates into one table (see consolidate_probe_results).
# Guarded so a missing/broken aggregator never blocks the probe itself.
try:
    from consolidate_probe_results import append_run
except Exception:  # pragma: no cover
    append_run = None


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
FEAT_DIR    = PDBBIND_DIR / "features"
LP_CSV      = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"
RESULTS_DIR = PDBBIND_DIR / "results"          # per-run probe/finetune CSVs (own subfolder)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Focused meeting-doc folder — default sink for single-seed scatter CSVs (pred vs true pK),
# i.e. they land next to 260625_meeting.html / autoresearch.html. Override with --scatter_dir.
MEETING_DIR = Path(__file__).resolve().parents[2] / "notebook" / "html" / "260625"
# Per-complex protein-ligand H-bond counts (HBGSA cache, npz field `n`); used as an
# alternative regression target via `probe --target hbonds`.
HBOND_CACHE_DIR = Path(__file__).resolve().parents[2] / "hbgsa_baseline" / "cache" / "hbonds"
# Per-complex pocket B-factor targets (built by dataset/extract_bfactors.py); used as
# alternative regression targets via `probe --target bfactor` / `--target bfactor_rel`.
BFACTOR_CACHE = PDBBIND_DIR / "bfactors.json"
# MISATO-derived targets (built by dataset/extract_misato_targets.py): partial charges
# from QM.hdf5; adaptability/rmsf/pose_stability/interaction_energy from MD.hdf5.
MISATO_DIR     = PDBBIND_DIR.parent / "misato"
MISATO_QM_JSON = MISATO_DIR / "misato_qm.json"
MISATO_MD_JSON = MISATO_DIR / "misato_md.json"
MISATO_SPLIT_DIR = MISATO_DIR / "misato_splits"   # official MISATO 8:1:1 MD split (train/val/test_MD.txt)

EXPS = {
    "atomblob":               Path("exps") / "260606_atomblob_vit_mae_40m_invfreq_pretrain",   # 260606 v5-ablation (atom-only, version-agnostic)
    # Atom-only matched control that uses element-wise ligand vdW radii. This is
    # intentionally a separate condition so the lig-vdW control cannot be lost by
    # forgetting a side-channel --atom_source flag during feature/probe runs.
    "atomblob_ligvdw":        Path("exps") / "260609_atomblob_vit_mae_40m_invfreq_v5_ligvdw_pretrain",
    "atomblob_density":       Path("exps") / "260526_atomblob_density_vit_mae_40m_pretrain",
    "atomblob_weighted":      Path("exps") / "260528_atomblob_vit_mae_40m_weighted_pretrain",
    "atomblob_merged_density": Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain",
    # gradmag/ligvdw: merged atoms + density + ‖∇ρ‖ (9-ch), element-wise vdW ligand
    # radii, v4 density. Only a v4 run exists (also in EXPS_OVERRIDE so it's flagged
    # v4-native and skips the OOD warning).
    "atomblob_merged_density_gradmag": Path("exps") / "260603_atomblob_merged_density_vit_mae_40m_weighted_v4_gradmag_ligvdw_pretrain",
    # SEPARATE-channel (non-merged) gradmag/ligvdw encoder (260604): 11 atoms + density
    # + ‖∇ρ‖ (13-ch), element-wise vdW radii, v1 density.
    "atomblob_density_gradmag": Path("exps") / "260604_atomblob_density_vit_mae_40m_weighted_v1_gradmag_ligvdw_pretrain",
    # ROLE-SPLIT (roleblob): 2 ROLE channels [ligand-occupancy, pocket-occupancy] +
    # density + gradmag (4-ch), element-wise vdW radii (atom_source=ligvdw, both lig &
    # poc). Built from the ligvdw atom precompute by summing the per-element channels
    # (identical to a single-channel voxelization). input_mode/with_gradmag come from
    # the encoder cfg.yaml (roleblob_density / true), so the label is just a result tag.
    "roleblob_density_gradmag":            Path("exps") / "260623_plinder_otf_roleblob_cdg_vit_mask050_pretrain",
    "roleblob_density_gradmag_channelvit": Path("exps") / "260623_plinder_otf_roleblob_cdg_channelvit_mask050_pretrain",
}

# Conditions whose encoder was retrained on the v2/v3 density distributions;
# the (condition, version) pair overrides the default exp dir from EXPS.
EXPS_OVERRIDE: dict[tuple[str, str], Path] = {
    ("atomblob_merged_density", "v2"): Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain",
    ("atomblob_merged_density", "v3"): Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_v3_pretrain",
    ("atomblob_density",        "v2"): Path("exps") / "260531_atomblob_density_vit_mae_40m_v2_pretrain",
    # 260606 v5-ablation (this box): separate-channel atomblob_density on v5 arcsinh density.
    ("atomblob_density",        "v5"): Path("exps") / "260606_atomblob_density_vit_mae_40m_invfreq_v5_pretrain",
    # Atom-only ligand-vdW matched control for the v5 ablation app/table.
    ("atomblob_ligvdw",         "v5"): Path("exps") / "260609_atomblob_vit_mae_40m_invfreq_v5_ligvdw_pretrain",
    # v4 has two merged_density retrains: 260601 (plain) and 260602 (dual-head,
    # separate atom/density recon heads). We point v4 at the dual-head encoder
    # being evaluated; swap to 260601_..._v4_pretrain to probe the plain variant.
    ("atomblob_merged_density", "v4"): Path("exps") / "260602_atomblob_merged_density_vit_mae_40m_weighted_v4_dualhead_pretrain",
    # gradmag/ligvdw is v4-native (trained on v4 density); mark it so the OOD warning is skipped.
    ("atomblob_merged_density_gradmag", "v4"): Path("exps") / "260603_atomblob_merged_density_vit_mae_40m_weighted_v4_gradmag_ligvdw_pretrain",
    # v1-normalization variant (260604): same gradmag/ligvdw recipe, only the density
    # normalization differs (per-crop ±3σ z-score). Probe with --voxel_version v1.
    ("atomblob_merged_density_gradmag", "v1"): Path("exps") / "260604_atomblob_merged_density_vit_mae_40m_weighted_v1_gradmag_ligvdw_pretrain",
    # SEPARATE-channel gradmag/ligvdw on v5 arcsinh density. 260606 v5-ablation (this box,
    # invfreq recipe, density/gradmag weights 0.1/0.1). [was 260605 svr7 swept-balanced]
    ("atomblob_density_gradmag", "v5"): Path("exps") / "260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain",
    # density-only (2ch: ρ + ‖∇ρ‖, no atoms) — CrossDocked-v5 pretrain (run #2).
    ("density_gradmag", "v5"): Path("exps") / "260606_density_vit_mae_40m_xray_v5_gradmag_pretrain",
    # PLINDER-OTF roleblob (role-split) encoders share the atomblob-CDG OTF density
    # recipe (xray_resample_plinder, normalize:false, density_source:xray) → v5-native,
    # exactly like the otf_cdg encoders probed at v5. Mark so the v1-native OOD warning
    # is skipped. Same exp dir as the base EXPS map (override only suppresses the warn).
    ("roleblob_density_gradmag",            "v5"): Path("exps") / "260623_plinder_otf_roleblob_cdg_vit_mask050_pretrain",
    ("roleblob_density_gradmag_channelvit", "v5"): Path("exps") / "260623_plinder_otf_roleblob_cdg_channelvit_mask050_pretrain",
}


def resolve_exp(condition: str, version: str) -> Path:
    """Pick the encoder exp dir for a (condition, voxel_version) pair.

    A (condition, version) override wins; otherwise fall back to the base EXPS
    map. Checked explicitly (not dict.get with an EXPS[...] default) so that
    conditions that exist ONLY in EXPS_OVERRIDE — e.g. density_gradmag, which
    has no version-agnostic EXPS entry — resolve instead of KeyError-ing on the
    eagerly-evaluated default.
    """
    if (condition, version) in EXPS_OVERRIDE:
        return EXPS_OVERRIDE[(condition, version)]
    return EXPS[condition]


def voxel_dir_for(version: str) -> Path:
    """Resolve the voxel directory for a given version tag."""
    if version == "v1":
        return PDBBIND_DIR / "voxels"
    return PDBBIND_DIR / f"voxels_{version}"


def version_suffix(version: str) -> str:
    """Output-filename suffix for the version. v1 is the default → no suffix."""
    return "" if version == "v1" else f"_{version}"


def clean_tag(tag: Optional[str]) -> str:
    """Sanitize a free-form run label for use in artifact filenames."""
    if not tag:
        return ""
    return "_".join(str(tag).split()).strip("_")


def feature_suffix(version: str, *, tag: Optional[str] = None) -> str:
    """Filename suffix for feature caches: density version plus optional run tag."""
    parts: list[str] = []
    if version != "v1":
        parts.append(version)
    clean = clean_tag(tag)
    if clean:
        parts.append(clean)
    return ("_" + "_".join(parts)) if parts else ""


def results_suffix(version: str, *, cl1_only: bool = False,
                   tag: Optional[str] = None) -> str:
    """Filename suffix for a results CSV: density version + filtered split + run tag.

    Like version_suffix (v1 stays bare for back-compat), but every part is
    argument-driven so the name is reproducible and parseable:
      • `vN`        the density version, always a clean token (never `v5filt`);
      • `_filtered` when --cl1_only restricts to the LP_PDBBind CL1 subset, so a
                    filtered run never overwrites the full-split file of its version;
      • `_<tag>`    an optional free-form run label (encoder variant, date, …).
    e.g. version=v5, cl1_only=True, tag="invfreq" → "_v5_filtered_invfreq"
         (probe_results_e99_v5_filtered_invfreq.csv). The leading `vN` stays a clean
    token so consolidate_probe_results.parse_run_meta can recover the version.
    """
    parts: list[str] = []
    if version != "v1":
        parts.append(version)
    if cl1_only:
        parts.append("filtered")
    clean = clean_tag(tag)
    if clean:
        parts.append(clean)
    return ("_" + "_".join(parts)) if parts else ""


@dataclass(frozen=True)
class FeatureSpec:
    """How a pretrained encoder expects PDBbind voxels to be assembled."""

    input_mode: str
    with_gradmag: bool
    ligand_radius: float
    atom_source: str
    needs_atoms: bool
    needs_density: bool
    expected_channels: int
    with_diff: bool = False   # mFo-DFc difference-map channels (diff + its gradmag)


def _fallback_input_mode(condition: str) -> str:
    """Map legacy condition labels to the training `input_mode` vocabulary."""
    if condition == "density_gradmag":
        return "density"
    if condition in ("atomblob_weighted", "atomblob_ligvdw"):
        return "atomblob"
    if condition == "atomblob_density_gradmag":
        return "atomblob_density"
    if condition == "atomblob_merged_density_gradmag":
        return "atomblob_merged_density"
    if condition in ("roleblob_density_gradmag", "roleblob_density_gradmag_channelvit"):
        return "roleblob_density"
    return condition


def _expected_channels(input_mode: str, with_gradmag: bool, with_diff: bool = False,
                       n_lig: int = 7, n_poc: int = 4) -> int:
    if input_mode == "density":
        n = 1
    elif input_mode in ("atomblob", "atomblob_density"):
        # n_lig+n_poc atom channels (per-element 7+4=11 default; atomblob8 = 8+4=12).
        n = (n_lig + n_poc) + (1 if input_mode.endswith("_density") else 0)
    elif input_mode in ("atomblob_merged", "atomblob_merged_density"):
        n = 7 + (1 if input_mode.endswith("_density") else 0)
    elif input_mode in ("roleblob", "roleblob_density"):
        n = 2 + (1 if input_mode.endswith("_density") else 0)   # [ligand, pocket] (+density)
    else:
        raise ValueError(f"unsupported input_mode={input_mode!r}")
    # +2 for the mFo-DFc difference map + its gradmag (order [..., diff, diff_gradmag],
    # matching train_common._channel_layout with_diff=True).
    return n + (1 if with_gradmag else 0) + (2 if with_diff else 0)


def infer_feature_spec(condition: str, cfg, atom_source_arg: str = "auto") -> FeatureSpec:
    """Infer PDBbind tensor construction from a pretrained cfg.yaml.

    The condition name remains a result label; cfg.yaml is the source of truth for
    channel layout and ligand radius. This prevents v5 ligvdW encoders from being
    fed uniform-0.5 ligand blobs just because the label lacks "gradmag".
    """
    input_mode = (
        cfg.get("input_mode", None)
        or OmegaConf.select(cfg, "model.input_mode", default=None)
        or _fallback_input_mode(condition)
    )
    input_mode = str(input_mode)
    with_gradmag_cfg = cfg.get("with_gradmag", None)
    if with_gradmag_cfg is None:
        with_gradmag_cfg = OmegaConf.select(cfg, "model.with_gradmag", default=None)
    with_gradmag = bool(
        condition.endswith("gradmag") if with_gradmag_cfg is None else with_gradmag_cfg
    )
    # density_input=False encoders (coords-only patch embed; density+gradmag are
    # reconstruction TARGETS only, not encoder input) must be fed atoms-only at probe
    # time — the density that input_mode names lives only in the training target. Strip
    # it from the INPUT spec. Default True → unchanged for every existing encoder.
    density_input = OmegaConf.select(cfg, "mae.density_input", default=True)
    if density_input is False and input_mode.endswith("_density"):
        input_mode = input_mode[: -len("_density")]      # atomblob_density -> atomblob
        with_gradmag = False
    ligand_radius = float(OmegaConf.select(cfg, "dset.ligand_radius", default=0.5))
    # mFo-DFc difference-map channels: encoders trained with mae.with_diff=true expect
    # 2 extra input channels (diff + its gradmag) after density+gradmag → n_in 13→15.
    with_diff = bool(OmegaConf.select(cfg, "mae.with_diff", default=False))
    # atomblob8 encoders carry an 8th "other-heavy" ligand channel (n_channels_ligand=8);
    # default 7+4 per-element for every existing encoder.
    n_lig = int(OmegaConf.select(cfg, "model.n_channels_ligand", default=7))
    n_poc = int(OmegaConf.select(cfg, "model.n_channels_pocket", default=4))
    expected = _expected_channels(input_mode, with_gradmag, with_diff, n_lig, n_poc)
    needs_density = input_mode in ("density", "atomblob_density", "atomblob_merged_density") \
        or with_gradmag or with_diff
    needs_atoms = input_mode != "density"

    if condition == "atomblob_ligvdw" and atom_source_arg == "default":
        raise ValueError(
            "condition=atomblob_ligvdw requires ligand-vdW atom voxels; "
            "omit --atom_source or pass --atom_source ligvdw"
        )

    if condition == "atomblob_ligvdw":
        atom_source = "ligvdw"
    elif atom_source_arg == "auto":
        atom_source = "ligvdw" if ligand_radius <= 0 else "default"
    else:
        atom_source = atom_source_arg

    return FeatureSpec(
        input_mode=input_mode,
        with_gradmag=with_gradmag,
        ligand_radius=ligand_radius,
        atom_source=atom_source,
        needs_atoms=needs_atoms,
        needs_density=needs_density,
        expected_channels=expected,
        with_diff=with_diff,
    )


def atom_dir_for(vox_dir: Path, atom_source: str) -> Path:
    if atom_source == "default":
        return vox_dir / "atoms"
    if atom_source == "ligvdw":
        return PDBBIND_DIR / "voxels_ligvdw" / "atoms"
    raise ValueError(f"unknown atom_source={atom_source!r}")


CACHE_METADATA_VERSION = 2
EXPECTED_VOXELIZE_METADATA_VERSION = 2
EXPECTED_CENTER_SOURCE = "ligand_sdf_all_non_h_geometric_centroid"
_PATH_SIGNATURE_KEYS = {
    "exp_dir",
    "atom_dir",
    "dens_dir",
    "stats_path",
    "reference_stats_json",
    "metadata_path",
    "index_csv",
    "struct_dir",
    "ccp4_dir",
}
_ATOM_METADATA_KEYS = (
    "metadata_version",
    "kind",
    "center_source",
    "grid_dim",
    "resolution",
    "cubes_around",
    "ligand_channels",
    "pocket_channels",
    "element_filter",
    "ligand_vdw",
    "ligand_radius",
    "pocket_radius",
    "density_enabled",
    "density_mode",
    "index_csv",
    "struct_dir",
    "ccp4_dir",
)
_DENSITY_STAT_KEYS = (
    "metadata_version",
    "center_source",
    "grid_dim",
    "resolution",
    "element_filter",
    "scheme",
    "formula",
    "stats_source",
    "reference_stats_json",
    "arcsinh_scale",
    "mu",
    "sigma",
    "max_abs",
    "clip_lo_raw",
    "clip_hi_raw",
    "mu_clip",
    "sigma_clip",
    "mu_a",
    "sigma_a",
    "pdbbind_fit_mu_a",
    "pdbbind_fit_sigma_a",
    "n_crops",
    "n_voxels_total",
    "raw_min",
    "raw_max",
)


def _canonical_path(path) -> Optional[str]:
    if path in (None, ""):
        return None
    return str(Path(path).expanduser().resolve())


def atom_voxel_metadata(atom_dir: Path) -> dict:
    """Comparable metadata snapshot for a PDBbind atom voxel cache."""
    meta_path = atom_dir.resolve().parent / "metadata.json"
    meta = {
        "metadata_path": _canonical_path(meta_path),
        "present": meta_path.exists(),
    }
    if not meta_path.exists():
        return meta

    raw = json.loads(meta_path.read_text())
    for key in _ATOM_METADATA_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key in _PATH_SIGNATURE_KEYS and value is not None:
            value = _canonical_path(value)
        meta[key] = value
    return meta


def density_stats_metadata(voxel_version: str, dens_dir: Path) -> Optional[dict]:
    """Small, comparable metadata snapshot for density-normalised voxel dirs."""
    if voxel_version == "v1":
        return None

    stats_path = dens_dir.parent / "stats.json"
    meta = {
        "voxel_version": voxel_version,
        "stats_path": _canonical_path(stats_path),
        "present": stats_path.exists(),
    }
    if not stats_path.exists():
        return meta

    stats = json.loads(stats_path.read_text())
    for key in _DENSITY_STAT_KEYS:
        if key not in stats:
            continue
        value = stats[key]
        if key in _PATH_SIGNATURE_KEYS and value is not None:
            value = _canonical_path(value)
        meta[key] = value
    return meta


def build_cache_signature(
    *,
    condition: str,
    epoch: int,
    voxel_version: str,
    exp_dir: Path,
    spec: FeatureSpec,
    n_in_channels: int,
    feature_dim: int,
    atom_dir: Path,
    dens_dir: Path,
    require_density_stats: bool,
) -> dict:
    atom_metadata = atom_voxel_metadata(atom_dir) if spec.needs_atoms else None
    if spec.needs_atoms and not (atom_metadata and atom_metadata.get("present")):
        raise FileNotFoundError(
            f"missing atom voxel metadata: {atom_dir.resolve().parent / 'metadata.json'}"
        )
    if atom_metadata and atom_metadata.get("present"):
        if atom_metadata.get("metadata_version") != EXPECTED_VOXELIZE_METADATA_VERSION:
            raise ValueError(
                f"atom voxel metadata_version={atom_metadata.get('metadata_version')!r}; "
                f"expected {EXPECTED_VOXELIZE_METADATA_VERSION}"
            )
        if atom_metadata.get("center_source") != EXPECTED_CENTER_SOURCE:
            raise ValueError(
                f"atom voxel center_source={atom_metadata.get('center_source')!r}; "
                f"expected {EXPECTED_CENTER_SOURCE!r}"
            )
        if spec.atom_source == "ligvdw" and atom_metadata.get("ligand_vdw") is not True:
            raise ValueError(f"{atom_dir} is not a ligand-vdW atom cache")
        if spec.atom_source == "default" and atom_metadata.get("ligand_vdw") is not False:
            raise ValueError(f"{atom_dir} is not a default ligand-radius atom cache")

    density_stats = density_stats_metadata(voxel_version, dens_dir) if spec.needs_density else None
    if require_density_stats and spec.needs_density and voxel_version != "v1" \
            and not (density_stats and density_stats.get("present")):
        raise FileNotFoundError(
            f"missing density stats for {voxel_version}: {dens_dir.parent / 'stats.json'}"
        )
    if density_stats and density_stats.get("present"):
        if density_stats.get("metadata_version") != EXPECTED_VOXELIZE_METADATA_VERSION:
            raise ValueError(
                f"density stats metadata_version={density_stats.get('metadata_version')!r}; "
                f"expected {EXPECTED_VOXELIZE_METADATA_VERSION}"
            )
        if density_stats.get("center_source") != EXPECTED_CENTER_SOURCE:
            raise ValueError(
                f"density stats center_source={density_stats.get('center_source')!r}; "
                f"expected {EXPECTED_CENTER_SOURCE!r}"
            )
        if atom_metadata and atom_metadata.get("present"):
            atom_filter = atom_metadata.get("element_filter")
            density_filter = density_stats.get("element_filter")
            # 'none'/'all'/None/'' are synonyms for "no element filtering" (the atom
            # cache labels it 'none', density stats label it 'all'); only a genuine
            # filter mismatch (e.g. 'heavy' vs 'all') should fail. The champion cache
            # uses this atom(none)+density(all) combo and is valid.
            _norm = lambda x: "all" if x in (None, "none", "all", "") else x
            if _norm(atom_filter) != _norm(density_filter):
                raise ValueError(
                    f"atom element_filter={atom_filter!r} but density element_filter={density_filter!r}"
                )

    return {
        "metadata_version": CACHE_METADATA_VERSION,
        "condition": condition,
        "epoch": int(epoch),
        "voxel_version": voxel_version,
        "exp_dir": _canonical_path(exp_dir),
        "input_mode": spec.input_mode,
        "with_gradmag": bool(spec.with_gradmag),
        "ligand_radius": float(spec.ligand_radius),
        "atom_source": spec.atom_source,
        "n_in_channels": int(n_in_channels),
        "feature_dim": int(feature_dim),
        "needs_atoms": bool(spec.needs_atoms),
        "needs_density": bool(spec.needs_density),
        "atom_dir": _canonical_path(atom_dir),
        "atom_metadata": atom_metadata,
        "dens_dir": _canonical_path(dens_dir) if spec.needs_density else None,
        "density_stats": density_stats,
    }


def expected_cache_signature(condition: str, args: argparse.Namespace) -> dict:
    exp_override = getattr(args, "exp_dir", None)
    if exp_override:
        conditions = getattr(args, "conditions", [condition])
        if len(conditions) != 1:
            raise ValueError("--exp_dir is only supported for one probe condition at a time")
        exp_dir = Path(exp_override)
    else:
        exp_dir = resolve_exp(condition, args.voxel_version)
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    spec = infer_feature_spec(condition, cfg, "auto")
    vox_dir = voxel_dir_for(args.voxel_version)
    atom_dir = atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    return build_cache_signature(
        condition=condition,
        epoch=args.epoch,
        voxel_version=args.voxel_version,
        exp_dir=exp_dir,
        spec=spec,
        n_in_channels=int(cfg.model.n_in_channels),
        feature_dim=int(cfg.model.dim),
        atom_dir=atom_dir,
        dens_dir=dens_dir,
        require_density_stats=True,
    )


def _normalise_signature_value(key: str, value):
    if key in _PATH_SIGNATURE_KEYS and value is not None:
        return _canonical_path(value)
    if isinstance(value, dict):
        return {k: _normalise_signature_value(k, v) for k, v in value.items()}
    return value


def _normalise_signature(sig: Optional[dict]) -> dict:
    if not isinstance(sig, dict):
        return {}
    return {k: _normalise_signature_value(k, v) for k, v in sig.items()}


def _legacy_cache_signature(bundle: dict) -> dict:
    """Best-effort signature for pre-v2 bundles so mismatches are explainable."""
    keys = (
        "condition", "epoch", "voxel_version", "exp_dir", "input_mode",
        "with_gradmag", "ligand_radius", "atom_source", "n_in_channels",
        "feature_dim", "atom_dir", "dens_dir",
    )
    sig = {k: bundle[k] for k in keys if k in bundle}
    sig["metadata_version"] = bundle.get("metadata_version", 1)
    return _normalise_signature(sig)


def _values_equal(actual, expected) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return bool(np.isclose(float(actual), float(expected), rtol=1e-6, atol=1e-8))
        except Exception:
            return False
    return actual == expected


def _compare_signatures(actual: dict, expected: dict, prefix: str = "") -> list[str]:
    issues: list[str] = []
    for key, exp_value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            issues.append(f"{name}: missing (expected {exp_value!r})")
            continue
        act_value = actual[key]
        if isinstance(exp_value, dict):
            if not isinstance(act_value, dict):
                issues.append(f"{name}: expected dict, got {type(act_value).__name__}")
                continue
            issues.extend(_compare_signatures(act_value, exp_value, name))
        elif not _values_equal(act_value, exp_value):
            issues.append(f"{name}: got {act_value!r}, expected {exp_value!r}")
    return issues


def validate_feature_bundle(bundle: dict, expected: dict, feat_path: Path) -> list[str]:
    issues: list[str] = []
    if bundle.get("metadata_version") != CACHE_METADATA_VERSION:
        issues.append(
            f"metadata_version: got {bundle.get('metadata_version')!r}, "
            f"expected {CACHE_METADATA_VERSION}"
        )

    actual = bundle.get("cache_signature")
    if not isinstance(actual, dict):
        issues.append("cache_signature: missing; regenerate features with this script")
        actual = _legacy_cache_signature(bundle)
    else:
        actual = _normalise_signature(actual)

    expected = _normalise_signature(expected)
    issues.extend(_compare_signatures(actual, expected))

    features = bundle.get("features")
    if not isinstance(features, dict):
        issues.append("features: missing or not a dict")
    elif features:
        first_key = next(iter(features))
        first = features[first_key]
        if getattr(first, "ndim", None) != 1:
            issues.append(f"features[{first_key!r}]: expected 1-D tensor, got shape={getattr(first, 'shape', None)}")
        elif int(first.shape[0]) != int(expected["feature_dim"]):
            issues.append(
                f"features[{first_key!r}]: dim {int(first.shape[0])}, "
                f"expected {expected['feature_dim']}"
            )

    if issues:
        return [f"{feat_path}: {issue}" for issue in issues]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# features — frozen-encoder mean-pooled patch tokens
# ═══════════════════════════════════════════════════════════════════════════════

def _radius_embed_from_cfg(m) -> "dict | None":
    """Reconstruct RadiusAtomEmbed kwargs from a saved cfg.model (None if radius_embed_k<=0).
    Mirrors the assembly in train_density.py so the probe rebuilds the SAME encoder arch
    (the learnable radius→k atom embedding lives in encoder.*, so it must be present to load)."""
    rek = int(m.get("radius_embed_k", 0))
    if rek <= 0:
        return None
    from voxbind.constants import vdw_radius
    LIG_EL = ["C", "O", "N", "S", "F", "Cl", "P"]
    POC_EL = ["C", "O", "N", "S"]
    n_lig = int(m.get("n_channels_ligand", 7))
    n_poc = int(m.get("n_channels_pocket", 4))
    def _radii(elems, n):
        return [vdw_radius(elems[i]) if i < len(elems) else vdw_radius("C") for i in range(n)]
    n_pt = int(m.n_in_channels) - (n_lig + n_poc)
    scales_cfg = m.get("radius_embed_scales", None)
    scales = [float(s) for s in scales_cfg] if scales_cfg is not None else [0.0]
    return {"lig_radii": _radii(LIG_EL, n_lig), "poc_radii": _radii(POC_EL, n_poc),
            "k": rek, "hidden": int(m.get("radius_embed_hidden", 16)), "n_passthrough": n_pt,
            "scales": scales}


def load_encoder(exp_dir: Path, epoch: int, device: str, cfg=None) -> DensityViT:
    """Instantiate DensityViT from the exp's cfg.yaml and load EMA weights."""
    cfg = cfg if cfg is not None else OmegaConf.load(exp_dir / "cfg.yaml")
    m = cfg.model

    encoder = DensityViT(
        grid_dim     = cfg.vox.grid_dim,
        patch_size   = m.patch_size,
        n_in_channels= m.n_in_channels,
        c_out        = m.n_channels // 2,
        dim          = m.dim,
        depth        = m.depth,
        n_heads      = m.heads,
        mlp_ratio    = m.mlp_ratio,
        dropout      = m.dropout,
        pos_encoding = m.get("pos_encoding", "learnable"),
        patch_embed_mode = m.get("patch_embed_mode",
                                 "channel_group" if m.get("channel_groups", None) else "fused"),
        channel_groups   = (tuple(m.channel_groups) if m.get("channel_groups", None) else None),
        n_memory_tokens  = int(m.get("n_memory_tokens", 0)),   # ChA-MAE encoders carry l memory tokens
        radius_embed     = _radius_embed_from_cfg(m),          # learnable radius→k atom embedding (opt-in)
    )

    ckpt_path = exp_dir / f"checkpoint_e{epoch:04d}.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # encoder_state_dict_ema keys are prefixed 'encoder.' from the wrapper.
    raw = ckpt["encoder_state_dict_ema"]
    stripped = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    missing = encoder.load_state_dict(stripped, strict=True)
    print(f"  loaded {len(stripped)} weights from {ckpt_path.name} (epoch={ckpt.get('epoch')})")
    return encoder.to(device).eval()


def forward_tokens(encoder: DensityViT, x: torch.Tensor) -> torch.Tensor:
    """(B, n_in, G, G, G) → (B, T_patch, D) post-norm patch tokens (memory tokens
    excluded) — the forward up to `encoder.norm` that the probe mean-pools. Stops
    before `decoder_proj` (the MAE recon head is irrelevant to the pooled rep).

    Delegates to `encoder.forward_features`, which handles BOTH the fused patch
    embed and channel_group / ChA encoders (where `encoder.patch_embed` is None and
    memory tokens are prepended through the trunk). Grad-capable (no @no_grad) so the
    end-to-end fine-tune path can backprop into the encoder. For a fused encoder this
    is identical to the previous inline patch_embed→blocks→norm path.
    """
    return encoder.forward_features(x)


@torch.no_grad()
def encode_tokens(encoder: DensityViT, x: torch.Tensor) -> torch.Tensor:
    """No-grad wrapper around `forward_tokens` for frozen feature extraction."""
    return forward_tokens(encoder, x)


def load_voxels_for(
    pid: str,
    condition: str,
    n_in_channels: int,
    atom_dir: Path,
    dens_dir: Path,
    input_mode: str = None,
    with_gradmag: bool = False,
    gradmag_dir: Path = None,
    with_diff: bool = False,
    diff_dir: Path = None,
) -> torch.Tensor:
    """Build the (n_in_channels, G, G, G) tensor for one complex."""
    input_mode = input_mode or _fallback_input_mode(condition)
    needs_atoms = input_mode != "density"
    needs_density = input_mode in (
        "density", "atomblob_density", "atomblob_merged_density", "roleblob_density",
    ) or with_gradmag or with_diff

    atoms_t = None
    if needs_atoms:
        atoms = np.load(atom_dir / f"{pid}.npy")             # (11, G, G, G) float16
        atoms_t = torch.from_numpy(atoms.astype(np.float32)) # promote to float32

    dens_t = None
    if needs_density:
        dens = np.load(dens_dir / f"{pid}.npy")              # (G, G, G) float16
        dens_t = torch.from_numpy(dens.astype(np.float32)).unsqueeze(0)

    if input_mode == "density":
        x = dens_t
    elif input_mode == "atomblob":
        x = atoms_t
    elif input_mode == "atomblob_density":
        x = torch.cat([atoms_t, dens_t], dim=0)              # (12, G, G, G)
    elif input_mode == "atomblob_merged":
        merged = atoms_t[:7].clone()
        merged[:4] += atoms_t[7:11]
        x = merged                                           # (7, G, G, G)
    elif input_mode == "atomblob_merged_density":
        # Pocket {C,O,N,S} (channels 7..10) folded into ligand {C,O,N,S} (0..3).
        # Ligand-only halogens + P (channels 4..6) stay as-is.
        merged = atoms_t[:7].clone()
        merged[:4] += atoms_t[7:11]
        x = torch.cat([merged, dens_t], dim=0)                # (8, G, G, G)
    elif input_mode in ("roleblob", "roleblob_density"):
        # Role-split: collapse the 7 ligand-element + 4 pocket-element channels into
        # 2 ROLE channels [ligand-occupancy, pocket-occupancy] by summation — exactly a
        # single-channel voxelization of each role (each atom's blob is channel-agnostic;
        # verified to float epsilon). Requires the ligvdw precompute (vdW-radius blobs) so
        # blob SIZE carries atom identity — atom_source resolves to ligvdw when
        # dset.ligand_radius<=0 (see infer_feature_spec). Order [ligand, pocket] matches
        # train_common._build_role_atoms.
        role = torch.cat([atoms_t[:7].sum(0, keepdim=True),
                          atoms_t[7:11].sum(0, keepdim=True)], dim=0)   # (2, G, G, G)
        x = role if input_mode == "roleblob" else torch.cat([role, dens_t], dim=0)
    else:
        raise ValueError(f"unknown input_mode={input_mode!r} for condition={condition!r}")

    if with_gradmag:
        if gradmag_dir is not None:
            # Density-ablation control: load precomputed matched-noise gradmag
            # instead of deriving ‖∇ρ‖ (which would be ‖∇noise‖, wrong distribution).
            g = torch.from_numpy(np.load(gradmag_dir / f"{pid}.npy").astype(np.float32)).unsqueeze(0)
        else:
            # ‖∇ρ‖ is derived from the already-normalised density crop exactly as
            # train_density.py does for xray source, then appended last.
            g = per_sample_zscore(gradient_magnitude3d(dens_t.unsqueeze(0))).squeeze(0)
        x = torch.cat([x, g], dim=0)

    if with_diff:
        # mFo-DFc difference-map crop (arcsinh+pool-z, same crop/pose as density) then
        # its ‖∇·‖ (per-sample z-score), appended last → order [..., diff, diff_gradmag],
        # matching train_common._channel_layout / crossdocked_density_box (with_diff=True).
        diff = np.load(diff_dir / f"{pid}.npy")                  # (G, G, G) float16
        diff_t = torch.from_numpy(diff.astype(np.float32)).unsqueeze(0)
        diff_g = per_sample_zscore(gradient_magnitude3d(diff_t.unsqueeze(0))).squeeze(0)
        x = torch.cat([x, diff_t, diff_g], dim=0)

    assert x.shape[0] == n_in_channels, (
        f"{condition}: built {x.shape[0]} channels from input_mode={input_mode}, "
        f"with_gradmag={with_gradmag}, but encoder expects {n_in_channels}"
    )
    return x


class _ExtractDataset(Dataset):
    """Per-complex voxel builder for frozen feature extraction.

    Returns (pid, x) so a DataLoader worker pool overlaps disk I/O + voxel
    assembly (np.load + float32 promote + cat + on-the-fly gradmag) with the
    GPU forward of the previous batch. A failed complex is surfaced as
    (pid, None, err) instead of raising, so one bad file doesn't kill the whole
    batch — preserving the old serial loop's per-pid error logging."""

    def __init__(self, pids, condition, n_in, atom_dir, dens_dir,
                 input_mode, with_gradmag, gradmag_dir,
                 with_diff=False, diff_dir=None):
        self.pids = list(pids)
        self.condition = condition
        self.n_in = n_in
        self.atom_dir = atom_dir
        self.dens_dir = dens_dir
        self.input_mode = input_mode
        self.with_gradmag = with_gradmag
        self.gradmag_dir = gradmag_dir
        self.with_diff = with_diff
        self.diff_dir = diff_dir

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int):
        pid = self.pids[i]
        try:
            x = load_voxels_for(pid, self.condition, self.n_in,
                                self.atom_dir, self.dens_dir,
                                input_mode=self.input_mode,
                                with_gradmag=self.with_gradmag,
                                gradmag_dir=self.gradmag_dir,
                                with_diff=self.with_diff,
                                diff_dir=self.diff_dir)
            return pid, x, ""
        except Exception as e:                                   # logged, not fatal
            return pid, None, repr(e)[:160]


def _extract_collate(batch):
    """Drop failed items, stack the survivors. Returns (pids, x|None, errors)."""
    good = [(pid, x) for pid, x, err in batch if x is not None]
    errs = [(pid, err) for pid, x, err in batch if x is None]
    if good:
        pids, xs = zip(*good)
        x = torch.stack(xs, dim=0)
    else:
        pids, x = (), None
    return list(pids), x, errs


def run_features(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = feature_suffix(args.voxel_version, tag=getattr(args, "tag", None))
    out_path = out_dir / f"{args.condition}_e{args.epoch}{suffix}.pt"

    exp_dir = Path(args.exp_dir) if getattr(args, "exp_dir", None) else \
        resolve_exp(args.condition, args.voxel_version)
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    spec = infer_feature_spec(args.condition, cfg, args.atom_source)

    vox_dir  = voxel_dir_for(args.voxel_version)
    # --atom_dir override: feed a custom atom-voxel dir (e.g. voxels_atomblob8/atoms for
    # atomblob8 8-lig encoders) instead of the version's default per-element atoms.
    atom_dir = Path(args.atom_dir) if getattr(args, "atom_dir", None) \
        else atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    # Density-ablation control: when --noise_voxels_dir is set, density and gradmag
    # are read from the precomputed matched-noise voxels instead of the real ones.
    gradmag_dir = None
    if getattr(args, "noise_voxels_dir", None):
        _nd = Path(args.noise_voxels_dir)
        dens_dir = _nd / "density"
        gradmag_dir = _nd / "gradmag"
    # --dens_dir: override just the density crop dir (gradmag still derived on-the-fly).
    # Used to feed RAW (un-normalized) density crops to a raw-density-trained encoder.
    if getattr(args, "dens_dir", None):
        dens_dir = Path(args.dens_dir)
    # mFo-DFc difference-map crops (parallel to dens_dir). Default: voxels_v5_diff/density
    # next to the density voxel dir; override with --diff_voxel_dir.
    diff_dir = None
    if spec.with_diff:
        diff_dir = Path(args.diff_voxel_dir) if getattr(args, "diff_voxel_dir", None) \
            else vox_dir.parent / "voxels_v5_diff" / "density"
    # availability.csv only exists in v1; v2/v3 reuse v1's since the pid set
    # is the same (they share the same successful-crop universe).
    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = PDBBIND_DIR / "voxels" / "availability.csv"

    print(f"=== PDBbind frozen-encoder features ===")
    print(f"  condition      : {args.condition}")
    print(f"  epoch          : {args.epoch}")
    print(f"  voxel_version  : {args.voxel_version}")
    print(f"  feature_tag    : {getattr(args, 'tag', None)}")
    print(f"  voxel_dir      : {vox_dir}")
    print(f"  exp_dir        : {exp_dir}")
    print(f"  input_mode     : {spec.input_mode}")
    print(f"  with_gradmag   : {spec.with_gradmag}")
    print(f"  ligand_radius  : {spec.ligand_radius}")
    print(f"  atom_source    : {spec.atom_source} ({atom_dir})")
    print(f"  dens_dir       : {dens_dir}")
    if spec.with_diff:
        print(f"  with_diff      : True  (mFo-DFc +2ch)")
        print(f"  diff_dir       : {diff_dir}")
    print(f"  device         : {args.device}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  out            : {out_path}")

    # ── Distribution-shift warning when the encoder wasn't retrained on the
    # selected density version. We override the exp dir for (condition, version)
    # combos that have a matching retrain; if no override exists for this combo
    # AND the condition reads density AND version != v1, the encoder is OOD.
    has_override = (args.condition, args.voxel_version) in EXPS_OVERRIDE
    if spec.needs_density and args.voxel_version != "v1" and not has_override:
        print(f"\n  [warn] {args.condition} encoder was pretrained on v1 density.")
        print(f"  [warn] --voxel_version {args.voxel_version} feeds the encoder OOD inputs.")
        print(f"  [warn] Probe results reflect distribution shift, not encoder quality.")
    if getattr(args, "exp_dir", None) and not clean_tag(getattr(args, "tag", None)):
        print("\n  [warn] --exp_dir without --tag writes to the shared condition cache path.")
        print("  [warn] Add --tag to keep multiple same-condition checkpoints side-by-side.")

    encoder = load_encoder(exp_dir, args.epoch, args.device, cfg=cfg)
    n_in = encoder.n_in_channels
    if n_in != spec.expected_channels:
        raise RuntimeError(
            f"cfg-derived layout builds {spec.expected_channels} channels "
            f"(input_mode={spec.input_mode}, with_gradmag={spec.with_gradmag}) "
            f"but encoder.n_in_channels={n_in}"
        )
    cache_signature = build_cache_signature(
        condition=args.condition,
        epoch=args.epoch,
        voxel_version=args.voxel_version,
        exp_dir=exp_dir,
        spec=spec,
        n_in_channels=n_in,
        feature_dim=encoder.dim,
        atom_dir=atom_dir,
        dens_dir=dens_dir,
        require_density_stats=True,
    )
    if cache_signature["density_stats"] is not None:
        ds = cache_signature["density_stats"]
        print(f"  density_stats  : {ds.get('stats_path')} "
              f"(source={ds.get('stats_source', 'n/a')}, present={ds.get('present')})")

    # Pick the right pdb_id pool: density-using conditions need has_density,
    # atom-only conditions (atomblob, atomblob_weighted) just need has_atoms.
    avail = pd.read_csv(avail_csv)
    if spec.needs_density:
        pool = avail[avail["has_atoms"] & avail["has_density"]].copy()
    else:
        pool = avail[avail["has_atoms"]].copy()
    if args.max_complexes:
        pool = pool.head(args.max_complexes)
    pids = pool["pdb_id"].tolist()
    print(f"  complexes  : {len(pids):,}")

    features: dict[str, torch.Tensor] = {}
    n_err = 0
    err_log: list[tuple[str, str]] = []

    # Worker-pooled, prefetched forward pass. Voxel assembly (np.load + float32
    # promote + cat + gradmag) runs in DataLoader workers and overlaps with the
    # GPU forward; H2D is async (pinned + non_blocking) and the only D2H sync is
    # a single .cpu() at the end. The frozen encoder is LayerNorm-only and the
    # rep is per-sample mean-pooled, so batch_size is mathematically
    # result-neutral — features are identical regardless of batching.
    on_cuda = str(args.device).startswith("cuda")
    num_workers = getattr(args, "num_workers", 8)
    ds = _ExtractDataset(pids, args.condition, n_in, atom_dir, dens_dir,
                         spec.input_mode, spec.with_gradmag, gradmag_dir,
                         with_diff=spec.with_diff, diff_dir=diff_dir)
    # pin_memory only with workers: pinning + the multiprocessing worker fd-share path
    # can flakily fail with "CUDA error: invalid argument" in the pin_memory thread
    # (fork-after-CUDA-init). num_workers=0 → single-process, no fork, no pin → robust.
    pin = on_cuda and num_workers > 0
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_extract_collate,
    )

    feat_chunks: list[torch.Tensor] = []                            # kept on-device
    ordered_pids: list[str] = []
    pbar = tqdm(loader, unit="batch", desc=f"encode {args.condition}")
    for batch_pids, x, errs in pbar:
        for pid, msg in errs:
            err_log.append((pid, msg))
            n_err += 1
        if x is None:
            continue
        x = x.to(args.device, non_blocking=on_cuda)                # (B, n_in, G, G, G)
        tokens = encode_tokens(encoder, x)                          # (B, N, D)
        feat_chunks.append(tokens.mean(dim=1))                      # (B, D), still on device
        ordered_pids.extend(batch_pids)
        pbar.set_postfix(saved=len(ordered_pids), err=n_err, refresh=False)

    if feat_chunks:                                                 # single D2H sync
        all_feats = torch.cat(feat_chunks, dim=0).cpu()
        for pid, vec in zip(ordered_pids, all_feats):
            features[pid] = vec.contiguous().clone()

    if out_path.is_symlink():
        out_path.unlink()
    torch.save({
        "metadata_version": CACHE_METADATA_VERSION,
        "cache_signature": cache_signature,
        "condition": args.condition,
        "epoch":     args.epoch,
        "n_in_channels": n_in,
        "feature_dim": encoder.dim,
        "exp_dir": cache_signature["exp_dir"],
        "voxel_version": args.voxel_version,
        "input_mode": spec.input_mode,
        "with_gradmag": spec.with_gradmag,
        "ligand_radius": spec.ligand_radius,
        "atom_source": spec.atom_source,
        "atom_dir": cache_signature["atom_dir"],
        "dens_dir": cache_signature["dens_dir"],
        "features":  features,
    }, out_path)
    print(f"\n  saved {len(features):,} features → {out_path}  "
          f"({out_path.stat().st_size/1e6:.1f} MB)")
    if err_log:
        err_path = out_dir / f"{args.condition}_e{args.epoch}{suffix}_errors.txt"
        err_path.write_text("\n".join(f"{p}\t{m}" for p, m in err_log) + "\n")
        print(f"  errors   : {n_err:,}  →  {err_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# probe — 2-layer MLP on frozen-encoder features
# ═══════════════════════════════════════════════════════════════════════════════

def load_lp_index(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def load_hbond_targets(cache_dir: Path = HBOND_CACHE_DIR) -> dict[str, float]:
    """Map pdb_id -> protein-ligand H-bond count from the HBGSA cache (npz field `n`).

    The cache is built by hbgsa_baseline/src/hbonds.py (geometric N/O/S donor-acceptor
    criteria, capped at 20). Used as an alternative `probe` regression target.
    """
    out: dict[str, float] = {}
    for npz in sorted(cache_dir.glob("*.npz")):
        try:
            out[npz.stem.lower()] = float(np.load(npz)["n"])
        except Exception:
            pass
    return out


def load_bfactor_targets(field: str, cache_path: Path = BFACTOR_CACHE) -> dict[str, float]:
    """Map pdb_id -> pocket B-factor target from the extract_bfactors.py cache.

    field='mean_pocket_b'  raw mean pocket B (absolute flexibility, but scales with
                           resolution / overall B-scaling — partly a resolution proxy).
    field='rel_pocket_b'   pocket-mean B / protein-mean B (per-structure scale cancels,
                           so resolution-robust; <1 = site more rigid than its protein).
    Non-finite / non-positive values are dropped, so a complex without usable B-factors
    is simply absent from the regression pool (build_dataset then skips it).
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"B-factor cache not found: {cache_path}\n"
            f"  build it with: python dataset/extract_bfactors.py"
        )
    raw = json.loads(cache_path.read_text())
    out: dict[str, float] = {}
    for pid, rec in raw.items():
        v = rec.get(field)
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v) and v > 0.0:
            out[pid.lower()] = v
    return out


def load_misato_target(field: str, json_path: Path) -> dict[str, float]:
    """Map pdb_id -> a MISATO-derived target from the extract_misato_targets.py JSON.

    Only finiteness is required (interaction_energy is negative, partial_charge positive),
    so a complex without the field is simply absent from the regression pool.
    """
    if not json_path.exists():
        src = "qm" if "qm" in json_path.name else "md"
        raise FileNotFoundError(
            f"MISATO cache not found: {json_path}\n"
            f"  build it with: <h5py-env python> dataset/extract_misato_targets.py --source {src}"
        )
    raw = json.loads(json_path.read_text())
    out: dict[str, float] = {}
    for pid, rec in raw.items():
        v = rec.get(field)
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out[pid.lower()] = v
    return out


def load_misato_split(split_dir: Path = MISATO_SPLIT_DIR) -> dict[str, str]:
    """pid(lower) -> 'train'/'val'/'test' from MISATO's official 8:1:1 MD split files."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        f = split_dir / f"{split}_MD.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"MISATO split file not found: {f}\n"
                f"  fetch data/MD/splits/*_MD.txt from the misato-dataset repo into {split_dir}"
            )
        for line in f.read_text().splitlines():
            pid = line.strip().lower()
            if pid:
                out[pid] = split
    return out


# Regression targets selectable via `probe --target`. pK is the LP_PDBBind default
# (target_map=None → build_dataset reads the pK column); every other target is a
# pdb_id -> value map regressed on the identical splits.
TARGET_LABELS = {
    "pK":                 "pK (binding affinity, LP_PDBBind)",
    "hbonds":             "protein-ligand H-bond count (HBGSA cache)",
    "bfactor":            "mean pocket B-factor (raw, resolution-confounded)",
    "bfactor_rel":        "relative pocket B-factor (pocket/protein, resolution-robust)",
    "partial_charges":    "ligand partial-charge magnitude (MISATO QM, mean|q|)",
    "electron_affinity":  "ligand electron affinity (MISATO QM)",
    "hardness":           "ligand chemical hardness (MISATO QM)",
    "adaptability":       "pocket adaptability (MISATO MD)",
    "rmsf":               "pocket RMSF (MISATO MD)",
    "pose_stability":     "ligand pose RMSD over MD (MISATO MD)",
    "interaction_energy": "protein-ligand interaction energy (MISATO MD)",
}


def load_target_map(target: str) -> Optional[dict[str, float]]:
    """Resolve `--target` to a pdb_id -> value map, or None for the pK default."""
    if target == "pK":
        return None
    if target == "hbonds":
        return load_hbond_targets()
    if target == "bfactor":
        return load_bfactor_targets("mean_pocket_b")
    if target == "bfactor_rel":
        return load_bfactor_targets("rel_pocket_b")
    if target == "partial_charges":
        return load_misato_target("partial_charge", MISATO_QM_JSON)
    if target == "electron_affinity":
        return load_misato_target("electron_affinity", MISATO_QM_JSON)
    if target == "hardness":
        return load_misato_target("hardness", MISATO_QM_JSON)
    if target == "adaptability":
        return load_misato_target("adaptability", MISATO_MD_JSON)
    if target == "rmsf":
        return load_misato_target("rmsf", MISATO_MD_JSON)
    if target == "pose_stability":
        return load_misato_target("pose_rmsd", MISATO_MD_JSON)
    if target == "interaction_energy":
        return load_misato_target("interaction_energy", MISATO_MD_JSON)
    raise ValueError(f"unknown target={target!r}")


def target_source(target: str):
    """Display source for a given --target's labels (run_probe logging)."""
    if target == "hbonds":
        return HBOND_CACHE_DIR
    if target in ("bfactor", "bfactor_rel"):
        return BFACTOR_CACHE
    if target in ("partial_charges", "electron_affinity", "hardness"):
        return MISATO_QM_JSON
    if target in ("adaptability", "rmsf", "pose_stability", "interaction_energy"):
        return MISATO_MD_JSON
    return "LP_PDBBind"


# ── Frozen, version-controlled splits (voxbind/splits) ───────────────────────────
# Map a --split flag to a committed manifest scheme. load_split() hash-verifies the
# pinned membership, so every server uses the IDENTICAL train/val/test set regardless
# of which crops/RSCC happen to have materialized locally.
FROZEN_SPLIT_SCHEMES = {
    # canonical affinity split — Kd/Ki-only (IC50 dropped 260623), ED + lig&poc RSCC≥0.8, sequence-dedup
    "lp_edrscc":    "lp_edrscc_v2",
    "lp_edrscc_v1": "lp_edrscc_v1",   # legacy: IC50-inclusive (5817/1498/2813)
    "lp_edrscc_v2": "lp_edrscc_v2",   # explicit alias for the canonical (3850/817/1320)
    # LP-PDBBind nested no-leak cleaning tiers (exact subsets of v2, filter on all splits)
    "lp_edrscc_v2_cl1":   "lp_edrscc_v2_cl1",    # +CL1        (2721/680/1166)
    "lp_edrscc_v2_cl12":  "lp_edrscc_v2_cl12",   # +CL1+CL2    (2643/659/1149)
    "lp_edrscc_v2_cl123": "lp_edrscc_v2_cl123",  # +CL1+CL2+CL3 (1559/410/733)
    "time":         "time_v1",        # temporal holdout (train ≤2016 / val 2017 / test ≥2018)
    "misato":       "misato_md_v1",   # MISATO official 8:1:1 MD split (committed mirror)
    # GEMS CleanSplit (leakage + intra-train redundancy filtered) ∩ ED+RSCC+Kd/Ki; test = CASF-2016
    "clean_ed":        "clean_ed_v1",         # 4099/1000/214 (CASF-2016 test)
    "clean_ed_v1":     "clean_ed_v1",
    "clean_ed_v1_indep": "clean_ed_v1_indep", # 4099/1000/109 (leakage-independent CASF-2016 test)
    # LBA-style protein-seq-identity splits on the lp_edrscc_v2 pool (Atom3D-LBA axis):
    # protein leak-proof (0% cross-split seq overlap), ligand-agnostic. Mirror image of
    # LP-PDBBind (which controls ligand/interaction, leaks protein). See dataset/make_lba_split.py.
    "lba30":        "lba30",           # mmseqs 30% id clustering (3846/823/1318)
    "lba60":        "lba60",           # mmseqs 60% id clustering (3828/821/1338)
}


def load_frozen_split_map(split_flag: str) -> tuple[dict[str, str], str]:
    """(pid -> 'train'/'val'/'test', scheme_name) from the frozen manifest for `split_flag`."""
    from voxbind.splits import load_split
    scheme = FROZEN_SPLIT_SCHEMES[split_flag]
    sp = load_split(scheme)                       # hash/count-verified; raises loudly on drift
    return {p: s for s in ("train", "val", "test") for p in sp[s]}, scheme


def build_dataset(
    features: dict[str, torch.Tensor],
    lp_df: pd.DataFrame,
    drop_covalent: bool,
    cl1_only: bool,
    target_map: Optional[dict[str, float]] = None,
    split_map: Optional[dict[str, str]] = None,
) -> dict:
    """Build train/val/test splits keyed by LP_PDBBind 'new_split'.

    Returns dict with X, y, pid for each split (numpy arrays). The regression target
    is LP_PDBBind pK by default; pass `target_map` (pdb_id -> value) to regress an
    alternative target (e.g. protein-ligand H-bond count) on the identical splits/
    filters — complexes absent from the map are dropped (they have no label).
    """
    if split_map is not None:
        # A frozen manifest (or MISATO) decides membership; y comes from target_map if
        # given, else LP pK (default affinity target — works for any frozen scheme).
        ymap = (target_map if target_map is not None
                else {p: v for p, v in zip(lp_df["pdb_id"], lp_df["pK"]) if pd.notna(v)})
        from voxbind.splits import check_local_availability
        usable = {p for p in features if p in ymap}        # locally present AND labeled
        dim = len(next(iter(features.values()))) if features else 512
        split_data = {}
        for split in ("train", "val", "test"):
            want = [p for p, s in split_map.items() if s == split]
            # Loud-warn policy: report (don't silently drop) any frozen pid missing here.
            avail = check_local_availability(want, usable, label=split,
                                             warn=lambda m: print("    " + m))
            pids = avail["used"]
            X = (np.stack([features[p].numpy() for p in pids]).astype(np.float32)
                 if pids else np.empty((0, dim), np.float32))
            y = np.asarray([ymap[p] for p in pids], dtype=np.float32)
            split_data[split] = {"X": X, "y": y, "pid": pids,
                                  "content_hash": avail["content_hash"],   # stamps effective set
                                  "n_frozen": avail["n_expected"]}
        return split_data

    df = lp_df.copy()
    if drop_covalent:
        df = df[~df["covalent"].astype(bool)]
    if cl1_only:
        df = df[df["CL1"].astype(bool)]
    df = df[df["pdb_id"].isin(features.keys())]
    df = df[df["new_split"].isin(["train", "val", "test"])]
    if target_map is None:
        df = df.dropna(subset=["pK"])
        ycol = "pK"
    else:
        df = df.assign(_target=df["pdb_id"].map(target_map)).dropna(subset=["_target"])
        ycol = "_target"

    split_data: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        sub  = df[df["new_split"] == split]
        pids = sub["pdb_id"].tolist()
        X    = np.stack([features[p].numpy() for p in pids]).astype(np.float32)
        y    = sub[ycol].astype(np.float32).to_numpy()
        split_data[split] = {"X": X, "y": y, "pid": pids}
    return split_data


# ── Token pooling: collapse (B, N, D) tokens → (B, D) for the affinity head ────
# `mean` is the original fixed pooling (and the only one the cached `probe`/frozen
# path can use, since those features are already mean-pooled on disk). `attn` is a
# learned-query attention pool: a single query attends over all N patch tokens, so
# the head can *select* binding-relevant tokens instead of averaging them away —
# only meaningful where the full token sequence is live (the `finetune` path).

class MeanPool(nn.Module):
    """Fixed mean over the token axis: (B, N, D) → (B, D)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class AttnPool(nn.Module):
    """Learned-query attention pooling: one query attends over N tokens → (B, D)."""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)        # (B, 1, D)
        out, _ = self.attn(q, x, x, need_weights=False)  # (B, 1, D)
        return self.norm(out.squeeze(1))                 # (B, D)


def make_pool(kind: str, dim: int, *, n_heads: int = 8, dropout: float = 0.0) -> nn.Module:
    if kind == "mean":
        return MeanPool()
    if kind == "attn":
        return AttnPool(dim, n_heads=n_heads, dropout=dropout)
    raise ValueError(f"unknown pool: {kind!r} (expected 'mean' or 'attn')")


class MLP2(nn.Module):
    """Probe head: input_dim → hidden → out_dim (out_dim=1 → squeezed scalar)."""
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


def soft_count_targets(y: torch.Tensor, num_classes: int, sigma: float) -> torch.Tensor:
    """Gaussian soft labels over counts 0..num_classes-1, centered on each integer y.

    Returns (N, num_classes) rows summing to 1; `sigma` is the smoothing width
    (sigma→0 → one-hot). softmax(−d²/2σ²) is the normalized discrete Gaussian, so the
    edge classes (0 and num_classes−1) simply keep their truncated, renormalized mass.
    """
    karange = torch.arange(num_classes, device=y.device, dtype=torch.float32)
    d = y.unsqueeze(1) - karange.unsqueeze(0)                       # (N, K)
    return torch.softmax(-(d * d) / (2.0 * sigma * sigma), dim=1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean per-class F1 over the labels present in true∪pred (numpy; no sklearn)."""
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    f1s = []
    for c in labels:
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> dict:
    """Multiclass metrics for integer counts 0..K-1 (numpy-only).

    accuracy, within-1 accuracy (|Δcount|≤1), macro-F1 (over present labels),
    balanced accuracy (mean recall over true-present classes), quadratic-weighted
    Cohen's κ over the full 0..K-1 ordinal scale, and the majority-class baseline.
    """
    y_true = np.rint(y_true).astype(int)
    y_pred = np.rint(np.clip(y_pred, 0, K - 1)).astype(int)
    acc = float((y_true == y_pred).mean())
    within1 = float((np.abs(y_true - y_pred) <= 1).mean())
    macro_f1 = _macro_f1(y_true, y_pred)
    # balanced accuracy = mean recall over classes present in y_true
    recalls = []
    for c in np.unique(y_true):
        denom = (y_true == c).sum()
        recalls.append(float(((y_pred == c) & (y_true == c)).sum()) / float(denom))
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    # quadratic-weighted kappa over the 0..K-1 rating scale
    cls = np.arange(K)
    O = np.zeros((K, K), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            O[t, p] += 1.0
    w = (cls[:, None] - cls[None, :]) ** 2 / float((K - 1) ** 2) if K > 1 else np.zeros((K, K))
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / max(O.sum(), 1.0)
    denom = float((w * E).sum())
    qwk = float(1.0 - (w * O).sum() / denom) if denom > 0 else 0.0
    _, counts = np.unique(y_true, return_counts=True)
    maj_acc = float(counts.max() / counts.sum())
    return {
        "n_classes":         int(K),
        "test_accuracy":     acc,
        "test_within1_acc":  within1,
        "test_macro_f1":     macro_f1,
        "test_balanced_acc": bal_acc,
        "test_qwk":          qwk,
        "majority_acc":      maj_acc,
    }


# ── probe-head loss functions (config-selectable via --probe_loss) ──────────
# A BASE regressor (mse/mae/huber) anchors the output SCALE so RMSE stays
# meaningful; an AUXILIARY term (corr/ccc/rank/softrank) pushes the ranking
# metrics (Pearson r / Spearman ρ). Compose as "<base>+<aux>", e.g. mse+corr,
# mae+rank; --aux_weight scales the aux term (base terms are fixed at 1.0).
# A scale-free aux ALONE (corr/rank/softrank) lets predictions drift off the pK
# scale → RMSE becomes meaningless, so keep a base term unless you only want ρ.
# References: GenScore (Pearson aux, Chem.Sci.2023), MBP (pairwise RankNet,
# arXiv:2306.04886), S1+P hybrid (arXiv:2407.15202), CCC loss (arXiv:2003.10724),
# Blondel soft-rank (ICML2020, arXiv:2002.08871, torchsort).
_BASE_LOSSES = {"mse", "mae", "huber"}
_AUX_LOSSES  = {"corr", "ccc", "rank", "softrank", "gar"}


def _pearson_term(p, y):
    """1 − Pearson r over the batch (GenScore / S1+P Pearson auxiliary). Scale-free."""
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


def _ccc_term(p, y):
    """1 − Lin's concordance correlation. Penalises mean/variance mismatch too, so
    (unlike raw Pearson) it is scale/shift-AWARE and usable standalone."""
    cov = ((p - p.mean()) * (y - y.mean())).mean()
    ccc = 2 * cov / (p.var(unbiased=False) + y.var(unbiased=False)
                     + (p.mean() - y.mean()) ** 2 + 1e-8)
    return 1.0 - ccc


def _ranknet_term(p, y):
    """RankNet pairwise logistic ranking loss (MBP-style within-batch ordering). Scale-free."""
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    tgt = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    mask = (y.unsqueeze(1) != y.unsqueeze(0)).float()
    loss = F.binary_cross_entropy_with_logits(dp, tgt, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _gar_term(p, y):
    """GAR-style pairwise-difference alignment (Gradient-Aligned Regression, ICML 2025,
    arXiv:2402.06104): match predicted pairwise gaps to the true gaps in DIRECTION AND
    MAGNITUDE — MSE over all pairs (pred_i−pred_j vs y_i−y_j). Unlike RankNet (sign only)
    it also fits the size of each gap. Shift-invariant, scale-sensitive → pairs with a
    base term. (Simplified surrogate: pairwise-difference MSE, not the full gradient-surgery.)"""
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    return ((dp - dy) ** 2).mean()


def _softrank_term(p, y):
    """1 − Spearman ρ via Blondel et al. (2020) differentiable soft-rank (torchsort).
    Needs `pip install torchsort`; raises a clear error if unavailable. Scale-free."""
    try:
        import torchsort
    except ImportError as e:
        raise RuntimeError("probe_loss term 'softrank' needs torchsort "
                           "(pip install torchsort)") from e
    pr = torchsort.soft_rank(p.reshape(1, -1)).reshape(-1)
    yr = torchsort.soft_rank(y.reshape(1, -1)).reshape(-1)
    pr = pr - pr.mean(); yr = yr - yr.mean()
    return 1.0 - (pr * yr).sum() / (pr.norm() * yr.norm() + 1e-8)


_LOSS_TERMS = {
    "mse":      lambda p, y: F.mse_loss(p, y),
    "mae":      lambda p, y: F.l1_loss(p, y),
    "huber":    lambda p, y: F.smooth_l1_loss(p, y),
    "corr":     _pearson_term,
    "ccc":      _ccc_term,
    "rank":     _ranknet_term,
    "gar":      _gar_term,
    "softrank": _softrank_term,
}


def build_probe_loss(spec: str, aux_weight: float = 1.0):
    """Return loss_fn(pred, target) for a '+'-joined spec like 'mse+corr'.

    Base terms (mse/mae/huber) get weight 1.0; aux terms (corr/ccc/rank/softrank)
    get weight `aux_weight`. Terms are summed. 'mse' (default) reproduces the
    prior nn.MSELoss behaviour bit-for-bit.
    """
    terms = [t.strip() for t in spec.split("+") if t.strip()]
    if not terms:
        raise ValueError(f"empty probe_loss spec {spec!r}")
    unknown = [t for t in terms if t not in _LOSS_TERMS]
    if unknown:
        raise ValueError(f"unknown probe_loss term(s) {unknown}; valid: {sorted(_LOSS_TERMS)}")

    def loss_fn(pred, target):
        total = pred.new_zeros(())
        for t in terms:
            w = aux_weight if t in _AUX_LOSSES else 1.0
            total = total + w * _LOSS_TERMS[t](pred, target)
        return total

    loss_fn.terms = terms
    loss_fn.needs_pairs = any(t in _AUX_LOSSES for t in terms)
    return loss_fn


def train_one(
    data: dict, *, seed: int, device: str, max_epochs: int, patience: int,
    batch_size: int, lr: float, weight_decay: float, hidden: int, dropout: float,
    head: str = "scalar", soft_sigma: float = 1.0, num_classes: Optional[int] = None,
    probe_loss: str = "mse", aux_weight: float = 1.0,
) -> dict:
    """Train a single MLP probe; return metrics dict.

    head="scalar"  — 1-D output, MSE on the raw value (default; unchanged behaviour).
    head="softmax" — K-class output over counts 0..K-1 (K = max target + 1), trained
                     with soft-label cross-entropy against a Gaussian target of width
                     `soft_sigma`. Predictions and all metrics use the expected value
                     E[count] = Σ k·softmax (continuous, so Spearman ρ is not crushed by
                     argmax ties). K, ρ/Pearson/RMSE are otherwise reported identically.
    """
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr, ytr = torch.from_numpy(data["train"]["X"]), torch.from_numpy(data["train"]["y"])
    Xva, yva = torch.from_numpy(data["val"  ]["X"]), torch.from_numpy(data["val"  ]["y"])
    Xte, yte = torch.from_numpy(data["test" ]["X"]), torch.from_numpy(data["test" ]["y"])
    Xtr, ytr, Xva, yva, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xva, yva, Xte, yte))

    if head == "softmax":
        ymax = max(ytr.max().item(), yva.max().item(), yte.max().item())
        K = num_classes or (int(round(ymax)) + 1)
        karange = torch.arange(K, device=device, dtype=torch.float32)
        soft_tr = soft_count_targets(ytr, K, soft_sigma)            # (n_train, K)
        out_dim = K
    else:
        out_dim = 1

    model = MLP2(Xtr.shape[1], hidden=hidden, dropout=dropout, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    _reg_loss = build_probe_loss(probe_loss, aux_weight)     # scalar-head training loss
    # ranking/correlation terms need ≥2 distinct samples per batch; skip a tiny trailing
    # batch only when such a term is active (default mse keeps min=1 → behaviour unchanged).
    _min_bs = 4 if _reg_loss.needs_pairs else 1

    def predict(X):                                                 # → continuous scalar per sample
        if head == "softmax":
            return (torch.softmax(model(X), dim=1) * karange).sum(dim=1)
        return model(X)

    n_train = Xtr.shape[0]
    best_val = -np.inf
    best_state = None
    best_epoch = -1
    epochs_since_best = 0
    # classification head: track a SECOND best state by val macro-F1 (argmax), so the
    # classification metrics get a class-selected model while the regression metrics keep
    # their E[count]-Spearman selection (existing behaviour unchanged). Early stopping
    # stays on Spearman so the softmax regression numbers are bit-identical.
    do_clf = head == "softmax"
    yva_int = np.rint(yva.cpu().numpy()).astype(int) if do_clf else None
    best_val_f1 = -np.inf
    best_state_clf = None
    best_epoch_clf = -1

    for epoch in range(max_epochs):
        # mini-batch SGD
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            idx = perm[s : s + batch_size]
            if idx.numel() < _min_bs:
                continue
            opt.zero_grad()
            out = model(Xtr[idx])
            if head == "softmax":
                loss = -(soft_tr[idx] * torch.log_softmax(out, dim=1)).sum(dim=1).mean()
            else:
                loss = _reg_loss(out, ytr[idx])
            loss.backward()
            opt.step()

        # val Spearman (on E[count] for the softmax head)
        model.eval()
        with torch.no_grad():
            pred_va = predict(Xva).cpu().numpy()
        val_spearman = spearmanr(pred_va, yva.cpu().numpy()).statistic
        if do_clf:
            with torch.no_grad():
                pred_va_cls = model(Xva).argmax(dim=1).cpu().numpy()
            val_f1 = _macro_f1(yva_int, pred_va_cls)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state_clf = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch_clf = epoch
        if val_spearman > best_val:
            best_val = val_spearman
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = predict(Xva).cpu().numpy()
        pred_te = predict(Xte).cpu().numpy()
    yva_np = yva.cpu().numpy()
    yte_np = yte.cpu().numpy()

    out = {
        "n_train": int(n_train),
        "n_val":   int(Xva.shape[0]),
        "n_test":  int(Xte.shape[0]),
        "best_val_spearman": float(best_val),
        "val_pearson":       float(pearsonr (pred_va, yva_np).statistic),
        "test_spearman":     float(spearmanr(pred_te, yte_np).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte_np).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte_np) ** 2).mean())),
        "test_mae":          float(np.abs(pred_te - yte_np).mean()),
        "epoch_stopped":     int(best_epoch),
    }
    # per-sample test predictions (popped by run_probe before the metrics DataFrame is built;
    # feeds the optional single-seed scatter CSV). Regression preds = predict(Xte), aligned
    # with data["test"]["pid"].
    out["_pred_te"] = np.asarray(pred_te, dtype=float)
    out["_yte"]     = np.asarray(yte_np, dtype=float)
    if do_clf:
        # argmax classification metrics from the macro-F1-selected model (counts as classes)
        model.load_state_dict(best_state_clf or best_state)
        model.eval()
        with torch.no_grad():
            pred_te_cls = model(Xte).argmax(dim=1).cpu().numpy()
        out.update(classification_metrics(yte_np, pred_te_cls, K))
        out["best_val_macro_f1"] = float(best_val_f1)
        out["clf_epoch_stopped"] = int(best_epoch_clf)
    return out


def run_probe(args: argparse.Namespace) -> None:
    suffix = feature_suffix(args.voxel_version, tag=args.feature_tag)
    target_map = load_target_map(args.target)            # None for pK (LP_PDBBind default)
    split_flag = getattr(args, "split", "lp")
    split_map, split_scheme = (None, None)
    if split_flag in FROZEN_SPLIT_SCHEMES:
        split_map, split_scheme = load_frozen_split_map(split_flag)
    tag_parts = ([] if args.target == "pK" else [args.target])
    if split_map is not None:
        tag_parts.append(f"{split_flag}split")          # misatosplit / lp_edrsccsplit / timesplit
    if args.head == "softmax":                           # head/loss variant token, orthogonal to target
        tag_parts.append("softmax")
    if getattr(args, "probe_loss", "mse") != "mse":      # non-default loss → own CSV, never clobber MSE
        tag_parts.append(f"loss-{args.probe_loss.replace('+', '-')}-w{args.aux_weight:g}")
    if getattr(args, "leak_test", "none") == "append":   # diagnostic leak — never clobber a clean CSV
        tag_parts.append("leaktest")
    if args.tag:
        tag_parts.append(args.tag)
    eff_tag = clean_tag("_".join(tag_parts)) if tag_parts else None
    csv_suffix = results_suffix(                         # results-CSV suffix: + filtered split + run tag
        args.voxel_version, cl1_only=args.cl1_only, tag=eff_tag)
    out_csv = Path(args.out_csv) if args.out_csv else (
        RESULTS_DIR / f"probe_results_e{args.epoch}{csv_suffix}.csv"
    )

    print(f"=== PDBbind frozen-encoder probe (pocket repr only) ===")
    print(f"  conditions    : {args.conditions}")
    print(f"  target        : {TARGET_LABELS.get(args.target, args.target)}")
    print(f"  head          : {args.head}{f'  (soft-label CE, sigma={args.soft_sigma})' if args.head == 'softmax' else ''}")
    print(f"  epoch         : {args.epoch}")
    print(f"  voxel_version : {args.voxel_version}")
    print(f"  seeds         : {args.seeds}")
    print(f"  device        : {args.device}")
    print(f"  intersect     : {not args.no_intersect}")
    print(f"  drop_covalent : {not args.no_covalent_filter}")
    print(f"  cl1_only      : {args.cl1_only}{'  (-> _filtered)' if args.cl1_only else ''}")
    print(f"  split         : {split_flag}"
          + (f"  (frozen → {split_scheme}, hash-verified)" if split_scheme
             else "  (LP_PDBBind, recomputed locally)"))
    print(f"  tag           : {args.tag}")
    print(f"  feature_tag   : {args.feature_tag}")
    print(f"  exp_dir       : {args.exp_dir}")
    print(f"  cache_check   : {'warn only' if args.allow_stale_features else 'strict'}")
    print(f"  out_csv       : {'disabled' if args.no_write_csv else out_csv}")

    lp_df = load_lp_index(LP_CSV)
    print(f"  LP rows       : {len(lp_df):,}")
    if target_map is not None:
        print(f"  target labels : {len(target_map):,}  (from {target_source(args.target)})")

    # ── Load all feature bundles upfront so we can intersect pdb_ids ─────────
    all_feats: dict[str, dict[str, torch.Tensor]] = {}
    for cond in args.conditions:
        feat_path = FEAT_DIR / f"{cond}_e{args.epoch}{suffix}.pt"
        if not feat_path.exists():
            print(f"\n[error] missing features: {feat_path}")
            print(f"        Run: python dataset/01c_pdbbind_probe.py features "
                  f"--condition {cond} --voxel_version {args.voxel_version}"
                  f"{' --tag ' + args.feature_tag if args.feature_tag else ''}")
            sys.exit(1)
        bundle = torch.load(feat_path, weights_only=False)
        try:
            expected = expected_cache_signature(cond, args)
            issues = validate_feature_bundle(bundle, expected, feat_path)
        except Exception as e:
            issues = [f"{feat_path}: could not build expected metadata ({e!r})"]
        if issues:
            print(f"\n[error] stale or incompatible feature cache for {cond}:")
            for issue in issues[:16]:
                print(f"        - {issue}")
            if len(issues) > 16:
                print(f"        - ... {len(issues) - 16} more mismatch(es)")
            print(f"        Regenerate: python dataset/01c_pdbbind_probe.py features "
                  f"--condition {cond} --voxel_version {args.voxel_version} --epoch {args.epoch}"
                  f"{' --tag ' + args.feature_tag if args.feature_tag else ''}"
                  f"{' --exp_dir ' + args.exp_dir if args.exp_dir else ''}")
            if not args.allow_stale_features:
                sys.exit(1)
            print("        continuing because --allow_stale_features was set")
        all_feats[cond] = bundle["features"]
        print(f"  loaded {cond:24s}: {len(all_feats[cond]):,} feats (dim={bundle['feature_dim']})")

    # --require_density: drop pids lacking a v5 density crop from EVERY condition, so an
    # atom-only condition (coords) is evaluated on the SAME has_atoms&has_density complexes
    # as C+D+G — a fair matched comparison (mainly affects the raw `lp` split).
    if getattr(args, "require_density", False):
        _av = pd.read_csv(PDBBIND_DIR / "voxels_v5" / "availability.csv")
        _dens = {str(p).lower() for p in _av.loc[_av["has_density"].astype(bool), "pdb_id"]}
        for _c in list(all_feats):
            _kept = {p: v for p, v in all_feats[_c].items() if str(p).lower() in _dens}
            print(f"  [require_density] {_c}: {len(all_feats[_c]):,} -> {len(_kept):,} (has_density only)")
            all_feats[_c] = _kept

    if args.no_intersect:
        shared = None
    else:
        shared = set.intersection(*(set(f.keys()) for f in all_feats.values()))
        print(f"  shared pids   : {len(shared):,}  (intersection across conditions)")

    # ── wandb: log probe metrics to the binding-affinity project ─────────────
    wb = None
    if not getattr(args, "no_wandb", False):
        try:
            import wandb as _wandb
            _extra = [t for t in (getattr(args, "wandb_tags", None) or "").split(",") if t]
            _name = f"probe_{'+'.join(args.conditions)}_e{args.epoch}"
            if eff_tag:
                _name += f"_{eff_tag}"
            wb = _wandb.init(
                project=getattr(args, "wandb_project", "binding-affinity"),
                entity="eddy26",
                name=_name, job_type="probe", reinit=True,
                tags=["probe", f"target:{args.target}", f"split:{split_flag}",
                      f"epoch:{args.epoch}", *_extra],
                config=dict(
                    kind="frozen_probe", conditions=list(args.conditions),
                    target=args.target, split=split_flag, split_scheme=split_scheme,
                    epoch=args.epoch, voxel_version=args.voxel_version,
                    seeds=args.seeds, head=args.head, lr=args.lr,
                    weight_decay=args.weight_decay, hidden=args.hidden,
                    dropout=args.dropout, tag=args.tag, feature_tag=args.feature_tag,
                    exp_dir=args.exp_dir,
                    probe_loss=args.probe_loss, aux_weight=args.aux_weight,
                ),
            )
        except Exception as e:
            print(f"  [wandb] init failed ({e!r}); continuing without wandb")
            wb = None

    rows = []
    for cond in args.conditions:
        feats = all_feats[cond]
        if shared is not None:
            feats = {p: v for p, v in feats.items() if p in shared}

        print(f"\n── {cond} ──────────────────────────────────────────────────────")
        data = build_dataset(
            feats, lp_df,
            drop_covalent   = not args.no_covalent_filter,
            cl1_only        = args.cl1_only,
            target_map      = target_map,
            split_map       = split_map,
        )
        print(f"  split sizes   : train={len(data['train']['pid']):,}  "
              f"val={len(data['val']['pid']):,}  test={len(data['test']['pid']):,}")
        print(f"  input dim     : {data['train']['X'].shape[1]}")

        # --test_affinity: score the SAME combined-trained model on a Kd-only/Ki-only TEST
        # subset (train/val untouched). mtype from LP 'kd/ki' column (Kd|Ki|IC50).
        if args.test_affinity != "all":
            _mt = (lp_df["kd/ki"].astype(str)
                   .str.extract(r"(?i)^\s*(Kd|Ki|IC50)", expand=False).str.lower())
            _keep = {str(p).lower() for p, m in zip(lp_df["pdb_id"], _mt)
                     if m == args.test_affinity.lower()}
            te = data["test"]
            idx = [i for i, p in enumerate(te["pid"]) if str(p).lower() in _keep]
            te["X"], te["y"] = te["X"][idx], te["y"][idx]
            te["pid"] = [te["pid"][i] for i in idx]
            print(f"  [test_affinity={args.test_affinity}] test → {len(idx)} pids")

        # --leak_test append: DIAGNOSTIC upper bound. Fold the test features+labels into
        # train (test stays intact for eval). The head now sees test labels during fitting,
        # so test metrics report an optimistic ceiling, not a generalisation estimate.
        if getattr(args, "leak_test", "none") == "append":
            tr, te = data["train"], data["test"]
            n_before = len(tr["pid"])
            tr["X"] = np.concatenate([tr["X"], te["X"]], axis=0)
            tr["y"] = np.concatenate([tr["y"], te["y"]], axis=0)
            tr["pid"] = list(tr["pid"]) + list(te["pid"])
            print(f"  [leak_test=append] train {n_before:,} → {len(tr['pid']):,} "
                  f"(+{len(te['pid']):,} test folded in; test still eval'd on itself — "
                  f"OPTIMISTIC, not a real result)")

        for seed in range(args.seeds):
            m = train_one(
                data, seed=seed, device=args.device,
                max_epochs=args.max_epochs, patience=args.patience,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden, dropout=args.dropout,
                head=args.head, soft_sigma=args.soft_sigma,
                probe_loss=args.probe_loss, aux_weight=args.aux_weight,
            )
            _pred_te = m.pop("_pred_te", None)    # per-sample preds (kept out of the metrics DataFrame)
            _yte     = m.pop("_yte", None)
            row = {"condition": cond, "seed": seed, **m}
            if split_scheme is not None:                  # cross-server reproducibility stamp
                row["split_scheme"]    = split_scheme
                row["split_test_hash"] = data["test"].get("content_hash", "")
                row["n_test_eff"]      = len(data["test"]["pid"])
            rows.append(row)
            # single-seed scatter CSV (pred vs true pK) → focused meeting folder by default.
            if (not args.no_scatter) and _pred_te is not None and seed == args.scatter_seed:
                sdir = Path(args.scatter_dir); sdir.mkdir(parents=True, exist_ok=True)
                lbl    = args.tag or args.feature_tag or cond
                scheme = split_scheme or split_flag
                sname  = f"scatter_e{args.epoch}_{args.voxel_version}_{scheme}_{lbl}_seed{seed}.csv"
                sdf = pd.DataFrame({"pid": list(data["test"]["pid"]),
                                    "y_true": _yte, "y_pred": _pred_te})
                sdf.insert(0, "condition", cond); sdf.insert(1, "seed", seed)
                sdf.to_csv(sdir / sname, index=False)
                print(f"  [scatter] {sdir / sname}  ({len(sdf)} pts, seed {seed})")
            print(f"  seed={seed}  ep_stop={m['epoch_stopped']:3d}  "
                  f"val_ρ={m['best_val_spearman']:.4f}  "
                  f"val_r={m['val_pearson']:.4f}  "
                  f"test_ρ={m['test_spearman']:.4f}  "
                  f"test_r={m['test_pearson']:.4f}  "
                  f"test_rmse={m['test_rmse']:.4f}")
            if wb is not None:
                wb.log({f"{cond}/seed/test_spearman":    m["test_spearman"],
                        f"{cond}/seed/test_pearson":     m["test_pearson"],
                        f"{cond}/seed/test_rmse":        m["test_rmse"],
                        f"{cond}/seed/best_val_spearman":m["best_val_spearman"],
                        f"{cond}/seed/val_pearson":      m["val_pearson"],
                        f"{cond}/seed/epoch_stopped":    m["epoch_stopped"],
                        "seed": seed})
            if "test_accuracy" in m:
                print(f"           clf(K={m['n_classes']}): acc={m['test_accuracy']:.3f}  "
                      f"±1acc={m['test_within1_acc']:.3f}  macroF1={m['test_macro_f1']:.3f}  "
                      f"balAcc={m['test_balanced_acc']:.3f}  QWK={m['test_qwk']:.3f}  "
                      f"(majority={m['majority_acc']:.3f})")

    df = pd.DataFrame(rows)

    print("\n── Summary (mean ± std across seeds) ───────────────────────────────")
    metric_cols = ["test_spearman", "test_pearson", "test_rmse", "test_mae",
                   "best_val_spearman", "val_pearson"]
    if "test_accuracy" in df.columns:
        clf_cols = ["test_accuracy", "test_within1_acc", "test_macro_f1",
                    "test_balanced_acc", "test_qwk", "majority_acc"]
        agg_clf = df.groupby("condition")[clf_cols].agg(["mean", "std"]).round(4)
        print("  [classification — counts as classes]")
        print(agg_clf.to_string())
        print("  [regression — E[count]]")
    agg = df.groupby("condition")[metric_cols].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    if wb is not None:
        for cond in df["condition"].unique():
            sub = df[df["condition"] == cond]
            wb.summary.update({
                f"{cond}/test_spearman_mean": float(sub["test_spearman"].mean()),
                f"{cond}/test_spearman_std":  float(sub["test_spearman"].std()),
                f"{cond}/test_pearson_mean":  float(sub["test_pearson"].mean()),
                f"{cond}/test_rmse_mean":     float(sub["test_rmse"].mean()),
                f"{cond}/val_spearman_mean":  float(sub["best_val_spearman"].mean()),
            })
        wb.finish()
    if args.no_write_csv:
        print("\n[write] disabled (--no_write_csv)")
    else:
        df.to_csv(out_csv, index=False)
        print(f"\n[write] {out_csv}")

    # Auto-consolidation disabled: the per-run CSVs (now cleanly version-named) plus
    # the live dashboard are the source of truth; the consolidated table is no longer
    # maintained. Rebuild on demand with: python dataset/consolidate_probe_results.py


# ═══════════════════════════════════════════════════════════════════════════════
# finetune — end-to-end encoder fine-tuning vs frozen control (matched splits)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `probe` freezes the encoder and trains an MLP head on cached, mean-pooled token
# features. `finetune` answers a different question: how much does letting
# gradients flow *into the encoder* buy on PDBbind affinity?
#
# To keep the comparison honest, both arms run over the SAME pdb_id splits and the
# SAME MLP2 head (512 → hidden → 1) on the SAME mean-pooled token rep:
#
#   • mode="frozen"   — encoder EMA weights fixed; encode every complex ONCE
#                       (no_grad) → train the head on cached features. Identical in
#                       expectation to re-running the frozen encoder each epoch,
#                       just far cheaper, and reproduces the `probe` baseline.
#   • mode="finetune" — encoder + head trained end-to-end; voxels are re-encoded
#                       every step so the encoder receives gradient. A low encoder
#                       LR (default 1e-5) + early stopping regularize the 40M-param
#                       backbone against the ~2.7k-complex train set.
#
# The only thing that differs is whether the encoder learns; the Δ in test ρ is the
# value of fine-tuning.


class FineTuneModel(nn.Module):
    """Encoder (DensityViT) → token pool → MLP2 affinity head.

    `pool="mean"` reproduces the cached-probe pooling; `pool="attn"` swaps in a
    learned-query attention pool over the live token sequence (its params train
    alongside the head). Pool params live outside the `encoder.` namespace so the
    optimiser routes them at `head_lr`.
    """

    def __init__(self, encoder: DensityViT, hidden: int, dropout: float,
                 pool: str = "mean", pool_heads: int = 8):
        super().__init__()
        self.encoder = encoder
        self.pool = make_pool(pool, encoder.dim, n_heads=pool_heads, dropout=dropout)
        self.head = MLP2(encoder.dim, hidden=hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = forward_tokens(self.encoder, x)   # (B, N, D) — grad flows into encoder
        z = self.pool(z)                      # (B, D)
        return self.head(z)                   # (B,)


class VoxelDataset(Dataset):
    """Lazily build the (n_in, G, G, G) input + pK target for each complex.

    Voxels load from disk per __getitem__ (workers + OS page cache make repeat
    epochs cheap). The gradmag channel is derived on the fly inside
    `load_voxels_for`, matching training and frozen feature extraction exactly.
    """

    def __init__(self, pids, ys, condition, n_in, atom_dir, dens_dir,
                 input_mode: str, with_gradmag: bool):
        self.pids = list(pids)
        self.ys = np.asarray(ys, dtype=np.float32)
        self.condition = condition
        self.n_in = n_in
        self.atom_dir = atom_dir
        self.dens_dir = dens_dir
        self.input_mode = input_mode
        self.with_gradmag = with_gradmag

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int):
        x = load_voxels_for(self.pids[i], self.condition, self.n_in,
                            self.atom_dir, self.dens_dir,
                            input_mode=self.input_mode,
                            with_gradmag=self.with_gradmag)
        return x, torch.tensor(self.ys[i], dtype=torch.float32)


def make_encoder_factory(exp_dir: Path, epoch: int):
    """Load the EMA checkpoint ONCE; return a thunk minting fresh encoders.

    Each fine-tune seed needs an independent encoder initialised from the same
    pretrained weights, so we cache the stripped state_dict and re-instantiate
    (cheap) rather than re-reading the ~0.5 GB checkpoint per seed.
    """
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    m = cfg.model
    ckpt_path = exp_dir / f"checkpoint_e{epoch:04d}.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["encoder_state_dict_ema"]
    stripped = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}

    def factory() -> DensityViT:
        enc = DensityViT(
            grid_dim     = cfg.vox.grid_dim,
            patch_size   = m.patch_size,
            n_in_channels= m.n_in_channels,
            c_out        = m.n_channels // 2,
            dim          = m.dim,
            depth        = m.depth,
            n_heads      = m.heads,
            mlp_ratio    = m.mlp_ratio,
            dropout      = m.dropout,
            pos_encoding = m.get("pos_encoding", "learnable"),
            patch_embed_mode = m.get("patch_embed_mode",
                                     "channel_group" if m.get("channel_groups", None) else "fused"),
            channel_groups   = (tuple(m.channel_groups) if m.get("channel_groups", None) else None),
            n_memory_tokens  = int(m.get("n_memory_tokens", 0)),   # ChA-MAE encoders carry l memory tokens
            radius_embed     = _radius_embed_from_cfg(m),          # learnable radius→k atom embedding (opt-in)
        )
        enc.load_state_dict(stripped, strict=True)
        return enc

    print(f"  loaded EMA encoder weights from {ckpt_path.name} "
          f"(epoch={ckpt.get('epoch')}, {len(stripped)} tensors)")
    return factory, cfg


def split_exists_filter(splits: dict, atom_dir: Path, dens_dir: Path,
                        needs_atoms: bool, needs_density: bool) -> dict:
    """Drop pids whose voxel files are missing so both arms share one pid pool.

    Loud-warn policy: report (don't silently shrink) any pid dropped for a missing
    local voxel file, with the effective-set content hash, so two servers can tell
    whether they actually fine-tuned on the identical complexes.
    """
    from voxbind.splits import content_hash
    out = {}
    for split, sdf in splits.items():
        keep = [pid for pid in sdf["pdb_id"]
                if (not needs_atoms or (atom_dir / f"{pid}.npy").exists())
                and (not needs_density or (dens_dir / f"{pid}.npy").exists())]
        dropped = len(sdf) - len(keep)
        if dropped:
            print(f"    [split:{split}] WARNING: {dropped}/{len(sdf)} pids missing voxel files "
                  f"locally → using {len(keep)} (effective N). "
                  f"used.content_hash={content_hash(keep)[:12]}")
        out[split] = sdf[sdf["pdb_id"].isin(keep)].reset_index(drop=True)
    return out


def build_splits(condition: str, *, drop_covalent: bool, cl1_only: bool,
                 atom_dir: Path, dens_dir: Path, avail_csv: Path,
                 spec: FeatureSpec, split_map: Optional[dict[str, str]] = None) -> dict:
    """Train/val/test pdb_id+pK frames for one condition.

    Default (split_map=None): legacy LP_PDBBind new_split recomputed from this server's
    availability.csv pool (density conditions need has_atoms & has_density; atom-only
    need has_atoms) — mirrors single-condition `probe`, but CAN drift across servers.

    When `split_map` is given (a frozen scheme, pid -> split), membership comes from the
    committed manifest instead; pK is joined from LP and label-less pids dropped. Either
    way split_exists_filter loud-warns any pid whose voxel files are missing locally.
    """
    df = load_lp_index(LP_CSV)
    if split_map is not None:
        pk_map = {p: v for p, v in zip(df["pdb_id"], df["pK"]) if pd.notna(v)}
        splits = {}
        for s in ("train", "val", "test"):
            pids = sorted(p for p, sp in split_map.items() if sp == s and p in pk_map)
            splits[s] = pd.DataFrame({"pdb_id": pids, "pK": [pk_map[p] for p in pids]})
    else:
        avail = pd.read_csv(avail_csv)
        pool = (avail[avail["has_atoms"] & avail["has_density"]] if spec.needs_density
                else avail[avail["has_atoms"]])
        pid_set = set(pool["pdb_id"])
        d = df
        if drop_covalent:
            d = d[~d["covalent"].astype(bool)]
        if cl1_only:
            d = d[d["CL1"].astype(bool)]
        d = d[d["pdb_id"].isin(pid_set)]
        d = d[d["new_split"].isin(["train", "val", "test"])].dropna(subset=["pK"])
        splits = {s: d[d["new_split"] == s][["pdb_id", "pK"]].reset_index(drop=True)
                  for s in ("train", "val", "test")}
    return split_exists_filter(splits, atom_dir, dens_dir,
                               needs_atoms=spec.needs_atoms,
                               needs_density=spec.needs_density)


@torch.no_grad()
def encode_split_features(encoder, splits, condition, n_in, atom_dir, dens_dir,
                          device, batch_size, input_mode, with_gradmag):
    """Frozen-encoder mean-pooled features → {split: {X, y, pid}} for train_one."""
    encoder = encoder.to(device).eval()
    data = {}
    for split, sdf in splits.items():
        pk_map = dict(zip(sdf["pdb_id"], sdf["pK"].astype(np.float32)))
        pids = sdf["pdb_id"].tolist()
        feats, used = [], []
        for start in tqdm(range(0, len(pids), batch_size),
                          desc=f"encode {split}", unit="batch", leave=False):
            tensors, ok = [], []
            for pid in pids[start : start + batch_size]:
                try:
                    tensors.append(load_voxels_for(pid, condition, n_in,
                                                   atom_dir, dens_dir,
                                                   input_mode=input_mode,
                                                   with_gradmag=with_gradmag)); ok.append(pid)
                except Exception:
                    pass
            if not tensors:
                continue
            x = torch.stack(tensors).to(device)
            feats.append(forward_tokens(encoder, x).mean(dim=1).cpu())
            used += ok
        data[split] = {
            "X": torch.cat(feats).numpy().astype(np.float32),
            "y": np.array([pk_map[p] for p in used], dtype=np.float32),
            "pid": used,
        }
    return data


def encode_split_tokens(encoder, splits, condition, n_in, atom_dir, dens_dir,
                        device, batch_size, input_mode, with_gradmag, num_workers,
                        token_source: str = "patch"):
    """Frozen-encoder tokens, cached ONCE → {split: {tokens, y}}.

    Unlike `encode_split_features` (which mean-pools to a vector), this keeps the
    full (N, T, D) token sequence so a *learnable* pool can train on it across
    many epochs and seeds without ever re-reading voxels or re-running the
    encoder. Stored fp16 on CPU. Uses the worker DataLoader (shuffle=False, so
    tokens stay row-aligned with y).

    token_source="patch"  — post-norm patch tokens (B, Np, D); fed to mean/attn pools.
    token_source="memory" — post-norm memory tokens (B, l, D) of a ChA-MAE encoder;
                            mean-pooled by the head (memtok pool). Tiny cache (l≈4)."""
    encoder = encoder.to(device).eval()
    extract = (encoder.forward_features_memory if token_source == "memory"
               else (lambda xb: forward_tokens(encoder, xb)))
    out = {}
    for split, sdf in splits.items():
        ds = VoxelDataset(sdf["pdb_id"], sdf["pK"].to_numpy(), condition, n_in,
                          atom_dir, dens_dir, input_mode=input_mode,
                          with_gradmag=with_gradmag)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
        toks, ys = [], []
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=f"encode {split}", unit="batch", leave=False):
                xb = xb.to(device, non_blocking=True)
                toks.append(extract(xb).half().cpu())                   # (B, T, D)
                ys.append(yb)
        out[split] = {"tokens": torch.cat(toks),
                      "y": torch.cat(ys).numpy().astype(np.float32)}
    return out


def set_encoder_trainable(encoder: DensityViT, scope: str, last_k: int) -> list:
    """Flag which encoder params get gradient; return the trainable list.

    scope="full"  — patch_embed + pos_embed + all blocks + norm (decoder_proj is
                    unused downstream, so it stays frozen/excluded).
    scope="lastk" — only the last `last_k` transformer blocks + final norm; the
                    patch embed, positional embedding and earlier blocks freeze.
    scope="none"  — whole encoder frozen; only the downstream pool + head train
                    (isolates a head/pooling change from any encoder update).
    """
    for p in encoder.parameters():
        p.requires_grad_(False)
    if scope == "full":
        for n, p in encoder.named_parameters():
            if not n.startswith("decoder_proj"):
                p.requires_grad_(True)
    elif scope == "lastk":
        for p in encoder.norm.parameters():
            p.requires_grad_(True)
        for blk in encoder.blocks[len(encoder.blocks) - last_k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    elif scope == "none":
        pass  # whole encoder stays frozen; only the pool + head train
    else:
        raise ValueError(f"unknown ft scope: {scope!r}")
    return [p for n, p in encoder.named_parameters()
            if p.requires_grad and not n.startswith("decoder_proj")]


@torch.no_grad()
def infer_loader(model, loader, device, amp, amp_dtype) -> np.ndarray:
    """Batched no-grad predictions over a DataLoader → (N,) numpy."""
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            preds.append(model(xb).float().cpu())
    return torch.cat(preds).numpy()


def train_finetune(splits, factory, *, seed, device, condition, n_in,
                   atom_dir, dens_dir, max_epochs, patience, batch_size,
                   accum_steps, head_lr, encoder_lr, weight_decay, grad_clip,
                   hidden, dropout, pool, pool_heads, num_workers, amp, ft_scope, last_k,
                   input_mode, with_gradmag) -> dict:
    """Fine-tune encoder + head end-to-end; return metrics (best val ρ → test)."""
    torch.manual_seed(seed); np.random.seed(seed)

    encoder = factory()
    enc_params = set_encoder_trainable(encoder, ft_scope, last_k)
    model = FineTuneModel(encoder, hidden=hidden, dropout=dropout,
                          pool=pool, pool_heads=pool_heads).to(device)

    # head + pool params (everything outside the encoder namespace) at head_lr;
    # encoder params at encoder_lr — and only when ft_scope leaves some unfrozen.
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    if enc_params:
        groups.append({"params": enc_params, "lr": encoder_lr, "weight_decay": weight_decay})
    opt = torch.optim.AdamW(groups)
    loss_fn = nn.MSELoss()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def make_loader(split, shuffle):
        sdf = splits[split]
        ds = VoxelDataset(sdf["pdb_id"], sdf["pK"].to_numpy(), condition, n_in,
                          atom_dir, dens_dir, input_mode=input_mode,
                          with_gradmag=with_gradmag)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    train_loader = make_loader("train", True)
    val_loader   = make_loader("val", False)
    test_loader  = make_loader("test", False)
    yva = splits["val"]["pK"].to_numpy().astype(np.float32)
    yte = splits["test"]["pK"].to_numpy().astype(np.float32)

    best_val, best_state, best_epoch, since = -np.inf, None, -1, 0
    for epoch in range(max_epochs):
        model.train()
        if ft_scope == "none":
            model.encoder.eval()   # frozen encoder → deterministic features (no dropout)
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss = loss_fn(model(xb), yb) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        pred_va = infer_loader(model, val_loader, device, amp, amp_dtype)
        val_spearman = spearmanr(pred_va, yva).statistic
        if val_spearman > best_val:
            best_val = val_spearman
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    pred_va = infer_loader(model, val_loader, device, amp, amp_dtype)
    pred_te = infer_loader(model, test_loader, device, amp, amp_dtype)
    return {
        "n_train": len(splits["train"]),
        "n_val":   len(splits["val"]),
        "n_test":  len(splits["test"]),
        "best_val_spearman": float(best_val),
        "val_pearson":       float(pearsonr (pred_va, yva).statistic),
        "test_spearman":     float(spearmanr(pred_te, yte).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte) ** 2).mean())),
        "test_mae":          float(np.abs(pred_te - yte).mean()),
        "epoch_stopped":     int(best_epoch),
    }


class PoolHead(nn.Module):
    """Token pool + MLP2 affinity head, trained on cached frozen-encoder tokens."""
    def __init__(self, dim: int, hidden: int, dropout: float, pool: str, pool_heads: int):
        super().__init__()
        self.pool = make_pool(pool, dim, n_heads=pool_heads, dropout=dropout)
        self.head = MLP2(dim, hidden=hidden, dropout=dropout)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:   # (B, Np, D) → (B,)
        return self.head(self.pool(tokens))


def _pool_infer(model: nn.Module, T: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    """Batched no-grad predictions over cached tokens (T fp16 on CPU)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, T.shape[0], batch_size):
            xb = T[s : s + batch_size].to(device, non_blocking=True).float()
            preds.append(model(xb).cpu())
    return torch.cat(preds).numpy()


def train_pool(data: dict, *, seed: int, device: str, pool: str, pool_heads: int,
               max_epochs: int, patience: int, batch_size: int, lr: float,
               weight_decay: float, hidden: int, dropout: float) -> dict:
    """Train a pool+head on cached frozen tokens (no encoder, no disk) → metrics.

    The token cache is shared across seeds; each seed gets a fresh head — exactly
    the `frozen`-baseline protocol, but over tokens so the pool can be learned."""
    torch.manual_seed(seed); np.random.seed(seed)
    Ttr = data["train"]["tokens"]; ytr = torch.from_numpy(data["train"]["y"])
    Tva, yva = data["val"]["tokens"],  data["val"]["y"]
    Tte, yte = data["test"]["tokens"], data["test"]["y"]
    dim = Ttr.shape[-1]
    model = PoolHead(dim, hidden, dropout, pool, pool_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_train = Ttr.shape[0]
    best_val, best_state, best_epoch, since = -np.inf, None, -1, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch_size):
            idx = perm[s : s + batch_size]
            xb = Ttr[idx].to(device, non_blocking=True).float()
            yb = ytr[idx].to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        pred_va = _pool_infer(model, Tva, device, batch_size)
        val_spearman = spearmanr(pred_va, yva).statistic
        if val_spearman > best_val:
            best_val = val_spearman
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; since = 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    pred_va = _pool_infer(model, Tva, device, batch_size)
    pred_te = _pool_infer(model, Tte, device, batch_size)
    return {
        "n_train": int(n_train),
        "n_val":   int(Tva.shape[0]),
        "n_test":  int(Tte.shape[0]),
        "best_val_spearman": float(best_val),
        "val_pearson":       float(pearsonr (pred_va, yva).statistic),
        "test_spearman":     float(spearmanr(pred_te, yte).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte) ** 2).mean())),
        "test_mae":          float(np.abs(pred_te - yte).mean()),
        "epoch_stopped":     int(best_epoch),
    }


def run_finetune(args: argparse.Namespace) -> None:
    device = args.device
    amp = (not args.no_amp) and str(device).startswith("cuda")
    exp_dir = Path(args.exp_dir) if getattr(args, "exp_dir", None) else \
        resolve_exp(args.condition, args.voxel_version)
    factory, cfg = make_encoder_factory(exp_dir, args.epoch)
    spec = infer_feature_spec(args.condition, cfg, args.atom_source)

    # memtok pool reads the encoder's learnable memory tokens (ChA-MAE); validate up front.
    n_mem = int(cfg.model.get("n_memory_tokens", 0))
    if args.pool == "memtok" and n_mem == 0:
        raise SystemExit(f"--pool memtok requires a ChA-MAE encoder (n_memory_tokens>0); "
                         f"{exp_dir.name} has none.")
    if args.pool == "memtok" and args.ft_scope != "none":
        raise SystemExit("--pool memtok is only supported with --ft_scope none "
                         "(frozen encoder, cached memory tokens → mean readout).")

    # Distribution-shift warning: density-consuming encoder fed an un-retrained
    # density version (same rationale as run_features).
    has_override = (args.condition, args.voxel_version) in EXPS_OVERRIDE
    if spec.needs_density and args.voxel_version != "v1" and not has_override:
        print(f"  [warn] {args.condition} encoder pretrained on v1 density; "
              f"--voxel_version {args.voxel_version} is OOD.")

    vox_dir  = voxel_dir_for(args.voxel_version)
    atom_dir = atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = PDBBIND_DIR / "voxels" / "availability.csv"

    csv_suffix = results_suffix(
        args.voxel_version, cl1_only=args.cl1_only, tag=args.tag)
    out_csv = Path(args.out_csv) if args.out_csv else (
        RESULTS_DIR / f"finetune_results_e{args.epoch}{csv_suffix}.csv")

    print(f"=== PDBbind frozen-vs-finetune ({args.condition}) ===")
    print(f"  modes          : {args.modes}")
    print(f"  epoch          : {args.epoch}")
    print(f"  voxel_version  : {args.voxel_version}")
    print(f"  exp_dir        : {exp_dir}")
    print(f"  input_mode     : {spec.input_mode}")
    print(f"  with_gradmag   : {spec.with_gradmag}")
    print(f"  ligand_radius  : {spec.ligand_radius}")
    print(f"  atom_source    : {spec.atom_source}")
    print(f"  atom_dir       : {atom_dir}")
    print(f"  dens_dir       : {dens_dir}")
    print(f"  seeds          : {args.seeds}")
    print(f"  device         : {device}  (amp={amp})")
    print(f"  ft_scope       : {args.ft_scope}"
          + (f" (last_k={args.last_k})" if args.ft_scope == "lastk" else ""))
    print(f"  pool           : {args.pool}"
          + (f" (heads={args.pool_heads})" if args.pool == "attn" else ""))
    print(f"  enc_lr/head_lr : {args.encoder_lr} / {args.head_lr}")
    print(f"  bsz x accum    : {args.batch_size} x {args.accum_steps}")
    print(f"  ft max_ep/pat  : {args.max_epochs} / {args.patience}")
    print(f"  out_csv        : {out_csv}")

    n_in = cfg.model.n_in_channels
    if n_in != spec.expected_channels:
        raise RuntimeError(
            f"cfg-derived layout builds {spec.expected_channels} channels "
            f"(input_mode={spec.input_mode}, with_gradmag={spec.with_gradmag}) "
            f"but cfg.model.n_in_channels={n_in}"
        )

    ft_split_flag = getattr(args, "split", "lp")
    ft_split_map, ft_scheme = (None, None)
    if ft_split_flag in FROZEN_SPLIT_SCHEMES:
        ft_split_map, ft_scheme = load_frozen_split_map(ft_split_flag)
    print(f"  split          : {ft_split_flag}"
          + (f"  (frozen → {ft_scheme}, hash-verified)" if ft_scheme
             else "  (LP_PDBBind, recomputed locally)"))
    splits = build_splits(args.condition,
                          drop_covalent=not args.no_covalent_filter,
                          cl1_only=args.cl1_only, atom_dir=atom_dir,
                          dens_dir=dens_dir, avail_csv=avail_csv,
                          spec=spec, split_map=ft_split_map)
    if args.max_complexes:
        splits = {s: sdf.head(args.max_complexes) for s, sdf in splits.items()}
        print(f"  [smoke] capped each split to {args.max_complexes} complexes")
    print(f"  split sizes    : train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")

    rows = []

    if "frozen" in args.modes:
        print("\n── frozen control (encode once → MLP head) ─────────────────────")
        enc = factory()
        data = encode_split_features(enc, splits, args.condition, n_in,
                                     atom_dir, dens_dir, device, args.feat_batch_size,
                                     spec.input_mode, spec.with_gradmag)
        del enc
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        for seed in range(args.seeds):
            m = train_one(data, seed=seed, device=device,
                          max_epochs=args.frozen_max_epochs,
                          patience=args.frozen_patience,
                          batch_size=args.frozen_batch_size, lr=args.head_lr,
                          weight_decay=args.weight_decay, hidden=args.hidden,
                          dropout=args.dropout)
            rows.append({"condition": args.condition, "mode": "frozen",
                         "ft_scope": "-", "pool": "mean", "seed": seed,
                         "encoder_lr": 0.0, "head_lr": args.head_lr, **m})
            print(f"  [frozen]   seed={seed} ep_stop={m['epoch_stopped']:3d} "
                  f"val_ρ={m['best_val_spearman']:.4f} "
                  f"test_ρ={m['test_spearman']:.4f} test_r={m['test_pearson']:.4f} "
                  f"rmse={m['test_rmse']:.4f}")

    if "finetune" in args.modes and args.ft_scope == "none":
        # Frozen encoder → encode tokens ONCE (shared across all seeds), then train
        # the pool+head on the cache. Skips re-reading voxels / re-running the
        # encoder every epoch & seed (the dominant cost), so a pooling A/B that
        # would take hours via the re-encode path finishes in minutes. Uses the
        # frozen-arm schedule so pool=mean here ≈ the cached `frozen` baseline.
        print("\n── finetune (frozen encoder, cached tokens → pool+head) ────────")
        # memtok: cache the encoder's l memory tokens and mean-read them (CLS-like
        # aggregate). mean/attn: cache the full patch-token sequence and pool it.
        token_source = "memory" if args.pool == "memtok" else "patch"
        head_pool    = "mean"   if args.pool == "memtok" else args.pool
        enc = factory()
        tok = encode_split_tokens(enc, splits, args.condition, n_in,
                                  atom_dir, dens_dir, device, args.feat_batch_size,
                                  spec.input_mode, spec.with_gradmag, args.num_workers,
                                  token_source=token_source)
        del enc
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        ntok = tok["train"]["tokens"].shape[1]
        print(f"  cached {ntok}-token reps ({token_source}, pool={args.pool}→{head_pool})  "
              f"(train={tok['train']['tokens'].shape[0]} "
              f"val={tok['val']['tokens'].shape[0]} "
              f"test={tok['test']['tokens'].shape[0]})")
        for seed in range(args.seeds):
            m = train_pool(tok, seed=seed, device=device,
                           pool=head_pool, pool_heads=args.pool_heads,
                           max_epochs=args.frozen_max_epochs, patience=args.frozen_patience,
                           batch_size=args.frozen_batch_size, lr=args.head_lr,
                           weight_decay=args.weight_decay,
                           hidden=args.hidden, dropout=args.dropout)
            rows.append({"condition": args.condition, "mode": "finetune",
                         "ft_scope": args.ft_scope, "pool": args.pool, "seed": seed,
                         "encoder_lr": 0.0, "head_lr": args.head_lr, **m})
            print(f"  [pool:{args.pool}] seed={seed} ep_stop={m['epoch_stopped']:3d} "
                  f"val_ρ={m['best_val_spearman']:.4f} "
                  f"test_ρ={m['test_spearman']:.4f} test_r={m['test_pearson']:.4f} "
                  f"rmse={m['test_rmse']:.4f}")

    elif "finetune" in args.modes:
        print("\n── finetune (encoder + head end-to-end) ────────────────────────")
        for seed in range(args.seeds):
            m = train_finetune(splits, factory, seed=seed, device=device,
                               condition=args.condition, n_in=n_in,
                               atom_dir=atom_dir, dens_dir=dens_dir,
                               max_epochs=args.max_epochs, patience=args.patience,
                               batch_size=args.batch_size, accum_steps=args.accum_steps,
                               head_lr=args.head_lr, encoder_lr=args.encoder_lr,
                               weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                               hidden=args.hidden, dropout=args.dropout,
                               pool=args.pool, pool_heads=args.pool_heads,
                               num_workers=args.num_workers, amp=amp,
                               ft_scope=args.ft_scope, last_k=args.last_k,
                               input_mode=spec.input_mode,
                               with_gradmag=spec.with_gradmag)
            rows.append({"condition": args.condition, "mode": "finetune",
                         "ft_scope": args.ft_scope, "pool": args.pool, "seed": seed,
                         "encoder_lr": args.encoder_lr, "head_lr": args.head_lr, **m})
            print(f"  [finetune] seed={seed} ep_stop={m['epoch_stopped']:3d} "
                  f"val_ρ={m['best_val_spearman']:.4f} "
                  f"test_ρ={m['test_spearman']:.4f} test_r={m['test_pearson']:.4f} "
                  f"rmse={m['test_rmse']:.4f}")

    df = pd.DataFrame(rows)
    df.insert(0, "voxel_version", args.voxel_version)
    df.insert(0, "epoch", args.epoch)
    df.to_csv(out_csv, index=False)

    print("\n── frozen vs finetune (mean ± std across seeds) ────────────────────")
    agg = df.groupby("mode")[["test_spearman", "test_pearson", "test_rmse",
                              "best_val_spearman"]].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    if {"frozen", "finetune"} <= set(df["mode"]):
        fz, ft = df[df["mode"] == "frozen"], df[df["mode"] == "finetune"]
        print(f"\n  Δ finetune − frozen :  "
              f"test_ρ {ft['test_spearman'].mean() - fz['test_spearman'].mean():+.4f}   "
              f"test_r {ft['test_pearson'].mean() - fz['test_pearson'].mean():+.4f}")
    print(f"\n[write] {out_csv}")


# ── CLI ────────────────────────────────────────────────────────────────────────

_CONDITION_CHOICES = ["atomblob", "atomblob_ligvdw", "atomblob_density", "atomblob_weighted",
                      "atomblob_merged_density", "atomblob_merged_density_gradmag",
                      "atomblob_density_gradmag", "density_gradmag",
                      "roleblob_density_gradmag", "roleblob_density_gradmag_channelvit"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDBbind frozen-encoder features + MLP affinity probe (features | probe)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser(
        "features",
        help="Extract frozen-encoder features for PDBbind refined complexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pf.add_argument("--condition", choices=_CONDITION_CHOICES, required=True)
    pf.add_argument("--epoch",      type=int, default=99,
                    help="Checkpoint epoch to use (matched across conditions)")
    pf.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v5",
                    help="Density-normalisation version: v1 = per-map z-score + "
                         "per-crop ±3σ clip; v2 = pocket-pool z-score; "
                         "v3 = pocket-pool symmetric max-abs; v4 = pocket-pool "
                         "clip + z-score; v5 = arcsinh + z-score. Atom source "
                         "is selected by --atom_source / pretrained cfg.")
    pf.add_argument("--batch_size", type=int, default=48,
                    help="Frozen-encoder forward batch. Result-neutral (LayerNorm-only "
                         "encoder + per-sample mean-pool → identical features at any batch); "
                         "purely a throughput/VRAM knob. Raise freely if VRAM allows.")
    pf.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader workers for voxel assembly (np.load + cat + gradmag). "
                         "Overlaps disk I/O with the GPU forward; 0 = synchronous.")
    pf.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    pf.add_argument("--out_dir",    default=str(FEAT_DIR))
    pf.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N (0 = all). Smoke testing.")
    pf.add_argument("--atom_source", choices=["auto", "default", "ligvdw"], default="auto",
                    help="PDBbind atom voxel source. auto follows cfg.dset.ligand_radius "
                         "(<=0 uses voxels_ligvdw/atoms); default uses the version "
                         "dir's atoms symlink; ligvdw forces voxels_ligvdw/atoms.")
    pf.add_argument("--exp_dir", default=None,
                    help="Override the encoder exp dir (default: resolve from "
                         "condition+voxel_version via EXPS/EXPS_OVERRIDE). Lets "
                         "multiple same-condition runs be probed against their own "
                         "checkpoints in an auto-research loop.")
    pf.add_argument("--tag", default=None,
                    help="Optional feature-cache label appended after the version token, "
                         "e.g. --tag 260608_w1 -> atomblob_density_gradmag_e99_v5_260608_w1.pt. "
                         "Use with --exp_dir when comparing same-condition checkpoints.")
    pf.add_argument("--noise_voxels_dir", default=None,
                    help="Density-ablation control: read density+gradmag from this dir's "
                         "density/ and gradmag/ subdirs (matched noise) instead of the real "
                         "voxels. Atoms still come from the real atom_dir.")
    pf.add_argument("--diff_voxel_dir", default=None,
                    help="mFo-DFc difference-map crop dir (contains {pid}.npy). Only used "
                         "when the encoder cfg has mae.with_diff=true (15-ch [7,4,2,2]). "
                         "Default: voxels_v5_diff/density next to the density voxel dir.")
    pf.add_argument("--atom_dir", default=None,
                    help="Override the atom-voxel dir (contains {pid}.npy). Use for encoders "
                         "with a non-default ligand layout, e.g. voxels_atomblob8/atoms for "
                         "atomblob8 (8 lig + 4 poc = 12 atom ch). Default: version's atoms.")
    pf.add_argument("--dens_dir", default=None,
                    help="Override just the density-crop dir (gradmag still derived on-the-fly). "
                         "Use to feed RAW un-normalized density (voxels_v5_raw/density) to a "
                         "raw-density-trained encoder.")
    pf.set_defaults(func=run_features)

    pr = sub.add_parser(
        "probe",
        help="2-layer MLP probe on frozen-encoder pocket features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pr.add_argument("--conditions", nargs="+",
                    default=_CONDITION_CHOICES, choices=_CONDITION_CHOICES)
    pr.add_argument("--epoch",         type=int,   default=99)
    pr.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v5",
                    help="Selects which density-normalisation variant's features "
                         "to probe. Adds matching suffix to feature paths + output CSV. "
                         "Default v5 = the canonical split (2172/480/839); matches `finetune`.")
    pr.add_argument("--target", choices=["pK", "hbonds", "bfactor", "bfactor_rel",
                                         "partial_charges", "electron_affinity", "hardness", "adaptability", "rmsf",
                                         "pose_stability", "interaction_energy"], default="pK",
                    help="Regression target on the identical LP_PDBBind splits: pK (binding "
                         "affinity, default); hbonds (protein-ligand H-bond count, HBGSA cache); "
                         "bfactor (raw mean pocket B-factor); bfactor_rel (pocket/protein mean B, "
                         "resolution-robust). A non-pK target adds a _<target> token to the CSV name "
                         "and needs its cache built (dataset/extract_bfactors.py for B-factors); "
                         "complexes lacking a label are dropped.")
    pr.add_argument("--split", choices=["lp", "lp_edrscc", "lp_edrscc_v1", "lp_edrscc_v2", "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123", "time", "misato", "clean_ed", "clean_ed_v1", "clean_ed_v1_indep", "lba30", "lba60"], default="lp",
                    help="Split source. lp = LP_PDBBind new_split recomputed locally (legacy; can drift "
                         "across servers). FROZEN, hash-verified manifests in voxbind/splits/ (identical "
                         "on every server): lp_edrscc (canonical = Kd/Ki-only v2, 3850/817/1320; "
                         "lp_edrscc_v1 = legacy IC50-inclusive 5817/1498/2813), "
                         "time (temporal holdout 8308/934/1182), misato (official 8:1:1 MD split). Frozen "
                         "schemes add a _<flag>split CSV token + split_test_hash/n_test_eff stamp.")
    pr.add_argument("--head",          choices=["scalar", "softmax"], default="scalar",
                    help="Probe head/objective: scalar (1-D output, MSE; default) or softmax "
                         "(K-class output over integer counts 0..max, soft-label cross-entropy with "
                         "Gaussian targets; metrics use E[count]=Σk·softmax). Intended for count "
                         "targets like hbonds; adds a _softmax token to the CSV name.")
    pr.add_argument("--soft_sigma",    type=float, default=1.0,
                    help="Gaussian width (in count units) for softmax-head soft labels; "
                         "smaller → closer to one-hot. Only used when --head softmax.")
    pr.add_argument("--probe_loss",    default="mse",
                    help="Scalar-head training loss as '+'-joined terms. BASE (scale-anchoring): "
                         "mse|mae|huber. AUX (ranking, ×--aux_weight): corr(1-Pearson)|ccc|"
                         "rank(RankNet)|softrank(soft-Spearman,needs torchsort). e.g. mse+corr, "
                         "mse+rank, mae+ccc. Default 'mse' = prior behaviour. Early stop stays on "
                         "val ρ. Non-default adds a _loss-… token to the CSV name. (Ignored for --head softmax.)")
    pr.add_argument("--aux_weight",    type=float, default=1.0,
                    help="Weight on the auxiliary ranking/correlation term(s) in --probe_loss "
                         "(base terms fixed at 1.0). Try 1–3; too high degrades RMSE.")
    pr.add_argument("--seeds",         type=int,   default=3)
    pr.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    pr.add_argument("--hidden",        type=int,   default=128)
    pr.add_argument("--dropout",       type=float, default=0.1)
    pr.add_argument("--lr",            type=float, default=1e-3)
    pr.add_argument("--weight_decay",  type=float, default=1e-4)
    pr.add_argument("--batch_size",    type=int,   default=64)
    pr.add_argument("--max_epochs",    type=int,   default=200)
    pr.add_argument("--patience",      type=int,   default=30)
    pr.add_argument("--no_intersect",  action="store_true",
                    help="Let each condition use its own pdb_id pool")
    pr.add_argument("--require_density", action="store_true",
                    help="Restrict EVERY condition's eval pool to has_density complexes "
                         "(voxels_v5/availability.csv), so an atom-only condition (coords) is scored on "
                         "the SAME has_atoms&has_density set as C+D+G — fair matched comparison.")
    pr.add_argument("--test_affinity", choices=["all", "Kd", "Ki"], default="all",
                    help="Stratify the TEST set by affinity measurement type (Kd|Ki from LP 'kd/ki'); "
                         "train/val untouched, so it's the SAME combined-trained, val-selected model scored "
                         "on the Kd-only or Ki-only test subset. 'all' = no filter (default).")
    pr.add_argument("--leak_test", choices=["none", "append"], default="none",
                    help="DIAGNOSTIC (not a real result): 'append' folds the TEST features+labels "
                         "INTO the train set, then still evaluates on that same test set. Val-based "
                         "early stopping is untouched. Reveals the optimistic ceiling — how much the "
                         "frozen encoder + tiny MLP can memorise/fit test when its labels are seen. "
                         "Adds a _leaktest token to the CSV name so it never overwrites a clean result.")
    pr.add_argument("--no_covalent_filter", action="store_true",
                    help="Keep covalent complexes (default: drop)")
    pr.add_argument("--cl1_only",      action="store_true",
                    help="Restrict to LP_PDBBind CL1=True (cleanest subset); adds _filtered to the CSV name")
    pr.add_argument("--tag",           default=None,
                    help="Optional run label appended after the version/_filtered token, e.g. "
                         "--voxel_version v5 --cl1_only --tag invfreq -> probe_results_e99_v5_filtered_invfreq.csv")
    pr.add_argument("--feature_tag",   default=None,
                    help="Optional feature-cache label to load, matching features --tag. "
                         "Result --tag is intentionally separate so old untagged feature "
                         "caches remain loadable.")
    pr.add_argument("--exp_dir",       default=None,
                    help="Override expected encoder exp dir for strict feature validation. "
                         "Use with one --conditions value when probing a custom same-condition checkpoint.")
    pr.add_argument("--allow_stale_features", action="store_true",
                    help="Warn instead of failing when feature-cache metadata does not "
                         "match the requested condition/epoch/voxel_version/atom source. "
                         "Use only for intentional legacy or custom-exp probes.")
    pr.add_argument("--out_csv",       default=None,
                    help="Override results CSV path")
    pr.add_argument("--no_write_csv",  action="store_true",
                    help="Print seed/summary metrics but do not write a results CSV.")
    pr.add_argument("--scatter_dir",   default=str(MEETING_DIR),
                    help="Directory for the per-sample SINGLE-SEED scatter CSV (pid,y_true,y_pred). "
                         "Defaults under the focused meeting doc (notebook/html/260625/), next to "
                         "260625_meeting.html. Use --no_scatter to skip.")
    pr.add_argument("--scatter_seed",  type=int, default=0,
                    help="Which seed's held-out test predictions to dump for the scatter CSV (default 0).")
    pr.add_argument("--no_scatter",    action="store_true",
                    help="Do not write the single-seed scatter CSV.")
    pr.add_argument("--no_wandb",      action="store_true",
                    help="Disable wandb logging (default: log to the binding-affinity project).")
    pr.add_argument("--wandb_project", default="binding-affinity",
                    help="wandb project for probe metrics (default: binding-affinity).")
    pr.add_argument("--wandb_tags",    default=None,
                    help="Comma-separated extra wandb tags for this probe run.")
    pr.set_defaults(func=run_probe)

    pt = sub.add_parser(
        "finetune",
        help="Frozen-encoder control vs end-to-end encoder fine-tuning on PDBbind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pt.add_argument("--condition", choices=_CONDITION_CHOICES,
                    default="atomblob_density_gradmag")
    pt.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v5",
                    help="Density-normalisation variant; picks the matching encoder "
                         "(EXPS_OVERRIDE) + voxel dir. Default v5 (separate gradmag/ligvdw).")
    pt.add_argument("--atom_source", choices=["auto", "default", "ligvdw"], default="auto",
                    help="PDBbind atom voxel source. auto follows cfg.dset.ligand_radius "
                         "(<=0 uses voxels_ligvdw/atoms); default uses the version "
                         "dir's atoms symlink; ligvdw forces voxels_ligvdw/atoms.")
    pt.add_argument("--epoch",   type=int, default=99, help="Encoder checkpoint epoch")
    pt.add_argument("--modes",   nargs="+", default=["frozen", "finetune"],
                    choices=["frozen", "finetune"],
                    help="Which arms to run. Both → matched frozen-vs-finetune comparison.")
    pt.add_argument("--seeds",   type=int, default=3)
    pt.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    # shared head architecture (identical across arms)
    pt.add_argument("--hidden",  type=int,   default=128)
    pt.add_argument("--dropout", type=float, default=0.1)
    pt.add_argument("--pool", choices=["mean", "attn", "memtok"], default="mean",
                    help="Token→vector pooling for the head. mean = fixed average "
                         "(matches the cached probe); attn = learned-query attention "
                         "pool over the live token sequence (trains with the head); "
                         "memtok = mean of the encoder's learnable memory tokens "
                         "(ChA-MAE only; needs --ft_scope none).")
    pt.add_argument("--pool_heads", type=int, default=8,
                    help="Heads in the attention pool (only used when --pool attn)")
    pt.add_argument("--exp_dir", default=None,
                    help="Override the encoder exp dir (default: resolve from "
                         "condition+voxel_version via EXPS/EXPS_OVERRIDE). Needed to "
                         "point at a specific pretrain (e.g. a mask050 ChA-MAE run).")
    # fine-tune scope + optimisation
    pt.add_argument("--ft_scope", choices=["full", "lastk", "none"], default="full",
                    help="full = whole encoder (low LR); lastk = only last --last_k "
                         "blocks + norm; none = freeze encoder, train pool+head only")
    pt.add_argument("--last_k",   type=int,   default=4, help="blocks to unfreeze when ft_scope=lastk")
    pt.add_argument("--head_lr",     type=float, default=1e-3, help="MLP head LR (both arms)")
    pt.add_argument("--encoder_lr",  type=float, default=1e-5, help="encoder LR (finetune arm)")
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--grad_clip",   type=float, default=1.0, help="0 disables grad clipping")
    pt.add_argument("--batch_size",  type=int,   default=16, help="finetune mini-batch")
    pt.add_argument("--accum_steps", type=int,   default=1,  help="grad-accum (effective bsz = bsz*accum)")
    pt.add_argument("--max_epochs",  type=int,   default=60, help="finetune epoch cap")
    pt.add_argument("--patience",    type=int,   default=15, help="finetune early-stop patience")
    pt.add_argument("--num_workers", type=int,   default=4)
    pt.add_argument("--no_amp",      action="store_true", help="disable autocast (bf16/fp16) mixed precision")
    # frozen-arm head schedule (reproduces the `probe` baseline)
    pt.add_argument("--frozen_max_epochs", type=int, default=200)
    pt.add_argument("--frozen_patience",   type=int, default=30)
    pt.add_argument("--frozen_batch_size", type=int, default=64)
    pt.add_argument("--feat_batch_size",   type=int, default=32, help="frozen-arm encode batch")
    pt.add_argument("--no_covalent_filter", action="store_true",
                    help="Keep covalent complexes (default: drop)")
    pt.add_argument("--cl1_only", action="store_true",
                    help="Restrict to LP_PDBBind CL1=True (cleanest subset); adds _filtered to the CSV name")
    pt.add_argument("--split", choices=["lp", "lp_edrscc", "lp_edrscc_v1", "lp_edrscc_v2", "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123", "time", "misato", "clean_ed", "clean_ed_v1", "clean_ed_v1_indep", "lba30", "lba60"], default="lp",
                    help="Split source (see `probe --split`). Frozen schemes (lp_edrscc/time/misato) load "
                         "the hash-verified manifest in voxbind/splits/ so both arms AND every server use "
                         "the identical pids; lp = legacy local recompute. Pair with --tag to name the CSV.")
    pt.add_argument("--tag", default=None,
                    help="Optional run label appended after the version/_filtered token (see `probe --tag`)")
    pt.add_argument("--max_complexes", type=int, default=0,
                    help="Head each split to N complexes (0 = all). Smoke testing.")
    pt.add_argument("--out_csv",  default=None, help="Override results CSV path")
    pt.set_defaults(func=run_finetune)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
