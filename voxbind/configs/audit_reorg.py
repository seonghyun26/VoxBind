"""audit_reorg.py — careful re-audit of the config reorg.

(1) Compose EVERY config in configs/ (catch any break after edits/archival).
(2) For each migrated preset, show the RAW (un-exempted) diff vs its old monolith, classify each
    difference, and FLAG anything that is not provably benign. This is the check that the gate's
    exemptions only hide non-behavioral differences — nothing real.
"""
import glob
import os
import re

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CD = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("config_train_atomblob_density_gradmag_vit_mae_40m_invfreq", "atomblob_density_gradmag_invfreq"),
    ("config_train_atomblob_density_gradmag_cha_mae_40m", "cha_gradmag"),
    ("config_train_atomblob_vit_mae_40m_invfreq", "coords_invfreq"),
    ("config_train_atomblob_density_gradmag_vit_mae_40m_invfreq_plinder_otf_p100", "plinder_otf_p100"),
]
RUNTIME = {"amp", "compile", "optimizer", "ema"}
RUNTIME_LEAF = {"channels_last", "num_workers", "prefetch_factor"}
PATH_LEAF = {"data_dir", "crops_dir", "ccp4_dir", "resample_dir", "pdb_dir", "out_root", "out_dir"}
INERT = {"density_channel_weight", "density_weight_warmup_epochs", "gradmag_channel_weight",
         "gradmag_noise", "gradmag_reconstruct"}


def raw(name, ov=None):
    with initialize_config_dir(config_dir=CD, version_base=None):
        cfg = compose(config_name=name, overrides=ov or [])
    d = OmegaConf.to_container(cfg, resolve=False)
    d.pop("hydra", None)
    return d


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}."))
    else:
        out[p[:-1]] = d
    return out


def npath(v):
    if isinstance(v, str):
        v = re.sub(r"\$\{oc\.env:VOXBIND_DATA_ROOT,([^}]*)\}", r"\1", v)
        v = re.sub(r"^.*?/?(dataset/data)\b", r"\1", v)
    return v


def classify(key, ov, nv):
    leaf = key.split(".")[-1]
    top = key.split(".")[0]
    if leaf in ("input_mode", "with_gradmag"):
        return "input_loc"            # P1b fold (value-preservation checked separately)
    if leaf == "model_name":
        return "model_name(log)"
    if leaf in PATH_LEAF:
        return "path" if npath(ov) == npath(nv) else "!!PATH_MISMATCH"
    if top in RUNTIME or leaf in RUNTIME_LEAF:
        return "runtime"
    if leaf in INERT and (ov == "<absent>" or nv == "<absent>"):
        return "inert_weight"
    if leaf == "density_weight_warmup_epochs" and 0 in (ov, nv):
        return "warmup0"
    if leaf == "gradmag_as_density" and not (ov is True or nv is True):
        return "feature_off"          # default-off dataset feature → inert when not True
    return "!!UNEXPLAINED"


print("=" * 70)
print("(1) COMPOSE EVERY CONFIG (break check)")
broke = []
for f in sorted(glob.glob(os.path.join(CD, "config_train_*.yaml")) + [os.path.join(CD, "pretrain.yaml")]):
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        raw(name)
    except Exception as e:
        broke.append((name, f"{type(e).__name__}: {str(e)[:80]}"))
for f in sorted(glob.glob(os.path.join(CD, "experiment", "*.yaml"))):
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        raw("pretrain", [f"+experiment={name}"])
    except Exception as e:
        broke.append((f"experiment/{name}", f"{type(e).__name__}: {str(e)[:80]}"))
if broke:
    print(f"  !! {len(broke)} configs FAILED to compose:")
    for n, e in broke:
        print(f"     {n}: {e}")
else:
    print("  all configs compose OK")

print("\n" + "=" * 70)
print("(2) RAW (un-exempted) DIFF per migrated preset — every diff must be a benign category")
clean = True
for old, new in PAIRS:
    o, n = flat(raw(old)), flat(raw("pretrain", [f"+experiment={new}"]))
    rows = []
    for k in sorted(set(o) | set(n)):
        ov, nv = o.get(k, "<absent>"), n.get(k, "<absent>")
        if ov != nv:
            rows.append((classify(k, ov, nv), k, ov, nv))
    # input-location value preservation: old top-level X == new model.X
    for key in ("input_mode", "with_gradmag"):
        ovv, nvv = o.get(key, "<absent>"), n.get(f"model.{key}", "<absent>")
        if ovv != "<absent>" and ovv != nvv:
            rows.append(("!!INPUT_VALUE_LOST", key, ovv, nvv))
    bad = [r for r in rows if r[0].startswith("!!")]
    clean = clean and not bad
    print(f"\n### {new}   ({len(rows)} raw diffs, {len(bad)} flagged)")
    for c, k, ov, nv in rows:
        print(f"   [{c:16s}] {k}: old={ov!r} new={nv!r}")

print("\n" + "=" * 70)
print("VERDICT:", "CLEAN — every exempted diff is a benign category, input values preserved"
      if clean and not broke else "!! PROBLEMS FOUND (see flags above)")
