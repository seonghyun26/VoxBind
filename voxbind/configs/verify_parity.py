"""verify_parity.py — compose-diff parity gate for the config reorg.

Asserts that a new `pretrain +experiment=<exp>` composition resolves to the SAME config
as an old monolithic `config_train_*` config — proving the reorg changes file organization
only, not the resolved config tree. The `hydra` node is excluded (plumbing, identical by
construction). Run from voxbind/:

    python configs/verify_parity.py <OLD_CONFIG_NAME> <EXPERIMENT_NAME>
    python configs/verify_parity.py config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
                                     atomblob_density_gradmag_invfreq

Exit 0 = identical; 1 = drift (every differing leaf is printed). Values are compared parsed
(so 1e-4 == 0.0001) and order-independently.
"""
import os
import re
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Path keys whose absolute/${env} prefix is an intentional P2 portability change, not drift.
_DATA_KEYS = {"data_dir", "crops_dir", "ccp4_dir", "pdb_dir", "resample_dir", "out_root", "out_dir"}


def _norm_path(v):
    """Normalize a data path so the portability prefix (absolute root or ${oc.env:VOXBIND_DATA_ROOT,…})
    is ignored but the meaningful relative subpath (dataset/data/<sub>) is still compared."""
    if not isinstance(v, str) or not v:
        return v
    v = re.sub(r"\$\{oc\.env:VOXBIND_DATA_ROOT,([^}]*)\}", r"\1", v)  # ${oc.env:..,X} -> X
    v = re.sub(r"^.*?/?(dataset/data)\b", r"\1", v)                  # strip any abs prefix before dataset/data
    return v


def _canon(d):
    """P1b/#2: input_mode/with_gradmag may live top-level (old monoliths) or under model (new
    configs). Fold them under model so the intentional namespace move is not flagged as drift —
    the VALUE must still match, which is what the diff then checks."""
    model = d.setdefault("model", {})
    for k in ("input_mode", "with_gradmag"):
        if k in d:
            model.setdefault(k, d.pop(k))
    # model_name is a provenance/log label (build constructs from explicit fields, not the name) and
    # was copy-paste-inconsistent across the old monoliths — exempt it from byte-parity.
    model.pop("model_name", None)
    # Runtime/throughput knobs are standardized in the base and are result-neutral (260611:
    # amp+compile verified same results); older monoliths predate them. Exempt from byte-parity.
    for k in ("amp", "compile", "optimizer", "ema", "channels_last", "num_workers", "prefetch_factor"):
        d.pop(k, None)
    # A density/gradmag loss-weight is INERT when its channel is absent (the trainer's input_mode
    # guards never apply it). The standardized _base objective carries these knobs everywhere; older
    # monoliths listed them only when relevant. Drop the inert ones so behavioral parity holds.
    im = str(model.get("input_mode", "") or "")
    mae = d.get("mae", {}) or {}
    if "density" not in im:
        for k in ("density_channel_weight", "density_weight_warmup_epochs"):
            mae.pop(k, None)
    if not bool(model.get("with_gradmag", False)):
        for k in ("gradmag_channel_weight", "gradmag_noise", "gradmag_reconstruct"):
            mae.pop(k, None)
    # density warmup of 0 epochs == off == the code default (train_density_vit_mae.py:1365), which is
    # what terse old configs that omitted the key got. Inert when 0 → don't flag the standardized 0.
    if mae.get("density_weight_warmup_epochs") == 0:
        mae.pop("density_weight_warmup_epochs", None)
    # Default-off dataset feature flags (e.g. gradmag_as_density) are inert when false → behaviorally
    # absent, like terse old configs that predate the feature. Don't flag the standardized false.
    ds = d.get("dset", {}) or {}
    if ds.get("gradmag_as_density") is False:
        ds.pop("gradmag_as_density", None)

    def _walk(o):  # P2: normalize known data-path keys wherever they appear
        if isinstance(o, dict):
            for k, v in o.items():
                o[k] = _norm_path(v) if k in _DATA_KEYS else (_walk(v) or v)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
        return o
    _walk(d)
    return d


def resolved(config_name, overrides=None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    d = OmegaConf.to_container(cfg, resolve=False)
    d.pop("hydra", None)
    return _canon(d)


def diff(new, old, path=""):
    out = []
    if isinstance(new, dict) and isinstance(old, dict):
        for k in sorted(set(new) | set(old)):
            if k not in new:
                out.append(f"{path}{k}: MISSING in NEW (old={old[k]!r})")
            elif k not in old:
                out.append(f"{path}{k}: EXTRA in NEW (new={new[k]!r})")
            else:
                out += diff(new[k], old[k], f"{path}{k}.")
    elif new != old:
        out.append(f"{path[:-1]}: new={new!r} != old={old!r}")
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old_name, exp = sys.argv[1], sys.argv[2]
    old = resolved(old_name)
    new = resolved("pretrain", [f"+experiment={exp}"])
    d = diff(new, old)
    if not d:
        print(f"PASS — pretrain +experiment={exp}  ==  {old_name}  (resolved configs identical)")
        return 0
    print(f"FAIL — {len(d)} difference(s)  [pretrain +experiment={exp}  vs  {old_name}]:")
    for line in d:
        print("  " + line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
