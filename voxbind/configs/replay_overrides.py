"""replay_overrides.py — CLI-override replay gate for the config reorg (P2/#6).

Scrapes every Hydra command-line override that scripts/*.sh pass to a training run, then checks
each override KEY against the new `pretrain` config. A bare `key=val` override requires the key to
already exist in the composed config — so this catches keys that the reorg moved/renamed (e.g. the
P1b move of `input_mode` -> `model.input_mode`) BEFORE a repointed launcher hits the failure live.

    python configs/replay_overrides.py            # report; exit 1 if any UNEXPECTED key is missing

Keys known to have moved in P1b (input_mode/with_gradmag -> model.*) are reported as expected
migrations (the fix lands when scripts are repointed in P3), not failures.
"""
import glob
import os
import re
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(CONFIG_DIR), "scripts")

# Override tokens: dotted group keys (unambiguously Hydra) + a whitelist of bare top-level keys
# (lowercase, so bash UPPERCASE vars like VOX=/CFG=/GPUS= never match).
_DOTTED = re.compile(r"\b((?:dset|model|mae|vox|optimizer|ema|amp|compile)\.[A-Za-z0-9_.]+)=")
_BARE = re.compile(r"(?:^|\s)((?:input_mode|with_gradmag|bsz|accum_steps|num_epochs|seed|lr|wd|aug|"
                   r"num_workers|prefetch_factor|exp_name|exp_dir|debug|channels_last)=)")

# Keys the reorg intentionally relocated; their fix is the repoint in P3, not a config bug.
_EXPECTED_MOVED = {"input_mode": "model.input_mode", "with_gradmag": "model.with_gradmag"}

# Only PRETRAIN-family launchers are in scope: those that reach train_density_{vit,cha}_mae.py
# (directly, via pretrain.sh / the watcher, or by naming a vit_mae/cha_mae config). Downstream
# (train_ddp.py / model/voxbind.yaml) and sampling scripts use a different config family.
_PRETRAIN_RE = re.compile(r"train_density_(?:vit|cha)_mae|config_train_[a-z0-9_]*(?:vit_mae|cha_mae)"
                          r"|pretrain\.sh|watch_n_launch")


def scrape_keys():
    keys = set()
    for path in sorted(glob.glob(os.path.join(SCRIPTS_DIR, "*.sh"))):
        text = open(path, encoding="utf-8", errors="ignore").read()
        if not _PRETRAIN_RE.search(text):
            continue
        keys |= set(_DOTTED.findall(text))
        keys |= {tok.rstrip("=").strip() for tok in _BARE.findall(text)}
    return sorted(keys)


def key_exists(cfg, dotted):
    cur = cfg
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        # use keys() not `in`: OmegaConf's __contains__ reports a ??? mandatory-missing key as absent
        if not isinstance(cur, DictConfig) or part not in list(cur.keys()):
            return False
        if i < len(parts) - 1:           # don't resolve the leaf (it may be a ??? mandatory value)
            try:
                cur = cur[part]
            except Exception:
                return False
    return True


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="pretrain")
    keys = scrape_keys()
    ok, moved, missing = [], [], []
    for k in keys:
        if key_exists(cfg, k):
            ok.append(k)
        elif k in _EXPECTED_MOVED:
            moved.append(k)
        else:
            missing.append(k)

    print(f"=== replay gate: {len(keys)} unique override keys scraped from scripts/ ===")
    print(f"[ ok ]      {len(ok)} keys resolve against pretrain")
    for k in ok:
        print(f"    ok   {k}")
    if moved:
        print(f"[migrate]   {len(moved)} keys moved in the reorg (repoint launchers in P3):")
        for k in moved:
            print(f"    {k}=  ->  {_EXPECTED_MOVED[k]}=")
    if missing:
        print(f"[MISSING]   {len(missing)} keys do NOT resolve and were NOT expected to move:")
        for k in missing:
            print(f"    !! {k}")
        print("\nFAIL — unexpected missing override keys (a launcher would crash on these).")
        return 1
    print("\nPASS — every script override resolves (modulo the expected P1b moves).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
