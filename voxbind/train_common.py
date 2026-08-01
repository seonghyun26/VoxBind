"""train_common.py — shared engine for the ViT density-encoder pre-trainers.

Pretext-INDEPENDENT scaffolding, extracted verbatim from ``train_density.py``
so the per-pretext trainers (``train_mae.py`` / ``train_electra.py`` /
``train_cross_modal.py``) share one copy instead of duplicating it. Holds:

  • DDP setup/teardown            (_setup_ddp, _cleanup_ddp, _unwrap)
  • bf16 AMP + torch.compile      (_amp_setup, _compile_options, maybe_compile_model)
  • atomic async checkpointing    (AsyncCheckpointSaver)
  • multi-channel input layout    (_VALID_INPUT_MODES, _build_merged_atoms, _channel_layout)
  • val-voxel cache               (_val_cache_path, precompute_val)
  • channel-freq inv-weights      (_ch_freq_cache_path, precompute_channel_weights)
  • metric logging                (log_metrics)
  • input-key reconciliation      (_reconcile_input_keys)

What STAYS per-pretext (it branches on the pretext): input corruption/masking
(today's ``MAEPrefetcher`` + ``_sample_op``), the loss heads (``compute_losses``),
the ``train_epoch`` / ``val_epoch`` bodies, and the ``@hydra.main`` entrypoint.

Status: additive foundation. ``train_density.py`` still carries its own
copies of these until the cutover (rewire its imports → here, add the pretext-
dispatch shim) is done behind a verifying one-step smoke run. Nothing imports this
module yet, so creating it changes no behavior.
"""
import copy
import hashlib
import json
import logging
import os
import threading
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger("voxbind.train_common")


# ── DDP ───────────────────────────────────────────────────────────────────────
def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _setup_ddp() -> tuple:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=2)
    )
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ── bf16 AMP + torch.compile ──────────────────────────────────────────────────
def _compile_options(cfg: DictConfig) -> dict:
    compile_cfg = cfg.get("compile", {}) or {}
    return {
        "enabled": bool(compile_cfg.get("enabled", False)),
        "backend": str(compile_cfg.get("backend", "inductor")),
        "mode": str(compile_cfg.get("mode", "reduce-overhead")),
        "fullgraph": bool(compile_cfg.get("fullgraph", False)),
        "dynamic": bool(compile_cfg.get("dynamic", False)),
    }


def _amp_setup(cfg: DictConfig):
    """Return (enabled, ctx_factory). ctx_factory() yields a bf16 autocast context
    (a no-op when disabled). bf16 needs no GradScaler; fp16 is intentionally unsupported."""
    amp_cfg = cfg.get("amp", {}) or {}
    enabled = bool(amp_cfg.get("enabled", False))
    dtype = str(amp_cfg.get("dtype", "bf16"))
    if enabled and dtype not in ("bf16", "bfloat16"):
        raise NotImplementedError(
            f"amp.dtype={dtype!r} unsupported — bf16 only (fp16 would need a GradScaler)")
    return enabled, (lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled))


def maybe_compile_model(model: torch.nn.Module, cfg: DictConfig, *, is_main: bool) -> torch.nn.Module:
    opts = _compile_options(cfg)
    if not opts["enabled"]:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile.enabled=true requested, but this PyTorch has no torch.compile")

    kwargs = {
        "backend": opts["backend"],
        "fullgraph": opts["fullgraph"],
        "dynamic": opts["dynamic"],
    }
    if opts["mode"] and opts["mode"] != "default":
        kwargs["mode"] = opts["mode"]
    if is_main:
        logger.info(
            "torch.compile enabled "
            f"(backend={opts['backend']}, mode={opts['mode']}, "
            f"fullgraph={opts['fullgraph']}, dynamic={opts['dynamic']})"
        )
    return torch.compile(model, **kwargs)


# ── Atomic async checkpointing (rank 0 only) ──────────────────────────────────
class AsyncCheckpointSaver:
    """Atomic + threaded checkpoint writer (rank 0 only). Mirrors train_ddp.py."""

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()

    @staticmethod
    def _write(state: dict, save_dir: str, chkp_name: str) -> None:
        tmp = os.path.join(save_dir, chkp_name + ".tmp")
        final = os.path.join(save_dir, chkp_name)
        try:
            torch.save(state, tmp)
            os.replace(tmp, final)
        except Exception as e:
            logger.warning(f"checkpoint write failed: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def save(self, state: dict, save_dir: str, chkp_name: str = "checkpoint.pth.tar") -> None:
        self.wait()
        frozen = copy.deepcopy(state)
        with self._lock:
            self._thread = threading.Thread(
                target=self._write,
                args=(frozen, save_dir, chkp_name),
                daemon=False,
            )
            self._thread.start()

    def wait(self) -> None:
        with self._lock:
            t = self._thread
        if t is not None:
            t.join()


# ── Multi-channel input / reconstruction layout ───────────────────────────────
_VALID_INPUT_MODES = (
    "density",
    "atomblob",
    "atomblob_density",
    "atomblob_merged",            # 7 atom ch: pocket C/O/N/S folded into ligand C/O/N/S
    "atomblob_merged_density",    # 8 ch: merged-7 atoms + 1 density
    "roleblob",                   # 2 atom ch: [ligand-occupancy, pocket-occupancy], vdW-radius blobs
    "roleblob_density",           # 3 ch: role-2 atoms + 1 density
)


def _build_merged_atoms(v_lig: torch.Tensor, v_poc: torch.Tensor) -> torch.Tensor:
    """Fold pocket atoms into the first 4 channels of the ligand 7-channel tensor.

    CrossDocked element vocab is C/O/N/S/F/Cl/P for ligand and C/O/N/S for
    pocket (pocket has no F/Cl/P). After this op every channel encodes
    'atoms-of-element-X' regardless of source — the lig-vs-pocket
    disambiguation drops out of the pretext.
    """
    # v_lig (B, 7, G³)  +  v_poc (B, 4, G³) summed into channels [0..4)
    v_merged = v_lig.clone()
    v_merged[:, :4] = v_merged[:, :4] + v_poc
    return v_merged                                                 # (B, 7, G³)


def _build_role_atoms(v_lig: torch.Tensor, v_poc: torch.Tensor) -> torch.Tensor:
    """Collapse per-element atom channels into 2 ROLE channels: [ligand, pocket].

    Summing a role's per-element channels is *exactly* equal to voxelizing all of
    that role's atoms into a single channel — each atom's Gaussian blob depends
    only on its coords + radius, not on which channel it lands in (verified to
    float epsilon). With element-wise vdW radii (ligand_radius=-1 / pocket_radius=-1)
    every atom contributes a blob sized by its own van-der-Waals diameter, so the
    occupancy channel encodes atom identity via blob SIZE instead of a one-hot
    element channel — letting the encoder ingest diverse atoms without a fixed
    element vocabulary. Channel order is [ligand, pocket] to match the atomblob
    convention (v_lig before v_poc). Kept lig/poc separable (not pre-merged) so a
    future generative path can role-collapse the pocket while keeping per-element
    ligand output for vox2mol.
    """
    return torch.cat(
        [v_lig.sum(dim=1, keepdim=True), v_poc.sum(dim=1, keepdim=True)], dim=1
    )                                                               # (B, 2, G³)


# Ligand / pocket element-channel counts = the voxelizer width per partner. Default 7/4
# (C,O,N,S,F,Cl,P / C,O,N,S). The per-element "+other" corpus uses 8/4 (a hybrid ligand
# channel catching the 0.25% rare tail). Set ONCE at run() startup via set_atom_channels();
# read by _channel_layout below (same module). The voxelize call-sites read cfg directly.
LIG_CH = 7
POC_CH = 4


def set_atom_channels(n_lig: int, n_poc: int) -> None:
    """Pin the ligand/pocket element-channel counts for _channel_layout (call once at startup)."""
    global LIG_CH, POC_CH
    LIG_CH, POC_CH = int(n_lig), int(n_poc)


# Number of electron-density SOURCES fed as (density, gradmag) channel pairs. Default 1 =
# 2Fo-Fc only (all prior runs bit-identical). 2 = 2Fo-Fc + mFo-DFc difference map, laid out
# per-source-interleaved [ …atoms…, dens0, grad0, dens1, grad1 ] so channel_groups=[…,2,2]
# groups each map's (density, gradmag) as its own ChannelViT modality. Set ONCE at run()
# startup via set_density_sources(); read by _channel_layout below.
N_DENSITY_SRC = 1


def set_density_sources(n: int) -> None:
    """Pin the # of density sources (2Fo-Fc [+ mFo-DFc]) for _channel_layout (call once at startup)."""
    global N_DENSITY_SRC
    N_DENSITY_SRC = int(n)


def _channel_layout(
    input_mode: str,
    with_gradmag: bool = False,
    gradmag_reconstruct: bool = True,
    density_input: bool = True,
) -> dict:
    """Single source of truth for the multi-channel input / reconstruction layout.

    Channel order is always  [ …atoms (n_atom)…, density (n_density), gradmag (n_gradmag) ].

      n_in     : encoder patch-embed input width.  Full atoms+density+gradmag when
                 density_input=True; just the atom channels when density_input=False
                 (density/gradmag are reconstruction TARGETS only, never fed to the
                 encoder — a coords-only encoder that must PREDICT the density field).
      n_recon  : MAE head / loss-target width.  Full atoms+density+gradmag when
                 gradmag is reconstructed; − n_gradmag when gradmag is input-only.
                 Independent of density_input, so n_recon > n_in is valid (extra
                 target-only channels).

    `density_idx` / `gradmag_idx` are the channel offsets of those trailing
    channels IN THE TARGET (x_clean) frame — unchanged by density_input, since the
    target always carries them.  When density_input=False they lie past n_in, so
    they must not be used to index the encoder input (gradmag_idx is None when
    with_gradmag is False).
    """
    if input_mode == "density":
        n_atom = 0
    elif input_mode in ("atomblob", "atomblob_density"):
        n_atom = LIG_CH + POC_CH                       # 7+4=11 default; 8+4=12 with the +other ligand channel
    elif input_mode in ("atomblob_merged", "atomblob_merged_density"):
        n_atom = LIG_CH
    elif input_mode in ("roleblob", "roleblob_density"):
        n_atom = 2                                    # [ligand-occupancy, pocket-occupancy]
    else:
        raise RuntimeError(f"unexpected input_mode={input_mode!r}")

    has_density = input_mode in (
        "density", "atomblob_density", "atomblob_merged_density", "roleblob_density",
    )
    n_density = 1 if has_density else 0
    if with_gradmag and not has_density:
        raise ValueError(
            f"with_gradmag=True requires a density-bearing input_mode "
            f"(density / *_density); got input_mode={input_mode!r}"
        )
    n_gradmag = 1 if with_gradmag else 0
    # Density SOURCES (2Fo-Fc [+ mFo-DFc]) each contribute a (density, gradmag) pair,
    # interleaved per source: [ …atoms…, dens0, grad0, dens1, grad1 ]. n_density / n_gradmag
    # stay PER-SOURCE (=1); the multi-source width scales by n_src. n_src=1 → identical to
    # the single-2Fo-Fc layout (all prior runs unaffected).
    n_src = N_DENSITY_SRC if has_density else 0
    per_src = n_density + n_gradmag                   # channels contributed per source
    n_full = n_atom + n_src * per_src                 # full assembled width (= target width source)
    # Reconstruction / loss-target width: gradmag dropped (per source) only when input-only.
    per_src_recon = n_density + (n_gradmag if gradmag_reconstruct else 0)
    n_recon = n_atom + n_src * per_src_recon
    # Encoder input width. density_input=False → atoms-only encoder; density+gradmag
    # are reconstruction targets the coords-only encoder must predict (n_in < n_recon).
    if density_input:
        n_in = n_full
    else:
        if not has_density:
            raise ValueError(
                f"density_input=False requires a density-bearing input_mode "
                f"(density / *_density); got input_mode={input_mode!r}"
            )
        n_in = n_atom
    return {
        "n_atom": n_atom,
        "n_density": n_density,           # PER SOURCE (1 if density-bearing)
        "n_gradmag": n_gradmag,           # PER SOURCE (1 if with_gradmag)
        "n_density_src": n_src,           # # density sources: 0 (atomblob) / 1 (2Fo-Fc) / 2 (+mFo-DFc)
        "n_in": n_in,
        "n_recon": n_recon,
        "density_input": density_input,
        "density_idx": n_atom,            # source-0 (2Fo-Fc) density channel
        "gradmag_idx": (n_atom + n_density) if with_gradmag else None,  # source-0 gradmag channel
    }


# ── Val-voxel cache ───────────────────────────────────────────────────────────
def _val_cache_path(cfg) -> str:
    key = {
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "cubes_around": cfg.vox.cubes_around,
        "ligand_radius": cfg.dset.ligand_radius,
        "pocket_radius": cfg.dset.pocket_radius,
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        # resample mode reads density from a different source (full-map crop at the
        # augmented pose) → must key the cache so it can't reuse a frozen-crops blob.
        "resample_dir": cfg.dset.get("resample_dir", ""),
        # 2nd density source (mFo-DFc) adds xray_diff_* tensors to the cache → must key it so
        # a single-source (13-ch) blob can't be reused for a 15-ch run and vice-versa.
        "density_sources": cfg.dset.get("density_sources", 1),
        "resample_dir_diff": cfg.dset.get("resample_dir_diff", ""),
        "density_source": cfg.mae.get("density_source", "synthetic"),
        "input_mode": cfg.get("input_mode", "density"),
        # with_gradmag adds an xray_gradmag tensor to the cache (xray source),
        # so it must key the cache to avoid reusing a gradmag-less blob.
        "with_gradmag": cfg.get("with_gradmag", False),
        # `density_mae_v1` is kept identical to cnn-MAE — the cached tensors are
        # just voxelized ligand+pocket (and optionally xray), no encoder-specific
        # content; input_mode only enters the assembly at val_epoch time.
        "task": "density_mae_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_mae_{h}.pt")


def precompute_val(loader_val, voxelizer, cfg) -> dict:
    path = _val_cache_path(cfg)
    if os.path.isfile(path):
        logger.info(f"loading pre-computed val voxels from {path}")
        return torch.load(path, weights_only=True)

    density_source = str(cfg.mae.get("density_source", "synthetic"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    cache_gradmag = with_gradmag and density_source == "xray"
    # 2nd density source (mFo-DFc): cache its (density, gradmag) too so val_epoch can
    # assemble the SAME 15-channel input as training. Single-source → all_diff stay empty.
    cache_diff = (int(cfg.dset.get("density_sources", 1)) > 1) and density_source == "xray"
    logger.info(
        f"pre-computing val voxels (density_source={density_source}, "
        f"with_gradmag={with_gradmag}, density_sources={int(cfg.dset.get('density_sources', 1))})..."
    )
    all_lig, all_poc, all_xray, all_gradmag = [], [], [], []
    all_diff, all_diff_gradmag = [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=int(cfg.model.get("n_channels_ligand", 7))).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=int(cfg.model.get("n_channels_pocket", 4))).cpu())
            if density_source == "xray":
                all_xray.append(batch["xray_density"].cpu())
            if cache_gradmag:
                all_gradmag.append(batch["xray_gradmag"].cpu())
            if cache_diff:
                all_diff.append(batch["xray_diff_density"].cpu())
                if cache_gradmag:
                    all_diff_gradmag.append(batch["xray_diff_gradmag"].cpu())
    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    if density_source == "xray":
        cache["xray_density"] = torch.cat(all_xray, 0)
    if cache_gradmag:
        cache["xray_gradmag"] = torch.cat(all_gradmag, 0)
    if cache_diff:
        cache["xray_diff_density"] = torch.cat(all_diff, 0)
        if cache_gradmag:
            cache["xray_diff_gradmag"] = torch.cat(all_diff_gradmag, 0)
    torch.save(cache, path)
    logger.info(f"saved val voxels to {path}")
    return cache


# ── Channel-frequency cache + inverse-frequency weights ───────────────────────
_CH_FREQ_POS_THRESH = 0.05            # voxel "atom-present" threshold (matches struct head)
_CH_FREQ_DEFAULT_N_SAMPLES = 2000     # samples scanned for the one-time precompute


def _ch_freq_cache_path(cfg, n_samples: int, merge_lig_poc: bool = False) -> str:
    key = {
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        "resample_dir": cfg.dset.get("resample_dir", ""),
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        # Blob radii set per-channel atom-positive-voxel frequencies, so they
        # must key the cache — else an element-wise-ligand run would reuse a
        # uniform-radius freq blob and get wrong inv-sqrt-freq weights.
        "ligand_radius": cfg.dset.get("ligand_radius", 0.5),
        "pocket_radius": cfg.dset.get("pocket_radius", -1),
        "n_samples": n_samples,
        "pos_thresh": _CH_FREQ_POS_THRESH,
        "merge_lig_poc": merge_lig_poc,
        "task": "channel_freq_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"channel_freq_{h}.pt")


def precompute_channel_weights(
    train_dataset, voxelizer, cfg, device,
    n_samples: int = _CH_FREQ_DEFAULT_N_SAMPLES,
    merge_lig_poc: bool = False,
    power: float = 0.5,
    clip_ratio: float = 0.0,
) -> torch.Tensor:
    """Per-channel atom-positive-voxel frequencies → inverse-frequency weights
    `w = (1 / freq) ** power`, normalized so sum = n_atom_channels (= 11 when
    merge_lig_poc=False; = 7 when True) so the overall loss scale matches the
    unweighted case.

        power = 0.5 → 1/sqrt(freq)  (gentle rebalancing)
        power = 1.0 → 1/freq        (strong: rare atoms much heavier)

    `clip_ratio > 0` caps the max/min weight ratio by raising the floor to
    `w.max() / clip_ratio` BEFORE normalization, so the rarest atom can't
    out-weight the most common one by more than `clip_ratio`×. Useful with
    1/freq, where voxel-dense pocket atoms would otherwise collapse to ~0.01.

    The per-channel frequencies are cached (keyed on dataset / grid / radius /
    merge — NOT on power or clip), and the weight transform is applied on top,
    so changing power or clip reuses the one freq scan.

    With `merge_lig_poc=True`, pocket atoms are folded into the first 4 channels
    of the ligand 7-vec before counting.
    """
    _nlig = int(cfg.model.get("n_channels_ligand", 7)); _npoc = int(cfg.model.get("n_channels_pocket", 4))
    n_atom_channels = _nlig if merge_lig_poc else _nlig + _npoc
    path = _ch_freq_cache_path(cfg, n_samples, merge_lig_poc=merge_lig_poc)
    if os.path.isfile(path):
        logger.info(f"loading channel-frequency cache from {path}")
        freq = torch.load(path, weights_only=True)["freq"].to(torch.float64)
    else:
        n = min(n_samples, len(train_dataset))
        logger.info(
            f"computing per-channel atom-voxel frequencies "
            f"(merge_lig_poc={merge_lig_poc}, n_channels={n_atom_channels}) "
            f"over {n} training samples..."
        )
        sum_per_ch = torch.zeros(n_atom_channels, dtype=torch.float64)
        n_voxels_total = 0
        with torch.no_grad():
            for i in range(n):
                sample = train_dataset[i]
                lig = {k: sample["ligand"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                poc = {k: sample["pocket"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                v_lig = voxelizer.forward(lig, num_channels=int(cfg.model.get("n_channels_ligand", 7)))
                v_poc = voxelizer.forward(poc, num_channels=int(cfg.model.get("n_channels_pocket", 4)))
                if merge_lig_poc:
                    atoms = _build_merged_atoms(v_lig, v_poc)          # (1, 7, G, G, G)
                else:
                    atoms = torch.cat([v_lig, v_poc], dim=1)           # (1, 11, G, G, G)
                pos = (atoms > _CH_FREQ_POS_THRESH).float()
                sum_per_ch += pos.sum(dim=(0, 2, 3, 4)).cpu().to(torch.float64)
                n_voxels_total += atoms.shape[2] * atoms.shape[3] * atoms.shape[4]
        freq = (sum_per_ch / max(1, n_voxels_total)).clamp(min=1e-8)   # pos-voxel frac
        torch.save({"freq": freq.to(torch.float32), "n_samples": n,
                    "pos_thresh": _CH_FREQ_POS_THRESH, "merge_lig_poc": merge_lig_poc}, path)
        logger.info(f"saved channel-frequency cache to {path}")

    w = (1.0 / freq).pow(power)
    if clip_ratio and clip_ratio > 0:
        # Raise the floor so max/min ≤ clip_ratio (keeps rare-atom emphasis,
        # gives crushed common/pocket channels a meaningful weight).
        w = w.clamp(min=float(w.max()) / float(clip_ratio))
    w = (w / w.sum() * float(n_atom_channels)).to(torch.float32)

    elem_lig = ["C", "O", "N", "S", "F", "Cl", "P"]
    elem_poc = ["C", "O", "N", "S"]
    names = ([f"merged_{e}" for e in elem_lig] if merge_lig_poc
             else [f"lig_{e}" for e in elem_lig] + [f"poc_{e}" for e in elem_poc])
    scheme = "1/freq" if power == 1.0 else ("1/sqrt(freq)" if power == 0.5 else f"1/freq^{power}")
    logger.info(
        f"per-channel atom weights ({scheme}, clip_ratio={clip_ratio or 'off'}, "
        f"norm to sum={n_atom_channels}; max/min={float(w.max()/w.min()):.1f}x):"
    )
    for i, name in enumerate(names):
        logger.info(f"  {name:>9}: freq={float(freq[i]):.3e}  weight={float(w[i]):.3f}")
    return w


# ── Logging ───────────────────────────────────────────────────────────────────
def log_metrics(epoch, train_metrics, val_metrics, dt):
    logger.info(f"epoch: {epoch} ({dt:.2f}s)")
    for split, m in zip(["train", "val"], [train_metrics, val_metrics]):
        if m is None:
            continue
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
        logger.info(f"[{split}] {parts}")


# ── Config reconciliation ─────────────────────────────────────────────────────
def _reconcile_input_keys(cfg: DictConfig) -> None:
    """P1b/#2: ``input_mode`` and ``with_gradmag`` canonically live under ``cfg.model`` — the model
    config fixes the input channel layout. Pre-P1b configs (and in-flight resumes that reload an old
    ``cfg.yaml``) put them at the top level. Mirror the two locations in place so ``cfg.model.*`` is
    authoritative when present, else inherits the legacy top-level value; a top-level copy is kept so
    the existing ``cfg.get('input_mode')`` readers keep working. Idempotent."""
    if "model" not in cfg or cfg.model is None:
        return
    with open_dict(cfg):
        for key, default in (("input_mode", "density"), ("with_gradmag", False)):
            in_model = OmegaConf.select(cfg, f"model.{key}", default=None)
            top = OmegaConf.select(cfg, key, default=None)
            val = in_model if in_model is not None else (top if top is not None else default)
            cfg.model[key] = val
            cfg[key] = val
