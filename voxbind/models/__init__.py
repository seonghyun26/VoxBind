import logging
import os
import types

import torch

from voxbind.models.voxbind import VoxBind
from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS

logger = logging.getLogger(__name__)


def create_model(cfg, device="cuda") -> VoxBind:
    """
    Create a VoxBind model.

    Args:
        cfg (Config): The configuration object containing model parameters.
        device (str): The device to use for model computation. Defaults to "cuda".

    Returns:
        VoxBind: The created VoxBind model.

    Notes
    -----
    When `cfg.model.density_pretrained_path` is set and `with_density=True`, the
    `density_encoder` is initialized from the 260521 density-MAE checkpoint
    (encoder slice from `state_dict_ema` or `encoder_state_dict_ema`). When
    `cfg.model.density_freeze` is also set, the encoder's parameters are frozen
    (requires_grad=False) and its train() method is overridden so the outer
    `model.train()` call doesn't flip dropout back on.
    """
    with_density = cfg.model.get("with_density", False)
    # Master gradmag flag lives at the top level (shared with the dataset /
    # pretraining), and widens the density branch 1→2 ch (density + gradmag).
    with_gradmag = bool(cfg.get("with_gradmag", False))
    density_encoder_type = cfg.model.get("density_encoder_type", "cnn")
    # vit knobs (only consulted when density_encoder_type == "vit"; safe defaults
    # otherwise so older configs without the `density_vit:` block still load).
    vit_cfg = cfg.model.get("density_vit", {}) or {}
    adapter_cfg = cfg.model.get("adapter", {}) or {}
    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=cfg.model.n_channels,
        ch_mults=cfg.model.ch_mults,
        is_attn=cfg.model.is_attn,
        n_blocks=cfg.model.n_blocks,
        n_groups=cfg.model.n_groups,
        dropout=cfg.model.dropout,
        smooth_sigma=cfg.smooth_sigma,
        with_density=with_density,
        with_gradmag=with_gradmag,
        density_encoder_type=density_encoder_type,
        density_encoder_blocks=int(cfg.model.get("density_encoder_blocks", 1)),
        density_grid_dim=int(cfg.vox.grid_dim),
        density_vit_patch=int(vit_cfg.get("patch", 8)),
        density_vit_dim=int(vit_cfg.get("dim", 192)),
        density_vit_depth=int(vit_cfg.get("depth", 6)),
        density_vit_heads=int(vit_cfg.get("heads", 6)),
        density_vit_mlp_ratio=int(vit_cfg.get("mlp_ratio", 4)),
        density_vit_dropout=float(vit_cfg.get("dropout", 0.1)),
        density_vit_patch_embed_mode=str(vit_cfg.get("patch_embed_mode", "fused")),
        density_vit_channel_groups=(tuple(vit_cfg.channel_groups)
                                    if vit_cfg.get("channel_groups", None) else None),
        density_vit_n_memory_tokens=int(vit_cfg.get("n_memory_tokens", 0)),
        # None → field mode (1/2 ch). 13 → full-voxel conditioning (Path B): reuse a
        # combined atomblob_density_gradmag ViT-MAE frozen, with the ligand masked.
        density_vit_n_in_channels=(int(vit_cfg["n_in_channels"])
                                   if vit_cfg.get("n_in_channels", None) is not None else None),
        # Leak-removal ablation (Path B): blank density+gradmag inside the clean ligand
        # footprint so the frozen encoder cannot read the co-crystal ligand's density.
        density_mask_ligand=bool(cfg.model.get("density_mask_ligand", False)),
        density_mask_threshold=float(cfg.model.get("density_mask_threshold", 0.2)),
        density_mask_dilate=int(cfg.model.get("density_mask_dilate", 2)),
        # bf16 autocast for the frozen encoder's forward ONLY (denoiser stays fp32).
        density_encoder_sees_ligand=bool(cfg.model.get("density_encoder_sees_ligand", False)),
        density_encoder_amp=bool(cfg.model.get("density_encoder_amp", False)),
        # Global density->noise blend: d = a*d + (1-a)*noise. 1.0 = no-op. Test-time knob
        # for measuring how much a trained model relies on the density channel.
        density_noise_alpha=float(cfg.model.get("density_noise_alpha", 1.0)),
        density_noise_sigma=float(cfg.model.get("density_noise_sigma", 0.0)),
        # Protein-only attenuation field: alpha(x) from pocket atoms, no ligand involved.
        density_drop_ligand_tokens=bool(cfg.model.get("density_drop_ligand_tokens", True)),
        density_attenuate=bool(cfg.model.get("density_attenuate", False)),
        density_attenuate_sigma=float(cfg.model.get("density_attenuate_sigma", 7.0)),
        density_attenuate_quantile=float(cfg.model.get("density_attenuate_quantile", 0.90)),
        density_attenuate_strength=float(cfg.model.get("density_attenuate_strength", 1.0)),
        density_attenuate_noise_sigma=float(cfg.model.get("density_attenuate_noise_sigma", 7.0)),
        density_proj_hidden=(int(cfg.model["density_proj_hidden"])
                             if cfg.model.get("density_proj_hidden", None) is not None else None),
        density_proj_kernel=int(cfg.model.get("density_proj_kernel", 1)),
        # "default": pocket_encoder + zero-init density_proj. "v3": frozen encoder replaces
        # pocket_encoder (pocket + apo density), fused via normal-init context_proj (early fusion).
        # "adapter": frozen VoxBind + frozen pretrained pocket encoder + trainable PocketAdapter.
        fusion=str(cfg.model.get("fusion", "default")),
        # adapter (fusion='adapter') knobs — match the pretrained pocket tower's ViT geometry.
        adapter_dim=int(adapter_cfg.get("dim", 512)),
        adapter_depth=int(adapter_cfg.get("depth", 12)),
        adapter_heads=int(adapter_cfg.get("heads", 8)),
        adapter_mlp_ratio=int(adapter_cfg.get("mlp_ratio", 4)),
        adapter_n_in=int(adapter_cfg.get("n_in_channels", 7)),
        adapter_channel_groups=(tuple(adapter_cfg.channel_groups)
                                if adapter_cfg.get("channel_groups", None) else None),
        adapter_patch=int(adapter_cfg.get("patch", 8)),
        adapter_grid_dim=int(cfg.vox.grid_dim),
        adapter_hidden=(int(adapter_cfg["hidden"]) if adapter_cfg.get("hidden", None) else None),
        adapter_mask_basis=str(adapter_cfg.get("mask_basis", "protein_vdw")),
        adapter_mask_thresh=float(adapter_cfg.get("mask_thresh", 0.2)),
    )

    # Optionally load + freeze the pretrained density encoder
    pretrained = cfg.model.get("density_pretrained_path", None)
    if with_density and pretrained:
        if not os.path.isfile(pretrained):
            raise FileNotFoundError(
                f"density_pretrained_path not found: {pretrained}"
            )
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        if "encoder_state_dict_ema" in ckpt and ckpt["encoder_state_dict_ema"]:
            enc_sd = ckpt["encoder_state_dict_ema"]
            source = "encoder_state_dict_ema"
        else:
            full = ckpt.get("state_dict_ema", ckpt.get("state_dict", {}))
            enc_sd = {k: v for k, v in full.items() if k.startswith("encoder.")}
            source = "state_dict_ema/state_dict (encoder.* slice)"
        # Strip the leading "encoder." so keys map to density_encoder.Sequential indices
        enc_sd = {k.replace("encoder.", "", 1): v for k, v in enc_sd.items()
                  if k.startswith("encoder.")}
        missing, unexpected = model.density_encoder.load_state_dict(enc_sd, strict=True)
        logger.info(
            f"loaded density encoder from {pretrained} ({source}); "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        if cfg.model.get("density_freeze", False):
            for p in model.density_encoder.parameters():
                p.requires_grad = False
            model.density_encoder.eval()
            # Override train() so outer model.train() doesn't reactivate dropout
            # on the frozen branch.
            def _no_train(self_mod, mode=True):
                return torch.nn.Module.train(self_mod, False)
            model.density_encoder.train = types.MethodType(
                _no_train, model.density_encoder,
            )
            n_frozen = sum(p.numel() for p in model.density_encoder.parameters())
            logger.info(f"density_encoder frozen ({n_frozen:,} params, dropout disabled)")

    # fusion='adapter': load the pretrained pocket-tower TRUNK into adapter_encoder, then FREEZE
    # all of VoxBind except the PocketAdapter (transfer per the handoff: frozen pocket encoder +
    # frozen VoxBind + trainable spatial adapter).
    if str(cfg.model.get("fusion", "default")) == "adapter":
        ap = cfg.model.get("adapter", {}) or {}
        pre = ap.get("pretrained_path", None)
        if pre:
            if not os.path.isfile(pre):
                raise FileNotFoundError(f"adapter.pretrained_path not found: {pre}")
            ck = torch.load(pre, map_location="cpu", weights_only=False)
            enc = ck.get("encoder_state_dict_ema") or {
                k: v for k, v in ck.get("state_dict_ema", ck.get("state_dict", {})).items()
                if k.startswith("encoder.")
            }
            enc = {k.replace("encoder.", "", 1): v for k, v in enc.items() if k.startswith("encoder.")}
            # forward_features never touches decoder_proj (the MAE recon head) — drop it so a
            # c_out mismatch can't break the load; the trunk (group_proj/blocks/norm/embeds) loads.
            enc = {k: v for k, v in enc.items() if not k.startswith("decoder_proj")}
            missing, unexpected = model.adapter_encoder.load_state_dict(enc, strict=False)
            trunk_missing = [k for k in missing if not k.startswith("decoder_proj")]
            logger.info(f"adapter_encoder loaded from {pre}: {len(enc)} keys, "
                        f"trunk_missing={len(trunk_missing)} unexpected={len(unexpected)}")
            if trunk_missing:
                raise RuntimeError(f"adapter_encoder trunk keys not loaded: {trunk_missing[:8]}")
        else:
            logger.warning("fusion='adapter' with no adapter.pretrained_path — encoder is RANDOM "
                           "(fine for a code smoke, NOT for training).")
        # Freeze everything, then re-enable ONLY the trainable adapter.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.pocket_adapter.parameters():
            p.requires_grad = True
        # Keep all frozen submodules in eval() and pin their train() so model.train() can't
        # reactivate dropout/norm updates on them; the adapter alone follows train/eval.
        def _no_train(self_mod, mode=True):
            return torch.nn.Module.train(self_mod, False)
        for name, mod in (("ligand_encoder", model.ligand_encoder),
                          ("pocket_encoder", model.pocket_encoder),
                          ("unet3d", model.unet3d),
                          ("final_ligand", model.final_ligand),
                          ("adapter_encoder", model.adapter_encoder)):
            mod.eval()
            mod.train = types.MethodType(_no_train, mod)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"fusion=adapter: trainable {n_train:,} / {n_total:,} params "
                    f"({100*n_train/max(1,n_total):.2f}%) — PocketAdapter only")

    model.to(device)
    return model
