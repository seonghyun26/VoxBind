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

    model.to(device)
    return model
