import torch

from voxbind.dataset.crossdocked import DatasetCrossdocked
from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray


def _make_dataset(cfg, split: str, aug: bool):
    """Instantiate the correct dataset class from cfg.dset.dset_name."""
    name = cfg.dset.dset_name
    if name == "crossdocked":
        return DatasetCrossdocked(
            data_dir=cfg.dset.data_dir,
            split=split,
            aug=aug,
            small=cfg.debug,
            ligand_radius=cfg.dset.ligand_radius,
            pocket_radius=cfg.dset.pocket_radius,
        )
    elif name == "crossdocked_xray":
        return DatasetCrossDockedXray(
            data_dir=cfg.dset.data_dir,
            crops_dir=cfg.dset.get("crops_dir", ""),
            ccp4_dir=cfg.dset.get("ccp4_dir", "dataset/data/ccp4"),
            split=split,
            use_xray=cfg.dset.get("use_xray", True),
            cache_size=cfg.dset.get("cache_size", 32),
            aug=aug,
            small=cfg.debug,
            ligand_radius=cfg.dset.ligand_radius,
            pocket_radius=cfg.dset.pocket_radius,
            normalize=cfg.dset.get("normalize", True),
        )
    else:
        raise NotImplementedError(f"Dataset '{name}' not implemented")


def create_dataloaders(cfg) -> tuple:
    """
    Create data loaders for training, validation, and sampling.

    Args:
        cfg (Config): Configuration object containing dataset and training parameters.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and sampling data loader.
    """
    # create train loader
    dset_train = _make_dataset(cfg, split="train", aug=cfg.aug)

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    # Keep a deeper worker queue to reduce main-process blocking on next(batch).
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))

    loader_train = torch.utils.data.DataLoader(
        dset_train,
        batch_size=cfg.bsz,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    # create val loader
    dset_val = _make_dataset(cfg, split="val", aug=False)
    loader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_size=cfg.bsz,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # create sampling loader
    loader_sampling = torch.utils.data.DataLoader(
        dset_val,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return loader_train, loader_val, loader_sampling


def create_sampling_dataloader(cfg, split="val") -> torch.utils.data.DataLoader:
    """
    Create a data loader for sampling.

    Args:
        cfg (dict): Configuration dictionary.
        split (str, optional): Split name. Defaults to "val".

    Returns:
        torch.utils.data.DataLoader: Data loader for sampling.
    """
    dset_val = _make_dataset(cfg, split=split, aug=False)
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))

    # create sampling loader
    loader_sampling = torch.utils.data.DataLoader(
        dset_val,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return loader_sampling
