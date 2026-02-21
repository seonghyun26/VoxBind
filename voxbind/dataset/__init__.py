import torch

from voxbind.dataset.crossdocked import DatasetCrossdocked


def create_dataloaders(cfg) -> tuple:
    """
    Create data loaders for training, validation, and sampling.

    Args:
        cfg (Config): Configuration object containing dataset and training parameters.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and sampling data loader.
    """
    if cfg.dset.dset_name == "crossdocked":
        Dataset = DatasetCrossdocked
    else:
        NotImplementedError(f"{cfg.dset.dset_name} Not implemented yet")

    # create train loader
    dset_train = Dataset(
        data_dir=cfg.dset.data_dir,
        split="train",
        aug=cfg.aug,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )

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
    dset_val = Dataset(
        data_dir=cfg.dset.data_dir,
        split="val",
        aug=False,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )
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
    if cfg.dset.dset_name == "crossdocked":
        Dataset = DatasetCrossdocked
    else:
        NotImplementedError(f"{cfg['dataset']['dset_name']} Not implemented yet")
    dset_val = Dataset(
        data_dir=cfg.dset.data_dir,
        split=split,
        aug=False,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )
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
