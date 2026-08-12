import torch

from voxbind.dataset.crossdocked import DatasetCrossdocked
from voxbind.dataset.crossdocked_density import DatasetCrossDockedDensity
from voxbind.dataset.crossdocked_density_box import DatasetCrossDockedDensityBox
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
            subset_n=cfg.dset.get("subset_n", None),
        )
    elif name == "crossdocked_xray":
        # On-the-fly RESAMPLE mode (opt-in): when dset.resample_dir is set, density is cropped
        # from the FULL map at the AUGMENTED pose each step (the s3 --resample output) instead of
        # loading frozen crops; atoms still voxelize on-GPU. Empty → legacy precomputed-crops path.
        resample_dir = cfg.dset.get("resample_dir", "")
        if resample_dir:
            density_kwargs = dict(
                resample_dir=resample_dir,
                data_dir=cfg.dset.data_dir,
                split=split,
                aug=aug,
                ligand_radius=cfg.dset.ligand_radius,
                pocket_radius=cfg.dset.pocket_radius,
                n_lig_ch=cfg.dset.get("n_lig_ch", 7),
                n_poc_ch=cfg.dset.get("n_poc_ch", 4),
                max_len=cfg.dset.get("max_len", 30),
                delta_translate=cfg.dset.get("delta_translate", 1.0),
                subset_n=cfg.dset.get("subset_n", None),
                subset_xray_only=cfg.dset.get("subset_xray_only", False),
                subset_val_n=cfg.dset.get("subset_val_n", None),
                return_gradmag=bool(cfg.get("with_gradmag", False)),
                # gradmag-only OTF: feed ‖∇ρ‖ into the single density channel (config flag;
                # also honored via the VOXBIND_OTF_GRADMAG_AS_DENSITY env var).
                gradmag_as_density=cfg.dset.get("gradmag_as_density", False),
                data_file=cfg.dset.get("data_file", "data_train.pt"),
                small=cfg.debug,
                cache_size=cfg.dset.get("cache_size", 32),
                # raw_density: skip arcsinh+z-score, feed native unclipped map values.
                raw_density=cfg.dset.get("raw_density", False),
                # v2.2: drop out-of-vocab-ligand complexes at load time (in-vocab-only corpus,
                # reuses v2 tuples+density; fixes the atomblob7 input/target mismatch).
                in_vocab_only=cfg.dset.get("in_vocab_only", False),
            )
            # Fast precomputed-box path (opt-in): when dset.box_path is set, resample the 64³
            # from a precomputed 96³ canonical box instead of the full ccp4 map — identical aug
            # semantics, ~85% util vs OTF's ~40%. n_lig_ch/n_poc_ch flow through unchanged.
            box_path = cfg.dset.get("box_path", "")
            if box_path:
                # 2nd density source (opt-in): dset.resample_dir_diff points at the mFo-DFc
                # difference-map recipe dir (its own box{g}.dat + norm). Adds xray_diff_*
                # channels → [ …atoms…, dens0, grad0, dens1, grad1 ]. Empty → single 2Fo-Fc.
                return DatasetCrossDockedDensityBox(
                    box_path=box_path,
                    resample_dir_diff=cfg.dset.get("resample_dir_diff", ""),
                    box_path_diff=cfg.dset.get("box_path_diff", ""),
                    **density_kwargs,
                )
            primary = DatasetCrossDockedDensity(**density_kwargs)
            # apo+holo concat (opt-in): dset.extra_resample_dir (+ extra_data_file) adds a SECOND
            # OTF corpus (e.g. v4 apo) to the TRAIN split via ConcatDataset — no file merge, no
            # extra storage; each sub-dataset self-aligns via its own manifest/size-filter. Val
            # stays the primary (holo) split so early-stop tracks affinity-relevant reconstruction.
            extra_rd = cfg.dset.get("extra_resample_dir", "")
            if extra_rd and split == "train":
                ek = dict(density_kwargs)
                ek["resample_dir"] = extra_rd
                ek["data_file"] = cfg.dset.get("extra_data_file", density_kwargs["data_file"])
                ek["subset_n"] = cfg.dset.get("extra_subset_n", None)   # None → all extra tuples
                extra = DatasetCrossDockedDensity(**ek)
                from torch.utils.data import ConcatDataset
                cat = ConcatDataset([primary, extra])
                print(f"[apoholo] ConcatDataset: primary(holo) {len(primary):,} + extra(apo) "
                      f"{len(extra):,} = {len(cat):,}", flush=True)
                return cat
            return primary
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
            # Opt-in: apply the pretraining density recipe (e.g. the PLINDER arcsinh+z from the
            # pocket tower's resample.json) to RAW crops instead of local normalize_crop, so the
            # frozen adapter encoder sees density on its pretraining scale.
            density_norm_recipe=cfg.dset.get("density_norm_recipe", ""),
            subset_n=cfg.dset.get("subset_n", None),
            subset_xray_only=cfg.dset.get("subset_xray_only", False),
            subset_val_n=cfg.dset.get("subset_val_n", None),
            # Master gradmag toggle lives at the top level (next to input_mode)
            # so the same flag drives both pre-training and downstream runs.
            return_gradmag=bool(cfg.get("with_gradmag", False)),
            # Optional precomputed (noise) gradmag source for the density-ablation
            # control; "" → derive ‖∇ρ‖ on the fly as usual.
            gradmag_crops_dir=cfg.dset.get("gradmag_crops_dir", ""),
            # Train/val source file; override (e.g. data_train_v7.pt) for the
            # combined CrossDocked∪PDBbind pretraining corpus.
            data_file=cfg.dset.get("data_file", "data_train.pt"),
        )
    else:
        raise NotImplementedError(f"Dataset '{name}' not implemented")


def create_dataloaders(cfg, distributed: bool = False, rank: int = 0, world_size: int = 1) -> tuple:
    """
    Create data loaders for training, validation, and sampling.

    Args:
        cfg (Config): Configuration object containing dataset and training parameters.
        distributed: when True, use DistributedSampler for the train loader so
                     each rank sees a disjoint shard of the dataset. Val and
                     sampling loaders are not sharded — caller should run those
                     on rank 0 only.
        rank, world_size: DDP rank metadata (ignored when distributed=False).

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

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dset_train, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        loader_train = torch.utils.data.DataLoader(
            dset_train,
            batch_size=cfg.bsz,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
    else:
        loader_train = torch.utils.data.DataLoader(
            dset_train,
            batch_size=cfg.bsz,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

    # create val loader (rank 0 only when distributed; not sharded)
    dset_val = _make_dataset(cfg, split="val", aug=False)
    loader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_size=cfg.bsz,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # create sampling loader (rank 0 only when distributed)
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
