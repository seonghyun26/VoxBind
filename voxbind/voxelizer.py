import os
import time

import numpy as np
import torch

from copy import deepcopy
from functools import partial
from scipy import ndimage as ndi
from pyuul import VolumeMaker


class Voxelizer(torch.nn.Module):
    """
    Voxelizer module for converting molecular structures to voxel representations.

    Args:
        grid_dim (int): The dimension of the voxel grid (default: 64).
        resolution (float): The resolution of the voxel grid (default: 0.25).
        radius (float): The radius used for voxelization (default: 0.5).
        cubes_around (int): The number of cubes around each atom used for voxelization (default: 8).
        device (str): The device to use for computation (default: "cuda").

    Attributes:
        grid_dim (int): The dimension of the voxel grid.
        device (str): The device used for computation.
        radius (float): The radius used for voxelization.
        resolution (float): The resolution of the voxel grid.
        cubes_around (int): The number of cubes around each atom used for voxelization.
        vol_maker (VolumeMaker.Voxels): The voxelization module.

    """

    def __init__(
            self,
            grid_dim: int = 64,
            resolution: float = 0.25,
            radius: float = 0.5,
            cubes_around: int = 8,
            device="cuda",
            backend: str = "pyuul",
    ):
        super(Voxelizer, self).__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.radius = radius
        self.resolution = resolution
        self.cubes_around = cubes_around
        self.backend = backend

        if backend == "pyuul":
            self.vol_maker = VolumeMaker.Voxels(device=device, sparse=False)
        elif backend == "torch":
            # Pre-compute neighborhood offsets once; stored as a buffer so they
            # move with the module and are never re-allocated during forward passes.
            r = torch.arange(-cubes_around, cubes_around + 1, dtype=torch.int32, device=device)
            dz, dy, dx = torch.meshgrid(r, r, r, indexing="ij")
            offsets = torch.stack([dx.flatten(), dy.flatten(), dz.flatten()], dim=1)
            self.register_buffer("_offsets", offsets)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'pyuul' or 'torch'.")

    def forward(self, batch: list, num_channels: int = 7) -> torch.Tensor:
        """
        Forward pass of the Voxelizer module.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """
        return self.mol2vox(batch, num_channels=num_channels)

    def mol2vox(self, batch: list, num_channels: int = 7) -> torch.Tensor:
        """
        Convert a batch of molecular structures to voxel representations.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """

        # H2D first (non_blocking so transfers overlap with GPU compute on the prefetch stream)
        batch = {
            "coords": batch["coords"].to(self.device, non_blocking=True),
            "radius": batch["radius"].to(self.device, non_blocking=True),
            "atoms_channel": batch["atoms_channel"].to(self.device, non_blocking=True),
        }

        if self.backend == "torch":
            return self._torch_voxelize(batch, num_channels)

        # --- PyUUL backend (original implementation) ---
        # add dumb coords on GPU (avoids a CPU torch.cat before the transfer)
        batch = self._add_dumb_coords(batch)

        batch_sz = batch["coords"].shape[0]
        n_chuncks = 4 if batch_sz > 16 else 1
        chk = (batch_sz + n_chuncks - 1) // n_chuncks
        voxels = torch.empty(
            (batch_sz, num_channels, self.grid_dim, self.grid_dim, self.grid_dim),
            device=self.device,
        )
        for i in range(n_chuncks):
            start = i * chk
            end = min((i + 1) * chk, batch_sz)
            if start >= end:
                break
            voxels_ = self.vol_maker(
                batch["coords"][start:end],
                batch["radius"][start:end],
                batch["atoms_channel"][start:end],
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=num_channels,
            )
            # extract center box (and get rid of dumb coordinates)
            c = voxels_.shape[-1] // 2
            box_min, box_max = c - self.grid_dim // 2, c + self.grid_dim // 2
            voxels[start:end] = voxels_[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
            del voxels_

        return voxels

    def _torch_voxelize(self, batch: dict, num_channels: int) -> torch.Tensor:
        """
        Pure-PyTorch Gaussian voxelizer.

        Writes directly into a (B, C, G, G, G) output grid using scatter_add,
        avoiding the large intermediate grid that PyUUL allocates internally.

        Each atom contributes a Gaussian blob to its (2*cubes_around+1)^3
        neighborhood. Sigma (in voxel units) is derived per-atom from its radius:
            sigma = radius / resolution

        Padded atoms (atoms_channel >= num_channels) are masked out automatically.
        """
        coords = batch["coords"]           # (B, N, 3) float, Angstroms
        radius = batch["radius"]           # (B, N) float
        atoms_channel = batch["atoms_channel"]  # (B, N) float

        B, N, _ = coords.shape
        G = self.grid_dim
        res = self.resolution

        # Convert Angstrom coords → fractional grid voxel coords
        # Origin (0 Å) maps to grid center (G/2); range [-G/2·res, +G/2·res] Å
        grid_coords = coords / res + G * 0.5  # (B, N, 3)

        # Nearest voxel center for each atom — serves as the neighborhood anchor
        atom_cell = grid_coords.round().to(torch.int32)  # (B, N, 3)

        # Per-atom sigma^2 in voxel^2: sigma = radius / resolution
        sigma_sq = (radius / res).clamp(min=1e-3).pow(2).unsqueeze(2)  # (B, N, 1)

        # Valid atom mask: channel must be in [0, num_channels)
        atoms_channel_long = atoms_channel.long()
        valid = (atoms_channel_long >= 0) & (atoms_channel_long < num_channels)  # (B, N)

        offsets = self._offsets  # (K, 3) — pre-computed in __init__
        K = offsets.shape[0]

        output = torch.zeros(B, num_channels, G * G * G, device=self.device, dtype=coords.dtype)

        # Chunk over offsets to bound peak memory.
        # With chunk=64: peak ~ (B × N × 64) elements — safe even for large pockets.
        chunk = 64
        for k0 in range(0, K, chunk):
            offs = offsets[k0:k0 + chunk]   # (C, 3)
            C = offs.shape[0]

            # Neighbor voxel integer coordinates: (B, N, C, 3)
            nbr = atom_cell.unsqueeze(2) + offs.view(1, 1, C, 3)

            # Squared distance from atom center to each neighbor cell center (voxel units)
            dist_sq = ((grid_coords.unsqueeze(2) - nbr.float()) ** 2).sum(-1)  # (B, N, C)

            # Gaussian weight; per-atom sigma via broadcasting
            weights = torch.exp(-dist_sq / (2.0 * sigma_sq))  # (B, N, C)

            # Zero out invalid atoms and out-of-bounds neighbors
            nx, ny, nz = nbr[..., 0], nbr[..., 1], nbr[..., 2]
            in_bounds = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G) & (nz >= 0) & (nz < G)
            weights = weights * (valid.unsqueeze(2) & in_bounds).to(coords.dtype)

            # Flat linear voxel index (clamped; masked weights are 0 so safe)
            lin = (nz * G * G + ny * G + nx).to(torch.int64).clamp(0, G * G * G - 1)
            lin_flat = lin.view(B, N * C)   # (B, N*C)

            # Scatter one channel at a time to keep memory low
            for c in range(num_channels):
                chan_w = (weights * (atoms_channel_long == c).unsqueeze(2).to(coords.dtype))
                output[:, c].scatter_add_(1, lin_flat, chan_w.view(B, N * C))

        return output.view(B, num_channels, G, G, G)

    def vox2mol(
        self,
        voxels: torch.Tensor,
        refine: bool = True,
        center_coords: torch.Tensor = None
    ) -> list:
        """
        Convert voxel representations back to molecular structures.

        Args:
            voxels (torch.Tensor): The input voxel representations.
            refine (bool): Whether to refine the coordinates using optimization (default: True).
            center_coords (torch.Tensor): The center coordinates for recentering the molecular structures.

        Returns:
            list: The reconstructed molecular structures.

        """
        assert len(voxels.shape) == 5

        peak_started = time.perf_counter()

        # Initialize coordinates with peak detection. Keeping this operation on
        # the GPU avoids 100 individual D2H copies + scipy maximum_filter calls.
        use_gpu_peaks = voxels.is_cuda and os.environ.get("VOXBIND_GPU_PEAKS", "1") != "0"
        if use_gpu_peaks:
            candidates = get_atom_coords_batch(
                voxels, rad=self.radius, resolution=self.resolution
            )
        else:
            candidates = [
                get_atom_coords(voxel.cpu(), rad=self.radius, resolution=self.resolution)
                for voxel in voxels
            ]

        mol_inits = []
        voxel_inits = []
        for voxel, mol_init in zip(voxels, candidates):
            if mol_init is not None and mol_init["coords"].shape[1] < 200:
                mol_inits.append(mol_init)
                voxel_inits.append(voxel.unsqueeze(0))

        peak_seconds = time.perf_counter() - peak_started
        if len(mol_inits) == 0:
            return None

        if not refine:
            mols = recenter_mols(mol_inits, center_coords)
            return mols
        voxel_inits = torch.cat(voxel_inits, axis=0)

        # refine coords
        optim_factory = partial(
            torch.optim.LBFGS, history_size=10, max_iter=4, line_search_fn="strong_wolfe",
        )

        refine_started = time.perf_counter()
        refine_batch = max(1, int(os.environ.get("VOXBIND_REFINE_BATCH", "25")))
        if refine_batch == 1:
            mols = self._refine_coords(mol_inits, voxel_inits, optim_factory, maxiter=10)
        else:
            mols = self._refine_coords_batched(
                mol_inits,
                voxel_inits,
                optim_factory,
                batch_size=refine_batch,
                maxiter=10,
            )
        refine_seconds = time.perf_counter() - refine_started
        del voxels, mol_inits
        torch.cuda.empty_cache()

        print(
            f"[vox2mol] input={len(candidates)} peaks={len(voxel_inits)} "
            f"peak_s={peak_seconds:.2f} refine_s={refine_seconds:.2f} "
            f"refine_batch={refine_batch}"
        )

        mols = recenter_mols(mols, center_coords)

        return mols

    def _refine_coords_batched(
        self,
        mol_inits: list,
        voxels: torch.Tensor,
        optim_factory,
        batch_size: int = 25,
        tol: float = 1e-6,
        maxiter: int = 15,
    ) -> list:
        """Refine several generated molecules in one GPU optimization batch."""
        assert len(voxels.shape) == 5, "voxels need a batch dimension"

        mols = []
        for start in range(0, len(mol_inits), batch_size):
            stop = min(start + batch_size, len(mol_inits))
            chunk = mol_inits[start:stop]
            target_voxels = voxels[start:stop]
            n_atoms = [mol["coords"].shape[1] for mol in chunk]
            max_atoms = max(n_atoms)
            device = target_voxels.device

            coords = target_voxels.new_full((len(chunk), max_atoms, 3), 999.0)
            channels = target_voxels.new_full((len(chunk), max_atoms), 999.0)
            radii = target_voxels.new_full((len(chunk), max_atoms), 999.0)
            for idx, mol in enumerate(chunk):
                count = n_atoms[idx]
                coords[idx, :count] = mol["coords"][0].to(device)
                channels[idx, :count] = mol["atoms_channel"][0].to(device)
                radii[idx, :count] = mol["radius"][0].to(device)

            coords.requires_grad_(True)
            optimizer = optim_factory([coords])

            def closure():
                optimizer.zero_grad()
                fitted = self.forward(
                    {"coords": coords, "atoms_channel": channels, "radius": radii}
                )
                loss = torch.nn.functional.mse_loss(target_voxels, fitted)
                loss.backward()
                return loss

            loss = 1e10
            failed = False
            for _ in range(maxiter):
                try:
                    previous = loss
                    loss = optimizer.step(closure)
                except Exception as exc:
                    print(f"batched coordinate refinement failed; using initial coordinates: {exc}")
                    failed = True
                    break
                if abs(loss.item() - previous) < tol:
                    break

            for idx, mol_init in enumerate(chunk):
                count = n_atoms[idx]
                refined = mol_init["coords"] if failed else coords[idx : idx + 1, :count]
                mols.append(
                    {
                        "coords": refined.detach().cpu(),
                        "atoms_channel": mol_init["atoms_channel"].detach().cpu(),
                        "radius": mol_init["radius"].detach().cpu(),
                    }
                )

        return mols

    def _refine_coords(
        self,
        mol_inits: list,
        voxels: torch.Tensor,
        optim_factory,
        tol: float = 1e-6,
        maxiter: int = 15,
        callback=None
    ) -> list:
        """
        Refine the coordinates of molecular structures using optimization.

        Args:
            mol_inits (list): The initial molecular structures.
            voxels (torch.Tensor): The voxel representations.
            optim_factory: The optimization algorithm used for refinement.
            tol (float): The tolerance for convergence (default: 1e-6).
            maxiter (int): The maximum number of iterations (default: 15).
            callback: The callback function for monitoring the refinement process.

        Returns:
            list: The refined molecular structures.

        """
        assert len(voxels.shape) == 5, "voxels need to have dimension 5 (including the batch dim.)"

        mols = []
        for i in range(voxels.shape[0]):
            mol_init = mol_inits[i]
            voxel = voxels[i].unsqueeze(0)

            mol = deepcopy(mol_init)
            mol["coords"].requires_grad = True

            optimizer = optim_factory([mol["coords"]])

            def closure():
                optimizer.zero_grad()
                voxel_fit = self.forward(mol)
                loss = torch.nn.functional.mse_loss(voxel, voxel_fit)
                loss.backward()
                return loss

            loss = 1e10
            for _ in range(maxiter):
                try:
                    prev_loss = loss
                    loss = optimizer.step(closure)
                except Exception:
                    print(
                        "refine coords diverges, so use initial cordinates...",
                        f"(coords min: {mol['coords'].min().item()}, max: {mol['coords'].max().item()})"
                    )
                    mol = deepcopy(mol_init)
                    break

                if callback is not None:
                    callback(mol)

                if abs(loss.item() - prev_loss) < tol:
                    break

            mols.append({
                "coords": mol["coords"].detach().cpu(),
                "atoms_channel": mol["atoms_channel"].detach().cpu(),
                "radius": mol["radius"].detach().cpu(),
            })

        return mols

    def _add_dumb_coords(self, batch: dict) -> dict:
        """
        Add dumb coordinates to the input batch for centering the ligand and pocket voxel.

        Args:
            batch (dict): The input batch of molecular structures.

        Returns:
            dict: The modified batch with dumb coordinates.

        """
        bsz = batch["coords"].shape[0]
        coords = batch["coords"]
        atoms_channel = batch["atoms_channel"]
        radius = batch["radius"]

        return {
            "coords": torch.cat(
                (coords, coords.new_full((bsz, 1, 3), -25), coords.new_full((bsz, 1, 3), 25)), 1
            ),
            "atoms_channel": torch.cat(
                (atoms_channel, atoms_channel.new_zeros((bsz, 2))), 1
            ),
            "radius": torch.cat(
                (radius, radius.new_full((bsz, 2), .5)), 1
            )
        }


########################################################################################
# aux functions
def local_maxima(
    data: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Find local maxima in a 3D array.

    Args:
        data (np.ndarray): The input 3D array.
        order (int, optional): The order of the local maxima. Defaults to 1.

    Returns:
        np.ndarray: The modified 3D array with local maxima set to 0.
    """
    data = data.numpy()
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel: torch.Tensor) -> torch.Tensor:
    """
    Find peaks in a voxel.

    Args:
        voxel (torch.Tensor): Input voxel.

    Returns:
        torch.Tensor: Peaks found in the voxel.
    """
    # default 0.25 (upstream); override at runtime via env var for diagnostics
    threshold = float(os.environ.get("VOXBIND_FIND_PEAKS_THRESHOLD", 0.25))
    print(
        f"[find_peaks] voxel stats before threshold={threshold}: "
        f"min={voxel.min():.4f}  max={voxel.max():.4f}  "
        f"mean={voxel.mean():.4f}  "
        f"p50={voxel.quantile(0.50):.4f}  p90={voxel.quantile(0.90):.4f}  "
        f"p95={voxel.quantile(0.95):.4f}  p99={voxel.quantile(0.99):.4f}  "
        f"frac>{threshold}={(voxel > threshold).float().mean().item():.4f}  "
        f"n_voxels_above={(voxel > threshold).sum().item()}"
    )
    voxel[voxel < threshold] = 0
    voxel = voxel.squeeze().clone()
    peaks = []
    for channel_idx in range(voxel.shape[0]):
        vox_in = voxel[channel_idx]
        peaks_ = local_maxima(vox_in, 1)
        peaks_ = torch.Tensor(peaks_).unsqueeze(0)
        peaks.append(peaks_)
    peaks = torch.concat(peaks, axis=0)
    return peaks


def get_atom_coords(
    grid: torch.Tensor,
    rad: float = 0.5,
    resolution: float = 0.25
) -> dict:
    """
    Get the coordinates of atoms from a grid.

    Args:
        grid (torch.Tensor): The input grid.
        rad (float, optional): The radius of the atoms. Defaults to 0.5.
        resolution (float, optional): The resolution of the grid. Defaults to 0.25.

    Returns:
        dict: A dictionary containing the coordinates, atom channels, and radii of the atoms.
    """
    peaks = find_peaks(grid.cpu())
    coords = []
    atoms_channel = []
    radius = []
    grid_dim = peaks.shape[-1]

    for channel_idx in range(peaks.shape[0]):
        px, py, pz = torch.where(peaks[channel_idx] > 0)
        px, py, pz = px.float(), py.float(), pz.float()
        coords.append(torch.cat([px.unsqueeze(1), py.unsqueeze(1), pz.unsqueeze(1)], axis=1))
        atoms_channel.append(torch.Tensor(px.shape[0]).fill_(channel_idx))
        radius.append(torch.Tensor(px.shape[0]).fill_(rad))
    coords = (torch.cat(coords, 0).unsqueeze(0) - (grid_dim - 1) / 2) * resolution

    if coords.shape[1] == 0:
        return None

    mol = {
        "coords": coords,
        "atoms_channel": torch.cat(atoms_channel, 0).unsqueeze(0),
        "radius": torch.cat(radius, 0).unsqueeze(0),
    }

    return mol


def get_atom_coords_batch(
    grids: torch.Tensor,
    rad: float = 0.5,
    resolution: float = 0.25,
) -> list:
    """Extract per-channel local maxima for a whole CUDA batch."""
    if len(grids.shape) != 5:
        raise ValueError("grids must have shape (batch, channels, x, y, z)")

    threshold = float(os.environ.get("VOXBIND_FIND_PEAKS_THRESHOLD", 0.25))
    pooled = torch.nn.functional.max_pool3d(grids, kernel_size=3, stride=1, padding=1)
    peak_mask = (grids >= threshold) & (grids == pooled)
    peak_indices = peak_mask.nonzero(as_tuple=False)
    grid_dim = grids.shape[-1]
    molecules = []

    for batch_idx in range(grids.shape[0]):
        selected = peak_indices[:, 0] == batch_idx
        indices = peak_indices[selected]
        if indices.shape[0] == 0:
            molecules.append(None)
            continue

        channels = indices[:, 1].to(dtype=grids.dtype)
        coords = indices[:, 2:5].to(dtype=grids.dtype)
        coords = (coords - (grid_dim - 1) / 2) * resolution
        molecules.append(
            {
                "coords": coords.unsqueeze(0),
                "atoms_channel": channels.unsqueeze(0),
                "radius": torch.full(
                    (1, indices.shape[0]), rad, device=grids.device, dtype=grids.dtype
                ),
            }
        )

    counts = [0 if mol is None else mol["coords"].shape[1] for mol in molecules]
    if counts:
        print(
            f"[gpu_peaks] batch={len(counts)} threshold={threshold} "
            f"atoms(min/mean/max)={min(counts)}/{sum(counts) / len(counts):.1f}/{max(counts)}"
        )
    return molecules


def recenter_mols(mols: list, center_coords: torch.Tensor) -> list:
    """
    Recenter the molecules based on the given center coordinates.

    Args:
        mols (list): List of molecules.
        center_coords (torch.Tensor): Center coordinates.

    Returns:
        list: List of recentered molecules.
    """
    centered_mols = []
    for mol in mols:
        coords = mol["coords"]
        if center_coords is not None:
            center_coords_ = center_coords.unsqueeze(0).tile((1, coords.shape[0], 1))
            coords += center_coords_
        centered_mols.append({
            "coords": coords,
            "atoms_channel": mol["atoms_channel"],
            "radius": mol["radius"]
        })

    return centered_mols
