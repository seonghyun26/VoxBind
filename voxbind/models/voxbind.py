import math
import random
import torch
from typing import Tuple, Union, List

from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS
from voxbind.models.unet3d import UNet3D, ResidualBlock


class VoxBind(torch.nn.Module):
    def __init__(
        self,
        n_channels_ligand: int = N_LIGAND_ELEMENTS,
        n_channels_pocket: int = N_POCKET_ELEMENTS,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        with_density: bool = False,
        verbose: bool = False
    ):
        """
        VoxBind is a class that represents the VoxBind model.

        Args:
            n_channels_ligand (int): Number of channels for the ligand input. Defaults to N_LIGAND_ELEMENTS.
            n_channels_pocket (int): Number of channels for the pocket input. Defaults to N_POCKET_ELEMENTS.
            n_channels (int): Number of channels in the model. Defaults to 64.
            ch_mults (Union[Tuple[int, ...], List[int]]): Channel multipliers for each block in the UNet3D.
            Defaults to (1, 2, 2, 4).
            is_attn (Union[Tuple[bool, ...], List[int]]): Attention flag for each block in the UNet3D.
            Defaults to (False, False, True, True).
            n_blocks (int): Number of blocks in the UNet3D. Defaults to 2.
            n_groups (int): Number of groups in the ResidualBlock. Defaults to 32.
            dropout (float): Dropout rate. Defaults to 0.1.
            smooth_sigma (float): Sigma value for smoothing. Defaults to 0.0.
            with_density (bool): If True, add a density_encoder branch that accepts a
                single-channel Gaussian density map (e.g. from pdb2vol). The branch is
                summed with the ligand and pocket encodings. Checkpoint-compatible: old
                checkpoints load fine with with_density=False (default). Defaults to False.
            verbose (bool): Flag to print the number of parameters in the model. Defaults to False.
        """
        super().__init__()

        self.unet3d = UNet3D(
            n_channels // 2,
            n_channels,
            n_channels,
            ch_mults,
            is_attn,
            n_blocks,
            n_groups,
            dropout,
            smooth_sigma,
            verbose=False
        )

        self.smooth_sigma = smooth_sigma
        self.n_channels_ligand = n_channels_ligand
        self.n_channels_pocket = n_channels_pocket
        self.with_density = with_density

        self.ligand_encoder = ResidualBlock(
            n_channels_ligand, n_channels // 2, n_groups=0, dropout=0
        )
        self.pocket_encoder = ResidualBlock(
            n_channels_pocket, n_channels // 2, n_groups=0, dropout=0
        )
        if with_density:
            # Lift 1-channel density to C/2, then a full ResidualBlock with GroupNorm.
            # GroupNorm can't be applied to 1 input channel, so we use a plain conv first.
            _dens_groups = min(n_groups, n_channels // 2) if n_groups > 0 else 0
            self.density_encoder = torch.nn.Sequential(
                torch.nn.Conv3d(1, n_channels // 2, kernel_size=3, padding=1),
                torch.nn.SiLU(),
                ResidualBlock(n_channels // 2, n_channels // 2,
                              n_groups=_dens_groups, dropout=dropout),
            )
            # Projects cat([ligand+pocket, density], dim=1) → C/2.
            # Zero-init so initial output is 0: x = x_backbone + 0 = x_backbone.
            # Training grows the density correction from zero, keeping early
            # iterations stable and identical to the baseline.
            self.density_proj = torch.nn.Conv3d(
                n_channels, n_channels // 2, kernel_size=1
            )
            torch.nn.init.zeros_(self.density_proj.weight)
            torch.nn.init.zeros_(self.density_proj.bias)

        self.final_ligand = torch.nn.Conv3d(
            n_channels, n_channels_ligand, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )

        if verbose:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f">> model has {(n_params/1e6):.02f}M parameters")

    def forward(
        self,
        ligand: torch.Tensor,
        pocket: torch.Tensor,
        density: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the VoxBind model.

        Args:
            ligand (torch.Tensor): Input tensor for the ligand. Shape (B, 7, G, G, G).
            pocket (torch.Tensor): Input tensor for the pocket. Shape (B, 4, G, G, G).
            density (torch.Tensor, optional): Single-channel Gaussian density map from
                pdb2vol, shape (B, 1, G, G, G). Only used when with_density=True.
                Defaults to None (standard baseline behaviour).

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        x = self.ligand_encoder(ligand) + self.pocket_encoder(pocket)
        if self.with_density and density is not None:
            x_dens = self.density_encoder(density)
            x = x + self.density_proj(torch.cat([x, x_dens], dim=1))

        x = self.unet3d(x, None)
        x = self.unet3d.act(x)
        x = self.final_ligand(x)

        return x

    def score(self, y: torch.Tensor, pocket: torch.Tensor,
              density: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the score function.

        Args:
            y (torch.Tensor): The y tensor.
            pocket (torch.Tensor): The pocket tensor.
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The calculated base score tensor.
        """
        xhat = self.forward(y, pocket, density=density)
        return (xhat - y) / (self.smooth_sigma ** 2)

    ####################################################################################
    # conditional walk-jump sampling methods
    def initialize_y_v(
        self,
        vox_pockets: torch.Tensor,
        ligand_gt: torch.Tensor,
        smooth_sigma: float,
        chain_init: str = "denovo"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the y and v tensors for the walk-jump sampling.

        Args:
            vox_pockets (torch.Tensor): The vox_pockets tensor.
            ligand_gt (torch.Tensor): The ligand_gt tensor.
            smooth_sigma (float): The smooth_sigma value.
            chain_init (str, optional): The chain initialization method. Defaults to "denovo".

        Returns:
            torch.Tensor: The initialized y tensor.
            torch.Tensor: The initialized v tensor.
        """
        n_channels = N_LIGAND_ELEMENTS
        grid_dim = vox_pockets.shape[-1]
        n_chains = vox_pockets.shape[0]

        # gaussian noise
        y = torch.cuda.FloatTensor(n_chains, n_channels, grid_dim, grid_dim, grid_dim)
        y.normal_(0, smooth_sigma)

        if chain_init == "ligand":
            y += ligand_gt
        elif chain_init == "denovo":
            mask_pocket = get_pocket_mask(vox_pockets, n_channels)
            noise = torch.cuda.FloatTensor(y.shape).uniform_(0, 1)
            noise[mask_pocket] = 0
            y += noise

        return y, torch.zeros_like(y)

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor, pocket: torch.Tensor,
                      density: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the jump step of the walk-jump sampling.

        Args:
            y (torch.Tensor): The y tensor.
            pocket (torch.Tensor): The pocket tensor.
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The estimated "clean" samples xhats.
        """
        return self.forward(y, pocket, density=density)

    @torch.no_grad()
    def wjs_walk_steps(
        self,
        y: torch.Tensor,
        v: torch.Tensor,
        pocket: torch.Tensor,
        mask: torch.Tensor = None,
        n_steps: int = 100,
        friction: float = 1.,
        lipschitz: float = 1.,
        density: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs `n_steps` walk steps of the walk-jump sampling.

        Args:
            y (torch.Tensor): The y tensor.
            v (torch.Tensor): The v tensor.
            pocket (torch.Tensor): The pocket tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
            n_steps (int, optional): The number of steps. Defaults to 100.
            friction (float, optional): The friction value. Defaults to 1..
            lipschitz (float, optional): The lipschitz value. Defaults to 1..
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The updated y tensor.
            torch.Tensor: The updated v tensor.
        """
        gamma = friction
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(-gamma)  # gamma = "effective friction"
        zeta2 = math.exp(-2 * gamma)
        delta = self.smooth_sigma / 2

        for _ in range(n_steps):
            with torch.no_grad():
                y += delta * v / 2
            psi = self.score(y, pocket, density=density)
            with torch.no_grad():
                noise = torch.randn_like(y)
                if mask is not None:
                    noise[mask] = 0.
                    psi[mask] = 0.
                v += u * delta * psi / 2
                v = zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * noise  # v_{t+1}
                y += delta * v / 2
        torch.cuda.empty_cache()
        return y, v

    def sample(
            self,
            pocket: torch.Tensor,
            ligand: torch.Tensor,
            warmup_wjs: int = 400,
            steps: int = 100,
            max_steps: int = 100,
            chain_init: str = "denovo",
            mask_pocket: bool = True,
            n_chains: int = 48,
            threshold: float = .2,
            density: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs the walk-jump sampling.

        Args:
            pocket (torch.Tensor): The pocket tensor.
            ligand (torch.Tensor): The ligand tensor.
            warmup_wjs (int, optional): The number of warmup wjs steps. Defaults to 0.
            steps (int, optional): The number of steps per iteration. Defaults to 100.
            max_steps (int, optional): The maximum number of steps. Defaults to 100.
            chain_init (str, optional): The chain initialization method. Defaults to "denovo".
            mask_pocket (bool, optional): Whether to mask the pocket. Defaults to True.
            threshold (float, optional): The threshold value. Defaults to .2.
            density (torch.Tensor, optional): X-ray density map (1,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The generated voxels.
        """
        self.eval()
        if ligand is None:
            chain_init = "denovo"

        # rotate pocket and ligand
        N = n_chains if n_chains < 10 else 10
        assert n_chains % N == 0, "n_chains must be divisible by N"
        pocket = pocket.repeat(N, 1, 1, 1, 1)
        ligand = ligand.repeat(N, 1, 1, 1, 1)
        if density is not None:
            density = density.repeat(N, 1, 1, 1, 1)
        rand_rots = [
            [random.choice([[2, 3], [3, 4], [2, 4], [3, 2], [4, 3], [4, 2], None]) for _ in range(N)],
            [random.randint(1, 4) for _ in range(N)]
        ]
        pocket = rotate_batch_voxel_grids(pocket, rand_rots)
        ligand = rotate_batch_voxel_grids(ligand, rand_rots)

        y, v = self.initialize_y_v(pocket, ligand, self.smooth_sigma, chain_init)

        # warm up
        if warmup_wjs > 0:
            mask_warmup = get_pocket_mask(pocket, n_channels=N_LIGAND_ELEMENTS)
            y, v = self.wjs_walk_steps(y, v, pocket, mask_warmup, warmup_wjs,
                                       density=density)
        y, v = y.repeat(n_chains // N, 1, 1, 1, 1), v.repeat(n_chains // N, 1, 1, 1, 1)
        pocket, ligand = pocket.repeat(n_chains // N, 1, 1, 1, 1), ligand.repeat(n_chains // N, 1, 1, 1, 1)
        if density is not None:
            density = density.repeat(n_chains // N, 1, 1, 1, 1)
        rand_rots[0] = rand_rots[0] * (n_chains // N)
        rand_rots[1] = rand_rots[1] * (n_chains // N)

        # mask
        mask = None
        if mask_pocket:
            mask = get_pocket_mask(pocket, n_channels=N_LIGAND_ELEMENTS)

        # sample
        voxels = []
        for _ in range(0, max_steps, steps):
            # walk `steps` steps
            y, v = self.wjs_walk_steps(y, v, pocket, mask, steps, density=density)

            # jump step
            xhats = self.wjs_jump_step(y, pocket, density=density)

            nz = (xhats > 0).float().mean().item()
            # quantile() rejects tensors > ~16M elements; subsample for stats.
            flat = xhats.flatten()
            if flat.numel() > 1_000_000:
                idx = torch.randint(flat.numel(), (1_000_000,), device=flat.device)
                stat_sample = flat[idx]
            else:
                stat_sample = flat
            print(
                f"[sample] xhats stats before threshold={threshold:.2f}: "
                f"min={xhats.min():.4f}  max={xhats.max():.4f}  "
                f"mean={xhats.mean():.4f}  "
                f"p25={stat_sample.quantile(0.25):.4f}  p50={stat_sample.quantile(0.50):.4f}  "
                f"p75={stat_sample.quantile(0.75):.4f}  p90={stat_sample.quantile(0.90):.4f}  "
                f"p95={stat_sample.quantile(0.95):.4f}  p99={stat_sample.quantile(0.99):.4f}  "
                f"frac>0={nz:.4f}  "
                f"frac>{threshold:.2f}={(xhats > threshold).float().mean().item():.4f}"
            )
            xhats[xhats < threshold] = 0
            xhats = unrotate_voxel_grids(xhats, rand_rots)
            voxels.append(xhats)

        voxels = torch.concat(voxels, axis=0)

        return voxels


def rotate_batch_voxel_grids(batch: torch.Tensor, rand_rots: list):
    """
    Rotate a batch of voxel grids based on random rotations.

    Args:
        batch (torch.Tensor): The input batch of voxel grids.
        rand_rots (list): A list of random rotations for each voxel grid in the batch.

    Returns:
        torch.Tensor: The rotated batch of voxel grids.
    """
    batch_sz = batch.shape[0]
    rot_batch = []
    for i in range(batch_sz):
        rand_rot, n_rots = rand_rots[0][i], rand_rots[1][i]
        if rand_rot is not None:
            # rot_i = torch.rot90(batch[i:i + 1], k=1, dims=rand_rot)
            rot_i = batch[i:i + 1]
            for j in range(n_rots):
                rot_i = torch.rot90(rot_i, k=1, dims=rand_rot)
        else:
            rot_i = batch[i:i + 1].clone()
        rot_batch.append(rot_i)

    return torch.cat(rot_batch, 0)


def unrotate_voxel_grids(xhats: torch.Tensor, rand_rots: list):
    """
    Unrotates the voxel grids based on the given random rotations.

    Args:
        xhats (torch.Tensor): The input voxel grids.
        rand_rots (list): The list of random rotations.

    Returns:
        torch.Tensor: The unrotated voxel grids.
    """
    unrot_gen_vox_ligands = []
    for i, xhat in enumerate(xhats):
        rand_rot = rand_rots[0][i]
        if rand_rot is None:
            unrot_i = xhat.unsqueeze(0)
        else:
            n_rots = rand_rots[1][i]
            unrot_i = xhat.unsqueeze(0)
            for j in range(n_rots):
                unrot_i = torch.rot90(unrot_i, k=1, dims=rand_rot[::-1])
        unrot_gen_vox_ligands.append(unrot_i)
    unrot_gen_vox_ligands = torch.concat(unrot_gen_vox_ligands)
    return unrot_gen_vox_ligands


def get_pocket_mask(pocket: torch.Tensor, n_channels: int = 7):
    """
    Generate a mask for the given pocket tensor.

    Args:
        pocket (torch.Tensor): The input pocket tensor.
        n_channels (int, optional): The number of channels in the mask. Defaults to 7.

    Returns:
        torch.Tensor: The generated mask tensor.
    """
    mask = ((pocket > 0).float().sum(1) > 0)
    # mask = ndimage.binary_dilation(mask.cpu())
    mask = torch.Tensor(mask).unsqueeze(1).repeat(1, n_channels, 1, 1, 1).cuda()
    return mask.bool()
