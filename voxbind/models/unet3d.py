import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union, List

from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 16,
        dropout: float = 0.1,
    ):
        """
        Residual block module for the UNet3D model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int, optional): Number of groups for group normalization. Defaults to 16.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()

        self.use_norm = n_groups > 0
        # first norm + conv layer
        if self.use_norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        # second norm + conv layer
        if self.use_norm:
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_norm:
            h = self.norm1(x)
            h = self.act1(h)
        else:
            h = self.act1(x)
        h = self.conv1(h)

        if self.use_norm:
            h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 16
    ):
        """
        Initializes an AttentionBlock.

        Args:
            n_channels (int): Number of input channels.
            n_heads (int, optional): Number of attention heads. Defaults to 1.
            d_k (int, optional): Dimensionality of the key and query vectors. Defaults to None.
                If None, it is set equal to n_channels.
            n_groups (int, optional): Number of groups for group normalization. Defaults to 16.
                If n_groups <= 0, no normalization is applied.
        """
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.use_norm = n_groups > 0
        self.n_heads = n_heads
        self.d_k = d_k

        if self.use_norm:
            self.norm = nn.GroupNorm(n_groups, n_channels)

        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_channels, height, width, depth).
        """
        batch_size, n_channels, height, width, depth = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd, bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum("bijh, bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width, depth)
        return res


class DensityCrossAttn(nn.Module):
    """UNet features (queries) attend to the FROZEN density ViT's patch tokens (keys/values).

    Why this exists: the `default`/`protein_first` fusions mix the density branch ONCE, at the
    UNet input, through a 1x1x1 conv -- so voxel (i,j,k) only ever sees the density feature at
    (i,j,k), linearly, and no block inside the UNet can consult the density again. Cross-attention
    makes the interaction non-local and content-addressed: a voxel can query the whole pocket.

    Cost is why it lives only at the attention levels. With patch=8, G=64 the encoder emits
    nG*8^3 tokens (1536 for groups [7,4,2], 1024 if the always-zero ligand group is dropped).
    Queries scale as S^3, so Q*K is 4.0e8 at 64^3 but 6.3e6 at 16^3 -- the latter is 2.7x CHEAPER
    than the self-attention already running at that level. Hence: levels with `has_attn` only.

    `out` is ZERO-INIT, so at step 0 the block is exactly the identity and the model reproduces
    the density-free baseline bit-for-bit -- the same discipline as `density_proj`. This is not
    cosmetic: the `v3` fusion used a normal-init projection, overfit, and was dropped.
    """

    def __init__(self, n_channels: int, ctx_dim: int, n_heads: int = 4,
                 d_head: int = 32, n_groups: int = 16):
        super().__init__()
        inner = n_heads * d_head
        self.n_heads, self.d_head = n_heads, d_head
        self.norm = nn.GroupNorm(min(n_groups, n_channels), n_channels)
        self.to_q = nn.Linear(n_channels, inner, bias=False)
        self.to_kv = nn.Linear(ctx_dim, 2 * inner, bias=False)
        self.out = nn.Linear(inner, n_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        if ctx is None:                       # density-free forward -> exact no-op
            return x
        B, C, H, W, D = x.shape
        h = self.norm(x).reshape(B, C, -1).permute(0, 2, 1)          # (B, S3, C)
        q = self.to_q(h).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.to_kv(ctx).chunk(2, dim=-1)                       # (B, T, inner) each
        k = k.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)                   # flash; no (S3 x T) in HBM
        o = o.transpose(1, 2).reshape(B, -1, self.n_heads * self.d_head)
        o = self.out(o).permute(0, 2, 1).reshape(B, C, H, W, D)
        return x + o


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool,
                 dropout: float, ctx_dim: int = None):
        """
        DownBlock class represents a block in the down-sampling path of a U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int): Number of groups for group normalization.
            has_attn (bool): Whether to include attention mechanism in the block.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()
        # Cross-attention to the frozen density tokens. Only at levels that already pay for
        # attention -- see DensityCrossAttn for the cost argument. None elsewhere so the
        # module isn't built at all (unused params are a DDP hazard).
        self.xattn = (DensityCrossAttn(out_channels, ctx_dim, n_groups=n_groups)
                      if (has_attn and ctx_dim) else None)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res(x)
        x = self.attn(x)
        if self.xattn is not None:
            x = self.xattn(x, ctx)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool,
                 dropout: float, ctx_dim: int = None):
        """
        UpBlock is a module that represents an upsampling block in a 3D U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int): Number of groups for group normalization.
            has_attn (bool): Whether to include attention mechanism in the block.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()
        # Cross-attention to the frozen density tokens. Only at levels that already pay for
        # attention -- see DensityCrossAttn for the cost argument. None elsewhere so the
        # module isn't built at all (unused params are a DDP hazard).
        self.xattn = (DensityCrossAttn(out_channels, ctx_dim, n_groups=n_groups)
                      if (has_attn and ctx_dim) else None)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the UpBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res(x)
        x = self.attn(x)
        if self.xattn is not None:
            x = self.xattn(x, ctx)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, n_groups: int, dropout: float,
                 ctx_dim: int = None):
        """
        Initializes a MiddleBlock instance.

        Args:
            n_channels (int): Number of input and output channels.
            n_groups (int): Number of groups for group normalization.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)
        self.attn = AttentionBlock(n_channels, n_groups=n_groups)
        # deepest level (8^3 = 512 queries) -- the cheapest place to cross-attend by far
        self.xattn = DensityCrossAttn(n_channels, ctx_dim, n_groups=n_groups) if ctx_dim else None
        self.res2 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        """
        Performs forward pass through the MiddleBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res1(x)
        x = self.attn(x)
        if getattr(self, "xattn", None) is not None:
            x = self.xattn(x, ctx)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        """
        Upsample module that performs 3D transposed convolution to upsample the input tensor.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Upsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        """
        Downsample module that performs 3D convolution with stride 2.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after 3D convolution with stride 2.
        """
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        n_inp_channels: int = N_POCKET_ELEMENTS + N_LIGAND_ELEMENTS,
        n_out_channels: int = N_LIGAND_ELEMENTS,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        ctx_dim: int = None,
        verbose: bool = False
    ):
        """
        3D U-Net model for voxel-based binding affinity prediction.

        Args:
            n_inp_channels (int): Number of input channels.
            n_out_channels (int): Number of output channels.
            n_channels (int): Number of channels in the model.
            ch_mults (Union[Tuple[int, ...], List[int]]): Channel multipliers for each resolution level.
            is_attn (Union[Tuple[bool, ...], List[int]]): Attention flag for each resolution level.
            n_blocks (int): Number of blocks in each resolution level.
            n_groups (int): Number of groups for group normalization.
            dropout (float): Dropout rate.
            smooth_sigma (float): Standard deviation for Gaussian smoothing.
            verbose (bool): Whether to print the number of parameters in the model.
        """
        super().__init__()

        self.smooth_sigma = smooth_sigma
        n_resolutions = len(ch_mults)

        self.grid_projection = nn.Conv3d(n_inp_channels, n_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups, is_attn[i], dropout, ctx_dim))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_groups, dropout, ctx_dim)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout, ctx_dim))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout, ctx_dim))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        if n_groups > 0:
            self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, n_out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        if verbose:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f">> model has {(n_params/1e6):.02f}M parameters")

    def forward(self, ligand: torch.Tensor, pocket: torch.Tensor,
                ctx: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the UNet3D model.

        Args:
            ligand (torch.Tensor): Input tensor representing the ligand.
            pocket (torch.Tensor): Input tensor representing the pocket.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        if pocket is not None:
            x = torch.cat((ligand, pocket), axis=1)
        else:
            x = ligand
        x = self.grid_projection(x)

        h = [x]
        for m in self.down:
            # Downsample takes no context; Down/UpBlock forward it to their xattn (if built)
            x = m(x, ctx) if isinstance(m, DownBlock) else m(x)
            h.append(x)

        x = self.middle(x, ctx)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, ctx)

        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x
