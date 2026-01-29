from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base_model import AutoCfdModel


class UnoOpBlock(nn.Module):
    """
    Lightweight operator block used inside U-NO.

    Depthwise conv provides spatial mixing, pointwise conv mixes channels,
    with a residual connection to stabilize deep stacks.
    """

    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(4, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class UnoStage(nn.Module):
    def __init__(self, dim: int, depth: int, kernel_size: int):
        super().__init__()
        self.blocks = nn.Sequential(*[UnoOpBlock(dim, kernel_size=kernel_size) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Uno2d(AutoCfdModel):
    """
    U-NO: U-shaped neural operator (encoder-decoder with skip connections).

    Conditioning follows FNO/CNO:
      - append mask
      - append spatial coords (x, y)
      - append case params broadcasted on grid
    """

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        n_case_params: int,
        loss_fn: nn.Module,
        base_dim: int = 64,
        levels: int = 4,
        depth_per_level: int = 2,
        bottleneck_depth: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__(loss_fn)
        assert levels >= 2
        assert base_dim % 4 == 0  # for GroupNorm

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.base_dim = base_dim
        self.levels = levels
        self.depth_per_level = depth_per_level
        self.bottleneck_depth = bottleneck_depth
        self.kernel_size = kernel_size

        in_proj_channels = in_chan + 1 + 2 + n_case_params  # inputs + mask + coords + props
        self.lift = nn.Conv2d(in_proj_channels, base_dim, kernel_size=1)

        # Encoder: stage -> downsample (stride-2 conv)
        enc_stages: List[nn.Module] = []
        downs: List[nn.Module] = []
        dims: List[int] = []
        cur_dim = base_dim
        for _ in range(levels - 1):
            enc_stages.append(UnoStage(cur_dim, depth=depth_per_level, kernel_size=kernel_size))
            dims.append(cur_dim)
            downs.append(nn.Conv2d(cur_dim, cur_dim * 2, kernel_size=3, stride=2, padding=1))
            cur_dim *= 2
        self.enc_stages = nn.ModuleList(enc_stages)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.bottleneck = UnoStage(cur_dim, depth=bottleneck_depth, kernel_size=kernel_size)

        # Decoder: upsample -> fuse skip -> stage
        ups: List[nn.Module] = []
        fuse: List[nn.Module] = []
        dec_stages: List[nn.Module] = []
        for skip_dim in reversed(dims):
            ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            # after upsample, channels are cur_dim; concatenate skip => cur_dim + skip_dim
            fuse.append(nn.Conv2d(cur_dim + skip_dim, skip_dim, kernel_size=1))
            cur_dim = skip_dim
            dec_stages.append(UnoStage(cur_dim, depth=depth_per_level, kernel_size=kernel_size))
        self.ups = nn.ModuleList(ups)
        self.fuse = nn.ModuleList(fuse)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.proj = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_dim, out_chan, kernel_size=1),
        )

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> Dict:
        bsz, _, height, width = inputs.shape

        if mask is None:
            mask = torch.ones((bsz, 1, height, width), device=inputs.device)
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        x = torch.cat([inputs, mask], dim=1)  # (B, in+1, H, W)

        props = case_params.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        props = props.repeat(1, 1, height, width)  # (B, P, H, W)
        grid = self.get_coords((bsz, 1, height, width), device=inputs.device)  # (B, 2, H, W)
        x = torch.cat([x, grid, props], dim=1)  # (B, in+1+2+P, H, W)

        x = self.lift(x)  # (B, base_dim, H, W)

        skips: List[Tensor] = []
        for stage, down in zip(self.enc_stages, self.downs):
            x = stage(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, fuse, stage, skip in zip(self.ups, self.fuse, self.dec_stages, reversed(skips)):
            x = up(x)
            # Handle odd sizes by padding/cropping to match skip
            if x.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - x.shape[-2]
                dw = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
                x = x[..., : skip.shape[-2], : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            x = stage(x)

        preds = self.proj(x)  # (B, out, H, W)
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(preds=preds, labels=label)
            return {"preds": preds, "loss": loss}
        return {"preds": preds}

    def get_coords(self, shape, device):
        bsz, _, size_x, size_y = shape
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid_x = grid_x.reshape(1, 1, size_x, 1).repeat([bsz, 1, 1, size_y])
        grid_y = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        grid_y = grid_y.reshape(1, 1, 1, size_y).repeat([bsz, 1, size_x, 1])
        coords = torch.cat([grid_x, grid_y], dim=1).to(device)
        return coords

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(inputs=inputs, case_params=case_params, mask=mask)
        return outputs["preds"]

    def generate_many(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Tensor,
        steps: int,
    ) -> List[Tensor]:
        assert len(inputs.shape) == len(case_params.shape) + 2
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            case_params = case_params.unsqueeze(0)
            mask = mask.unsqueeze(0)
        cur = inputs
        preds: List[Tensor] = []
        for _ in range(steps):
            cur = self.generate(inputs=cur, case_params=case_params, mask=mask)
            preds.append(cur)
        return preds

