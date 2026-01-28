from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base_model import AutoCfdModel


class CnoBlock(nn.Module):
    """
    A simple convolutional neural operator block.

    This block applies a depthwise-then-pointwise convolution (to better
    capture non-local mixing) followed by a residual connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        padding: int = 2,
        act_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_fn = act_fn or nn.GELU()

        # Depthwise + pointwise factorization gives a richer operator
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(4, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act_fn(x)
        return x + residual


class Cno2d(AutoCfdModel):
    """
    2D Convolutional Neural Operator for autoregressive CFDBench.

    The design mirrors FNO's conditioning:
      - Input channels: previous frame(s) + geometry mask
      - Extra channels: spatial coordinates (x, y) + case parameters
      - Core: a stack of CNO blocks operating in the spatial domain
    """

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        n_case_params: int,
        loss_fn: nn.Module,
        num_layers: int,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        padding: int = 2,
    ):
        super().__init__(loss_fn)

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.act_fn = nn.GELU()

        # +1 for mask, +2 for coordinates, +n_case_params for physical params
        in_proj_channels = in_chan + 1 + 2 + n_case_params
        self.fc0 = nn.Conv2d(in_proj_channels, hidden_dim, kernel_size=1)

        blocks: List[nn.Module] = []
        for _ in range(num_layers):
            blocks.append(
                CnoBlock(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    act_fn=self.act_fn,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_dim, out_chan, kernel_size=1)

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> Dict:
        """
        Args:
            inputs: (B, C_in, H, W)
            case_params: (B, P)
            mask: (B, 1, H, W) or (B, H, W), 1 for interior, 0 for obstacles
            label: (B, C_out, H, W)
        """
        batch_size, _, height, width = inputs.shape

        if mask is None:
            mask = torch.ones((batch_size, 1, height, width), device=inputs.device)
        else:
            if mask.dim() == 3:  # (B, H, W)
                mask = mask.unsqueeze(1)  # (B, 1, H, W)

        # Concatenate mask
        x = torch.cat([inputs, mask], dim=1)  # (B, C_in+1, H, W)

        # Broadcast physical parameters across the grid
        props = case_params.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        props = props.repeat(1, 1, height, width)  # (B, P, H, W)

        # Append spatial coordinates
        grid = self.get_coords(x.shape, x.device)  # (B, 2, H, W)
        x = torch.cat([x, grid, props], dim=1)  # (B, C_in+1+2+P, H, W)

        # Lift to hidden_dim
        x = self.fc0(x)
        if self.padding > 0:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        x = self.blocks(x)

        if self.padding > 0:
            x = x[..., self.padding : -self.padding, self.padding : -self.padding]

        x = self.fc1(x)
        x = self.act_fn(x)
        preds = self.fc2(x)  # (B, C_out, H, W)

        # Apply geometry mask
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(preds=preds, labels=label)
            return {"preds": preds, "loss": loss}
        return {"preds": preds}

    def get_coords(self, shape, device):
        """
        Return a tensor of shape (B, 2, H, W) with normalized (x, y) coordinates.
        """
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
        preds: Tensor = outputs["preds"]
        return preds

    def generate_many(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Tensor,
        steps: int,
    ) -> List[Tensor]:
        """
        Autoregressively generate `steps` frames.

        Args:
            inputs: (C, H, W) or (B, C, H, W)
            case_params: (P,) or (B, P)
            mask: (H, W) or (B, H, W)
        """
        assert len(inputs.shape) == len(case_params.shape) + 2
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            case_params = case_params.unsqueeze(0)
            mask = mask.unsqueeze(0)
        assert inputs.shape[0] == case_params.shape[0] == mask.shape[0]

        cur_frame = inputs
        preds: List[Tensor] = []
        for _ in range(steps):
            cur_frame = self.generate(
                inputs=cur_frame,
                case_params=case_params,
                mask=mask,
            )
            preds.append(cur_frame)
        return preds

