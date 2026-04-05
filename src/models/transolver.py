import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base_model import AutoCfdModel


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act in ACTIVATION:
            act_fn = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not supported")
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=64, W=64, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()

        # (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous() \
            .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous() \
            .reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        H=64,
        W=64,
        kernel=3,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
            kernel=kernel,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        n_case_params: int,
        loss_fn: nn.Module,
        n_hidden: int = 128,
        n_layers: int = 5,
        n_head: int = 8,
        slice_num: int = 32,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = 'gelu',
        height: int = 64,
        width: int = 64,
        kernel: int = 3,
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.n_hidden = n_hidden
        self.height = height
        self.width = width

        # fun_dim: velocity channels + mask + case_params
        # space_dim: 2 (x, y coordinates)
        fun_dim = in_chan + 1 + n_case_params
        space_dim = 2

        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_chan,
                slice_num=slice_num,
                H=height,
                W=width,
                kernel=kernel,
                last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_coords(self, shape, device):
        bsz, c, size_x, size_y = shape
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid_x = grid_x.reshape(1, 1, size_x, 1).repeat([bsz, 1, 1, size_y])
        grid_y = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        grid_y = grid_y.reshape(1, 1, 1, size_y).repeat([bsz, 1, size_x, 1])
        coords = torch.cat([grid_x, grid_y], dim=1).to(device)
        return coords

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> Dict:
        batch_size, n_chan, height, width = inputs.shape

        if mask is None:
            mask = torch.ones((batch_size, 1, height, width), device=inputs.device)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Broadcast case_params to spatial dims: (B, P) -> (B, P, H, W)
        props = case_params.unsqueeze(-1).unsqueeze(-1)
        props = props.repeat(1, 1, height, width)

        # Grid coordinates: (B, 2, H, W)
        grid = self.get_coords(inputs.shape, inputs.device)

        # Assemble function values: (B, in_chan+1+P, H, W) -> (B, H*W, fun_dim)
        fx = torch.cat([inputs, mask, props], dim=1)
        fx = fx.permute(0, 2, 3, 1).contiguous().reshape(batch_size, height * width, -1)

        # Spatial positions: (B, 2, H, W) -> (B, H*W, 2)
        x_pos = grid.permute(0, 2, 3, 1).contiguous().reshape(batch_size, height * width, 2)

        # Concatenate position and function values, preprocess
        fx = torch.cat([x_pos, fx], dim=-1)
        fx = self.preprocess(fx)

        # Apply Transolver blocks
        for block in self.blocks:
            fx = block(fx)

        # Reshape output: (B, H*W, out_chan) -> (B, out_chan, H, W)
        preds = fx.reshape(batch_size, height, width, self.out_chan) \
            .permute(0, 3, 1, 2).contiguous()

        # Apply mask
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(preds=preds, labels=label)
            return dict(preds=preds, loss=loss)
        return dict(preds=preds)

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
        assert inputs.shape[0] == case_params.shape[0] == mask.shape[0]

        cur_frame = inputs
        preds = []
        for _ in range(steps):
            cur_frame = self.generate(
                inputs=cur_frame, case_params=case_params, mask=mask
            )
            preds.append(cur_frame)
        return preds
