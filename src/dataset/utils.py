import json
import torch
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
from torch import Tensor


def normalize_physics_props(case_params: Dict[str, float]):
    """
    Normalize the physics properties in-place.
    """
    density_mean = 5
    density_std = 4
    viscosity_mean = 0.00238
    viscosity_std = 0.005
    case_params["density"] = (case_params["density"] - density_mean) / density_std
    case_params["viscosity"] = (
        case_params["viscosity"] - viscosity_mean
    ) / viscosity_std


def normalize_bc(case_params: Dict[str, float], key: str):
    """
    Normalize the boundary conditions in-place.
    """
    case_params[key] = case_params[key] / 50 - 0.5


def plot_contour(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    plt.tricontourf(x, y, z)
    plt.colorbar()
    plt.show()


def dump_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load a JSON object from a file"""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def plot(inputs, outputs, labels, output_file: Path):
    # Create a figure with 6 subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom,
    # width, height]

    axs = axs.flatten()

    last_im = None

    def sub_plot(idx, data, title):
        nonlocal last_im
        ax = axs[idx - 1]
        im = ax.imshow(data.cpu().detach().numpy())
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        last_im = im

    sub_plot(1, inputs[0], "input u")
    sub_plot(4, inputs[1], "input v")
    sub_plot(2, labels[0], "label u")
    sub_plot(5, labels[1], "label v")
    sub_plot(3, outputs[0], "output u")
    sub_plot(6, outputs[1], "output v")

    # # Add a common colorbar
    # fig.colorbar(last_im, cax=cbar_ax)

    # # Add some spacing between the subplots
    # fig.tight_layout()

    plt.savefig(output_file)
    plt.clf()
    plt.close()


def plot_loss(losses, out: Path):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(out)
    plt.clf()
    plt.close()


def generate_edge_mask(input_shape: tuple, edge_width: int) -> Tensor:
    """
    Generate a mask for the edges of the input data.
    """
    # Create a mask with the same shape as the input data
    mask = torch.zeros(input_shape, dtype=torch.float32)

    # Set the edges to 1
    mask[..., :edge_width, :] = 1
    mask[..., -edge_width:, :] = 1
    mask[..., :, :edge_width] = 1
    mask[..., :, -edge_width:] = 1

    return mask


def generate_block_mask(
    input_shape: tuple, block_size: int, num_blocks: int, device=None
) -> Tensor:

    B, H, W = input_shape[0], input_shape[-2], input_shape[-1]
    mask = torch.zeros(input_shape, dtype=torch.float32, device=device)

    for b in range(B):
        for _ in range(num_blocks):
            top = torch.randint(0, H - block_size + 1, (1,))
            left = torch.randint(0, W - block_size + 1, (1,))
            mask[b, ..., top : top + block_size, left : left + block_size] = 1

    return mask


def add_noise(
    input_data: Tensor,
    noise_mask: Tensor,
    noise_mean: float = 0.0,
    noise_std: float = 0.0,
):

    input_shape = input_data.shape

    data_min = input_data.amin(dim=(-2, -1), keepdim=True)
    data_max = input_data.amax(dim=(-2, -1), keepdim=True)
    scale = (data_max - data_min).clamp(min=1e-6)  # avoid division by zero
    input_norm = (input_data - data_min) / scale

    noise = torch.normal(
        mean=noise_mean, std=noise_std, size=input_shape, device=input_data.device
    )
    noise = noise * noise_mask

    input_noisy_norm = input_norm + noise
    input_noisy_norm = input_noisy_norm.clamp(0.0, 1.0)
    input_noisy = input_noisy_norm * scale + data_min

    return input_noisy


def apply_mask_fill(
    input_data: Tensor, mask: Tensor, fill_value: float = 0.0
) -> Tensor:
    return input_data * (1 - mask) + fill_value * mask


def mask_and_noise_process(
    input_data: Tensor,
    mask_range: str,
    noise_mode: str,
    noise_mean: float = 0.0,
    noise_std: float = 0.0,
    block_size: int = 0,
    num_blocks: int = 0,
    edge_width: int = 0,
) -> Tensor:
    """
    Apply mask and noise to the input data.
    """
    input_shape = input_data.shape

    if mask_range == "edge":
        mask = generate_edge_mask(input_shape, edge_width)
    elif mask_range == "global":
        mask = generate_block_mask(input_shape, block_size, num_blocks)
    else:
        raise ValueError(f"Unknown mask range: {mask_range}")

    if noise_mode == "noise":
        input_data = add_noise(input_data, mask, noise_mean, noise_std)
    elif noise_mode == "zero":
        input_data = apply_mask_fill(input_data, mask, fill_value=0.0)
    else:
        raise ValueError(f"Unknown noise mode: {noise_mode}")

    return input_data
