import json
from pathlib import Path
from typing import Union, Optional
import typing

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

from args import Args


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


@typing.no_type_check
def plot_predictions(
    label: Tensor,
    pred: Tensor,
    out_dir: Path,
    step: int,
    inp: Optional[Tensor] = None,  # non-autoregressive input func. is not plottable.
):
    assert all([isinstance(x, Tensor) for x in [label, pred]])
    assert label.shape == pred.shape, f"{label.shape}, {pred.shape}"

    if inp is not None:
        assert inp.shape == label.shape
        assert isinstance(inp, Tensor)
        inp_dir = out_dir / "input"
        inp_dir.mkdir(exist_ok=True, parents=True)
        inp_arr = inp.cpu().detach().numpy()
    label_dir = out_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True, parents=True)

    pred_arr = pred.cpu().detach().numpy()
    label_arr = label.cpu().detach().numpy()

    # Plot and save images
    if inp is not None:
        u_min = min(inp_arr.min(), pred_arr.min(), label_arr.min())  # type: ignore  # noqa
        u_max = max(inp_arr.max(), pred_arr.max(), label_arr.max())  # type: ignore  # noqa
    else:
        u_min = min(pred_arr.min(), label_arr.min())  # type: ignore  # noqa
        u_max = max(pred_arr.max(), label_arr.max())  # type: ignore  # noqa

    if inp is not None:
        plt.axis("off")
        plt.imshow(
            inp_arr, vmin=inp_arr.min(), vmax=inp_arr.max(), cmap="coolwarm"  # type: ignore  # noqa
        )
        plt.savefig(
            inp_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0  # type: ignore  # noqa
        )
        plt.clf()

    plt.axis("off")
    plt.imshow(label_arr, vmin=label_arr.min(), vmax=label_arr.max(), cmap="coolwarm")
    plt.savefig(label_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    plt.axis("off")
    plt.imshow(pred_arr, vmin=pred_arr.min(), vmax=pred_arr.max(), cmap="coolwarm")
    plt.savefig(pred_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()


def plot(inp: Tensor, label: Tensor, pred: Tensor, out_path: Path):
    assert all([isinstance(x, Tensor) for x in [inp, label, pred]])
    assert (
        inp.shape == label.shape == pred.shape
    ), f"{inp.shape}, {label.shape}, {pred.shape}"

    tensor_dir = out_path.parent / "tensors"
    tensor_dir.mkdir(exist_ok=True, parents=True)
    tensor_path = tensor_dir / (out_path.stem + ".pt")
    torch.save((inp, label, pred), tensor_path)

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1, wspace=0, hspace=0)
    # [left, bottom, width, height]
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])

    axs = axs.flatten()

    last_im = None

    inp_arr = inp.cpu().detach().numpy()
    pred_arr = pred.cpu().detach().numpy()
    label_arr = label.cpu().detach().numpy()
    u_min = min(inp_arr.min(), pred_arr.min(), label_arr.min())
    u_max = max(inp_arr.max(), pred_arr.max(), label_arr.max())

    def sub_plot(idx, data: np.ndarray, title):
        nonlocal last_im
        ax = axs[idx - 1]
        ax.set_axis_off()
        im = ax.imshow(data, vmin=u_min, vmax=u_max, cmap="coolwarm")
        # fig.colorbar(im, ax=ax)
        # ax.set_title(title)
        last_im = im

    # print("plotting input")
    sub_plot(1, inp_arr, "Input")
    sub_plot(2, label_arr, "Label")
    sub_plot(3, pred_arr, "Prediction")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # type: ignore
    fig.colorbar(last_im, cax=cbar_ax)  # type: ignore
    # # Add a common colorbar
    # fig.colorbar(last_im, cax=cbar_ax)

    # Add some spacing between the subplots
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_loss(losses, out: Path, fontsize: int = 12, linewidth: int = 2):
    plt.plot(losses, linewidth=linewidth)
    plt.xlabel("Step", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.savefig(out)
    plt.clf()
    plt.close()


def plot_flow_field(u, v, color: str, label: str, ax):

    # Create a meshgrid for the coordinates
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

    # Create a quiver plot in the given axis
    ax.quiver(x, y, u, v, scale=30, color=color, width=0.005)
    ax.set_title(f"Flow Field - {label}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


def generate_frame(
    u_real_frame,
    v_real_frame,
    u_pred_frame,
    v_pred_frame,
    save_path: Path,
    dir_frame_range: list[tuple],  # containing id range of frames to be visualized
):
    assert (
        u_real_frame.shape
        == v_real_frame.shape
        == u_pred_frame.shape
        == v_pred_frame.shape
    )

    for start_idx, end_idx in dir_frame_range:
        for i in range(start_idx, end_idx + 1):
            v_r = v_real_frame[i, :, :]
            u_r = u_real_frame[i, :, :]
            v_p = v_pred_frame[i, :, :]
            u_p = u_pred_frame[i, :, :]

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            plot_flow_field(u_r, v_r, "g", f"Real Field - Frame {i+1}", axs[0])

            plot_flow_field(u_p, v_p, "g", f"Predicted Field - Frame {i+1}", axs[1])

            # Adjust layout and show the figure
            plt.tight_layout()
            plt.savefig(save_path / f"Frame_{i+1}.png")
            plt.clf()
            plt.close()


def get_best_ckpt(output_dir: Path) -> Union[Path, None]:
    """
    Returns None if there is no ckpt-* directory in output_dir
    """
    ckpt_dirs = sorted(output_dir.glob("ckpt-*"))
    best_loss = float("inf")
    best_ckpt_dir = None
    for ckpt_dir in ckpt_dirs:
        scores = load_json(ckpt_dir / "scores.json")
        dev_loss = scores["dev_loss"]
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_ckpt_dir = ckpt_dir
    return best_ckpt_dir


def load_ckpt(model, ckpt_path: Path) -> None:
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))


def get_output_dir(args: Args, is_auto: bool = False) -> Path:
    output_dir = Path(
        args.output_dir,
        "auto" if is_auto else "non-auto",
        args.data_name,
        f"dt{args.delta_time}",
        args.model,
    )

    if args.model == "deeponet":
        dir_name = (
            f"lr{args.lr}"
            + f"_width{args.deeponet_width}"
            + f"_depthb{args.branch_depth}"
            + f"_deptht{args.trunk_depth}"
            + f"_normprop{args.norm_props}"
            + f"_act{args.act_fn}"
            + f"-{args.act_scale_invariant}"
            + f"-{args.act_on_output}"
        )
    elif args.model == "unet":
        dir_name = (
            f"lr{args.lr}" f"_d{args.unet_dim}" f"_cp{args.unet_insert_case_params_at}"
        )
    elif args.model == "fno":
        dir_name = (
            f"lr{args.lr}"
            + f"_d{args.fno_depth}"
            + f"_h{args.fno_hidden_dim}"
            + f"_m1{args.fno_modes_x}"
            + f"_m2{args.fno_modes_y}"
        )
    elif args.model == "resnet":
        dir_name = (
            f"lr{args.lr}" f"_d{args.resnet_depth}" f"_w{args.resnet_hidden_chan}"
        )
    elif args.model == "auto_edeeponet":
        dir_name = (
            f"lr{args.lr}"
            + f"_width{args.autoedeeponet_width}"
            + f"_depthb{args.autoedeeponet_depth}"
            + f"_deptht{args.autoedeeponet_depth}"
            + f"_normprop{args.norm_props}"
            + f"_act{args.autoedeeponet_act_fn}"
            # + f"-{args.act_scale_invariant}"
            # + f"-{args.act_on_output}"
        )
    elif args.model == "auto_deeponet":
        dir_name = (
            f"lr{args.lr}"
            f"_width{args.deeponet_width}"
            f"_depthb{args.branch_depth}"
            f"_deptht{args.trunk_depth}"
            f"_normprop{args.norm_props}"
            f"_act{args.act_fn}"
        )
    elif args.model == "auto_ffn":
        dir_name = (
            f"lr{args.lr}" f"_width{args.autoffn_width}" f"_depth{args.autoffn_depth}"
        )
    elif args.model == "auto_deeponet_cnn":
        dir_name = f"lr{args.lr}" f"_depth{args.autoffn_depth}"
    elif args.model == "ffn":
        dir_name = f"lr{args.lr}" f"_width{args.ffn_width}" f"_depth{args.ffn_depth}"
    else:
        raise NotImplementedError

    if args.velocity_dim == 0:
        velocity_dim = "u"
    else:
        velocity_dim = "v"

    output_dir /= dir_name
    if args.model == "auto_deeponet":
        output_dir /= velocity_dim
    return output_dir


def get_robustness_dir_name(args: Args) -> Path:
    suffix_parts = []
    if args.noise_std is not None:
        suffix_parts.append(f"ns={args.noise_std}")
    if args.edge_width is not None:
        suffix_parts.append(f"ew={args.edge_width}")
    if args.block_size is not None:
        suffix_parts.append(f"bs={args.block_size}")
    if args.num_blocks is not None:
        suffix_parts.append(f"nb={args.num_blocks}")
    if args.mask_range:
        suffix_parts.append(f"mask_{args.mask_range}")
    if args.noise_mode:
        suffix_parts.append(f"mode_{args.noise_mode}")
    dir_name = "_".join(suffix_parts)

    return dir_name


def load_best_ckpt(model, output_dir: Path):
    print(f"Finding the best checkpoint from {output_dir}")
    best_ckpt_dir = get_best_ckpt(output_dir)
    assert best_ckpt_dir is not None
    print(f"Loading best checkpoint from {best_ckpt_dir}")
    ckpt_path = best_ckpt_dir / "model.pt"
    load_ckpt(model, ckpt_path)


def get_dir_nums(dir: Path):
    return sum(1 for f in dir.iterdir() if f.is_dir())


def check_file_exists(file_path):
    return Path(file_path).is_file()


def check_path_exists(path):
    return Path(path).exists()
