"""Evaluate CFDBench velocity predictions for the FLOWFORGE rebuttal."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _load(path: Path):
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        archive = np.load(path)
        if {"u", "v"} <= set(archive.files):
            return np.stack([archive["u"], archive["v"]], axis=1)
        frames = [archive[key] for key in sorted(archive.files, key=_frame_number)]
        return np.asarray(frames)
    raise ValueError(f"Unsupported field file: {path}")


def _frame_number(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def as_velocity(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] % 2:
            raise ValueError(
                "flattened CFDBench predictions require an even frame count"
            )
        array = array.reshape(-1, 2, *array.shape[-2:])
    elif array.ndim >= 4:
        array = array.reshape(-1, *array.shape[-3:])
    else:
        raise ValueError(f"Unsupported velocity shape: {array.shape}")
    if array.shape[1] < 2:
        raise ValueError("u and v channels are required")
    return array[:, :2]


def _gradient(field, spacing, axis):
    return np.gradient(field, spacing, axis=axis, edge_order=2)


def _metrics(target, prediction, dx, dy, margin=0):
    target = as_velocity(target)
    prediction = as_velocity(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {target.shape} vs {prediction.shape}")

    def divergence(field):
        return _gradient(field[:, 0], dx, -1) + _gradient(field[:, 1], dy, -2)

    def vorticity(field):
        return _gradient(field[:, 1], dx, -1) - _gradient(field[:, 0], dy, -2)

    def crop(field):
        return field if margin == 0 else field[..., margin:-margin, margin:-margin]

    def rms(field):
        return float(np.sqrt(np.mean(np.square(field))))

    target_div = crop(divergence(target))
    prediction_div = crop(divergence(prediction))
    target_vort = crop(vorticity(target))
    prediction_vort = crop(vorticity(prediction))
    target_eval = crop(target)
    prediction_eval = crop(prediction)
    epsilon = 1e-12
    spectrum_delta = np.log(np.abs(np.fft.fft2(target_eval)) + 1e-8) - np.log(
        np.abs(np.fft.fft2(prediction_eval)) + 1e-8
    )
    return {
        "rmse": rms(prediction_eval - target_eval),
        "relative_vorticity_error": float(
            np.linalg.norm(prediction_vort - target_vort)
            / (np.linalg.norm(target_vort) + epsilon)
        ),
        "spectral_error": rms(spectrum_delta),
        "d_gt": rms(target_div),
        "d_pred": rms(prediction_div),
        "e_div": rms(prediction_div - target_div),
        "d_ratio": rms(prediction_div) / (rms(target_div) + epsilon),
    }


def evaluate(target, prediction, dx, dy, margin=0):
    target = as_velocity(target)
    prediction = as_velocity(prediction)
    aggregate = _metrics(target, prediction, dx, dy, margin)
    per_frame = [
        {
            "frame": index,
            **_metrics(
                target[index : index + 1], prediction[index : index + 1], dx, dy, margin
            ),
        }
        for index in range(target.shape[0])
    ]
    return aggregate, per_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--interior-margin", type=int, default=0)
    args = parser.parse_args()

    aggregate, rows = evaluate(
        _load(args.target),
        _load(args.prediction),
        args.dx,
        args.dy,
        args.interior_margin,
    )
    aggregate.update(
        {
            "dx": args.dx,
            "dy": args.dy,
            "boundary": "second-order-one-sided",
            "interior_margin": args.interior_margin,
            "num_frames": len(rows),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rebuttal_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "rebuttal_metrics_per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
