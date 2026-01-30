from pathlib import Path
from args import Args
import time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_json, check_file_exists, generate_frame, get_robustness_dir_name
from dataset.cavity import CavityFlowAutoDataset
from dataset.base import CfdAutoDataset
from models.base_model import AutoCfdModel


def get_dev_losses(output_dir: Path):
    ckpt_dirs = output_dir.glob("ckpt-*")
    dev_scores = {
        "mse": [],
        "nmse": [],
    }
    for ckpt_dir in ckpt_dirs:
        scores = load_json(ckpt_dir / "dev_scores.json")["mean"]
        for key in dev_scores:
            dev_scores[key].append(scores[key])
    return dev_scores


def get_test_scores(output_dir: Path):
    test_dir = output_dir / "test"
    scores = load_json(test_dir / "scores.json")
    # print(list(scores.keys()))
    if "scores" in scores:
        scores = scores["scores"]
    if "mean" in scores:
        return scores["mean"]
    # print(scores['mean'])
    avgs = {}
    for key in scores:
        avgs[key] = np.mean(scores[key])
    return avgs


def get_data_result(data_param_dir: Path, model_pattern: str = "*") -> list[list]:
    print(f"getting result for {data_param_dir}")
    scores = []
    # run_dirs = [run_dir for run_dir in run_dirs if 'depthb4' not in run_dir.name]
    for model_dir in sorted(data_param_dir.glob(model_pattern)):
        run_dirs = sorted(model_dir.glob("*"))
        run_dirs = [run_dir for run_dir in run_dirs]
        for run_dir in run_dirs:
            """
            Each run_dir correspond to one set of hyperparameters. E.g.,
            dam_bc_geo/dt0.1/fno/lr_0.0001_d4_h32_m112_m212
            """
            if not run_dir.is_dir():
                continue
            try:
                test_scores = get_test_scores(run_dir)
                scores.append([str(run_dir)] + list(test_scores.values()))
            except FileNotFoundError:
                print(run_dir.name, "not found")
                pass
    return scores


def get_result(result_dir: Path, data_pattern: str, model_pattern: str):
    data_dirs = result_dir.glob(data_pattern)
    scores = []
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            continue
        # if 'prop' in data_dir.name:
        #     continue
        # Loop data subdirs such as `dt0.1`
        for data_param_dir in data_dir.iterdir():
            if data_param_dir.is_dir():
                scores += get_data_result(data_param_dir, model_pattern=model_pattern)
    table = [[str(score) for score in line] for line in scores]
    # table = sorted(table)
    # transpose
    rows = []
    n_cols = len(table[0])
    for c in range(n_cols):
        row = [line[c] for line in table]
        rows.append(row)

    lines = ["\t".join(line) for line in rows]

    print(*lines)
    data_pattern = data_pattern.replace("*", "+")
    out_path = result_dir / f"{data_pattern}_{model_pattern}.txt"
    print(*lines, sep="\n", file=open(out_path, "w", encoding="utf8"))


def get_visualize_result(
    test_data: CavityFlowAutoDataset,
    prediction_path: Path,
    data_to_visualize: str,
    is_autodeeponet: bool = False,
    if_robustness_test: bool = False,
    args: Args = None,
):
    count = 0
    dir_frame_range = []
    frame_num_list = test_data.frame_num_list
    name_list = test_data.case_name_list

    for i in range(len(name_list)):
        dir_name = name_list[i]
        if data_to_visualize in dir_name:
            dir_frame_range.append((count, count + frame_num_list[i] - 1))
            count += frame_num_list[i]
        else:
            count += frame_num_list[i]

    print("Getting visualing result...")

    # Determine the directory name based on robustness_test
    if if_robustness_test and args is not None:
        robustness_test_dir = get_robustness_dir_name(args)
        # Build path properly using Path components instead of string concatenation
        robustness_test_path = Path("robustness_test") / robustness_test_dir
    else:
        robustness_test_path = Path("test")

    if is_autodeeponet:
        u_prediction_path = prediction_path / "u" / robustness_test_path / "preds.pt"
        v_prediction_path = prediction_path / "v" / robustness_test_path / "preds.pt"
        print(f"Loading predictions from:")
        print(f"  u: {u_prediction_path}")
        print(f"  v: {v_prediction_path}")
        u_prediction = torch.load(
            u_prediction_path, weights_only=True
        )  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(
            v_prediction_path, weights_only=True
        )  # v_prediction: (all_frames, h, w)
    else:
        result_path = prediction_path / robustness_test_path / "preds.pt"
        print(f"Loading predictions from: {result_path}")
        print(f"  prediction_path: {prediction_path}")
        print(f"  robustness_test_path: {robustness_test_path}")
        if args is not None:
            print(f"  model: {args.model}")
        prediction = torch.load(
            result_path, weights_only=True
        )  # prediction: (all_frames, h, w)
        h, w = prediction.shape[1], prediction.shape[2]
        prediction = prediction.reshape(-1, 2, h, w)
        u_prediction = prediction[:, 0, :, :]  # u_prediction: (all_frames, h, w)
        v_prediction = prediction[:, 1, :, :]  # v_prediction: (all_frames, h, w)

    u_real = test_data.labels[:, 0]  # u_real: (all_frames, h, w)
    v_real = test_data.labels[:, 1]  # v_real: (all_frames, h, w)
    assert u_real.shape == v_real.shape
    assert u_prediction.shape == v_prediction.shape

    image_save_path = prediction_path / "visualize_result" / data_to_visualize
    image_save_path.mkdir(exist_ok=True, parents=True)
    # Get model name from args
    if args is not None:
        model_name = args.model
    else:
        model_name = "model"
    generate_frame(
        u_real, v_real, u_prediction, v_prediction, image_save_path, dir_frame_range, model_name=model_name
    )

    print("Getting visualing result finished.")


def get_frame_accuracy(u_p_frame, v_p_frame, u_r_frame, v_r_frame):
    # Calculate the prediction accuracy for one frame
    vel_p = torch.sqrt(u_p_frame**2 + v_p_frame**2)
    vel_r = torch.sqrt(u_r_frame**2 + v_r_frame**2)
    angle_p = torch.arctan2(v_p_frame, u_p_frame)
    angle_r = torch.arctan2(v_r_frame, u_r_frame)

    delta_angle = angle_p - angle_r

    SI = torch.cos(delta_angle)
    mask_1 = SI > 0.8

    MI = 1 - abs(vel_p - vel_r) / (vel_r + vel_p)
    mask_2 = MI > 0.8

    mask = mask_1 * mask_2
    good_pred_rate = torch.sum(mask) / mask.numel()

    return good_pred_rate


def get_case_accuracy(
    test_data: CavityFlowAutoDataset,
    prediction_path: Path,
    result_save_path: Path,
    args: Args,
    is_autodeeponet: bool = False,
    robustness_test: bool = False,
):
    if robustness_test:
        robustness_test_dir = get_robustness_dir_name(args)
        dir_name = "robustness_test/" + robustness_test_dir
    else:
        dir_name = "test"

    if is_autodeeponet:
        u_prediction_path = prediction_path / "u" / dir_name / "preds.pt"
        v_prediction_path = prediction_path / "v" / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)

        u_prediction = torch.load(
            u_prediction_path, weights_only=True
        )  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(
            v_prediction_path, weights_only=True
        )  # v_prediction: (all_frames, h, w)
    else:
        prediction_path = prediction_path / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)
        prediction = torch.load(
            prediction_path, weights_only=True
        )  # prediction: (all_frames, h, w)
        h, w = prediction.shape[1], prediction.shape[2]
        prediction = prediction.reshape(-1, 2, h, w)
        u_prediction = prediction[:, 0, :, :]  # u_prediction: (all_frames, h, w)
        v_prediction = prediction[:, 1, :, :]  # v_prediction: (all_frames, h, w)

    u_real = test_data.labels[:, 0]  # u_real: (all_frames, h, w)
    v_real = test_data.labels[:, 1]  # v_real: (all_frames, h, w)
    assert u_real.shape == v_real.shape
    assert u_prediction.shape == v_prediction.shape

    name_list = test_data.case_name_list
    frame_num_list = test_data.frame_num_list

    count = 0
    accuracy_list = []  # store the accuracy for each frame in one case
    case_accuracy_list = []  # store the accuracy for each case

    for i in range(u_prediction.shape[0]):
        u_p_frame = u_prediction[i, :, :]
        v_p_frame = v_prediction[i, :, :]
        u_r_frame = u_real[i, :, :]
        v_r_frame = v_real[i, :, :]
        accuracy = get_frame_accuracy(u_p_frame, v_p_frame, u_r_frame, v_r_frame)
        accuracy_list.append(accuracy)

        if len(accuracy_list) == frame_num_list[count]:
            case_accuracy = np.mean(accuracy_list)
            case_accuracy_list.append(case_accuracy)
            count += 1
            accuracy_list = []

    with open(result_save_path / "case_accuracy.txt", "w") as f:
        assert len(name_list) == len(case_accuracy_list)
        for i in range(len(case_accuracy_list)):
            f.write(f"{name_list[i]}: {case_accuracy_list[i]}\n")

        f.write(f"Average case accuracy: {np.mean(case_accuracy_list)}\n")

    print(f"Case accuracy saved in {result_save_path}")


def compute_vorticity(u, v, dx=1.0, dy=1.0):
    """
    Compute vorticity field (curl of velocity).
    """
    dv_dx = np.gradient(v, axis=-1, edge_order=2) / dx
    du_dy = np.gradient(u, axis=-2, edge_order=2) / dy
    return dv_dx - du_dy


def compute_divergence(u, v, dx=1.0, dy=1.0):
    """
    Compute divergence field.
    """
    du_dx = np.gradient(u, axis=-1, edge_order=2) / dx
    dv_dy = np.gradient(v, axis=-2, edge_order=2) / dy
    return du_dx + dv_dy


def calculate_metrics(y_true, y_pred, dx=1.0, dy=1.0):
    """
    Compute Reviewer-Proof metrics for flow fields.
    y_true, y_pred: ndarray of shape (n, c, h, w)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n, c, h, w = y_true.shape
    epsilon = 1e-8
    metrics = {}

    # RMSE 
    # Baseline metric for absolute pixel-wise accuracy.
    # Calculated globally to avoid sample-size bias.
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    metrics["RMSE"] = rmse

    # Global Relative L2 Error 
    # We use the Norm of Difference / Norm of True (Global Aggregation).
    diff_norm = np.linalg.norm(y_true - y_pred)
    true_norm = np.linalg.norm(y_true)
    rel_l2 = diff_norm / (true_norm + epsilon)
    metrics["Rel L2 Error"] = rel_l2

    # Spectral Error (Log Spectral Distance) 
    # Measures the preservation of high-frequency details (texture/turbulence).
    # 2D FFT over spatial dimensions (-2, -1)
    fft_true = np.fft.fft2(y_true, axes=(-2, -1))
    fft_pred = np.fft.fft2(y_pred, axes=(-2, -1))

    # Use Log magnitude to balance high/low frequency contribution
    log_true = np.log(np.abs(fft_true) + epsilon)
    log_pred = np.log(np.abs(fft_pred) + epsilon)

    # RMSE of the log spectra
    spectral_error = np.sqrt(np.mean((log_true - log_pred) ** 2))
    metrics["Spectral Error"] = spectral_error

    # Only applicable if we have both u and v components (c >= 2)
    if c >= 2:
        u_true, v_true = y_true[:, 0], y_true[:, 1]
        u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]

        # Relative Vorticity Error (Global)
        # Checks if the model captures the rotational dynamics (eddies) correctly.
        vor_true = compute_vorticity(u_true, v_true, dx, dy)
        vor_pred = compute_vorticity(u_pred, v_pred, dx, dy)

        vor_diff_norm = np.linalg.norm(vor_true - vor_pred)
        vor_true_norm = np.linalg.norm(vor_true)

        rel_vor_error = vor_diff_norm / (vor_true_norm + epsilon)
        metrics["Rel Vorticity Error"] = rel_vor_error

        # Divergence Consistency Error (RMSE)
        # Instead of assuming div=0, we check if pred matches ground truth divergence.
        div_true = compute_divergence(u_true, v_true, dx, dy)
        div_pred = compute_divergence(u_pred, v_pred, dx, dy)

        # RMSE of divergence difference
        div_error = np.sqrt(np.mean((div_true - div_pred) ** 2))
        metrics["Div Error"] = div_error

    return metrics


def cal_loss(
    test_data: CavityFlowAutoDataset,
    prediction_path: Path,
    result_save_path: Path,
    args: Args,
    is_autodeeponet: bool = False,
    robustness_test: bool = False,
):
    if robustness_test:
        robustness_test_dir = get_robustness_dir_name(args)
        dir_name = "robustness_test/" + robustness_test_dir
    else:
        dir_name = "test"

    if is_autodeeponet:
        u_prediction_path = prediction_path / "u" / dir_name / "preds.pt"
        v_prediction_path = prediction_path / "v" / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)

        u_prediction = torch.load(
            u_prediction_path, weights_only=True
        )  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(
            v_prediction_path, weights_only=True
        )  # v_prediction: (all_frames, h, w)
    else:
        prediction_path = prediction_path / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)
        prediction = torch.load(
            prediction_path, weights_only=True
        )  # prediction: (all_frames, h, w)
        h, w = prediction.shape[1], prediction.shape[2]
        prediction = prediction.reshape(-1, 2, h, w)
        u_prediction = prediction[:, 0, :, :]  # u_prediction: (all_frames, h, w)
        v_prediction = prediction[:, 1, :, :]  # v_prediction: (all_frames, h, w)

    u_real = test_data.labels[:, 0]  # u_real: (all_frames, h, w)
    v_real = test_data.labels[:, 1]  # v_real: (all_frames, h, w)
    assert u_real.shape == v_real.shape
    assert u_prediction.shape == v_prediction.shape

    prediction = torch.stack([u_prediction, v_prediction], dim=1)
    real = torch.stack([u_real, v_real], dim=1)

    loss = calculate_metrics(real, prediction)
    with open(result_save_path / "loss.txt", "w") as f:
        for key, value in loss.items():
            f.write(f"{key}: {value}\n")

    print(f"Loss saved in {result_save_path}")


def cal_time(
    path: Path,
    result_save_path: Path,
    is_autodeeponet: bool = False,
    type: str = "predict",
):
    if is_autodeeponet:
        u_time_file = path / "u" / "test" / f"{type}_time.txt"
        v_time_file = path / "v" / "test" / f"{type}_time.txt"

        with open(u_time_file, "r") as f:
            u_time = 0.0
            line = f.readline()
            colon_index = line.find(":")
            u_time = float(line[colon_index + 1 :].strip())

        with open(v_time_file, "r") as f:
            v_time = 0.0
            line = f.readline()
            colon_index = line.find(":")
            v_time = float(line[colon_index + 1 :].strip())

        time = u_time + v_time

    else:
        file_path = path / "test" / f"{type}_time.txt"
        with open(file_path, "r") as f:
            time = 0.0
            line = f.readline()
            colon_index = line.find(":")
            time = float(line[colon_index + 1 :].strip())

    with open(result_save_path / f"{type}_time.txt", "w") as f:
        f.write(f"Total {type}_time: {time}")

    print(f"{type} time saved in {result_save_path}")


def collate_fn(batch: list):
    # Collate function for DataLoader
    inputs, labels, case_params = zip(*batch)
    inputs = torch.stack(inputs)  # (b, 3, h, w)
    labels = torch.stack(labels)  # (b, 3, h, w)
    labels = labels[:, :-1]  # (b, 2, h, w)
    mask = inputs[:, -1:]  # (b, 1, h, w)
    inputs = inputs[:, :-1]  # (b, 2, h, w)
    keys = [x for x in case_params[0].keys() if x not in ["rotated", "dx", "dy"]]
    case_params_vec = []
    for case_param in case_params:
        case_params_vec.append([case_param[k] for k in keys])
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    return dict(
        inputs=inputs.cuda(),
        label=labels.cuda(),
        mask=mask.cuda(),
        case_params=case_params.cuda(),
    )


def measure_predict_time(
    model: AutoCfdModel,
    dataset: CfdAutoDataset,
    num_frames: int = 200,
    num_runs: int = 10,
    warmup_runs: int = 5,
    batch_size: int = 1,
) -> float:
    """
    Measure prediction time for specified number of frames.
    Only predicts first num_frames, runs multiple times and returns average time.

    Args:
        model: Model to use for prediction
        dataset: Dataset to predict on
        num_frames: Number of frames to predict (default: 200)
        num_runs: Number of runs for averaging (default: 10)
        warmup_runs: Number of warmup runs (default: 5)
        batch_size: Batch size for prediction (default: 1)

    Returns:
        Average prediction time in seconds
    """
    # Limit dataset to first num_frames
    limited_dataset = torch.utils.data.Subset(
        dataset, range(min(num_frames, len(dataset)))
    )
    loader = DataLoader(
        limited_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    all_run_times: list[float] = []

    print("=== Measuring Prediction Time ===")
    print(f"# examples: {len(limited_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"# batches: {len(loader)}")

    # Warmup runs
    print("Warming up GPU...")
    with torch.inference_mode():
        it = iter(loader)
        for _ in range(warmup_runs):
            try:
                batch = next(it)
            except StopIteration:
                if len(limited_dataset) == 0:
                    print("Warning: Empty dataset, skipping warmup.")
                    break
                it = iter(loader)
                batch = next(it)
            _ = model(**batch)
    torch.cuda.synchronize()
    print("GPU is ready!")

    # Measure time over multiple runs
    for run_idx in range(num_runs):
        current_run_total_time = 0.0

        with torch.inference_mode():
            for step, batch in enumerate(
                tqdm(loader, desc=f"Run {run_idx + 1}/{num_runs}")
            ):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                inputs = batch["inputs"]  # (b, 2, h, w)
                labels = batch["label"]  # (b, 2, h, w)

                # Compute the prediction
                outputs: dict = model(**batch)
                loss: dict = outputs["loss"]
                preds: Tensor = outputs["preds"]
                height, width = inputs.shape[2:]
                preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
                preds_cpu = preds.cpu().detach()

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                current_run_total_time += end_time - start_time

        all_run_times.append(current_run_total_time)
        print(f"Run {run_idx + 1}/{num_runs}: {current_run_total_time:.4f}s")

    if not all_run_times:
        inference_time = 0.0
    else:
        inference_time = sum(all_run_times) / len(all_run_times)

    print(f"Average prediction time for {num_frames} frames: {inference_time:.4f}s")
    return inference_time


if __name__ == "__main__":
    result_dir = Path("result/auto")
    data_pattern = "dam*"
    model_pattern = "auto_edeeponet"
    # model_pattern = "auto_ffn"
    model_pattern = "auto_deeponet_cnn"
    model_pattern = "fno"
    # model_pattern = '*'
    # model_pattern = "ffn"
    # model_pattern = "deeponet"

    get_result(result_dir, data_pattern, model_pattern)
