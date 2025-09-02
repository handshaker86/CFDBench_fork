from pathlib import Path
from args import Args

import numpy as np
import torch

from utils import load_json, check_file_exists, generate_frame, get_robustness_dir_name
from dataset.cavity import CavityFlowAutoDataset


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

    if is_autodeeponet:
        u_prediction_path = prediction_path / "u" / "test" / "preds.pt"
        v_prediction_path = prediction_path / "v" / "test" / "preds.pt"
        u_prediction = torch.load(u_prediction_path, weights_only=True)  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(v_prediction_path, weights_only=True)  # v_prediction: (all_frames, h, w)
    else:
        result_path = prediction_path / "test" / "preds.pt"
        prediction = torch.load(result_path, weights_only=True)  # prediction: (all_frames, h, w)
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
    generate_frame(
        u_real, v_real, u_prediction, v_prediction, image_save_path, dir_frame_range
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

        u_prediction = torch.load(u_prediction_path, weights_only=True)  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(v_prediction_path, weights_only=True)  # v_prediction: (all_frames, h, w)
    else:
        prediction_path = prediction_path / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)
        prediction = torch.load(prediction_path, weights_only=True)  # prediction: (all_frames, h, w)
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


def calculate_metrics(y_true, y_pred):
    """
    y_true: Tensor of shape (n, c, h, w), true values
    y_pred: Tensor of shape (n, c, h, w), predicted values
    """
    # RMSE (Root Mean Squared Error) over all dimensions
    rmse_mat = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=(-1, -2)))
    rmse = torch.mean(rmse_mat)

    # Normalized RMSE (nRMSE)
    range_y = torch.amax(y_true, dim=(-1, -2)) - torch.amin(y_true, dim=(-1, -2))
    mean_y = torch.mean(y_true, dim=(-1, -2))
    nrmse_range = torch.mean(rmse_mat / range_y)
    nrmse_mean = torch.mean(rmse_mat / mean_y)

    # Maximum Error
    max_err_mat = torch.amax(torch.abs(y_true - y_pred), dim=(-1, -2))
    max_error = torch.mean(max_err_mat)

    return {
        "RMSE": rmse.item(),
        "nRMSE (range)": nrmse_range.item(),
        "nRMSE (mean)": nrmse_mean.item(),
        "Max Error": max_error.item(),
    }


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

        u_prediction = torch.load(u_prediction_path, weights_only=True)  # u_prediction: (all_frames, h, w)
        v_prediction = torch.load(v_prediction_path, weights_only=True)  # v_prediction: (all_frames, h, w)
    else:
        prediction_path = prediction_path / dir_name / "preds.pt"
        result_save_path.mkdir(exist_ok=True, parents=True)
        prediction = torch.load(prediction_path, weights_only=True)  # prediction: (all_frames, h, w)
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
    type: str = "predict"   # "predict" or "compute"
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
