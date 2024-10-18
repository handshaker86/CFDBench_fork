import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
import datetime
from pathlib import Path
from args import Args


def visualize_flow_field(u, v, color: str, label: str, ax):

    # Create a meshgrid for the coordinates
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

    # Create a quiver plot in the given axis
    ax.quiver(x, y, u, v, scale=30, color=color, width=0.005)
    ax.set_title(f"Flow Field - {label}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


def generate_frame(u_real_frame, v_real_frame, u_pred_frame, v_pred_frame):
    for i in range(v_real_frame.shape[0]):
        v_r = v_real_frame[i, :, :]
        u_r = u_real_frame[i, :, :]
        v_p = v_pred_frame[i, :, :]
        u_p = u_pred_frame[i, :, :]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        visualize_flow_field(u_r, v_r, "r", f"Real Field - Frame {i+1}", axs[0])

        visualize_flow_field(u_p, v_p, "g", f"Predicted Field - Frame {i+1}", axs[1])

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()


def evaluate_frame(
    u_prediction_frame, v_prediction_frame, u_real_frame, v_real_frame
):  # take a whole case's frame as inout
    num_frames = u_prediction_frame.shape[0]
    num_points = 16 * 16

    rate_list = []

    for i in range(num_frames):
        u_p_frame = u_prediction_frame[i, :, :]
        v_p_frame = v_prediction_frame[i, :, :]
        u_r_frame = u_real_frame[i, :, :]
        v_r_frame = v_real_frame[i, :, :]

        vel_p = np.sqrt(u_p_frame**2 + v_p_frame**2)
        vel_r = np.sqrt(u_r_frame**2 + v_r_frame**2)
        angle_p = np.arctan2(v_p_frame, u_p_frame)
        angle_r = np.arctan2(v_r_frame, u_r_frame)

        delta_angle = angle_p - angle_r

        SI = np.cos(delta_angle)
        mask_1 = SI > 0.8

        MI = 1 - abs(vel_p - vel_r) / (vel_r + vel_p)
        mask_2 = MI > 0.8

        mask = mask_1 * mask_2
        good_pred_rate = np.sum(mask) / num_points
        rate_list.append(good_pred_rate)

    average_rate = sum(rate_list) / len(rate_list)

    return average_rate


def plot_average_rate(case_names: list, average_rate: list, save_path: Path):
    plt.scatter(case_names, average_rate)
    plt.title("good prediction average rate")
    plt.xlabel("Cases")
    plt.ylabel("Rate")
    plt.xticks(case_names)
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    plt.savefig(save_path / file_name)
    plt.show()


def get_test_dirs(case_dirs: Path, time_step: int, seed: int = 0):

    case_dirs_list = []
    test_case_dirs = []

    for name in ["bc", "geo", "prop"]:
        # for name in ["geo"]:
        case_dir = case_dirs / name
        this_case_dirs = sorted(
            case_dir.glob("case*"),
            key=lambda x: int(x.name[4:]),  # arrange case_dirs in order
        )
        case_dirs_list += this_case_dirs  # get a list of path

    assert len(case_dirs_list) > 0
    num_cases = len(case_dirs_list)
    print(f"num_cases: {num_cases}")
    random.seed(seed)
    random.shuffle(case_dirs_list)

    num_train = round(num_cases * 0.8)
    num_dev = round(num_cases * 0.1)

    for i in range(num_train + num_dev, num_cases):
        test_case_dir = case_dirs_list[i]
        u_file = test_case_dir / "u.npy"
        u_data = np.load(u_file)

        if u_data.shape[0] <= time_step:
            continue

        test_case_dirs.append(test_case_dir)

    print(f"test_cases:{len(test_case_dirs)}")

    return test_case_dirs


def get_frames(case_dirs, seed, time_step, u_pred_path, v_pred_path):
    u_real_frames = []  # each element is a 3d-frame
    v_real_frames = []
    u_pred_frames = []
    v_pred_frames = []
    name_list = []
    frame_num_list = []

    test_case_dirs = get_test_dirs(case_dirs, time_step, seed)

    for test_case in test_case_dirs:
        case_name = os.path.basename(test_case)
        case_name = case_name[5:]
        name_list.append(case_name)

        u_data = np.load(test_case / "u.npy")
        frame_num_list.append(u_data.shape[0])
        u_real_frame = u_data[time_step:, ::4, ::4]

        v_data = np.load(test_case / "v.npy")
        v_real_frame = v_data[time_step:, ::4, ::4]

        u_real_frames.append(u_real_frame)
        v_real_frames.append(v_real_frame)

    u_preds = torch.load(u_pred_path)  # u_preds is a list
    v_preds = torch.load(v_pred_path)  # v_preds is a list

    for u_pred in u_preds:
        u_pred = u_pred.reshape(-1, 16, 16).numpy()
        u_pred_frames.append(u_pred)

    for v_pred in v_preds:
        v_pred = v_pred.reshape(-1, 16, 16).numpy()
        v_pred_frames.append(v_pred)

    return (
        u_real_frames,
        v_real_frames,
        u_pred_frames,
        v_pred_frames,
        name_list,
        frame_num_list,
    )


def evaluate_many_frames(
    case_dirs, seed, time_step, u_pred_path: Path, v_pred_path: Path, save_path: Path
):
    average_list = []
    u_frame_list = []
    v_frame_list = []

    (
        u_real_frames,
        v_real_frames,
        u_pred_frames,
        v_pred_frames,
        name_list,
        frame_num_list,
    ) = get_frames(
        case_dirs, seed, time_step, u_pred_path, v_pred_path
    )  # four lists

    for u_real_frame in u_real_frames:
        for i in range(u_real_frame.shape[0]):
            u_frame_list.append(np.expand_dims(u_real_frame[i], 0))

    for v_real_frame in v_real_frames:
        for i in range(v_real_frame.shape[0]):
            v_frame_list.append(np.expand_dims(v_real_frame[i], 0))

    u_real_frames = u_frame_list
    v_real_frames = v_frame_list

    assert (
        len(u_real_frames)
        == len(v_real_frames)
        == len(u_pred_frames)
        == len(v_pred_frames)
    )

    for j in range(len(frame_num_list)):
        frame_num = frame_num_list[j]
        for i in range(frame_num):
            u_real_frame = u_real_frames[i]
            v_pred_frame = v_pred_frames[i]
            u_pred_frame = u_pred_frames[i]
            v_real_frame = v_real_frames[i]
            average = evaluate_frame(
                u_pred_frame, v_pred_frame, u_real_frame, v_real_frame
            )
            average_list.append(average)
        accuracy = sum(average_list) / len(average_list)
        case_name = name_list[j]
        print(f"{case_name} accuracy : {accuracy}")

    # for i in range(len(u_real_frames)):
    #     u_real_frame = u_real_frames[i]
    #     v_pred_frame = v_pred_frames[i]
    #     u_pred_frame = u_pred_frames[i]
    #     v_real_frame = v_real_frames[i]

    #     average = evaluate_frame(u_pred_frame,v_pred_frame, u_real_frame,v_real_frame)
    #     average_list.append(average)

    # overall_accuracy = sum(average_list) / len(average_list)
    # print(f"average accuracy: {overall_accuracy}")

    # data_nums = len(average_list)
    # groups = data_nums // 20
    # remainder = data_nums % 20
    # if remainder != 0:
    #     groups += 1

    # for j in range(groups):
    #     start_idx = j * 20
    #     end_idx = min((i + 1) * 20, data_nums-1)
    #     name = name_list[start_idx:end_idx]
    #     average = average_list[start_idx:end_idx]
    #     plot_average_rate(name_list,average_list,save_path)


seed = 0
time_step = 10
args = Args().parse_args()
case_dirs = Path(args.data_dir + "/cavity")
save_path = Path(
    "/dssg/home/acct-iclover/iclover/xiaowen_ML_test/CFDBench/src/result/deeponet"
)
u_pred_path = Path(
    "/dssg/home/acct-iclover/iclover/xiaowen_ML_test/CFDBench/src/result/deeponet/u_result(16)/auto/cavity_prop_geo_bc/dt1.0/auto_deeponet/lr0.001_width100_depthb8_deptht8_normprop1_actrelu/test/preds.pt"
)
v_pred_path = Path(
    "/dssg/home/acct-iclover/iclover/xiaowen_ML_test/CFDBench/src/result/deeponet/v_result(16)/auto/cavity_prop_geo_bc/dt1.0/auto_deeponet/lr0.001_width100_depthb8_deptht8_normprop1_actrelu/test/preds.pt"
)

evaluate_many_frames(case_dirs, seed, time_step, u_pred_path, v_pred_path, save_path)
