import os
import argparse
import itertools
import subprocess


# noise_std_list = [0.01, 0.1, 0.3, 0.5, 0.8]
# edge_width_list = [2, 4, 8, 16, 32]

# block_configs = [
#     {"block_size": 4, "num_blocks": 5},  # 2% masking
#     {"block_size": 8, "num_blocks": 5},  # 8% masking
#     {"block_size": 12, "num_blocks": 10},  # 36% masking
#     {"block_size": 16, "num_blocks": 10},  # 64% masking
#     {"block_size": 16, "num_blocks": 15},  # 96% masking
# ]

noise_std_list = [0.01, 0.02, 0.05, 0.08, 0.1]
edge_width_list = [1, 2, 4, 8, 10]

block_configs = [
    {"block_size": 3, "num_blocks": 5},  # 1% masking
    {"block_size": 4, "num_blocks": 5},  # 2% masking
    {"block_size": 4, "num_blocks": 10},  # 4% masking
    {"block_size": 8, "num_blocks": 5},  # 8% masking
    {"block_size": 8, "num_blocks": 6},  # 10% masking
]


def run_robustness_test(
    data_name,
    dataset_path,
    model,
    delta_time,
    noise_std,
    edge_width,
    block_config=None,
    mask_range=None,
    noise_mode=None,
    velocity_dim=0,
):

    # Construct log file path
    log_dir = f"output_logs/robustness_test/{model}/{data_name}/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

    # Build descriptive suffix
    suffix_parts = []
    if noise_std is not None:
        suffix_parts.append(f"ns={noise_std}")
    if edge_width is not None:
        suffix_parts.append(f"ew={edge_width}")
    if block_config is not None:
        suffix_parts.append(
            f"bs={block_config['block_size']}_nb={block_config['num_blocks']}"
        )
    if mask_range:
        suffix_parts.append(f"mask_{mask_range}")
    if noise_mode:
        suffix_parts.append(f"mode_{noise_mode}")

    log_file_name = "_".join(suffix_parts) + ".txt"
    output_file = os.path.join(log_dir, log_file_name)

    command = [
        "python",
        "src/train_auto.py",
        "--mode",
        "test",
        "--model",
        f"{model}",
        "--data_name",
        f"{data_name}",
        "--data_dir",
        f"{dataset_path}",
        "--delta_time",
        f"{delta_time}",
        "--robustness_test",
        "--mask_range",
        f"{mask_range}",
        "--noise_mode",
        f"{noise_mode}",
        "--noise_std",
        str(noise_std),
        "--edge_width",
        str(edge_width),
        "--block_size",
        str(block_config["block_size"]) if block_config else "0",
        "--num_blocks",
        str(block_config["num_blocks"]) if block_config else "0",
        "--velocity_dim",
        str(velocity_dim),
    ]

    command_str = " ".join(command)
    print(f"Running experiment: {command_str} (Logging to {output_file})")

    with open(output_file, "w") as log_file:
        subprocess.run(command, stdout=log_file, stderr=log_file)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU you want to use, or "0" for CPU 

def main():
    parser = argparse.ArgumentParser(description="Run robustness tests.")
    parser.add_argument(
        "--data_name",
        default="cavity_bc_geo_prop",
        help="Specify the name of the dataset. Default is 'cavity_bc_geo_prop'.",
    )
    parser.add_argument(
        "--dataset_path",
        default="data/cfdbench_dataset",
        help="Specify the path to the dataset.",
    )
    parser.add_argument(
        "--model",
        default="MLP",
        help="Specify the model pool to use. Default is 'MLP'.",
    )
    parser.add_argument(
        "--delta_time",
        type=float,
        default=1.0,
        help="Specify the delta time for the dataset. Default is 1.0.",
    )
    parser.add_argument(
        "--mask_range",
        choices=["edge", "global"],
        default=None,
        help="Specify the mask range. Default is 'edge'.",
    )
    parser.add_argument(
        "--noise_mode",
        choices=["noise", "zero"],
        default=None,
        help="Specify the noise mode. Default is 'noise'.",
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    model = args.model
    mask_range = args.mask_range
    noise_mode = args.noise_mode
    data_name = args.data_name
    delta_time = args.delta_time

    if mask_range == "edge" and noise_mode == "noise":
        for edge_width, noise_std in itertools.product(edge_width_list, noise_std_list):
            run_robustness_test(
                data_name=data_name,
                dataset_path=dataset_path,
                model=model,
                delta_time=delta_time,
                noise_std=noise_std,
                edge_width=edge_width,
                block_config=None,
                mask_range=mask_range,
                noise_mode=noise_mode,
            )
            if model == "auto_deeponet":
                run_robustness_test(
                    data_name=data_name,
                    dataset_path=dataset_path,
                    model=model,
                    delta_time=delta_time,
                    noise_std=noise_std,
                    edge_width=edge_width,
                    block_config=None,
                    mask_range=mask_range,
                    noise_mode=noise_mode,
                    velocity_dim=1,
                )
    elif mask_range == "edge" and noise_mode == "zero":
        for edge_width in edge_width_list:
            run_robustness_test(
                data_name=data_name,
                dataset_path=dataset_path,
                model=model,
                delta_time=delta_time,
                noise_std=0.0,
                edge_width=edge_width,
                block_config=None,
                mask_range=mask_range,
                noise_mode=noise_mode,
            )
            if model == "auto_deeponet":
                run_robustness_test(
                    data_name=data_name,
                    dataset_path=dataset_path,
                    model=model,
                    delta_time=delta_time,
                    noise_std=0.0,
                    edge_width=edge_width,
                    block_config=None,
                    mask_range=mask_range,
                    noise_mode=noise_mode,
                    velocity_dim=1,
                )

    elif mask_range == "global" and noise_mode == "noise":
        for cfg, noise_std in itertools.product(block_configs, noise_std_list):
            run_robustness_test(
                data_name=data_name,
                dataset_path=dataset_path,
                model=model,
                delta_time=delta_time,
                noise_std=noise_std,
                edge_width=0,
                block_config=cfg,
                mask_range=mask_range,
                noise_mode=noise_mode,
            )
            if model == "auto_deeponet":
                run_robustness_test(
                    data_name=data_name,
                    dataset_path=dataset_path,
                    model=model,
                    delta_time=delta_time,
                    noise_std=noise_std,
                    edge_width=0,
                    block_config=cfg,
                    mask_range=mask_range,
                    noise_mode=noise_mode,
                    velocity_dim=1,
                )

    elif mask_range == "global" and noise_mode == "zero":
        for cfg in block_configs:
            run_robustness_test(
                data_name=data_name,
                dataset_path=dataset_path,
                model=model,
                delta_time=delta_time,
                noise_std=0.0,
                edge_width=0,
                block_config=cfg,
                mask_range=mask_range,
                noise_mode=noise_mode,
            )
            if model == "auto_deeponet":
                run_robustness_test(
                    data_name=data_name,
                    dataset_path=dataset_path,
                    model=model,
                    delta_time=delta_time,
                    noise_std=0.0,
                    edge_width=0,
                    block_config=cfg,
                    mask_range=mask_range,
                    noise_mode=noise_mode,
                    velocity_dim=1,
                )
    else:
        raise ValueError(
            "Invalid combination of mask_range and noise_mode. Please check the arguments."
        )


if __name__ == "__main__":
    main()
