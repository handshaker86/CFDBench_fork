from pathlib import Path
from typing import List
import time
from shutil import copyfile
from copy import deepcopy

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from dataset.base import CfdAutoDataset
from dataset import get_auto_dataset
from models.base_model import AutoCfdModel
from models.auto_deeponet import AutoDeepONet
from models.auto_edeeponet import AutoEDeepONet
from models.auto_deeponet_cnn import AutoDeepONetCnn
from models.auto_ffn import AutoFfn
from utils import (
    dump_json,
    plot,
    plot_loss,
    get_output_dir,
    get_robustness_dir_name,
    load_best_ckpt,
    plot_predictions,
    check_path_exists,
)
from utils_auto import init_model
from args import Args
from get_result import (
    get_visualize_result,
    get_case_accuracy,
    cal_loss,
    cal_time,
)


def collate_fn(batch: list):
    # batch is a list of tuples (input_frame, label_frame, case_params)
    inputs, labels, case_params = zip(*batch)
    inputs = torch.stack(inputs)  # (b, 3, h, w)
    labels = torch.stack(labels)  # (b, 3, h, w)

    # The last channel from features is the binary mask.
    labels = labels[:, :-1]  # (b, 2, h, w)
    mask = inputs[:, -1:]  # (b, 1, h, w)
    inputs = inputs[:, :-1]  # (b, 2, h, w)

    # Case params is a dict, turn it into a tensor
    keys = [x for x in case_params[0].keys() if x not in ["rotated", "dx", "dy"]]
    case_params_vec = []
    for case_param in case_params:
        case_params_vec.append([case_param[k] for k in keys])
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    # Build the kwargs dict for the model's forward method
    return dict(
        inputs=inputs.cuda(),
        label=labels.cuda(),
        mask=mask.cuda(),
        case_params=case_params.cuda(),
    )


def evaluate(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    batch_size: int = 2,
    plot_interval: int = 1,
    measure_time: bool = False,
):
    if measure_time:
        assert batch_size == 1

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    scores = {name: [] for name in model.loss_fn.get_score_names()}
    input_scores = deepcopy(scores)
    all_preds: List[Tensor] = []
    inference_time = 0.0
    compute_time = 0.0
    print("=== Evaluating ===")
    print(f"# examples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"# batches: {len(loader)}")
    print(f"Plot interval: {plot_interval}")
    print(f"Output dir: {output_dir}")
    model.eval()

    # inference warm up
    print("Warming up GPU...")
    with torch.inference_mode():
        for _ in range(10):
            batch = next(iter(loader))
            _ = model(**batch)
    torch.cuda.synchronize() 
    print("GPU is ready!")

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(loader)):
            # inputs, labels, case_params = batch
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            inputs = batch["inputs"]  # (b, 2, h, w)
            labels = batch["label"]  # (b, 2, h, w)

            # Compute the prediction
            start_compute_time = time.perf_counter()
            outputs: dict = model(**batch)
            torch.cuda.synchronize()
            end_compute_time = time.perf_counter()
            loss: dict = outputs["loss"]
            preds: Tensor = outputs["preds"]
            height, width = inputs.shape[2:]

            # When using DeepONetAuto, the prediction is a flattened.
            preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
            # preds = preds.repeat(1, 3, 1, 1)
            all_preds.append(preds.cpu().detach())
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_time += end_time - start_time
            compute_time += end_compute_time - start_compute_time
            # loss = model.loss_fn(labels=labels[:, :1], preds=preds)

            # Compute difference between the input and label
            input_loss: dict = model.loss_fn(labels=labels[:, :1], preds=inputs[:, :1])
            for key in input_scores:
                input_scores[key].append(input_loss[key].cpu().tolist())
            for key in scores:
                scores[key].append(loss[key].cpu().tolist())

            if step % plot_interval == 0 and not measure_time:
                # Dump input, label and prediction flow images.
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=inputs[0][0],
                    label=labels[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )
    with open(output_dir / "predict_time.txt", "w") as f:
        f.write(f"Time taken for generating prediction: {inference_time}")
    with open(output_dir / "compute_time.txt", "w") as f:
        f.write(f"Time taken for model computation: {compute_time}")
    print(f"Predict has been saved to {output_dir/'predict_time.txt'}")
    print(f"Computation time has been saved to {output_dir/'compute_time.txt'}")

    if measure_time:
        print("Memory usage:")
        print(torch.cuda.memory_summary("cuda"))
        print("Time usage:")
        time_per_step = 1000 * (end_time - start_time) / len(loader)
        print(f"Time (ms) per step: {time_per_step:.3f}")
        exit()

    avg_scores = {}
    for key in scores:
        mean = np.mean(scores[key])
        input_mean = np.mean(input_scores[key])
        avg_scores[key] = mean
        avg_scores[f"input_{key}"] = input_mean
        print(f"Prediction {key}: {mean}")
        print(f"     Input {key}: {input_mean}")

    plot_loss(scores["nmse"], output_dir / "loss.png")
    return dict(
        preds=torch.cat(all_preds, dim=0),  # preds: (all_frames, 1, h, w)
        scores=dict(
            mean=avg_scores,
            all=scores,
        ),
    )


def test(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    infer_steps: int = 200,
    plot_interval: int = 10,
    batch_size: int = 1,
    measure_time: bool = False,
):
    assert infer_steps > 0
    assert plot_interval > 0
    output_dir.mkdir(exist_ok=True, parents=True)
    print("=== Testing ===")
    print(f"batch_size: {batch_size}")
    print(f"Plot interval: {plot_interval}")
    result = evaluate(
        model,
        data,
        output_dir=output_dir,
        batch_size=batch_size,
        plot_interval=plot_interval,
        measure_time=measure_time,
    )
    preds = torch.squeeze(result["preds"], 1)  # preds: (all_frames, h, w)
    scores = result["scores"]
    torch.save(preds, output_dir / "preds.pt")
    dump_json(scores, output_dir / "scores.json")
    print("=== Testing done ===")


def train(
    model: AutoCfdModel,
    train_data: CfdAutoDataset,
    dev_data: CfdAutoDataset,
    output_dir: Path,
    num_epochs: int = 400,
    lr: float = 1e-3,
    lr_step_size: int = 1,
    lr_gamma: float = 0.9,
    batch_size: int = 2,
    eval_batch_size: int = 2,
    log_interval: int = 10,
    eval_interval: int = 2,
    measure_time: bool = False,
):
    """
    Main function for training.

    ### Parameters
    - model
    - train_data
    - dev_data
    - output_dir
    ...
    - log_interval: log loss, learning rate etc. every `log_interval` steps.
    - measure_time: if `True`, will only run one epoch and print the time.
    """
    train_start_time = time.time()
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    print("====== Training ======")
    print(f"# batch: {batch_size}")
    print(f"# examples: {len(train_data)}")
    print(f"# step: {len(train_loader)}")
    print(f"# epoch: {num_epochs}")

    start_time = time.time()
    global_step = 0
    train_losses = []

    for ep in range(num_epochs):
        ep_start_time = time.time()
        ep_train_losses = []
        for step, batch in enumerate(train_loader):
            # Forward
            outputs: dict = model(**batch)
            if step == 0 and not measure_time:
                out_file = Path("example.png")
                inputs = batch["inputs"]
                labels = batch["label"]
                preds = outputs["preds"]
                if any(
                    isinstance(model, t)
                    for t in [
                        AutoDeepONet,
                        AutoEDeepONet,
                        AutoFfn,
                        AutoDeepONetCnn,
                    ]
                ):
                    plot(inputs[0][0], labels[0][0], labels[0][0], out_file)
                else:
                    plot(inputs[0][0], labels[0][0], preds[0][0], out_file)

            # Backward
            loss: dict = outputs["loss"]
            # print(loss)
            loss["nmse"].backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            ep_train_losses.append(loss["nmse"].item())
            global_step += 1
            if global_step % log_interval == 0:
                log_info = dict(
                    ep=ep,
                    step=step,
                    mse=f"{loss['mse'].item():.3e}",
                    nmse=f"{loss['nmse'].item():.3e}",
                    lr=f"{scheduler.get_last_lr()[0]:.3e}",
                    time=round(time.time() - start_time),
                )
                print(log_info)

        if measure_time:
            print("Memory usage:")
            print(torch.cuda.memory_summary("cuda"))
            print("Time usage:")
            print(time.time() - ep_start_time)
            exit()

        scheduler.step()
        train_losses += ep_train_losses

        # Plot
        if (ep + 1) % eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            result = evaluate(model, dev_data, ckpt_dir, batch_size=eval_batch_size)
            dev_scores = result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save checkpoint
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            if ckpt_path.exists():
                ckpt_backup_path = ckpt_dir / "backup_model.pt"
                print(f"Backing up old checkpoint to {ckpt_backup_path}")
                copyfile(ckpt_path, ckpt_backup_path)
            torch.save(model.state_dict(), ckpt_path)

            # Save average scores
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["all"]["nmse"]),  # type: ignore
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")
    train_end_time = time.time()
    with open(output_dir / "train_time.txt", "w") as f:
        f.write(f"Time taken for training: {train_end_time - train_start_time}")
    print(f"Training time has been saved to {output_dir/'train_time.txt'}")
    print("====== Training done ======")
    dump_json(train_losses, output_dir / "train_losses.json")
    plot_loss(train_losses, output_dir / "train_losses.png")


def main():
    args = Args().parse_args()
    print("#" * 80)
    print(args)
    print("#" * 80)

    output_dir = get_output_dir(args, is_auto=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    robustness_test = args.robustness_test
    args.save(str(output_dir / "args.json"))

    kwargs = dict(
        mask_range=args.mask_range,
        noise_mode=args.noise_mode,
        noise_std=args.noise_std,
        edge_width=args.edge_width,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
    )

    # Data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, dev_data, test_data = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        robustness_test=robustness_test,
        **kwargs,
    )

    assert train_data is not None
    assert dev_data is not None
    assert test_data is not None
    print(f"# train examples: {len(train_data)}")
    print(f"# dev examples: {len(dev_data)}")
    print(f"# test examples: {len(test_data)}")

    # Model
    print("Loading model")
    model = init_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")

    if "train" in args.mode:
        args.save(str(output_dir / "train_args.json"))
        train(
            model,
            train_data=train_data,
            dev_data=dev_data,
            output_dir=output_dir,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
        )
    if "test" in args.mode:
        # Test
        args.save(str(output_dir / "test_args.json"))
        load_best_ckpt(model, output_dir)

        # process test_dir
        if robustness_test:
            dir_name = get_robustness_dir_name(args)
            test_dir = output_dir / "robustness_test" / dir_name
        else:
            test_dir = output_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        test(
            model,
            test_data,
            test_dir,
            batch_size=len(test_data),
            infer_steps=20,
            plot_interval=10,
        )

    # Calculate prediction accuracy
    if args.cal_case_accuracy:
        # init output_dir
        if args.model == "auto_deeponet":
            output_dir = output_dir.parent

        # init result_save_path
        if "cylinder" in args.data_name:
            time_step = int(args.delta_time / 0.001)
        else:
            time_step = int(args.delta_time / 0.1)

        if len(args.model.split("_")) > 1:
            model_name = args.model.split("_")[1]
        else:
            model_name = args.model
        data_name = args.data_name
        if robustness_test:
            prefix = "robustness_test_"
            test_dir_name = get_robustness_dir_name(args)
        else:
            prefix = ""
            test_dir_name = ""
        result_save_path = Path(
            prefix
            + f"results/time_step={time_step}/{model_name}/{data_name}/{test_dir_name}"
        )

        # check if the results exists
        if args.model == "auto_deeponet":
            u_result_path = Path(output_dir / "u")
            v_result_path = Path(output_dir / "v")
            if not check_path_exists(u_result_path):
                raise FileNotFoundError(
                    f"[Warning] u velocity results in {output_dir} not found, please run the test first"
                )
            elif not check_path_exists(v_result_path):
                raise FileNotFoundError(
                    f"[Warning] v velocity results in {output_dir} not found, please run the test first"
                )
        else:
            if not check_path_exists(output_dir):
                raise FileNotFoundError(
                    f"[Warning] {output_dir} not found, please run the test first"
                )

        is_autodeeponet = args.model == "auto_deeponet"
        get_case_accuracy(
            test_data,
            output_dir,
            result_save_path,
            args,
            is_autodeeponet,
            robustness_test,
        )
        cal_loss(
            test_data,
            output_dir,
            result_save_path,
            args,
            is_autodeeponet,
            robustness_test,
        )
        if not args.robustness_test:
            cal_time(output_dir, result_save_path, is_autodeeponet, type="predict")
            cal_time(output_dir, result_save_path, is_autodeeponet, type="compute")

    # Visualize prediction
    if args.visualize:
        # init output_dir
        if args.model == "auto_deeponet":
            output_dir = output_dir.parent

        # check if the results exists
        if args.model == "auto_deeponet":
            u_result_path = Path(output_dir / "u")
            v_result_path = Path(output_dir / "v")
            if not check_path_exists(u_result_path):
                raise FileNotFoundError(
                    f"[Warning] u velocity results in {output_dir} not found, please run the test first"
                )
            elif not check_path_exists(v_result_path):
                raise FileNotFoundError(
                    f"[Warning] v velocity results in {output_dir} not found, please run the test first"
                )
        else:
            if not check_path_exists(output_dir):
                raise FileNotFoundError(
                    f"[Warning] {output_dir} not found, please run the test first"
                )
        is_autodeeponet = args.model == "auto_deeponet"
        get_visualize_result(
            test_data, output_dir, args.data_to_visualize, is_autodeeponet
        )


if __name__ == "__main__":
    main()
