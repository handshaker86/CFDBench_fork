from pathlib import Path
from typing import List, Tuple, Optional
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
    load_json,
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
    measure_predict_time,
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


def _parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _read_best_dev_nmse_from_run(run_dir: Path) -> float:
    best = float("inf")
    for ckpt_dir in sorted(run_dir.glob("ckpt-*")):
        scores_path = ckpt_dir / "scores.json"
        if not scores_path.exists():
            continue
        scores = load_json(scores_path)
        dev_loss = scores.get("dev_loss", None)
        if dev_loss is None:
            continue
        best = min(best, float(dev_loss))
    return best


def _sweep_cno_and_pick_best(
    args: Args,
    train_data: CfdAutoDataset,
    dev_data: CfdAutoDataset,
    base_output_dir: Path,
) -> Tuple[int, int, int, List[dict]]:
    """
    Tiny CNO sweep that reuses already-loaded datasets.

    Returns: (best_depth, best_hidden_dim, best_kernel_size, all_results)
    """
    assert args.model == "cno"
    depths = _parse_int_list(args.cno_sweep_depths)
    hiddens = _parse_int_list(args.cno_sweep_hidden_dims)
    kernels = _parse_int_list(args.cno_sweep_kernel_sizes)
    assert depths and hiddens and kernels

    sweep_dir = base_output_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    best_cfg: Optional[Tuple[int, int, int]] = None
    best_dev_nmse: float = float("inf")

    print("=== CNO tiny sweep (reusing loaded data) ===")
    print(f"Sweep epochs per config: {args.cno_sweep_epochs}")
    print(f"depths={depths}, hidden_dims={hiddens}, kernel_sizes={kernels}")
    print("Default-like config included (if in grid): d=6,h=64,k=5")

    # For sweep speed: only evaluate once per config (at the end).
    # This avoids paying for repeated dev passes while still letting us
    # rank configs by a consistent dev metric.
    sweep_eval_interval = max(1, args.cno_sweep_epochs)

    for hidden_dim in hiddens:
        for depth in depths:
            for kernel_size in kernels:
                sweep_args = deepcopy(args)
                sweep_args.cno_hidden_dim = hidden_dim
                sweep_args.cno_depth = depth
                sweep_args.cno_kernel_size = kernel_size
                # Ensure spatial shapes stay consistent for residual blocks:
                # for stride=1, use "same" padding p=(k-1)//2.
                sweep_args.cno_padding = (kernel_size - 1) // 2

                cfg_name = (
                    f"lr{sweep_args.lr}_d{depth}_h{hidden_dim}_k{kernel_size}_p{sweep_args.cno_padding}"
                )
                cfg_out = sweep_dir / cfg_name
                cfg_out.mkdir(parents=True, exist_ok=True)
                sweep_args.save(str(cfg_out / "args.json"))

                model = init_model(sweep_args)
                train(
                    model=model,
                    train_data=train_data,
                    dev_data=dev_data,
                    output_dir=cfg_out,
                    num_epochs=sweep_args.cno_sweep_epochs,
                    lr=sweep_args.lr,
                    lr_step_size=sweep_args.lr_step_size,
                    batch_size=sweep_args.batch_size,
                    eval_batch_size=sweep_args.eval_batch_size,
                    eval_interval=sweep_eval_interval,
                    log_interval=sweep_args.log_interval,
                    measure_time=False,
                )

                cfg_best = _read_best_dev_nmse_from_run(cfg_out)
                rec = dict(
                    depth=depth,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=sweep_args.cno_padding,
                    best_dev_nmse=cfg_best,
                    out_dir=str(cfg_out),
                )
                results.append(rec)
                print(f"[sweep] {cfg_name}: best_dev_nmse={cfg_best}")

                if cfg_best < best_dev_nmse:
                    best_dev_nmse = cfg_best
                    best_cfg = (depth, hidden_dim, kernel_size)

    assert best_cfg is not None
    best_depth, best_hidden, best_kernel = best_cfg
    print("=== CNO sweep done ===")
    print(
        f"Best (proxy) config: depth={best_depth}, hidden_dim={best_hidden}, kernel={best_kernel}, dev_nmse={best_dev_nmse}"
    )
    return best_depth, best_hidden, best_kernel, results


def evaluate(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    batch_size: int = 2,
    plot_interval: int = 1,
    measure_time: bool = False,
    warmup_runs: int = 5,
    average_runs: int = 10,
):

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    scores = {name: [] for name in model.loss_fn.get_score_names()}
    input_scores = deepcopy(scores)
    all_preds: List[Tensor] = []
    all_run_times: List[float] = []

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
        it = iter(loader)
        for _ in range(warmup_runs):
            try:
                batch = next(it)
            except StopIteration:
                if len(data) == 0:
                    print("Warning: Empty dataset, skipping warmup.")
                    break
                it = iter(loader)
                batch = next(it)
            _ = model(**batch)
    torch.cuda.synchronize()
    print("GPU is ready!")

    for run_idx in range(average_runs):

        is_collecting_run = (
            run_idx == 0
        )  # Only the first run collects scores and predictions
        if is_collecting_run:
            # Reset collectors
            scores = {name: [] for name in model.loss_fn.get_score_names()}
            input_scores = deepcopy(scores)
            all_preds = []

        current_run_total_time = 0.0

        with torch.inference_mode():
            for step, batch in enumerate(tqdm(loader)):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                inputs = batch["inputs"]  # (b, 2, h, w)
                labels = batch["label"]  # (b, 2, h, w)

                # Compute the prediction
                outputs: dict = model(**batch)
                torch.cuda.synchronize()
                loss: dict = outputs["loss"]
                preds: Tensor = outputs["preds"]
                height, width = inputs.shape[2:]
                preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
                preds_cpu = preds.cpu().detach()
                if is_collecting_run:
                    all_preds.append(preds_cpu)

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                current_run_total_time += end_time - start_time

                if is_collecting_run:
                    # loss = model.loss_fn(labels=labels[:, :1], preds=preds)

                    # Compute difference between the input and label
                    input_loss: dict = model.loss_fn(
                        labels=labels[:, :1], preds=inputs[:, :1]
                    )
                    for key in input_scores:
                        input_scores[key].append(input_loss[key].cpu().tolist())
                    for key in scores:
                        scores[key].append(loss[key].cpu().tolist())

                    # if step % plot_interval == 0 and not measure_time:
                    # # Dump input, label and prediction flow images.
                    #     image_dir = output_dir / "images"
                    #     image_dir.mkdir(exist_ok=True, parents=True)
                    #     plot_predictions(
                    #         inp=inputs[0][0],
                    #         label=labels[0][0],
                    #         pred=preds[0][0],
                    #         out_dir=image_dir,
                    #         step=step,
                    #     )

            all_run_times.append(current_run_total_time)

    if not all_run_times:
        inference_time = 0.0
    elif measure_time:
        inference_time = sum(all_run_times) / len(all_run_times)
    else:
        inference_time = all_run_times[0]

    with open(output_dir / "predict_time.txt", "w") as f:
        f.write(f"Time taken for generating prediction: {inference_time}")

    print(f"Predict time has been saved to {output_dir/'predict_time.txt'}")

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
    - measure_time: if `True`, will only run several epochs and save average time.
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

    # Model (optionally sweep CNO hyperparams first, without reloading data)
    if args.model == "cno" and args.cno_sweep and "train" in args.mode:
        best_d, best_h, best_k, sweep_results = _sweep_cno_and_pick_best(
            args=args,
            train_data=train_data,
            dev_data=dev_data,
            base_output_dir=output_dir,
        )
        sweep_out = output_dir / "sweep"
        dump_json(sweep_results, sweep_out / "sweep_results.json")
        dump_json(
            dict(
                best=dict(
                    cno_depth=best_d,
                    cno_hidden_dim=best_h,
                    cno_kernel_size=best_k,
                    cno_padding=args.cno_padding,
                )
            ),
            sweep_out / "best_config.json",
        )
        # Stick to the selected config for the main run
        args.cno_depth = best_d
        args.cno_hidden_dim = best_h
        args.cno_kernel_size = best_k
        args.cno_padding = (best_k - 1) // 2

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
            # Training should not exit early unless explicitly requested.
            measure_time=False,
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
            # NOTE: Using the full test set as a single batch can easily OOM.
            # Use eval_batch_size to keep memory bounded.
            batch_size=args.eval_batch_size,
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

    # Visualize prediction
    if args.visualize:
        if args.model == "auto_deeponet":
            # Temporarily set velocity_dim to get base directory, then remove u/v
            visualize_output_dir = get_output_dir(args, is_auto=True)
            # Remove u or v from path (get_output_dir adds it based on velocity_dim)
            if visualize_output_dir.name in ["u", "v"]:
                visualize_output_dir = visualize_output_dir.parent
        else:
            visualize_output_dir = get_output_dir(args, is_auto=True)
        
        # Handle robustness_test path
        if robustness_test:
            dir_name = get_robustness_dir_name(args)
            if args.model == "auto_deeponet":
                u_result_path = Path(visualize_output_dir / "u" / "robustness_test" / dir_name)
                v_result_path = Path(visualize_output_dir / "v" / "robustness_test" / dir_name)
            else:
                result_path = Path(visualize_output_dir / "robustness_test" / dir_name)
        else:
            if args.model == "auto_deeponet":
                u_result_path = Path(visualize_output_dir / "u" / "test")
                v_result_path = Path(visualize_output_dir / "v" / "test")
            else:
                result_path = Path(visualize_output_dir / "test")

        # check if the results exists
        if args.model == "auto_deeponet":
            if not check_path_exists(u_result_path):
                raise FileNotFoundError(
                    f"[Warning] u velocity results in {u_result_path} not found, please run the test first"
                )
            elif not check_path_exists(v_result_path):
                raise FileNotFoundError(
                    f"[Warning] v velocity results in {v_result_path} not found, please run the test first"
                )
        else:
            if not check_path_exists(result_path):
                raise FileNotFoundError(
                    f"[Warning] {result_path} not found, please run the test first"
                )
        is_autodeeponet = args.model == "auto_deeponet"
        get_visualize_result(
            test_data, visualize_output_dir, args.data_to_visualize, is_autodeeponet, robustness_test, args
        )

    # Measure prediction time
    if args.measure_predict_time:
        print("=== Measuring Prediction Time ===")
        num_frames = args.measure_predict_num_frames
        is_autodeeponet = args.model == "auto_deeponet"

        if is_autodeeponet:
            # For auto_deeponet, measure u and v separately
            output_dir_parent = (
                output_dir.parent if output_dir.name in ["u", "v"] else output_dir
            )

            # Measure u model time
            print("Measuring u model prediction time...")
            args_u = deepcopy(args)
            args_u.velocity_dim = 0
            model_u = init_model(args_u)
            output_dir_u = get_output_dir(args_u, is_auto=True)
            if "test" not in args.mode:
                load_best_ckpt(model_u, output_dir_u)
            avg_time_u = measure_predict_time(
                model=model_u,
                dataset=test_data,
                num_frames=num_frames,
                num_runs=10,
                warmup_runs=5,
                batch_size=num_frames,
            )

            # Measure v model time
            print("Measuring v model prediction time...")
            args_v = deepcopy(args)
            args_v.velocity_dim = 1
            model_v = init_model(args_v)
            output_dir_v = get_output_dir(args_v, is_auto=True)
            if "test" not in args.mode:
                load_best_ckpt(model_v, output_dir_v)
            avg_time_v = measure_predict_time(
                model=model_v,
                dataset=test_data,
                num_frames=num_frames,
                num_runs=10,
                warmup_runs=5,
                batch_size=num_frames,
            )

            # Total time is sum of u and v
            avg_time = avg_time_u + avg_time_v

            # Save results
            if robustness_test:
                dir_name = get_robustness_dir_name(args)
                result_save_path_u = output_dir_u / "robustness_test" / dir_name
                result_save_path_v = output_dir_v / "robustness_test" / dir_name
                result_save_path = output_dir_parent / "robustness_test" / dir_name
            else:
                result_save_path_u = output_dir_u / "test"
                result_save_path_v = output_dir_v / "test"
                result_save_path = output_dir_parent / "test"

            result_save_path_u.mkdir(exist_ok=True, parents=True)
            result_save_path_v.mkdir(exist_ok=True, parents=True)
            result_save_path.mkdir(exist_ok=True, parents=True)

            with open(result_save_path_u / "measure_predict_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time_u:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time_u / num_frames:.6f}s\n")

            with open(result_save_path_v / "measure_predict_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time_v:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time_v / num_frames:.6f}s\n")

            with open(result_save_path / "measure_predict_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time / num_frames:.6f}s\n")
                f.write(f"u model time: {avg_time_u:.4f}s\n")
                f.write(f"v model time: {avg_time_v:.4f}s\n")

            # Save final result to benchmark directory
            if len(args.model.split("_")) > 1:
                model_name = args.model.split("_")[1]
            else:
                model_name = args.model
            data_name = args.data_name
            benchmark_path = Path(f"results/benchmark/{model_name}/{data_name}")
            benchmark_path.mkdir(exist_ok=True, parents=True)
            with open(benchmark_path / "prediction_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time / num_frames:.6f}s\n")
                f.write(f"u model time: {avg_time_u:.4f}s\n")
                f.write(f"v model time: {avg_time_v:.4f}s\n")

            print(
                f"Prediction time measurement saved to {result_save_path / 'measure_predict_time.txt'}"
            )
            print(
                f"Final benchmark result saved to {benchmark_path / 'prediction_time.txt'}"
            )
        else:
            # For other models, measure normally
            if "test" not in args.mode:
                load_best_ckpt(model, output_dir)

            avg_time = measure_predict_time(
                model=model,
                dataset=test_data,
                num_frames=num_frames,
                num_runs=10,
                warmup_runs=5,
                batch_size=num_frames,
            )

            # Save result
            if robustness_test:
                dir_name = get_robustness_dir_name(args)
                result_save_path = output_dir / "robustness_test" / dir_name
            else:
                result_save_path = output_dir / "test"
            result_save_path.mkdir(exist_ok=True, parents=True)
            with open(result_save_path / "measure_predict_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time / num_frames:.6f}s\n")

            # Save final result to benchmark directory
            if len(args.model.split("_")) > 1:
                model_name = args.model.split("_")[1]
            else:
                model_name = args.model
            data_name = args.data_name
            benchmark_path = Path(f"results/benchmark/{model_name}/{data_name}")
            benchmark_path.mkdir(exist_ok=True, parents=True)
            with open(benchmark_path / "prediction_time.txt", "w") as f:
                f.write(
                    f"Average prediction time for {num_frames} frames: {avg_time:.4f}s\n"
                )
                f.write(f"Time per frame: {avg_time / num_frames:.6f}s\n")

            print(
                f"Prediction time measurement saved to {result_save_path / 'measure_predict_time.txt'}"
            )
            print(
                f"Final benchmark result saved to {benchmark_path / 'prediction_time.txt'}"
            )


if __name__ == "__main__":
    main()
