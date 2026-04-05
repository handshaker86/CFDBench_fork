"""
Tiny tuning sweep for CNO on CFDBench Dam.

Goal:
- Run a small grid over a few meaningful axes (hidden_dim, depth, kernel_size)
- Train only a few epochs (default: 5) to pick a reasonable config quickly
- Keep a "default-like" config in the sweep list to avoid fairness complaints

Example:
python3 CFDBench_fork/src/sweep_cno_dam.py --epochs 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SweepCfg:
    depth: int
    hidden_dim: int
    kernel_size: int

    def to_args(self) -> list[str]:
        return [
            f"--cno_depth={self.depth}",
            f"--cno_hidden_dim={self.hidden_dim}",
            f"--cno_kernel_size={self.kernel_size}",
        ]

    def name(self) -> str:
        return f"d={self.depth},h={self.hidden_dim},k={self.kernel_size}"


def _read_best_dev_nmse(run_dir: Path) -> Optional[float]:
    """
    We store per-eval checkpoints under run_dir/ckpt-*/scores.json.
    Pick the smallest dev_loss among them.
    """
    best = None
    for ckpt_dir in sorted(run_dir.glob("ckpt-*")):
        scores_path = ckpt_dir / "scores.json"
        if not scores_path.exists():
            continue
        with open(scores_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dev_loss = data.get("dev_loss", None)
        if dev_loss is None:
            continue
        dev_loss = float(dev_loss)
        if best is None or dev_loss < best:
            best = dev_loss
    return best


def _get_run_dir(
    output_root: Path,
    data_name: str,
    delta_time: float,
    cfg: SweepCfg,
    lr: float,
    padding: int,
) -> Path:
    # Match get_output_dir structure: result/auto/<data_name>/dt<dt>/<model>/<dir_name>
    # For cno, get_output_dir uses: lr{lr}_d{depth}_h{hidden}_k{kernel}_p{padding}
    dir_name = (
        f"lr{lr}_d{cfg.depth}_h{cfg.hidden_dim}_k{cfg.kernel_size}_p{padding}"
    )
    return output_root / "auto" / data_name / f"dt{delta_time}" / "cno" / dir_name


def _run_one(
    *,
    python: str,
    train_auto_py: Path,
    output_root: Path,
    data_name: str,
    data_dir: str,
    delta_time: float,
    epochs: int,
    eval_interval: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    lr_step_size: int,
    seed: int,
    cno_padding: int,
    cfg: SweepCfg,
) -> tuple[Path, int]:
    cmd = [
        python,
        str(train_auto_py),
        "--model=cno",
        f"--data_name={data_name}",
        f"--data_dir={data_dir}",
        f"--delta_time={delta_time}",
        f"--num_epochs={epochs}",
        f"--eval_interval={eval_interval}",
        f"--batch_size={batch_size}",
        f"--eval_batch_size={eval_batch_size}",
        f"--lr={lr}",
        f"--lr_step_size={lr_step_size}",
        f"--seed={seed}",
        "--mode=train",
        f"--output_dir={str(output_root)}",
        f"--cno_padding={cno_padding}",
        *cfg.to_args(),
    ]

    run_dir = _get_run_dir(output_root, data_name, delta_time, cfg, lr, cno_padding)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running {cfg.name()} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return run_dir, proc.returncode


def _default_sweep() -> list[SweepCfg]:
    """
    12 configs: 3x2x2 = 12 (hidden_dim x depth x kernel_size)
    Includes the default-like config (d=6,h=64,k=5).
    """
    hidden_dims = [64, 96, 128]
    depths = [4, 6, 8]
    kernel_sizes = [3, 5]
    return [
        SweepCfg(depth=d, hidden_dim=h, kernel_size=k)
        for h, d, k in itertools.product(hidden_dims, depths, kernel_sizes)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default="python3")
    p.add_argument(
        "--train_auto_py",
        default=str(Path(__file__).resolve().parent / "train_auto.py"),
    )
    p.add_argument("--output_root", default="result_sweeps")
    p.add_argument("--data_name", default="dam_bc_geo_prop")
    p.add_argument("--data_dir", default="data/cfdbench_dataset")
    p.add_argument("--delta_time", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_step_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cno_padding", type=int, default=2)
    args = p.parse_args()

    train_auto_py = Path(args.train_auto_py).resolve()
    output_root = Path(args.output_root).resolve()

    sweep: Iterable[SweepCfg] = _default_sweep()

    results: list[tuple[SweepCfg, Optional[float], Path, int]] = []

    print("### Tiny CNO sweep (Dam)")
    print(f"- train_auto: {train_auto_py}")
    print(f"- output_root: {output_root}")
    print(f"- data_name: {args.data_name}")
    print(f"- delta_time: {args.delta_time}")
    print(f"- epochs: {args.epochs}")
    print(
        "- sweep axes: hidden_dim in {64,96,128}, depth in {4,6,8}, kernel_size in {3,5}"
    )
    print("- default-like config included: d=6,h=64,k=5")

    for cfg in sweep:
        run_dir, code = _run_one(
            python=args.python,
            train_auto_py=train_auto_py,
            output_root=output_root,
            data_name=args.data_name,
            data_dir=args.data_dir,
            delta_time=args.delta_time,
            epochs=args.epochs,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            seed=args.seed,
            cno_padding=args.cno_padding,
            cfg=cfg,
        )
        dev_nmse = _read_best_dev_nmse(run_dir)
        results.append((cfg, dev_nmse, run_dir, code))
        print(f"-> return_code={code}, best_dev_nmse={dev_nmse}, dir={run_dir}")

    # Rank configs (lower is better). Put failures at the bottom.
    ranked = sorted(
        results,
        key=lambda x: (float("inf") if x[1] is None else x[1]),
    )

    print("\n### Ranked results (best dev_nmse first)")
    for i, (cfg, dev, run_dir, code) in enumerate(ranked, start=1):
        dev_str = "N/A" if dev is None else f"{dev:.6g}"
        print(f"{i:02d}. dev_nmse={dev_str}  rc={code}  {cfg.name()}  dir={run_dir}")

    best_cfg, best_dev, best_dir, best_code = ranked[0]
    print("\n### Best config (5-epoch proxy)")
    print(
        f"- cfg: {best_cfg.name()}\n- dev_nmse: {best_dev}\n- rc: {best_code}\n- dir: {best_dir}"
    )


if __name__ == "__main__":
    main()

