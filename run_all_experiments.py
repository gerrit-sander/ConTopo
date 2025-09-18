"""Run one experiment script for every model folder inside a root directory.

Usage
-----
    python run_all_experiments.py exp_actmaps.py ./save/ShallowCNN/models/CE_runs

The command above will execute:
    python exp_actmaps.py <model_folder>
for every immediate subdirectory inside the provided models root.  Extra
arguments supplied after `--` are forwarded to each experiment invocation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an experiment for every model folder in a directory.",
    )
    parser.add_argument(
        "experiment",
        help="Path to the experiment script to execute (e.g. exp_generateRDM.py)",
    )
    parser.add_argument(
        "models_root",
        help="Directory whose immediate subfolders will be passed to the experiment.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for launching the experiment (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to append to every experiment invocation. Use `--` to end the script arguments.",
    )
    args = parser.parse_args()

    # Drop the explicit "--" separator if present in REMAINDER.
    if args.experiment_args and args.experiment_args[0] == "--":
        args.experiment_args = args.experiment_args[1:]

    return args


def resolve_paths(experiment: str, models_root: str) -> tuple[Path, Path]:
    exp_path = Path(experiment).expanduser().resolve()
    models_root_path = Path(models_root).expanduser().resolve()

    if not exp_path.exists():
        raise SystemExit(f"Experiment script not found: {exp_path}")
    if not models_root_path.is_dir():
        raise SystemExit(f"Models root is not a directory: {models_root_path}")

    return exp_path, models_root_path


def collect_model_folders(models_root: Path) -> list[Path]:
    folders = [p for p in models_root.iterdir() if p.is_dir()]
    if not folders:
        raise SystemExit(f"No model folders found inside: {models_root}")
    return sorted(folders)


def run_experiment(
    python_exe: str,
    experiment: Path,
    model_dir: Path,
    extra_args: list[str] | None,
    dry_run: bool,
) -> int:
    cmd = [python_exe, str(experiment), str(model_dir)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=experiment.parent)
    return completed.returncode


def main() -> None:
    args = parse_args()
    exp_path, models_root = resolve_paths(args.experiment, args.models_root)
    model_folders = collect_model_folders(models_root)

    failures = 0
    total = len(model_folders)
    for idx, folder in enumerate(model_folders, 1):
        print(f"[{idx}/{total}] {folder}")
        rc = run_experiment(
            python_exe=args.python,
            experiment=exp_path,
            model_dir=folder,
            extra_args=args.experiment_args,
            dry_run=args.dry_run,
        )
        if rc != 0:
            print(f"    -> exit code {rc}")
            failures += 1

    if failures:
        raise SystemExit(f"{failures} run(s) failed out of {total}.")


if __name__ == "__main__":
    main()
