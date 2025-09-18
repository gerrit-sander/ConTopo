"""Run one experiment script for every model folder inside a root directory.

Usage
-----
    python run_all_experiments.py exp_actmaps.py ./save/ShallowCNN/models/CE_runs

The command above will execute:
    python exp_actmaps.py <model_folder>
for every immediate subdirectory inside the provided models root.  Extra
arguments supplied after `--` are forwarded to each experiment invocation.

Pass `--log-dir ./logs` to also append combined stdout/stderr for every run to
`./logs/<experiment>.log`.
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
        "--log-dir",
        help="Optional directory to append combined stdout/stderr logs.",
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


def build_log_path(log_dir: Path, experiment: Path) -> Path:
    filename = f"{experiment.stem}.log"
    return log_dir / filename


def run_experiment(
    python_exe: str,
    experiment: Path,
    model_dir: Path,
    extra_args: list[str] | None,
    dry_run: bool,
    log_dir: Path | None,
) -> int:
    cmd = [python_exe, str(experiment), str(model_dir)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    if dry_run:
        if log_dir:
            log_path = build_log_path(log_dir, experiment)
            prepend = "\n" if log_path.exists() and log_path.stat().st_size > 0 else ""
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    prepend
                    + "=== DRY RUN ===\n"
                    f"Model directory: {model_dir}\n"
                    f"Command: {' '.join(cmd)}\n"
                )
        return 0

    if log_dir:
        log_path = build_log_path(log_dir, experiment)
        completed = subprocess.run(
            cmd,
            cwd=experiment.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = completed.stdout or ""
        if output:
            print(output, end="")
        prepend = "\n" if log_path.exists() and log_path.stat().st_size > 0 else ""
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                prepend
                + "=== RUN START ===\n"
                f"Model directory: {model_dir}\n"
                f"Command: {' '.join(cmd)}\n"
                "--- Output ---\n"
            )
            handle.write(output)
            if not output.endswith("\n"):
                handle.write("\n")
            handle.write("=== RUN END ===\n")
    else:
        completed = subprocess.run(cmd, cwd=experiment.parent)
    return completed.returncode


def main() -> None:
    args = parse_args()
    exp_path, models_root = resolve_paths(args.experiment, args.models_root)
    model_folders = collect_model_folders(models_root)
    log_dir: Path | None = None
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

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
            log_dir=log_dir,
        )
        if rc != 0:
            print(f"    -> exit code {rc}")
            failures += 1

    if failures:
        raise SystemExit(f"{failures} run(s) failed out of {total}.")


if __name__ == "__main__":
    main()
