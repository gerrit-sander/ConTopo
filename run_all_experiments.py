"""
Run an experiment over saved models.

Modes:
  - checkpoint: run <experiment>.py for every checkpoint file under REPO_ROOT/save (default)
  - modeldir:   run <experiment>.py for every immediate subfolder under REPO_ROOT/save/models

Usage examples:
  # from repo root (uses ./save), checkpoint mode (default)
  python run_all_experiments.py exp_actmaps.py

  # explicitly choose mode and repo root
  python run_all_experiments.py exp_generateRDM.py --mode modeldir --repo-root /path/to/repo
"""

import sys
import argparse
import subprocess
from pathlib import Path

CKPT_PATTERNS = ("*.pth", "*.pt", "*.ckpt")


def collect_checkpoints(repo_root: Path) -> list[Path]:
    save_dir = repo_root / "save"
    ckpts = sorted({
        p.relative_to(repo_root)
        for pat in CKPT_PATTERNS
        for p in save_dir.rglob(pat)
        if p.is_file()
    })
    return ckpts


def collect_modeldirs(repo_root: Path) -> list[Path]:
    """Return immediate subdirectories under <repo_root>/save/models as relative paths."""
    models_root = repo_root / "save" / "models"
    if not models_root.exists():
        return []
    dirs = [p for p in models_root.iterdir() if p.is_dir()]
    return sorted(p.relative_to(repo_root) for p in dirs)


def main():
    parser = argparse.ArgumentParser(description="Run experiment over saved models")
    parser.add_argument("experiment", help="Path to experiment script (e.g., exp_generateRDM.py)")
    parser.add_argument("--mode", choices=["checkpoint", "modeldir"], default="checkpoint",
                        help="Iteration mode: per-checkpoint or per model directory under save/models")
    parser.add_argument("--repo-root", default=None,
                        help="Path to repo root (defaults to current working directory)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    save_dir = repo_root / "save"

    # Resolve experiment script path relative to repo_root (or accept absolute)
    exp_script_path = Path(args.experiment)
    if not exp_script_path.is_absolute():
        exp_script_path = (repo_root / exp_script_path).resolve()

    if not exp_script_path.exists():
        sys.exit(f"Can't find experiment script at: {exp_script_path}")
    if not save_dir.exists():
        sys.exit(f"Can't find 'save' directory under: {repo_root}")

    # Collect items based on mode
    if args.mode == "checkpoint":
        items = collect_checkpoints(repo_root)
        item_kind = "checkpoint"
    else:
        items = collect_modeldirs(repo_root)
        item_kind = "model folder"

    if not items:
        where = save_dir if args.mode == "checkpoint" else (save_dir / "models")
        sys.exit(f"No {item_kind}s found under {where}")

    print(f"Found {len(items)} {item_kind}(s).")

    # Make experiment path the way we'll pass it to Python (relative if possible)
    try:
        exp_for_cmd = str(exp_script_path.relative_to(repo_root))
    except ValueError:
        exp_for_cmd = str(exp_script_path)  # not under repo_root; use absolute

    # Run sequentially from the repo root so relative paths resolve like manual usage
    for i, rel_path in enumerate(items, 1):
        arg_path = str(rel_path)
        cmd = [sys.executable, exp_for_cmd, arg_path]
        print(f"[{i}/{len(items)}] Running (cwd={repo_root}): {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as e:
            print(f"    -> failed with return code {e.returncode}")

if __name__ == "__main__":
    main()
