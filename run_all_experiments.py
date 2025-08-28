"""
Run <experiment>.py for every checkpoint under REPO_ROOT/save.

Usage:
  # from repo root (uses ./save)
  python run_all_experiments.py exp_actmaps.py

  # or point to repo root explicitly
  python run_all_experiments.py exp_tsne.py /path/to/my/repo
"""

import sys
import subprocess
from pathlib import Path

CKPT_PATTERNS = ("*.pth", "*.pt", "*.ckpt")

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_all_experiments.py <experiment_py> [repo_root]")

    exp_arg = sys.argv[1]  # e.g., "exp_actmaps.py" or "experiments/exp_tsne.py"
    repo_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
    save_dir = repo_root / "save"

    # Resolve experiment script path relative to repo_root (or accept absolute)
    exp_script_path = Path(exp_arg)
    if not exp_script_path.is_absolute():
        exp_script_path = (repo_root / exp_arg).resolve()

    if not exp_script_path.exists():
        sys.exit(f"Can't find experiment script at: {exp_script_path}")
    if not save_dir.exists():
        sys.exit(f"Can't find 'save' directory under: {repo_root}")

    # collect checkpoints as RELATIVE paths (from repo root), de-duplicated and sorted
    ckpts = sorted({
        p.relative_to(repo_root)
        for pat in CKPT_PATTERNS
        for p in save_dir.rglob(pat)
        if p.is_file()
    })
    if not ckpts:
        sys.exit(f"No checkpoints found under {save_dir}")

    print(f"Found {len(ckpts)} checkpoint(s) under {save_dir}.\n")

    # Make experiment path the way we'll pass it to Python (relative if possible)
    try:
        exp_for_cmd = str(exp_script_path.relative_to(repo_root))
    except ValueError:
        exp_for_cmd = str(exp_script_path)  # not under repo_root; use absolute

    # run sequentially from the repo root so relative paths resolve like manual usage
    for i, ckpt_rel in enumerate(ckpts, 1):
        ckpt_arg = str(ckpt_rel)  # e.g., "save/.../e2e_best.pth"
        cmd = [sys.executable, exp_for_cmd, ckpt_arg]
        print(f"[{i}/{len(ckpts)}] Running (cwd={repo_root}): {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as e:
            print(f"    -> failed with return code {e.returncode}")

if __name__ == "__main__":
    main()