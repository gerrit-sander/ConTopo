import os
import re
import argparse
import csv
from typing import List, Tuple, Dict, Any

import torch
import matplotlib.pyplot as plt


def _find_model_folders(models_root: str) -> List[str]:
    """Return immediate subdirectories of `models_root` as candidate model folders."""
    models_root = os.path.abspath(models_root)
    if not os.path.isdir(models_root):
        raise NotADirectoryError(f"Models root is not a directory: {models_root}")
    out: List[str] = []
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if os.path.isdir(path):
            out.append(path)
    return out


def _load_unaveraged_rdms(model_folder: str) -> torch.Tensor:
    """
    Load per-trial upper-triangular RDM vectors for a model folder and
    return as a [n_trials, m] tensor on CPU.

    Expected file: RDM_{base}.pt where base = basename(model_folder).
    Supports payload formats produced by exp_generateRDM.py.
    """
    base = os.path.basename(os.path.normpath(model_folder))
    rdm_file = os.path.join(model_folder, f"RDM_{base}.pt")
    if not os.path.isfile(rdm_file):
        raise FileNotFoundError(f"Missing unaveraged RDM file: {rdm_file}")

    obj = torch.load(rdm_file, map_location="cpu")

    # Newer format: dict with key 'rdms_upper' (list of 1D tensors)
    if isinstance(obj, dict) and "rdms_upper" in obj:
        rdms: List[torch.Tensor] = obj["rdms_upper"]
        if len(rdms) == 0:
            return torch.empty((0, 0), dtype=torch.float32)
        # Ensure float32 CPU and consistent length
        rdms_proc: List[torch.Tensor] = [r.to(dtype=torch.float32, device="cpu").view(-1) for r in rdms]
        m = rdms_proc[0].numel()
        for i, r in enumerate(rdms_proc):
            if r.numel() != m:
                raise ValueError(
                    f"RDM vector length mismatch in {rdm_file}: index {i} has {r.numel()}, expected {m}"
                )
        return torch.stack(rdms_proc, dim=0)

    # Legacy or alternative: already a tensor [n, m] or list of vectors
    if torch.is_tensor(obj):
        T = obj.detach().to(dtype=torch.float32, device="cpu")
        if T.ndim == 1:
            T = T.view(1, -1)
        elif T.ndim != 2:
            raise ValueError(f"Unsupported RDM tensor shape in {rdm_file}: {tuple(T.shape)}")
        return T

    if isinstance(obj, (list, tuple)):
        rdms_proc = [torch.as_tensor(x, dtype=torch.float32, device="cpu").view(-1) for x in obj]
        if not rdms_proc:
            return torch.empty((0, 0), dtype=torch.float32)
        m = rdms_proc[0].numel()
        for i, r in enumerate(rdms_proc):
            if r.numel() != m:
                raise ValueError(
                    f"RDM vector length mismatch in {rdm_file}: index {i} has {r.numel()}, expected {m}"
                )
        return torch.stack(rdms_proc, dim=0)

    raise ValueError(f"Unrecognized RDM payload format in {rdm_file}")


def _load_consistency(model_folder: str) -> Tuple[float | None, Dict[str, Any]]:
    """Load RDM consistency stats and return (mean, raw_dict)."""
    base = os.path.basename(os.path.normpath(model_folder))
    stats_file = os.path.join(model_folder, f"RDMConsistency_{base}.pt")
    if not os.path.isfile(stats_file):
        return None, {}
    obj = torch.load(stats_file, map_location="cpu")
    if isinstance(obj, dict):
        mean = obj.get("mean")
        if mean is None and obj.get("num_pairs", 0) == 0:
            # Not enough trials to compute consistency
            return None, obj
        # Ensure python float or None
        try:
            mean = float(mean) if mean is not None else None
        except Exception:
            mean = None
        return mean, obj
    return None, {}


def main():
    parser = argparse.ArgumentParser(description="Aggregate RDMs and consistency across model folders.")
    parser.add_argument(
        "models_root",
        nargs="?",
        default=".",
        help="Path to the directory that contains model folders (default: current directory).",
    )
    parser.add_argument(
        "--output-prefix",
        default="RSA",
        help="Prefix for output files placed under models_root (default: 'RSA').",
    )
    parser.add_argument(
        "--skip-mismatch",
        action="store_true",
        help="Skip model folders whose RDM vector length does not match the first encountered length.",
    )
    parser.add_argument(
        "--exclude-substring",
        default="0.0dropout",
        help=(
            "Produce a second set of outputs excluding rows whose model folder name contains this substring. "
            "Default: '0.0dropout'."
        ),
    )
    args = parser.parse_args()

    models_root = os.path.abspath(args.models_root)
    model_folders = _find_model_folders(models_root)
    # Sort model folders hierarchically: by loss tag, then numeric rho ascending, then name
    def _sort_key(path: str) -> Tuple[str, float, str]:
        name = os.path.basename(path.rstrip(os.sep))
        # loss tag assumed to be prefix before first underscore
        loss_tag = name.split("_", 1)[0]
        # extract numeric rho between underscores followed by 'rho'
        m = re.search(r"_(\d+(?:\.\d+)?)rho(?:_|$)", name)
        rho_val = float(m.group(1)) if m else float("inf")
        return (loss_tag, rho_val, name)

    model_folders = sorted(model_folders, key=_sort_key)
    if not model_folders:
        raise SystemExit(f"No model folders found under: {models_root}")

    # Aggregate per-trial RDM vectors across models
    all_rows: List[torch.Tensor] = []
    index_rows: List[Dict[str, Any]] = []
    vector_len: int | None = None

    # Collect consistency summary
    consistency_rows: List[Tuple[str, float | None]] = []

    for model_path in model_folders:
        model_name = os.path.basename(model_path.rstrip(os.sep))

        # Load consistency
        mean_consistency, _raw = _load_consistency(model_path)
        consistency_rows.append((model_name, mean_consistency))

        # Load unaveraged RDMs and append to global tensor
        try:
            T = _load_unaveraged_rdms(model_path)  # [n_trials, m]
        except Exception as e:
            print(f"[WARN] Skipping {model_name}: {e}")
            continue

        if T.numel() == 0:
            print(f"[WARN] No RDMs found for {model_name}; skipping.")
            continue

        n, m = T.shape
        if vector_len is None:
            vector_len = m
        elif m != vector_len:
            msg = (
                f"RDM vector length mismatch for {model_name}: got {m}, expected {vector_len}"
            )
            if args.skip_mismatch:
                print(f"[WARN] {msg}. Skipping this model.")
                continue
            else:
                raise ValueError(msg)

        # Global row index starts at total rows aggregated so far
        start_idx = sum(chunk.shape[0] for chunk in all_rows)
        all_rows.append(T)
        for i in range(n):
            index_rows.append({
                "global_index": start_idx + i,
                "model": model_name,
                "trial_index": i,
            })

    if not all_rows:
        raise SystemExit("No RDM rows aggregated. Nothing to save.")

    big = torch.cat(all_rows, dim=0)  # [g, m]
    g, m = big.shape

    # Derive and save model order summary (first appearance order)
    def _loss_rho_from_name(name: str) -> Tuple[str, float]:
        loss_tag = name.split("_", 1)[0]
        m = re.search(r"_(\d+(?:\.\d+)?)rho(?:_|$)", name)
        rho_val = float(m.group(1)) if m else float("inf")
        return loss_tag, rho_val

    seen_models: set[str] = set()
    models_in_order: List[str] = []
    for row in index_rows:
        mname = row["model"]
        if mname not in seen_models:
            seen_models.add(mname)
            models_in_order.append(mname)

    # Sanity check: rho ascending within each loss group
    def _check_and_write_model_order(models: List[str], out_csv_path: str) -> None:
        from collections import defaultdict
        # count trials per model
        trial_counts: Dict[str, int] = defaultdict(int)
        for r in index_rows:
            trial_counts[r["model"]] += 1

        # write order csv
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["loss", "rho", "model", "n_trials"])
            for mname in models:
                loss, rho = _loss_rho_from_name(mname)
                writer.writerow([loss, rho, mname, trial_counts.get(mname, 0)])

        # check ordering
        groups: Dict[str, List[float]] = {}
        for mname in models:
            loss, rho = _loss_rho_from_name(mname)
            groups.setdefault(loss, []).append(rho)
        for loss, rhos in groups.items():
            asc = sorted(rhos)
            if rhos != asc:
                print(f"[WARN] Rho order not ascending for loss='{loss}' in unfiltered set: {rhos} (expected {asc})")

    out_order_csv = os.path.join(models_root, f"{args.output_prefix}_ModelOrder.csv")
    _check_and_write_model_order(models_in_order, out_order_csv)

    # Save aggregated tensor and index mapping as a .pt payload
    out_pt = os.path.join(models_root, f"{args.output_prefix}_AllRDMs.pt")
    payload = {
        "matrix": big,                 # [g, m] float32 CPU
        "index": index_rows,           # list of {global_index, model, trial_index}
        "vector_length": m,
        "note": "Upper-triangular RDM vectors concatenated across all models and trials.",
    }
    torch.save(payload, out_pt)

    # Save index mapping as CSV for easy inspection
    out_idx_csv = os.path.join(models_root, f"{args.output_prefix}_AllRDMs_index.csv")
    with open(out_idx_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_index", "model", "trial_index"]) 
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    # Save consistency overview CSV: model_name, consistency_mean
    out_cons_csv = os.path.join(models_root, f"{args.output_prefix}_Consistency.csv")
    with open(out_cons_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "consistency_mean"])  # keep minimal as requested
        for model_name, mean_val in consistency_rows:
            writer.writerow([model_name, "" if mean_val is None else f"{mean_val:.6f}"])

    print(f"Aggregated RDMs: {g} trials across {len(model_folders)} model folders")
    print(f"Saved tensor payload: {out_pt}")
    print(f"Saved index CSV:     {out_idx_csv}")
    print(f"Saved model order CSV: {out_order_csv}")
    print(f"Saved consistency CSV: {out_cons_csv}")

    # Also produce a filtered variant excluding models that contain the given substring
    excl = args.exclude_substring
    if excl:
        keep_indices = [i for i, row in enumerate(index_rows) if excl not in row["model"]]
        if not keep_indices:
            print(f"[WARN] No rows remain after excluding models containing '{excl}'. Skipping filtered outputs.")
            return

        big_f = big[keep_indices, :]
        index_rows_f = []
        for new_idx, old_idx in enumerate(keep_indices):
            row = index_rows[old_idx]
            index_rows_f.append({
                "global_index": new_idx,
                "model": row["model"],
                "trial_index": row["trial_index"],
            })

        tag = excl.replace(os.sep, "_")
        suffix = f"no_{tag}"

        out_pt_f = os.path.join(models_root, f"{args.output_prefix}_AllRDMs_{suffix}.pt")
        payload_f = {
            "matrix": big_f,
            "index": index_rows_f,
            "vector_length": big_f.shape[1],
            "note": (
                "Filtered: rows excluded where 'model' contains '" + excl + "'. "
                "Upper-triangular RDM vectors concatenated across remaining models and trials."
            ),
        }
        torch.save(payload_f, out_pt_f)

        out_idx_csv_f = os.path.join(models_root, f"{args.output_prefix}_AllRDMs_index_{suffix}.csv")
        with open(out_idx_csv_f, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_index", "model", "trial_index"]) 
            writer.writeheader()
            for row in index_rows_f:
                writer.writerow(row)

        # Save filtered model-order report and check ordering
        seen_models_f: set[str] = set()
        models_in_order_f: List[str] = []
        for row in index_rows_f:
            mname = row["model"]
            if mname not in seen_models_f:
                seen_models_f.add(mname)
                models_in_order_f.append(mname)

        out_order_csv_f = os.path.join(models_root, f"{args.output_prefix}_ModelOrder_{suffix}.csv")
        # Reuse the checker, but count trials within filtered set
        def _check_and_write_model_order_filtered(models: List[str], out_csv_path: str) -> None:
            from collections import defaultdict
            trial_counts_f: Dict[str, int] = defaultdict(int)
            for r in index_rows_f:
                trial_counts_f[r["model"]] += 1
            with open(out_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["loss", "rho", "model", "n_trials"])
                for mname in models:
                    loss, rho = _loss_rho_from_name(mname)
                    writer.writerow([loss, rho, mname, trial_counts_f.get(mname, 0)])
            # check ordering
            groups_f: Dict[str, List[float]] = {}
            for mname in models:
                loss, rho = _loss_rho_from_name(mname)
                groups_f.setdefault(loss, []).append(rho)
            for loss, rhos in groups_f.items():
                asc = sorted(rhos)
                if rhos != asc:
                    print(f"[WARN] Rho order not ascending for loss='{loss}' in filtered set: {rhos} (expected {asc})")

        _check_and_write_model_order_filtered(models_in_order_f, out_order_csv_f)

        print(f"Saved filtered tensor payload: {out_pt_f}")
        print(f"Saved filtered index CSV:     {out_idx_csv_f}")
        print(f"Saved filtered model order CSV: {out_order_csv_f}")

        # --- Second-level RSA over filtered rows ---
        # Ensure consistent dimensionality
        assert big_f.ndim == 2, f"Expected 2D tensor, got shape {tuple(big_f.shape)}"
        g_f, m_f = big_f.shape
        if g_f >= 1:
            # Pearson correlation across rows: center each row, normalize, then dot-product
            X = big_f.to(dtype=torch.float32, device="cpu")
            Xc = X - X.mean(dim=1, keepdim=True)
            eps = 1e-8
            norms = Xc.norm(dim=1, keepdim=True).clamp_min(eps)
            Y = Xc / norms
            rsa_mat = Y @ Y.t()  # [g_f, g_f]
            # Numeric hygiene: clamp to [-1, 1]
            rsa_mat = rsa_mat.clamp(-1.0, 1.0)

            # Save matrix with shape-specific prefix
            rsa_pt = os.path.join(models_root, f"{args.output_prefix}_{g_f}x{g_f}.pt")
            torch.save({
                "rsa_matrix": rsa_mat,  # [g, g]
                "index": index_rows_f,  # matches row/col order
                "note": "Pearson correlation over upper-triangular RDM vectors (filtered)",
            }, rsa_pt)

            # Save heatmap
            rsa_png = os.path.join(models_root, f"{args.output_prefix}_{g_f}x{g_f}.png")
            plt.figure(figsize=(8, 7))
            im = plt.imshow(rsa_mat.numpy(), cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
            plt.title(f"Model-by-model RSA (g={g_f})")
            plt.xlabel("RDM index")
            plt.ylabel("RDM index")
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Pearson r")
            plt.tight_layout()
            plt.savefig(rsa_png, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved RSA matrix: {rsa_pt}")
            print(f"Saved RSA heatmap: {rsa_png}")


if __name__ == "__main__":
    main()
