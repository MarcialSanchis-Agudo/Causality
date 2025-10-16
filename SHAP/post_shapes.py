#!/usr/bin/env python3
"""
Post-process Gradient-SHAP outputs:
- Merge per-shard HDF5 files into one.
- Print sizes of input/merged/mean files.
- Plot SHAP histograms and mid-depth heatmaps for all channels.
- Compute mean SHAP field across samples (N axis) for each channel & latent.

Examples
--------
# Merge all shard files from a job and plot first 3 modes of 4 samples
python post_shapes.py \
  --inputs "outputs_jax_sh*/shap_values_mean_job5196390_sh*-of-*.h5" \
  --output "shap_values_mean_job5196390_merged.h5" \
  --figdir "figs_shap_5196390" \
  --samples 0 1 2 3 \
  --max-modes 3 \
  --thr 95

# If you already have a merged file, skip merging:
python post_shapes.py \
  --merged "shap_values_mean_job5196390_merged.h5" \
  --figdir "figs_shap_5196390" \
  --samples 0 5 10 \
  --max-modes 6 \
  --thr 97.5
"""
import os
import glob
import argparse
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------- helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def fmt_size(num_bytes: int) -> str:
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < step:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} PB"


def file_size(path: str) -> str:
    try:
        return fmt_size(os.path.getsize(path))
    except Exception:
        return "n/a"


def merge_hdf5(inputs, output):
    """Merge many per-shard HDF5s into `output` without loading everything in RAM."""
    files = sorted(inputs)
    assert files, "No input files matched."

    print("[merge] input files:")
    for p in files:
        print(f"  {p} ({file_size(p)})")

    # Inspect first file for dataset keys & shapes
    with h5py.File(files[0], "r") as h0:
        keys = list(h0.keys())
        assert keys, f"No datasets found in {files[0]}"
        # Expect datasets of shape (N, H, W, S, D)
        H, W, S, D = h0[keys[0]].shape[1:]

    # Create output
    with h5py.File(output, "w") as ho:
        out_dsets = {}
        for k in keys:
            out_dsets[k] = ho.create_dataset(
                k, shape=(0, H, W, S, D), maxshape=(None, H, W, S, D),
                dtype="float32", chunks=(1, H, W, max(1, S // 2), D),
                compression="gzip", shuffle=True, fletcher32=True
            )

        # Append per file
        for path in files:
            with h5py.File(path, "r") as hi:
                for k in keys:
                    src = hi[k]             # (n,H,W,S,D)
                    n = int(src.shape[0])
                    if n == 0:
                        continue
                    dst = out_dsets[k]
                    old = int(dst.shape[0])
                    dst.resize(old + n, axis=0)
                    # chunked copy to keep memory low
                    bs = max(1, min(8, n))  # small batch appends
                    w0 = old
                    for i0 in range(0, n, bs):
                        i1 = min(n, i0 + bs)
                        dst[w0 + i0 : w0 + i1] = src[i0:i1]

    print(f"[merge] wrote: {output} ({file_size(output)})")
    return keys


def middle_slice(a4d):
    """Return middle depth slice along S: (H,W,S,D) -> (H,W,D)."""
    S = a4d.shape[2]
    return a4d[:, :, S // 2, :]


def global_percentile_threshold(h5path, keys, thr=95.0, max_per_key=None):
    """Compute a single global |SHAP| threshold across all keys/datasets."""
    vals = []
    with h5py.File(h5path, "r") as hf:
        for k in keys:
            ds = hf[k]
            N = ds.shape[0]
            if N == 0:
                continue
            take = N if (max_per_key is None) else min(N, max_per_key)
            # Use a light subsample for global percentile to avoid huge RAM
            idxs = np.linspace(0, N - 1, take, dtype=int)
            for i in idxs:
                x = np.abs(ds[i])  # (H,W,S,D)
                vals.append(np.percentile(x, thr))
    if not vals:
        return None
    return float(np.median(vals))


def plot_histograms(h5path, figdir, keys, bins=80):
    """Global histograms of |SHAP| for each key (channel)."""
    ensure_dir(figdir)
    for k in keys:
        dat = []
        with h5py.File(h5path, "r") as hf:
            ds = hf[k]
            N = ds.shape[0]
            if N == 0:
                continue
            # Subsample a few items to keep memory low
            take = max(1, min(16, N))
            idxs = np.linspace(0, N - 1, take, dtype=int)
            for i in idxs:
                d = np.abs(ds[i])  # (H,W,S,D)
                dat.append(d.ravel())
        if not dat:
            continue
        arr = np.concatenate(dat)
        plt.figure(figsize=(8, 4))
        plt.hist(arr, bins=bins)
        plt.title(f"|SHAP| Histogram · {k}")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"hist_{k}.png"), dpi=150)
        plt.close()


def plot_mid_depth_heatmaps(h5path, figdir, keys, samples, max_modes, vmin=None, vmax=None, thr=None, prefix=""):
    """
    Plot mid-depth heatmaps for selected samples and up to `max_modes` latent dims
    for every key (channel). If `thr` is set, mask values below this |SHAP| threshold.
    """
    ensure_dir(figdir)
    with h5py.File(h5path, "r") as hf:
        for k in keys:
            ds = hf[k]   # (N,H,W,S,D) OR (H,W,S,D) if mean file
            shape = ds.shape
            if len(shape) == 5:
                N, H, W, S, D = shape
                single = False
            else:
                H, W, S, D = shape
                single = True
                N = 1

            if (not single) and N == 0:
                print(f"[plot] {k}: empty, skipping.")
                continue

            modes = min(max_modes, D)
            idx_iter = samples if not single else [0]
            for n in idx_iter:
                if not single and (n < 0 or n >= N):
                    print(f"[plot] {k}: sample {n} out of range [0,{N-1}], skipping.")
                    continue
                x = ds[:] if single else ds[n]   # (H,W,S,D)
                mid = middle_slice(x)            # (H,W,D)

                # dynamic min/max per key if not given
                if vmin is None or vmax is None:
                    vmin_k = float(np.min(mid))
                    vmax_k = float(np.max(mid))
                else:
                    vmin_k, vmax_k = vmin, vmax

                # optional masking by global |SHAP| threshold
                if thr is not None:
                    mask = (np.abs(mid) >= thr)
                    mid = np.where(mask, mid, 0.0)

                # grid of modes (cols) → one row figure
                fig, axes = plt.subplots(1, modes, figsize=(3.2*modes, 3.2), squeeze=False)
                axes = axes[0]
                for md in range(modes):
                    im = axes[md].imshow(mid[:, :, md], origin="lower",
                                         cmap="plasma", vmin=vmin_k, vmax=vmax_k)
                    axes[md].set_title(f"{k} · z{md+1}")
                    axes[md].set_xticks([]); axes[md].set_yticks([])
                    div = make_axes_locatable(axes[md])
                    cax = div.append_axes("right", size="4%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                title = f"{k} · {'mean' if single else f'sample {n}'} · mid-depth"
                fig.suptitle(title)
                fig.tight_layout()
                base = f"{prefix}heat_{k}_{'mean' if single else f'n{n}'}_m{modes}.png"
                fig.savefig(os.path.join(figdir, base), dpi=150)
                plt.close(fig)


def compute_mean_fields(merged_path: str, mean_out: str):
    """
    Compute mean over samples (axis 0) for each dataset in the merged file.
    Writes a new HDF5 where each dataset has shape (H,W,S,D).
    Also returns the list of keys.
    """
    with h5py.File(merged_path, "r") as hi, h5py.File(mean_out, "w") as ho:
        keys = list(hi.keys())
        for k in keys:
            ds = hi[k]  # (N,H,W,S,D)
            if ds.shape[0] == 0:
                continue
            # streaming mean: μ = sum/N
            N = ds.shape[0]
            H, W, S, D = ds.shape[1:]
            acc = np.zeros((H, W, S, D), dtype=np.float64)  # double for accumulation
            bs = max(1, min(16, N))
            for i0 in range(0, N, bs):
                i1 = min(N, i0 + bs)
                acc += np.sum(ds[i0:i1].astype(np.float64), axis=0)
            mean = (acc / N).astype(np.float32)
            ho.create_dataset(
                k, data=mean, dtype="float32",
                chunks=(H, W, max(1, S // 2), D),
                compression="gzip", shuffle=True, fletcher32=True
            )
    print(f"[mean] wrote: {mean_out} ({file_size(mean_out)})")
    return keys


def main():
    parser = argparse.ArgumentParser(description="Merge shard SHAP files, compute means, and plot.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="+", help="Glob pattern(s) to per-shard HDF5s, e.g. 'outputs_jax_sh*/shap_values_mean_jobXXXX_sh*-of-*.h5'")
    g.add_argument("--merged", help="Path to an already merged HDF5")

    parser.add_argument("--output", default="shap_values_merged.h5", help="Output merged HDF5 (if --inputs used)")
    parser.add_argument("--mean-out", default=None, help="Output HDF5 with per-key mean fields; default = <merged_basename>_mean.h5")
    parser.add_argument("--figdir", default="figs_shap", help="Directory for plots")

    parser.add_argument("--samples", type=int, nargs="+", default=[0,1,2,3], help="Sample indices to plot")
    parser.add_argument("--max-modes", type=int, default=6, help="Number of latent dims (modes) to plot per channel")
    parser.add_argument("--thr", type=float, default=95.0, help="Percentile threshold for |SHAP| masking")
    parser.add_argument("--thr-sample-cap", type=int, default=8, help="Per-key sample cap when computing global percentile")
    parser.add_argument("--no-merge", action="store_true", help="If set with --inputs, just plot first file without merging")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed colorbar vmin for heatmaps (default: auto per key)")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed colorbar vmax for heatmaps (default: auto per key)")
    args = parser.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # safer on shared FS

    # 1) Find/merge
    merged_path = args.merged
    keys = None

    if args.inputs:
        # Expand globs
        file_list = []
        for pat in args.inputs:
            file_list.extend(glob.glob(pat, recursive=True))
        file_list = sorted(set(file_list))
        if not file_list:
            raise SystemExit("No files matched your --inputs pattern(s).")

        if args.no_merge:
            merged_path = file_list[0]
            print(f"[info] --no-merge: using first matched file:\n  {merged_path} ({file_size(merged_path)})")
            with h5py.File(merged_path, "r") as h:
                keys = list(h.keys())
        else:
            print(f"[merge] {len(file_list)} shard files → {args.output}")
            keys = merge_hdf5(file_list, args.output)
            merged_path = args.output
            print(f"[merge] keys: {keys}")
    else:
        # Use existing merged file
        assert os.path.isfile(args.merged), f"Merged file not found: {args.merged}"
        merged_path = args.merged
        print(f"[info] merged file: {merged_path} ({file_size(merged_path)})")
        with h5py.File(merged_path, "r") as h:
            keys = list(h.keys())
        print(f"[info] keys: {keys}")

    assert keys, "No datasets to plot."

    # 2) Histograms per key
    ensure_dir(args.figdir)
    print("[plot] histograms…")
    plot_histograms(merged_path, args.figdir, keys, bins=100)

    # 3) Compute a global |SHAP| threshold (percentile over all keys)
    print(f"[plot] computing global |SHAP| percentile ({args.thr}th)…")
    gthr = global_percentile_threshold(merged_path, keys, thr=args.thr, max_per_key=args.thr_sample_cap)
    if gthr is None:
        print("[plot] WARNING: threshold computation yielded None; skipping masking.")
    else:
        print(f"[plot] global threshold |SHAP| ≈ {gthr:.3e}")

    # 4) Heatmaps per key for chosen samples and modes
    print("[plot] heatmaps (individual samples)…")
    plot_mid_depth_heatmaps(
        merged_path, args.figdir, keys,
        samples=args.samples, max_modes=args.max_modes,
        vmin=args.vmin, vmax=args.vmax, thr=gthr, prefix=""
    )

    # 5) Mean over samples for each key → write mean file and plot mean heatmaps
    mean_out = args.mean_out
    if mean_out is None:
        base, ext = os.path.splitext(merged_path)
        mean_out = f"{base}_mean{ext if ext else '.h5'}"

    print("[mean] computing per-key mean fields over samples…")
    mean_keys = compute_mean_fields(merged_path, mean_out)

    print("[plot] heatmaps (per-key means)…")
    # For mean fields there is no 'N' axis; reuse plot with prefix and same thr/vmin/vmax
    plot_mid_depth_heatmaps(
        mean_out, args.figdir, mean_keys,
        samples=[], max_modes=args.max_modes,
        vmin=args.vmin, vmax=args.vmax, thr=gthr, prefix="mean_"
    )

    print("[done] Plots in:", args.figdir)
    print("[done] Merged file :", merged_path, f"({file_size(merged_path)})")
    print("[done] Mean file   :", mean_out,    f"({file_size(mean_out)})")


if __name__ == "__main__":
    main()
