#!/usr/bin/env python3
# ============================
# Cluster/runtime hygiene
# ============================
import os as _os

# Shard from Slurm env (works with --ntasks>1). No Python multiprocessing.
SHARD_COUNT = int(_os.environ.get("SLURM_NTASKS", "1"))
SHARD_INDEX = int(_os.environ.get("SLURM_PROCID", "0"))

# Platform (set to 'gpu' in SBATCH; keep default here)
_os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

# Memory/allocator knobs
_os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
_os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # avoid file locks on shared FS
_os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0")

# Thread caps per task
_cpus = int(_os.environ.get("SLURM_CPUS_PER_TASK", _os.environ.get("OMP_NUM_THREADS", "8")))
_os.environ["OMP_NUM_THREADS"] = str(_cpus)
_os.environ["MKL_NUM_THREADS"] = str(_cpus)
_os.environ["OPENBLAS_NUM_THREADS"] = str(_cpus)
_os.environ["NUMEXPR_NUM_THREADS"] = str(_cpus)

# Headless plotting
import matplotlib
matplotlib.use("Agg")

# ============================
# Imports
# ============================
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

import flax.linen as nn
from src.data.dataloader import data_loader
from src.models.autoencoders.vae_gan import VAE_GAN
import src.inference.inference_vae_gan as inference_vae_gan


# ===== GPU keepalive (prevents “idle GPU” cancellations) =====
import threading, time

GPU_KEEPALIVE = int(_os.environ.get("GPU_KEEPALIVE", "0"))     # 0=off, 1=on
GPU_KEEPALIVE_PERIOD = float(_os.environ.get("GPU_KEEPALIVE_PERIOD", "45"))  # seconds

_keepalive_stop = threading.Event()

if GPU_KEEPALIVE and any(d.platform == "gpu" for d in jax.devices()):
    # small on-device buffers captured by the jitted fn
    KA_M = int(_os.environ.get("GPU_KEEPALIVE_SIZE", "1024"))
    _A = jnp.ones((KA_M, KA_M), dtype=jnp.float16)
    _B = jnp.ones((KA_M, KA_M), dtype=jnp.float16)

    @jit
    def _ka_step(a, b):
        # ~few ms GEMM; tiny memory footprint with fp16
        return jnp.tanh(a @ b).sum()

    # one warmup to compile once
    try:
        _ka_step(_A, _B).block_until_ready()
    except Exception as e:
        print("[KEEPALIVE] Warmup failed; continuing without keepalive:", e)
        GPU_KEEPALIVE = 0

    def _keepalive_loop():
        while not _keepalive_stop.wait(GPU_KEEPALIVE_PERIOD):
            try:
                _ka_step(_A, _B).block_until_ready()
            except Exception as e:
                print("[KEEPALIVE] step failed:", e)
                # don’t kill the job—just retry next tick

    if GPU_KEEPALIVE:
        threading.Thread(target=_keepalive_loop, daemon=True).start()
        print(f"[KEEPALIVE] started: period={GPU_KEEPALIVE_PERIOD}s, size={KA_M}x{KA_M}")
# =============================================================


# ============================
# Config
# ============================
WORK_DIR       = _os.getcwd()
DATA_ROOT      = _os.path.join(WORK_DIR, 'src/data/datasets/flow')
DATASET_NAME   = 'minimal_channel_flow'
BATCH_SIZE     = 16

# Model / checkpoint
LATENT_DIM     = 9
vae_gan_model  = VAE_GAN(latent_dim=LATENT_DIM, filter_chn=24,
                         activation_fn=nn.silu, bias=True,
                         resize_method='lanczos3', output_chn=3)
checkpoint_path = _os.path.join(
    WORK_DIR, 'pretrained_models/checkpoints_vae_gan/ld_9_fc_24_lambda_0.0001'
)
inference_module = inference_vae_gan.VAEGANInference(generator_model=vae_gan_model,
                                                     checkpoint_dir=checkpoint_path)

# SHAP settings (memory-friendly defaults; scale up later)
N_REPEATS    = int(_os.environ.get("N_REPEATS", "2"))   # repeats over random baselines
R_PER        = int(_os.environ.get("R_PER", "2"))       # baselines per repeat
NSAMPLES     = int(_os.environ.get("NSAMPLES", "8"))    # steps along path baseline→input
sigma_xy     = float(_os.environ.get("SHAP_SMOOTH", "1.0"))  # 2D blur per slice; 0 disables
shap_base    = "shap_values_mean"

# Optional: cap samples per shard for quick sanity
MAX_SAMPLES_PER_SHARD = int(_os.environ.get("MAX_SAMPLES_PER_SHARD", "20"))

# Output dirs (per shard)
OUT_DIR       = _os.path.join(WORK_DIR, f"outputs_jax_sh{SHARD_INDEX}-of-{SHARD_COUNT}")
FIG_DIR       = _os.path.join(OUT_DIR, "figs")
NPY_DIR       = _os.path.join(OUT_DIR, "arrays")
_os.makedirs(FIG_DIR, exist_ok=True)
_os.makedirs(NPY_DIR, exist_ok=True)

print(f"jax {jax.__version__}")
print("devices:", jax.devices())

# ============================
# Data loader
# ============================
# Important: no multiprocessing in Python; Slurm gives us N tasks already.
test_loader = data_loader(dataset_name=DATASET_NAME,
                          mode='test',
                          root=DATA_ROOT,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          drop_last=False,
                          workers=0,
                          world_size=1,
                          rank=0)

# ============================
# Utils
# ============================
def mse(a,b): return float(np.mean((a-b)**2))
def psnr(a,b):
    dr = float(np.max(a) - np.min(a) + 1e-12)
    return 20.0*np.log10(dr) - 10.0*np.log10(mse(a,b)+1e-12)

def middle_slice_along_depth(x_hwsc):
    """Return the middle slice along depth S. x: (H,W,S,C) -> (H,W,C)"""
    H, W, S, C = x_hwsc.shape
    return x_hwsc[:, :, S//2, :]

def smooth_phi(phi_hwscd: np.ndarray, sigma: float) -> np.ndarray:
    """phi_hwscd: (H,W,S,C,D); apply Gaussian on (H,W) for every (S,C,D)."""
    if sigma <= 0.0:
        return phi_hwscd
    H,W,S,C,D = phi_hwscd.shape
    out = np.empty_like(phi_hwscd)
    for s in range(S):
        for c in range(C):
            for d in range(D):
                out[:, :, s, c, d] = gaussian_filter(phi_hwscd[:, :, s, c, d],
                                                     sigma=sigma, mode="nearest")
    return out

# ============================
# 1) Streaming inference per shard
#    We save per-shard outputs; we do NOT hold whole dataset in RAM.
# ============================
BG_RES_MAX = max(512, R_PER * 128)  # modest reservoir; not huge
bg_reservoir = []
Z_per_shard = []
total_batches = len(test_loader)

for batch_idx, batch_data in enumerate(test_loader):
    if (batch_idx % SHARD_COUNT) != SHARD_INDEX:
        continue  # not this shard

    t0b = time.time()
    mean, variance, shape = inference_module.encode(batch_data)   # (B,1,1,1,D)
    reconstructed = inference_module.decode(mean, shape)          # (B,H,W,S,C)

    X_cpu   = jax.device_get(batch_data)                          # (B,H,W,S,C)
    Z_cpu   = np.squeeze(jax.device_get(mean), axis=(1,2,3))      # (B,D)
    Rec_cpu = jax.device_get(reconstructed)                       # (B,H,W,S,C)

    Z_per_shard.append(Z_cpu)

    # Grow background reservoir (stop at cap)
    if len(bg_reservoir) < BG_RES_MAX:
        need = BG_RES_MAX - len(bg_reservoir)
        take = min(need, X_cpu.shape[0])
        bg_reservoir.append(X_cpu[:take].astype(np.float32, copy=False))

    # A few recon panels on first local batch only
    if batch_idx // SHARD_COUNT == 0:
        H, W, S, C = X_cpu.shape[1:]
        show_idx = min(3, X_cpu.shape[0])
        for i in range(show_idx):
            X2D    = middle_slice_along_depth(X_cpu[i])
            Xrec2D = middle_slice_along_depth(Rec_cpu[i])
            fig, axes = plt.subplots(C, 2, figsize=(6, 3*C), constrained_layout=True)
            if C == 1: axes = np.array([axes])
            for c in range(C):
                vmin = min(X2D[..., c].min(), Xrec2D[..., c].min())
                vmax = max(X2D[..., c].max(), Xrec2D[..., c].max())
                im0 = axes[c,0].imshow(X2D[..., c], origin="lower", vmin=vmin, vmax=vmax)
                axes[c,0].set_title(f"Orig ch{c+1}@midS"); axes[c,0].set_xticks([]); axes[c,0].set_yticks([])
                fig.colorbar(im0, ax=axes[c,0], fraction=0.046, pad=0.04)
                im1 = axes[c,1].imshow(Xrec2D[..., c], origin="lower", vmin=vmin, vmax=vmax)
                axes[c,1].set_title(f"Recon ch{c+1}@midS"); axes[c,1].set_xticks([]); axes[c,1].set_yticks([])
                fig.colorbar(im1, ax=axes[c,1], fraction=0.046, pad=0.04)
            fig.suptitle(f"Shard {SHARD_INDEX} · sample {i} in batch {batch_idx}")
            fig.savefig(_os.path.join(FIG_DIR, f"recon_sh{SHARD_INDEX}_b{batch_idx}_i{i}.png"), dpi=150)
            plt.close(fig)

    print(f"[Shard {SHARD_INDEX}] batch {batch_idx} forward+recon in {time.time()-t0b:.2f}s")

# Save Z time series and quick correlation for this shard
if len(Z_per_shard) == 0:
    raise SystemExit(f"[Shard {SHARD_INDEX}] Nothing to do. Check your loader or shard setup.")

Z_sh = np.concatenate(Z_per_shard, axis=0)  # (N_shard, D)
np.save(_os.path.join(NPY_DIR, f"Z_shard_{SHARD_INDEX}-of-{SHARD_COUNT}.npy"), Z_sh)

fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
for j in range(Z_sh.shape[1]):
    ax.plot(Z_sh[:, j], lw=0.9, label=f"z{j+1}")
ax.set_title(f"Latent time series · shard {SHARD_INDEX}/{SHARD_COUNT}")
ax.set_xlabel("sample (local)"); ax.set_ylabel("value")
ax.legend(ncol=min(6, Z_sh.shape[1]), fontsize=8)
fig.savefig(_os.path.join(FIG_DIR, f"latent_timeseries_sh{SHARD_INDEX}.png"), dpi=150); plt.close(fig)

R = np.corrcoef(Z_sh.T)
fig, ax = plt.subplots(figsize=(4+0.3*R.shape[0], 4+0.3*R.shape[0]), constrained_layout=True)
im = ax.imshow(R, vmin=-1, vmax=1, origin="lower")
ax.set_title(f"Latent corr · shard {SHARD_INDEX}")
ax.set_xticks(range(R.shape[0])); ax.set_yticks(range(R.shape[0]))
ax.set_xticklabels([f"{k+1}" for k in range(R.shape[0])]); ax.set_yticklabels([f"{k+1}" for k in range(R.shape[0])])
for i in range(R.shape[0]):
    for j in range(R.shape[0]):
        ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(_os.path.join(FIG_DIR, f"latent_corr_sh{SHARD_INDEX}.png"), dpi=150); plt.close(fig)

# Build background pool from our reservoir
bg_pool = np.concatenate(bg_reservoir, axis=0) if len(bg_reservoir) else None
if bg_pool is None or bg_pool.shape[0] == 0:
    raise SystemExit(f"[Shard {SHARD_INDEX}] No background samples collected.")
print(f"[Shard {SHARD_INDEX}] Background pool: {bg_pool.shape}")

# ============================
# 2) JAX functions for Gradient-SHAP
# ============================
def encode_mean_single(x_hwsc_np):
    """(H,W,S,C) → (D,)"""
    x = jnp.asarray(x_hwsc_np, dtype=jnp.float32)[None, ...]   # (1,H,W,S,C)
    mean, var, shp = inference_module.encode(x)                # (1,1,1,1,D)
    return jnp.squeeze(mean, axis=(0,1,2,3))                   # (D,)

encode_mean_single_jit = jit(lambda x: encode_mean_single(x))

def grad_latent_j_wrt_x(x_hwsc, j_idx: int):
    f = lambda inp: encode_mean_single_jit(inp)[j_idx]         # scalar
    return grad(f)(x_hwsc)                                     # (H,W,S,C)

def gradient_shap_one_latent(x_hwsc, baselines_hwsc, j_idx: int, m_steps: int):
    x = jnp.asarray(x_hwsc, dtype=jnp.float32)                 # (H,W,S,C)
    b = jnp.asarray(baselines_hwsc, dtype=jnp.float32)         # (R,H,W,S,C)
    ks = jnp.linspace(0.0, 1.0, m_steps + 1, dtype=jnp.float32)[1:]  # (m,)

    def attr_for_one_baseline(b_i):
        dx = x - b_i
        def grad_at_k(k):
            xk = b_i + k * dx
            return grad_latent_j_wrt_x(xk, j_idx)              # (H,W,S,C)
        grads = vmap(grad_at_k)(ks)                            # (m,H,W,S,C)
        avg_grad = jnp.mean(grads, axis=0)                     # (H,W,S,C)
        return dx * avg_grad                                   # (H,W,S,C)

    phi_all = vmap(attr_for_one_baseline)(b)                   # (R,H,W,S,C)
    return jnp.mean(phi_all, axis=0)                           # (H,W,S,C)

def gradient_shap_one_sample(x_hwsc, baselines_hwsc, m_steps: int, D: int):
    outs = []
    for j_idx in range(D):
        outs.append(gradient_shap_one_latent(x_hwsc, baselines_hwsc, j_idx, m_steps))
    return jnp.stack(outs, axis=-1)  # (H,W,S,C,D)

rng = np.random.default_rng(42 + SHARD_INDEX)  # distinct per shard
def sample_backgrounds(bg_all: np.ndarray, k: int):
    k = min(k, bg_all.shape[0])
    idx = rng.choice(bg_all.shape[0], size=k, replace=False)
    return bg_all[idx].astype(np.float32, copy=False)          # (k,H,W,S,C)

# ============================
# 3) SHAP pass over my shard (streaming; per-sample; write HDF5 as we go)
# ============================
H,W,S,C = bg_pool.shape[1:]
D = LATENT_DIM

jobid = _os.environ.get("SLURM_JOB_ID", "nojid")
out_path = _os.path.join(OUT_DIR, f"{shap_base}_job{jobid}_sh{SHARD_INDEX}-of-{SHARD_COUNT}.h5")

with h5py.File(out_path, "w") as hf:
    # Create extendable datasets
    dsets = {}
    for ch in range(C):
        name = ["u","v","w"][ch] if ch < 3 else f"ch{ch}"
        dsets[name] = hf.create_dataset(
            f"{name}_shap",
            shape=(0, H, W, S, D),
            maxshape=(None, H, W, S, D),
            dtype="float32",
            chunks=(1, H, W, max(1, S//2), D),
            compression="gzip", shuffle=True, fletcher32=True
        )

    shard_sample_counter = 0

    for batch_idx, batch_data in enumerate(test_loader):
        if (batch_idx % SHARD_COUNT) != SHARD_INDEX:
            continue

        X_cpu = jax.device_get(batch_data).astype(np.float32, copy=False)  # (B,H,W,S,C)

        for b in range(X_cpu.shape[0]):
            if shard_sample_counter >= MAX_SAMPLES_PER_SHARD:
                print(f"[Shard {SHARD_INDEX}] Reached MAX_SAMPLES_PER_SHARD={MAX_SAMPLES_PER_SHARD}, stopping.")
                break

            t0 = time.time()
            x_i = jnp.asarray(X_cpu[b])

            # Online mean over repeats
            phi_mean = np.zeros((H, W, S, C, D), dtype=np.float32)
            for r in range(1, N_REPEATS + 1):
                baselines = sample_backgrounds(bg_pool, R_PER)    # (R,H,W,S,C)
                phi_hwscd = gradient_shap_one_sample(x_i, baselines, NSAMPLES, D)
                phi_np = np.asarray(phi_hwscd, dtype=np.float32)
                phi_mean += (phi_np - phi_mean) / r
                print(f"[Shard {SHARD_INDEX}] sample {shard_sample_counter+1} repeat {r}/{N_REPEATS} done")

            if sigma_xy > 0.0:
                phi_mean = smooth_phi(phi_mean, sigma_xy)

            # Append to datasets and flush
            for ch in range(C):
                key = ["u","v","w"][ch] if ch < 3 else f"ch{ch}"
                ds = dsets[key]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1, ...] = phi_mean[:, :, :, ch, :]

            hf.flush()
            shard_sample_counter += 1
            print(f"[Shard {SHARD_INDEX}] wrote sample #{shard_sample_counter} in {time.time()-t0:.2f}s")

        else:
            continue
        break  # stop outer loop if we hit MAX_SAMPLES_PER_SHARD

print(f"[Shard {SHARD_INDEX}] Done. SHAP written to: {out_path}")
print(f"[Shard {SHARD_INDEX}] Z saved to: {NPY_DIR}/Z_shard_{SHARD_INDEX}-of-{SHARD_COUNT}.npy")

# ============================
# Quick local plots from our shard HDF5
# ============================
with h5py.File(out_path, "r") as hf:
    key = "u_shap" if "u_shap" in hf else list(hf.keys())[0]
    dat = hf[key][:]  # (N_shard, H, W, S, D)
    if dat.shape[0] > 0:
        shap_abs = np.abs(dat)
        thr = np.percentile(shap_abs, 95)
        midS = shap_abs.shape[3] // 2
        for md in range(min(3, shap_abs.shape[-1])):  # a few modes to visualize
            fig, ax = plt.subplots(figsize=(6,6))
            plane = np.where(shap_abs[0, :, :, midS, md] > thr, shap_abs[0, :, :, midS, md], 0.0)
            cf = ax.contourf(plane, cmap='plasma',
                             vmin=shap_abs.min(), vmax=shap_abs.max())
            make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cf)
            ax.set_title(f"Masked SHAP (key={key}) · z{md+1} · shard {SHARD_INDEX}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(_os.path.join(FIG_DIR, f"SHAP_heat_{key}_z{md+1}_sh{SHARD_INDEX}.png"))
            plt.close()
# Gracefully stop keepalive
try:
    _keepalive_stop.set()
except NameError:
    pass
