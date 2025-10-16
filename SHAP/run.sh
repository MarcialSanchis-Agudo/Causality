#!/bin/env bash
#SBATCH -A NAISS2025-5-144 ###########NAISS2024-3-30
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:4
#SBATCH --ntasks-per-node=4
#SBATCH -t 10:00:00
#SBATCH -J ShapeJAX-2GPU
#SBATCH --output logs/shap.out
#SBATCH --error  errors/shap.error

module purge
module load virtualenv/20.24.6-GCCcore-13.2.0
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0
source /mimer/NOBACKUP/groups/deepmechalvis/carlos/envvae/bin/activate

export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HDF5_USE_FILE_LOCKING=FALSE
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 ${XLA_FLAGS}"

export N_REPEATS=10 R_PER=10 NSAMPLES=10 MAX_SAMPLES_PER_SHARD=200000

# each task should see exactly 1 device
python - << 'PY'
import os, jax
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("devices:", jax.devices())
PY

srun --ntasks=4 --gpu-bind=single:1 python3 -u shapes_f.py
