#!/bin/env bash
#SBATCH -A NAISS2025-5-144              # Project name
#SBATCH -p alvis                        # Cluster name
# Error file
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:4        # Type and number of GPUs to use per node -C MEM512 Request a node with 512GB
#SBATCH -t 04:00:00
#SBATCH -J ShapeJAX
#SBATCH --output logs/shap_post.out
#SBATCH --error  errors/shap_post.error

set -euo pipefail

# ---- user knobs for plotting ----
FIGDIR_BASE="figs_shap"      # final dir becomes ${FIGDIR_BASE}_${JOBID}
SAMPLES=("0" "1" "2" "3")
MAX_MODES=6
THR=95.0
THR_SAMPLE_CAP=8
# Optionally pin a specific job id (e.g., export JOBID_FORCE=5196390 before sbatch)
JOBID_FORCE=5199899
# ---------------------------------

module purge
module load virtualenv/20.24.6-GCCcore-13.2.0
module load CUDA/12.8.0            # for libcudart.so.12 needed by h5py
source /mimer/NOBACKUP/groups/deepmechalvis/carlos/envvae/bin/activate

# CPU-only (do not reserve GPUs for post-processing)
unset CUDA_VISIBLE_DEVICES || true
export JAX_PLATFORM_NAME=cpu
export JAX_DISABLE_JAX_PLUGIN_DISCOVERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE

CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}
export OPENBLAS_NUM_THREADS=${CPUS}
export NUMEXPR_NUM_THREADS=${CPUS}

mkdir -p logs errors

# sanity
python - << 'PY'
import sys, numpy
print("python:", sys.version.split()[0])
print("numpy :", numpy.__version__)
PY

# ---- discover shard files for all jobs (or a specific job if forced) ----
shopt -s nullglob globstar
if [[ -n "${JOBID_FORCE}" ]]; then
  # Only pick files for that job id
  mapfile -t ALL_H5 < <(ls -1 **/shap_values_mean_job${JOBID_FORCE}_sh*-of-*.h5 2>/dev/null || true)
else
  mapfile -t ALL_H5 < <(ls -1 **/shap_values_mean_job*_sh*-of-*.h5 2>/dev/null || true)
fi
shopt -u nullglob

if (( ${#ALL_H5[@]} == 0 )); then
  echo "ERROR: no shard files found."
  if [[ -n "${JOBID_FORCE}" ]]; then
    echo "Looked for **/shap_values_mean_job${JOBID_FORCE}_sh*-of-*.h5"
  else
    echo "Looked for **/shap_values_mean_job*_sh*-of-*.h5"
  fi
  exit 2
fi

# Build map: jobid -> files, newest mtime per job
declare -A MAP_FILES MAP_COUNT MAP_MTIME
for f in "${ALL_H5[@]}"; do
  base=$(basename "$f")
  if [[ "$base" =~ shap_values_mean_job([0-9]+)_sh([0-9]+)-of-([0-9]+)\.h5 ]]; then
    jid="${BASHREMATCH[1]}"
    MAP_FILES["$jid"]+="${f}"$'\n'
    MAP_COUNT["$jid"]=$(( ${MAP_COUNT["$jid"]:-0} + 1 ))
    mt=$(stat -c %Y "$f")
    if [[ -z "${MAP_MTIME["$jid"]:-}" || $mt -gt ${MAP_MTIME["$jid"]} ]]; then
      MAP_MTIME["$jid"]=$mt
    fi
  fi
done

if (( ${#MAP_COUNT[@]} == 0 )); then
  echo "ERROR: no files matched expected naming pattern."
  exit 3
fi

# Select best job id
best_jid=""
best_count=-1
best_mtime=-1
for jid in "${!MAP_COUNT[@]}"; do
  cnt=${MAP_COUNT["$jid"]}
  mt=${MAP_MTIME["$jid"]}
  if (( cnt > best_count )) || { (( cnt == best_count )) && (( mt > best_mtime )); }; then
    best_jid="$jid"
    best_count=$cnt
    best_mtime=$mt
  fi
done

echo "Selected job id: ${best_jid}  (files: ${best_count})"
mapfile -t FILES < <(printf '%s' "${MAP_FILES["$best_jid"]}" | sed '/^$/d' | sort -V)

echo "Files to process:"
printf '  %s\n' "${FILES[@]}"

FIGDIR="${FIGDIR_BASE}_${best_jid}"
MERGED_OUT="shap_values_mean_job${best_jid}_merged.h5"
mkdir -p "${FIGDIR}"

# samples array → args
SAMPLE_ARGS=()
for s in "${SAMPLES[@]}"; do SAMPLE_ARGS+=( "$s" ); done

# If one file only, just plot from it; else merge then plot
if (( ${#FILES[@]} == 1 )); then
  python -u post_shapes.py \
    --merged "${FILES[0]}" \
    --figdir "${FIGDIR}" \
    --samples "${SAMPLE_ARGS[@]}" \
    --max-modes ${MAX_MODES} \
    --thr ${THR} \
    --thr-sample-cap ${THR_SAMPLE_CAP}
else
  python -u post_shapes.py \
    --inputs "${FILES[@]}" \
    --output "${MERGED_OUT}" \
    --figdir "${FIGDIR}" \
    --samples "${SAMPLE_ARGS[@]}" \
    --max-modes ${MAX_MODES} \
    --thr ${THR} \
    --thr-sample-cap ${THR_SAMPLE_CAP}
fi

echo "Done. Plots in: ${FIGDIR}"
