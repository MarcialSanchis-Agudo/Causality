#!/bin/env bash
#SBATCH -A NAISS2024-5-129              # Project name
#SBATCH -p alvis                        # Cluster name
#SBATCH --output logs/VAE_3D_TRY.out    # Log file
#SBATCH --error errors/VAE_3D_TRY.error # Error file
#SBATCH -t 7-00:00:00                   # Max execution time
#SBATCH --gpus-per-node=A100:2         # Type and number of GPUs to use per node -C MEM512 Request a node with 512GB
## #SBATCH -c 16                           # 16 Cores per task
## #SBATCH -C MEM512                       # Request a node with 512GB
#SBATCH -N 1                            # Number of nodes
#SBATCH -J "VAE 3F"                     # Job name

ml purge
module load virtualenv/20.24.6-GCCcore-13.2.0 make/4.4.1-GCCcore-13.2.0 CMake/3.27.6-GCCcore-13.2.0 CUDA/12.4.0 UCX/1.15.0-GCCcore-13.2.0 UCC/1.2.0-GCCcore-13.2.0 Xvfb/21.1.9-GCCcore-13.2.0

source /mimer/NOBACKUP/groups/deepmechalvis/carlos/envvae/bin/activate
echo "INFO: All module has been loaded"

## Run DistributedDataParallel with srun (NCCL backend)
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python VAE_3D_TRY.py
echo "Training Finished"

echo "INFO: End interactive job"


