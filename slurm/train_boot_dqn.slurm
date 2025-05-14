#!/bin/bash
#SBATCH --job-name=boot-dqn-seaquest
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --output=logs/boot_dqn_seaquest_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

WORKDIR=logs/boot_dqn/seaquest
mkdir -p "$WORKDIR"

python train.py \
  --algo bootstrapped_dqn \
  --env SeaquestNoFrameskip-v4 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --n-timesteps 250000 \
  --log-folder "$WORKDIR" \
  --save-freq 250000 \
  --log-interval 5000 \
  --tensorboard-log "$WORKDIR" \
  --verbose 0

echo "Training completed. Results saved to $WORKDIR" 