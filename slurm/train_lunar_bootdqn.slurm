#!/bin/bash
#SBATCH --job-name=boot-dqn-lunar
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --output=logs/boot_dqn_lunar_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

WORKDIR=logs/lunar_bootdqn
mkdir -p "$WORKDIR"

python train.py \
  --algo bootstrapped_dqn \
  --env LunarLander-v3 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --n-timesteps 100000 \
  --log-folder "$WORKDIR" \
  --save-freq 2500 \
  --log-interval 5000 \
  --tensorboard-log "$WORKDIR" \
  --verbose 0

echo "Training completed. Results saved to $WORKDIR" 