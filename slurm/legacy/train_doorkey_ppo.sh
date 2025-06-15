#!/bin/bash
#SBATCH --job-name=doorkey_ppo
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --begin=now
#SBATCH --output=logs/ppo_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

WORKDIR=logs/doorkey_ppo/
mkdir -p "$WORKDIR"

python train.py \
  --algo ppo \
  --env MiniGrid-DoorKey-5x5-v0 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --log-folder "$WORKDIR" \
  --tensorboard-log "$WORKDIR" \

echo "Training completed. Results saved to $WORKDIR" 