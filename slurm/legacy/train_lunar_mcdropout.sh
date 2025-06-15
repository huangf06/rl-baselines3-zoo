#!/bin/bash
#SBATCH --job-name=lunar_mc_dropout
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --begin=now
#SBATCH --output=logs/mc_dropout_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

WORKDIR=logs/lunar_mc_dropout/
mkdir -p "$WORKDIR"

python train.py \
  --algo mcdropout_dqn \
  --env LunarLander-v3 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --log-folder "$WORKDIR" \
  --tensorboard-log "$WORKDIR"

echo "MC Dropout DQN training completed. Results saved to $WORKDIR"
