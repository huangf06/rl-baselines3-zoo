#!/bin/bash
#SBATCH --job-name=bootstrapped-dqn-1m
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --output=/var/scratch/fhu100/rlzoo-logs/bootstrapped_dqn_1m_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

WORKDIR=/var/scratch/fhu100/rlzoo-logs/bootstrapped_dqn/seed_101_1m
mkdir -p "$WORKDIR"

cd /home/fhu100/workspace/RL_UQ_Experiments/rl-baselines3-zoo

python train.py \
  --algo bootstrapped_dqn \
  --env SeaquestNoFrameskip-v4 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --n-timesteps 1000000 \
  --eval-freq 250000 \
  --eval-episodes 5 \
  --log-folder "$WORKDIR" \
  --save-freq 250000 \
  --log-interval 5000 \
  --tensorboard-log "$WORKDIR" \
  --verbose 1

echo "Training completed. Results saved to $WORKDIR" 