#!/bin/bash
#SBATCH --job-name=mcdropout-lunar-array
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --array=0-9
#SBATCH --output=/var/scratch/fhu100/rlzoo-logs/%x_%A_%a.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

SEED_LIST=(101 307 911 1747 2029 2861 3253 4099 7919 9011)
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/var/scratch/fhu100/rlzoo-logs/lunar_mc_dropout/seed_${SEED}
mkdir -p "$WORKDIR"

python /home/fhu100/workspace/RL_UQ_Experiments/rl-baselines3-zoo/train.py \
  --algo mcdropout_dqn \
  --env LunarLander-v3 \
  --seed ${SEED} \
  --device cuda \
  --vec-env dummy \
  --n-timesteps 100000 \
  --log-folder "$WORKDIR" \
  --save-freq 25000 \
  --tensorboard-log "$WORKDIR" \
  --verbose 1

echo "Training for seed=${SEED} completed. Results saved to $WORKDIR"