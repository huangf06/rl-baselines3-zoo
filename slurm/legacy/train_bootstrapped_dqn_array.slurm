#!/bin/bash
#SBATCH --job-name=bootdqn-test
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --begin=now
#SBATCH --array=0-2
#SBATCH --output=logs/bootdqn_test_%A_%a.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

# Use multiple seeds for better statistical significance
SEED_LIST=(101 1747 42)
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

WORKDIR=logs/bootstrapped_dqn_test/seed_${SEED}
mkdir -p "$WORKDIR"

python /home/fhu100/workspace/RL_UQ_Experiments/rl-baselines3-zoo/train.py \
  --algo bootstrapped_dqn \
  --env SeaquestNoFrameskip-v4 \
  --seed ${SEED} \
  --device cuda \
  --vec-env subproc \
  --n-timesteps 250000 \
  --eval-freq 25000 \
  --eval-episodes 5 \
  --log-folder "$WORKDIR" \
  --save-freq 25000 \
  --log-interval 1000 \
  --num-threads 2 \
  --optimize-memory-usage \
  --tensorboard-log "$WORKDIR" \
  --verbose 1 \
  --n-eval-envs 1

echo "Training for seed=${SEED} completed. Results saved to $WORKDIR" 