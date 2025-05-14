#!/bin/bash
#SBATCH --job-name=dqn-seaquest
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --output=logs/dqn_%j.out

. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source ~/.bashrc
conda activate rlzoo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

WORKDIR=logs/dqn/test_run
mkdir -p "$WORKDIR"

python train.py \
  --algo dqn \
  --env SeaquestNoFrameskip-v4 \
  --seed 101 \
  --device cuda \
  --vec-env subproc \
  --n-timesteps 250000 \
  --log-folder "$WORKDIR" \
  --save-freq 250000 \
  --log-interval 5000 \
  --n-envs 8 \
  --optimize-memory-usage \
  --tensorboard-log "$WORKDIR" \
  --verbose 1

echo "Training completed. Results saved to $WORKDIR" 