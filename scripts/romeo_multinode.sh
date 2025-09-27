#!/bin/bash
#SBATCH --job-name=physae-opt
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=romeo

set -euo pipefail

romeo_load_armgpu_env
spack load python@3.13.0/gzl2pkh cuda@12.6.2
source /home/cosmic_86/envs/pytorch_arm_test/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-12345}
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"

echo "[$(date)] Lancement de l'optimisation stage A"
srun python -m physae.cli optimise \
  --stages A \
  --n-trials 50 \
  --output-dir results/optuna \
  --show-progress

echo "[$(date)] Lancement de l'optimisation stage B"
srun python -m physae.cli optimise \
  --stages B \
  --n-trials 40 \
  --output-dir results/optuna \
  --show-progress

echo "[$(date)] Entraînement multi-stage"
srun python -m physae.cli train \
  --stages A B \
  --studies-dir results/optuna \
  --ckpt-dir results/checkpoints \
  --accelerator gpu \
  --strategy ddp \
  --devices 4 \
  --num-nodes $SLURM_JOB_NUM_NODES \
  --metrics-out results/checkpoints/train_metrics.json

FINAL_CKPT=$(ls -1t results/checkpoints/stage_*.ckpt | head -n 1)

echo "[$(date)] Inférence sur le jeu de validation"
srun python -m physae.cli infer \
  --checkpoint "$FINAL_CKPT" \
  --output results/inference/val_predictions.csv \
  --metrics-out results/inference/val_metrics.json \
  --device cuda

echo "[$(date)] Pipeline terminé."
