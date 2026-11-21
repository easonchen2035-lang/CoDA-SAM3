#!/bin/bash
#SBATCH -p all
#SBATCH -x gmicloud-loki-gpu-001,gmicloud-loki-gpu-006
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/%x-%j.out
# Usage: sbatch -J <name> scripts/infer.sbatch <run_infer.py args...>   (one GPU, nodes 004/005 only)
cd "${SLURM_SUBMIT_DIR:?Submit the job from the project root}"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date -Is)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -u scripts/run_infer.py "$@" 2>&1 | grep --line-buffered -vE "Loading weights|deprecated"
echo "end $(date -Is)"
