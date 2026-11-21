#!/bin/bash
#SBATCH -p all
#SBATCH -x gmicloud-loki-gpu-001,gmicloud-loki-gpu-006
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/slurm/%x-%j.out
# Usage: sbatch -J <name> scripts/gpu_py.sbatch <script.py> <args...>  (1 GPU, nodes 004/005)
cd "${SLURM_SUBMIT_DIR:?Submit the job from the project root}"
echo "host $(hostname) start $(date -Is)"
.venv/bin/python -u "$@" 2>&1 | grep --line-buffered -vE "Loading weights|deprecated"
echo "end $(date -Is)"
