#!/bin/bash
#SBATCH -p all
#SBATCH -x gmicloud-loki-gpu-001,gmicloud-loki-gpu-006
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/%x-%j.out
cd "${SLURM_SUBMIT_DIR:?Submit the job from the project root}"
echo "host $(hostname) start $(date -Is)"
.venv/bin/python -u scripts/run_select.py "$@" 2>&1 | grep --line-buffered -vE "Loading weights|deprecated"
echo "end $(date -Is)"
