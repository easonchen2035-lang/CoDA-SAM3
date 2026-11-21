#!/bin/bash
#SBATCH -p all
#SBATCH -x gmicloud-loki-gpu-001,gmicloud-loki-gpu-006
#SBATCH -c 64
#SBATCH --mem=600G
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/%x-%j.out
# Usage: sbatch -J <name> scripts/cpu.sbatch <python script> <args...>   (CPU-only, nodes 004/005)
cd "${SLURM_SUBMIT_DIR:?Submit the job from the project root}"
echo "host $(hostname) start $(date -Is)"
.venv/bin/python -u "$@"
echo "end $(date -Is)"
