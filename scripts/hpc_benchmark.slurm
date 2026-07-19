#!/usr/bin/env bash
#SBATCH --job-name=metbit-bench
#SBATCH --output=metbit-bench-%j.out
#SBATCH --error=metbit-bench-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # request enough cores to sweep OpenMP threads
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# For a GPU node, uncomment and adjust to your cluster's partition/gres:
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#
# Submit from the repository root:
#     sbatch scripts/hpc_benchmark.slurm
#
# Building AND running inside the allocation guarantees that -march=native
# targets the actual compute-node CPU (safe on heterogeneous clusters).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Load your cluster's toolchain here (edit to match your site) ---------
# module purge
# module load GCC/12.3.0
# module load Python/3.11.3
# module load CUDA/12.2          # only on a GPU node
# source .venv/bin/activate      # or your virtualenv/conda env

# Sweep up to the cores SLURM actually granted this job.
export THREADS="1 2 4 8 ${SLURM_CPUS_PER_TASK:-16}"
export PYTHON="${PYTHON:-python3}"
# REPS=7 gives slightly steadier numbers on a shared cluster; optional.
export REPS="${REPS:-7}"

bash scripts/hpc_benchmark.sh
