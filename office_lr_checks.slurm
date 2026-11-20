#!/bin/bash
#SBATCH --job-name=cida-office-lr
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=office-lr-%j.out
#SBATCH --error=office-lr-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export CIDA_DEVICE_DIAGNOSTICS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
"$PWD/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
"$PWD/.venv/bin/python" -u diagnose_office_training.py --comparison ft-lr --device cuda "$@"
