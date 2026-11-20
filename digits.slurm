#!/bin/bash
#SBATCH --job-name=cida-digits
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=digits-%j.out
#SBATCH --error=digits-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
SOURCE_DOMAIN="${1:-usps}"
TARGET_DOMAIN="${2:-mnist}"
case "$SOURCE_DOMAIN:$TARGET_DOMAIN" in
    mnist:usps|usps:mnist|svhn:mnist) ;;
    *) echo 'Use mnist usps, usps mnist, or svhn mnist.' >&2; exit 2 ;;
esac
PYTHON="$PWD/.venv/bin/python"
SOURCE="runs/digits_best/${SOURCE_DOMAIN}_source.pt"
OUTPUT="runs/digits_best/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
mkdir -p runs/digits_best

if [[ ! -f "$SOURCE" ]]; then
    "$PYTHON" -u train_source.py --benchmark digits --domain "$SOURCE_DOMAIN" \
        --preset friend --data data/digits --device cuda --batch-size 64 \
        --lr 0.0001 --epochs 10 --seed 0 --cpu-threads 4 --output "$SOURCE"
fi
"$PYTHON" -u adapt_target.py --source "$SOURCE" --target-domain "$TARGET_DOMAIN" \
    --data data/digits --device cuda --protocol held-out --batch-size 64 \
    --lr 0.0001 --epochs 30 --projection-steps 2000 --confidence-fraction 0.2 \
    --la1-samples confident --seeds 0 1 2 3 4 --cpu-threads 4 --output "$OUTPUT"
