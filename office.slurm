#!/bin/bash
#SBATCH --job-name=cida-office
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=office-%j.out
#SBATCH --error=office-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON="$PWD/.venv/bin/python"
DATA="$PWD/data/office31"
OUTPUT="$PWD/runs/office_residual3_no_la1"
SOURCE="$OUTPUT/source.pt"
export TORCH_HOME="$PWD/.cache/torch"
export PYTHONUNBUFFERED=1
export CIDA_DEVICE_DIAGNOSTICS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
mkdir -p "$OUTPUT"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'

# Stage 1: train one source model, or reuse its completed checkpoint.
if [[ ! -f "$SOURCE" ]]; then
    "$PYTHON" -u train_source.py \
        --benchmark office31 --preset friend --domain amazon \
        --office-shared 20 --office-weights imagenet \
        --data "$DATA" --device cuda --batch-size 8 --lr 0.0001 \
        --epochs 50 --seed 0 --workers 0 --cpu-threads 4 \
        --output "$SOURCE"
fi

# Stage 2: five sequential target trials. Completed seeds are retained.
for SEED in 0 1 2 3 4; do
    TRIAL="$OUTPUT/target/seed_$SEED"
    if [[ -f "$TRIAL/metrics.json" ]]; then
        echo "Skipping completed seed $SEED"
        continue
    fi
    echo "Training target seed $SEED"
    "$PYTHON" -u adapt_target.py \
        --source "$SOURCE" --target-domain dslr --data "$DATA" \
        --device cuda --batch-size 8 --lr 0.0001 --epochs 60 \
        --projection-steps 2000 --confidence-fraction 0.2 \
        --disable-separability --eval-every 1 \
        --workers 0 --cpu-threads 4 --seeds "$SEED" --output "$TRIAL"
done

"$PYTHON" scripts/slurm/summarize_office.py "$OUTPUT"
