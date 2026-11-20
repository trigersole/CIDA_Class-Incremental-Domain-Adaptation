#!/usr/bin/env bash
#SBATCH --job-name=cida-office
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/office-%j.out
#SBATCH --error=logs/office-%j.err

set -Eeuo pipefail
cd "${CIDA_REPO:?Submit with bash scripts/slurm/submit_office.sh}"
export PYTHONUNBUFFERED=1
export CIDA_DEVICE_DIAGNOSTICS=1
# Keep BLAS and PyTorch within the four allocated CPUs; no loader workers.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-${CIDA_REPO}/.cache/torch}"
PYTHON="${CIDA_PYTHON:?Python interpreter was not exported}"
DATA="${OFFICE_DATA:-${CIDA_REPO}/data/office31}"
OUTPUT="${OFFICE_OUTPUT:-${CIDA_REPO}/runs/office_mlp3_no_la1}"
SOURCE="${OFFICE_SOURCE:-${CIDA_REPO}/runs/office_residual3_no_la1/source.pt}"

echo "Job=${SLURM_JOB_ID:-unset}; host=$(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Requested: one GPU, four CPUs, 32G host RAM. Actual allocation:"
scontrol show job "${SLURM_JOB_ID}" || true
# Explicit step resources; the CUDA probe runs in the same step as training.
GPU_STEP=(srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1)
trap 'status=$?; echo "Batch exit status=${status}; date=$(date -Is)" >&2' EXIT
for domain in amazon dslr; do
    [[ -d "${DATA}/${domain}" ]] || { echo "Missing dataset: ${DATA}/${domain}" >&2; exit 2; }
done
mkdir -p "${OUTPUT}"
echo "Stage: ${1:-}; data: ${DATA}; source: ${SOURCE}; output: ${OUTPUT}"

train_source_stage() {
        [[ ! -e "${SOURCE}" ]] || { echo "Source already exists; submit again to reuse it." >&2; exit 2; }
        "${GPU_STEP[@]}" "${PYTHON}" -u scripts/slurm/gpu_step.py train_source.py \
            --benchmark office31 --preset friend --domain amazon \
            --office-shared 20 --office-weights imagenet \
            --data "${DATA}" --device cuda --batch-size 8 --lr 0.0001 \
            --epochs 50 --seed 0 --workers 0 --cpu-threads 4 \
            --output "${SOURCE}"
}

train_target_stage() {
        SEED="${1:?Pass the target seed}"
        [[ "${SEED}" =~ ^[0-4]$ ]] || { echo "Expected seed 0-4." >&2; exit 2; }
        TRIAL="${OUTPUT}/target/seed_${SEED}"
        [[ ! -e "${TRIAL}/metrics.json" ]] || { echo "Completed result exists: ${TRIAL}; choose another OFFICE_OUTPUT." >&2; exit 2; }
        "${PYTHON}" - "${SOURCE}" <<'PY'
import sys
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert (c.get("stage") == "source" and c.get("benchmark") == "office31"
        and c.get("source_domain") == "amazon" and len(c.get("shared_classes", [])) == 20
        and c.get("config", {}).get("preset") == "friend"
        and c.get("imagenet_pretrained") is True), "Expected an ImageNet-initialized Amazon/20-class friend source"
PY
        "${GPU_STEP[@]}" "${PYTHON}" -u scripts/slurm/gpu_step.py adapt_target.py \
            --source "${SOURCE}" --target-domain dslr --data "${DATA}" \
            --device cuda --batch-size 8 --lr 0.0001 --epochs 60 \
            --projection-steps 2000 --confidence-fraction 0.2 \
            --disable-separability --eval-every 1 \
            --workers 0 --cpu-threads 4 --seeds "${SEED}" --output "${TRIAL}"
}

case "${1:-all}" in
    source) train_source_stage ;;
    target) train_target_stage "${2:?Pass a seed}" ;;
    all)
        if [[ -f "${SOURCE}" ]]; then
            echo "Reusing source: ${SOURCE}"
        else
            train_source_stage
        fi
        for SEED in 0 1 2 3 4; do
            if [[ -f "${OUTPUT}/target/seed_${SEED}/metrics.json" ]]; then
                echo "Skipping completed seed ${SEED}"
            else
                echo "Starting seed ${SEED}"
                train_target_stage "${SEED}"
            fi
        done
        "${PYTHON}" scripts/slurm/summarize_office.py "${OUTPUT}"
        ;;
    *) echo "Expected all, source, or target stage." >&2; exit 2 ;;
esac
