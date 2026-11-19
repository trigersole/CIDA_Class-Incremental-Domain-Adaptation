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
# Keep BLAS and PyTorch within the four allocated CPUs; no loader workers.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-${CIDA_REPO}/.cache/torch}"
PYTHON="${CIDA_PYTHON:?Python interpreter was not exported}"
DATA="${OFFICE_DATA:-${CIDA_REPO}/data/office31}"
OUTPUT="${OFFICE_OUTPUT:-${CIDA_REPO}/runs/office_residual3_no_la1}"
SOURCE="${OFFICE_SOURCE:-${OUTPUT}/source.pt}"

"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check the Python environment and GPU allocation"; print("GPU:", torch.cuda.get_device_name(0))'
for domain in amazon dslr; do
    [[ -d "${DATA}/${domain}" ]] || { echo "Missing dataset: ${DATA}/${domain}" >&2; exit 2; }
done
mkdir -p "${OUTPUT}"
echo "Stage: ${1:-}; data: ${DATA}; source: ${SOURCE}; output: ${OUTPUT}"

case "${1:-}" in
    source)
        [[ ! -e "${SOURCE}" ]] || { echo "Source already exists; submit again to reuse it." >&2; exit 2; }
        srun "${PYTHON}" -u train_source.py \
            --benchmark office31 --preset friend --domain amazon \
            --office-shared 20 --office-weights imagenet \
            --data "${DATA}" --device cuda --batch-size 8 --lr 0.0001 \
            --epochs 50 --seed 0 --workers 0 --cpu-threads 4 \
            --output "${SOURCE}"
        ;;
    target)
        SEED="${2:?Pass the target seed as the second script argument}"
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
        srun "${PYTHON}" -u adapt_target.py \
            --source "${SOURCE}" --target-domain dslr --data "${DATA}" \
            --device cuda --batch-size 8 --lr 0.0001 --epochs 60 \
            --projection-steps 2000 --confidence-fraction 0.2 \
            --disable-separability --eval-every 1 \
            --workers 0 --cpu-threads 4 --seeds "${SEED}" --output "${TRIAL}"
        ;;
    *) echo "Expected source or target stage." >&2; exit 2 ;;
esac
