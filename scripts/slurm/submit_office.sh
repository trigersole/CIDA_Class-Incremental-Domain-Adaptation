#!/usr/bin/env bash
# Run after setup_office.sh; environment activation is unnecessary.
set -Eeuo pipefail
export CIDA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${CIDA_REPO}"
export CIDA_PYTHON="${CIDA_PYTHON:-${CIDA_REPO}/.venv/bin/python}"
[[ -x "${CIDA_PYTHON}" ]] || { echo 'Run bash scripts/slurm/setup_office.sh first, or set CIDA_PYTHON to an absolute Python path.' >&2; exit 2; }
export OFFICE_DATA="${OFFICE_DATA:-${CIDA_REPO}/data/office31}"
export OFFICE_OUTPUT="${OFFICE_OUTPUT:-${CIDA_REPO}/runs/office_residual3_no_la1}"
export OFFICE_SOURCE="${OFFICE_SOURCE:-${OFFICE_OUTPUT}/source.pt}"
export TORCH_HOME="${TORCH_HOME:-${CIDA_REPO}/.cache/torch}"
mkdir -p logs "${OFFICE_OUTPUT}"
DEPENDENCY=()
if [[ -f "${OFFICE_SOURCE}" ]]; then
    echo "Reusing source: ${OFFICE_SOURCE}"
else
    SOURCE_JOB=$(sbatch --parsable "$@" scripts/slurm/office.sbatch source)
    SOURCE_JOB="${SOURCE_JOB%%;*}"
    echo "Source job: ${SOURCE_JOB}"
    DEPENDENCY=("--dependency=afterok:${SOURCE_JOB}" --kill-on-invalid-dep=yes)
fi
TARGET_JOB=$(sbatch --parsable "$@" "${DEPENDENCY[@]}" --array=0-4%1 scripts/slurm/office.sbatch target)
echo "Target array: ${TARGET_JOB%%;*} (seeds 0-4, one at a time)"
echo "After all five finish: ${CIDA_PYTHON} scripts/slurm/summarize_office.py '${OFFICE_OUTPUT}'"
