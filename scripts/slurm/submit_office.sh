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
# Only one submission per call, with no arrays or dependent queued jobs.
ACTIVE=$(squeue --noheader --user="${USER}" --name=cida-office --format=%A)
if [[ -n "${ACTIVE}" ]]; then
    echo "An Office job is already queued/running: ${ACTIVE}"
    echo "Wait for it to finish, then run this command again."
    exit 0
fi
JOB=$(sbatch --parsable "$@" scripts/slurm/office.sbatch all)
echo "Office pipeline job: ${JOB%%;*}"
echo "Runs source then seeds 0-4 sequentially, followed by the combined summary."
echo "If the six-hour limit interrupts it, rerun this command; completed stages are skipped."
