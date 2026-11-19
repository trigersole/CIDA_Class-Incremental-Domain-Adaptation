#!/usr/bin/env bash
# Run once on a login node with Internet access. No GPU required for setup.
set -Eeuo pipefail
CIDA_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${CIDA_SETUP_ROOT}"
command -v uv >/dev/null || { echo 'Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2; exit 2; }
uv venv --python 3.11 .venv
# Explicit CUDA backend: auto can select CPU wheels on GPU-less login nodes.
uv pip install --python .venv/bin/python --torch-backend cu121 -r requirements-slurm.txt
export TORCH_HOME="${TORCH_HOME:-${CIDA_SETUP_ROOT}/.cache/torch}"
.venv/bin/python - <<'PY'
import torch
from torchvision.models import ResNet50_Weights
assert torch.version.cuda is not None, "Expected CUDA-enabled PyTorch"
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
print("Environment and ImageNet weights are ready. Submit with bash scripts/slurm/submit_office.sh")
PY
