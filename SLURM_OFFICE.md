# Office-31 on a campus Slurm cluster

This runs Amazon-to-DSLR with 20 shared and 11 private classes, the new Office
three-layer FC+ELU residual projections, and La1 disabled. Digits is unchanged.

Each job requests one GPU, four CPUs, 32 GB **host RAM**, and six hours. GPU VRAM
depends on your cluster's GPU allocation; `--mem` does not select GPU VRAM.
Source training is one job (one warm-up + 50 epochs), followed by an array of five
target jobs (60 epochs each, seeds 0-4). `%1` runs only one array task at a time.
The array waits for successful source completion. Runtime on your cluster is
unmeasured: each stage must finish within six hours. The current trainer saves
model checkpoints at stage completion and does not support partial-stage resume;
an interrupted source or seed must restart. Epoch diagnostics are saved during
target training. Separate completed seeds remain available.

## Setup on the login node

Activate your cluster's CUDA-enabled Python environment first (Python 3.10+).
Use the campus module/Conda instructions for that environment. Then:

```bash
git clone https://github.com/trigersole/CIDA_Class-Incremental-Domain-Adaptation.git
cd CIDA_Class-Incremental-Domain-Adaptation
python -m pip install -r requirements.txt
```

If already cloned, run `git pull --ff-only` instead. Use the existing CUDA-enabled
PyTorch installation if your campus provides one; do not replace it with CPU-only
PyTorch. The submit script exports the absolute active Python interpreter to jobs,
so activate your environment before submitting. Load any required CUDA modules in
that same shell; environment variables are exported by sbatch.

Copy your Office dataset into `data/office31/amazon/images/<class>/...` and
`data/office31/dslr/images/<class>/...` (31 original class folders per domain).
The intermediate `images` directory is optional. `webcam` is optional for A-to-D.

Pre-download ImageNet weights on a node with Internet access, using the same
shared cache as the jobs:

```bash
export TORCH_HOME="$PWD/.cache/torch"
python -c 'from torchvision.models import ResNet50_Weights; ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)'
```

No Internet access is needed during training once dependencies, dataset, and
weights are present. Put the repository/data/outputs on storage accessible to
compute nodes, not login-node-only temporary storage.

## Submit

```bash
bash scripts/slurm/submit_office.sh
```

If your cluster requires a partition or account, pass the actual campus values:

```bash
bash scripts/slurm/submit_office.sh --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT
```

These are placeholders: no campus-specific partition/account was found in the
existing project Slurm script. The new script follows its `sbatch`/`srun` pattern
with the smaller allocation. It does not reuse the old 32-GPU resource request.

To reuse your saved matching Office **friend source** checkpoint, set its path
before submitting; it must be the full `source.pt`, not the notebook file:

```bash
export OFFICE_SOURCE="/absolute/path/to/source.pt"
bash scripts/slurm/submit_office.sh
```

Optional alternate paths (use absolute paths):

```bash
export OFFICE_DATA="/absolute/path/to/office31"
export OFFICE_OUTPUT="/absolute/path/to/runs/office_residual3_no_la1"
```

## Monitor and collect

```bash
squeue -u "$USER"
ls logs/
python scripts/slurm/summarize_office.py runs/office_residual3_no_la1
```

Run the final command after all five seeds complete; it refuses missing seeds or
inconsistent settings. Use your `OFFICE_OUTPUT` path if changed. It writes overall
`metrics.json` containing the five trials and mean/sample standard deviation.
Each seed has independent checkpoints and epoch JSON/CSV under
`runs/office_residual3_no_la1/target/seed_N/`. Completed trial folders are protected
from accidental overwriting. To restart just a failed seed (for example seed 2),
activate the same environment, set the variables below, and submit directly:

```bash
export CIDA_REPO="$PWD"
export CIDA_PYTHON="$(command -v python)"
sbatch --array=2 scripts/slurm/office.sbatch target
```

Include your required partition/account options and the same custom OFFICE_*
paths, if any. A completed source checkpoint is required for this command.
