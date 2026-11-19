# Office-31 on a campus Slurm cluster

## Simple submission (use this on CCDS-TC2)

Run from the repository root after the uv setup and dataset extraction:

```bash
sbatch office.slurm
```

`office.slurm` is a standalone batch script: direct `.venv/bin/python` commands
run source training then seeds 0-4 sequentially inside one GPU allocation, followed
by the combined summary. No submission helper, `srun`, Python execution wrapper,
or job array is involved. It uses your observed partition `MGPU-TC2`, one GPU,
four CPUs, 32G RAM, and six hours. Logs are `office-JOBID.out` and
`office-JOBID.err` in the repository root. Model/input/output device logging remains
enabled. Training settings and the existing output folder are unchanged.

The six-hour limit still applies. Resubmitting skips completed source/seeds;
an interrupted stage restarts. This simpler launch removes the earlier job-step
layer, but has not been verified live on campus and does not establish the cause
of previous SIGKILLs. The sections below also document the earlier helper scripts;
use the direct `sbatch office.slurm` command for the simple launch.

This runs Amazon-to-DSLR with 20 shared and 11 private classes, the new Office
three-layer FC+ELU residual projections, and La1 disabled. Digits is unchanged.

Each job requests one GPU, four CPUs, 32 GB **host RAM**, and six hours. GPU VRAM
depends on your cluster's GPU allocation; `--mem` does not select GPU VRAM.
The launcher submits **one job** that runs source training (one warm-up + 50
epochs), then all five target seeds (60 epochs each, seeds 0-4) sequentially,
then the combined result summary. There are no job arrays or queued dependencies.
Completed source/seed outputs are skipped when resubmitting the same experiment.
It refuses another submission while a `cida-office` job is queued or running.
Runtime on your cluster is
unmeasured: the entire pipeline shares one six-hour allocation. It may need
resubmission to complete all stages. Each individual stage must fit within six
hours. The current trainer saves
model checkpoints at stage completion and does not support partial-stage resume;
an interrupted source or seed must restart. Epoch diagnostics are saved during
target training. Separate completed seeds remain available.

## Setup on the login node

Install uv once if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Then clone the repository and set up its environment:

```bash
git clone https://github.com/trigersole/CIDA_Class-Incremental-Domain-Adaptation.git
cd CIDA_Class-Incremental-Domain-Adaptation
bash scripts/slurm/setup_office.sh
```

If already cloned, run `git pull --ff-only` instead. Setup uses uv to create
`.venv` with Python 3.11, installs torch 2.5.1 and torchvision 0.20.1 with CUDA 12.1
wheels, and downloads the ImageNet weights. An explicit CUDA backend avoids CPU
wheels on login nodes without GPUs. The compute nodes need an NVIDIA driver
compatible with that PyTorch CUDA build. If your campus requires CUDA modules,
load them before submission according to its instructions.

The submit script automatically exports the absolute `.venv/bin/python` path;
no activation, uv invocation, or package installation is needed inside GPU jobs.
For a campus-provided environment instead, set `CIDA_PYTHON` to its absolute
Python path. Setup must run in a Linux cluster shell; a Windows `.venv` cannot
be copied to Linux. Keep this environment on storage shared with compute nodes.

Copy your Office dataset into `data/office31/amazon/images/<class>/...` and
`data/office31/dslr/images/<class>/...` (31 original class folders per domain).
The intermediate `images` directory is optional. `webcam` is optional for A-to-D.

Setup caches the ImageNet weights under `.cache/torch` (or your exported
`TORCH_HOME`). No Internet access is needed during training once dependencies,
dataset, and weights are present. Put the repository/data/outputs on storage accessible to
compute nodes, not login-node-only temporary storage.

## Submit

```bash
bash scripts/slurm/submit_office.sh
```

This single job automatically runs source training and seeds 0-4, then generates
the summary. If the time limit stops the pipeline, run the same command again:
completed stages are skipped, while the interrupted stage restarts. Do not change
settings or source contents while reusing an output folder. Use a new
`OFFICE_OUTPUT` for a different experiment.

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

Every training `srun` step explicitly requests one GPU and four CPUs. Its
`gpu_step.py` launcher performs a CUDA matrix multiplication inside that step
before starting training. Look for `slurm_gpu_step` with
`cuda_compute_verified: true`, followed by `training_device` reporting model,
input, and output on `cuda:0` (the local CUDA index of the allocated GPU).
The job also logs `scontrol show job` so actual GPU/RAM allocations can be compared
with requests. A CPU-only step fails before training; visibility is never forced
by overwriting `CUDA_VISIBLE_DEVICES`.

These diagnostics do not establish the cause of a previous SIGKILL. An empty
`squeue` means the job has ended, not that it ran on CPU. Slurm may record a step
as cancelled under the job owner's UID when the parent batch fails. Check both
the batch status and error log before attributing termination to a user action.

```bash
squeue -u "$USER"
ls logs/
.venv/bin/python scripts/slurm/summarize_office.py runs/office_residual3_no_la1
```

Run the final command after all five seeds complete; it refuses missing seeds or
inconsistent settings. Use your `OFFICE_OUTPUT` path if changed. It writes overall
`metrics.json` containing the five trials and mean/sample standard deviation.
Each seed has independent checkpoints and epoch JSON/CSV under
`runs/office_residual3_no_la1/target/seed_N/`. Completed trial folders are protected
from accidental overwriting. To restart just a failed seed (for example seed 2),
set the variables below, and submit directly:

```bash
export CIDA_REPO="$PWD"
export CIDA_PYTHON="$PWD/.venv/bin/python"
sbatch scripts/slurm/office.sbatch target 2
```

Include your required partition/account options and the same custom OFFICE_*
paths, if any. A completed source checkpoint is required for this command.
