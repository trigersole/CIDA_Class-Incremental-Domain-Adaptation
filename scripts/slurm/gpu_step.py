"""Verify CUDA inside the Slurm step, then replace this process with training."""
import json
import os
import socket
import sys

import torch


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gpu_step.py TRAINING_SCRIPT [ARGS...]")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable INSIDE srun; inspect GPU allocation, driver and environment.")
    # Exercise the device, rather than checking only that a GPU name is visible.
    with torch.no_grad():
        probe = torch.ones((32, 32), device="cuda")
        product = probe @ probe
        torch.cuda.synchronize()
        assert product.is_cuda and product[0, 0].item() == 32
    print(json.dumps({
        "stage": "slurm_gpu_step", "host": socket.gethostname(),
        "python": sys.executable, "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_compute_verified": True,
        **{key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")},
    }), flush=True)
    # Preserve the PID/exit status without keeping a second CUDA context alive.
    os.execv(sys.executable, [sys.executable, "-u", *sys.argv[1:]])
