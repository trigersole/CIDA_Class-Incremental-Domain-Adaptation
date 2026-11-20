# Three paper digit transfers with the successful modified preset

The model and training losses are unchanged from the successful digits friend
preset: convolutional BatchNorm, two-layer residual projections, AdaBN,
log-density, confident-only La1, all five adaptation losses, source warm-up + 10
epochs, target 30 epochs, batch 64, Adam 1e-4, projection steps 2000, seeds 0-4.
Shared classes are 0-4 and private classes 5-9, with one support per private class.
Held-out evaluation uses target train for adaptation/supports and target test for
scoring. Support selection continues to use a local PyTorch generator per seed.

## Campus cluster

After `git pull --ff-only`, install the SVHN reader dependency and download the
datasets on the login node (Internet needed for this step):

```bash
uv pip install --python .venv/bin/python 'scipy>=1.10'
.venv/bin/python - <<'PY'
from cida.data import digits
for name in ('mnist', 'usps', 'svhn'):
    for train in (True, False):
        ds = digits(name, 'data/digits', train, download=True, preset='friend')
        print(name, 'train' if train else 'test', len(ds))
PY
```

Submit one transfer at a time to respect the campus job limit:

```bash
sbatch digits.slurm usps mnist
# After that job finishes:
sbatch digits.slurm svhn mnist
```

For the original transfer use `sbatch digits.slurm mnist usps`. Each job trains
one source checkpoint (seed 0), then runs five adaptation seeds. Source checkpoints
are saved separately by source domain and reused on a subsequent invocation.
Target reruns overwrite files in that transfer's output directory; preserve old
outputs before repeating an experiment. There is no partial-stage resume.
All stages in a job share the six-hour limit.

Outputs: `runs/digits_best/usps_to_mnist/metrics.json` and
`runs/digits_best/svhn_to_mnist/metrics.json`, plus individual checkpoints and
per-epoch diagnostics. Logs: `digits-JOBID.out` and `digits-JOBID.err`.

## Direct commands (Kaggle, workstation, or an allocated GPU shell)

Choose a pair by changing the first two variables:

```bash
SOURCE=usps
TARGET=mnist
python train_source.py --benchmark digits --domain "$SOURCE" --preset friend --download --data data/digits --device cuda --batch-size 64 --lr 0.0001 --epochs 10 --seed 0 --output "runs/digits_best/${SOURCE}_source.pt"
python adapt_target.py --source "runs/digits_best/${SOURCE}_source.pt" --target-domain "$TARGET" --download --data data/digits --device cuda --protocol held-out --batch-size 64 --lr 0.0001 --epochs 30 --projection-steps 2000 --confidence-fraction 0.2 --la1-samples confident --seeds 0 1 2 3 4 --output "runs/digits_best/${SOURCE}_to_${TARGET}"
```

Use `SOURCE=svhn TARGET=mnist` or `SOURCE=mnist TARGET=usps` for the other pairs.
Use your environment's Python; on the cluster it is `.venv/bin/python`.
SVHN retains RGB, uses its standard train split without the extra split, and uses
torchvision's 0-9 label mapping. MNIST/USPS preprocessing is unchanged.
`evaluate.py` reads the target domain from new checkpoints; legacy target
checkpoints without a target domain still evaluate on USPS. `debug_digits.py`
remains an M-to-U-only diagnostic and rejects other transfers explicitly.
These are modified-method experiments, not exact paper replications.
