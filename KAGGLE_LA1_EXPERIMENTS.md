# Controlled La1 experiments on Kaggle

Use this updated code in /kaggle/working/CIDA. Keep your existing **friend-preset**
source checkpoint: no source retraining is needed for these experiments. If it is
attached under /kaggle/input, set SOURCE to that exact source.pt path instead.

```python
%cd /kaggle/working/CIDA
SOURCE = "/kaggle/working/runs/digits_friend/source.pt"
DATA = "/kaggle/working/data/digits"
```

Run seed 0 first. Compare all three settings at the same 30-epoch budget, then
repeat with seeds 0 1 2 3 4. Omit --download if Internet is unavailable and the
MNIST/USPS datasets are already cached at DATA.

## 1. Full method: La1 on all unlabeled targets

```python
!python adapt_target.py --source "{SOURCE}" --data "{DATA}" --download --device cuda \
    --epochs 30 --projection-steps 2000 --seeds 0 \
    --la1-samples all --eval-every 1 \
    --output /kaggle/working/runs/la1_comparison/all
```

## 2. La1 on confident targets only (first candidate)

```python
!python adapt_target.py --source "{SOURCE}" --data "{DATA}" --download --device cuda \
    --epochs 30 --projection-steps 2000 --seeds 0 \
    --la1-samples confident --eval-every 1 \
    --output /kaggle/working/runs/la1_comparison/confident
```

Images are drawn from the current confident subset B_t. Nearest-guide pseudo-labels
are recomputed for those images at each La1 step, just as for all-target La1.
The Gaussian prototypes are still fitted on all unlabeled target features.
The La1 iterator is separate from Lc/La2's iterator; five loss slots are preserved.
Empty confident sets skip the affected loss slots instead of training on other images.

## 3. No La1 (diagnostic ablation)

```python
!python adapt_target.py --source "{SOURCE}" --data "{DATA}" --download --device cuda \
    --epochs 30 --projection-steps 2000 --seeds 0 \
    --disable-separability --eval-every 1 \
    --output /kaggle/working/runs/la1_comparison/no_la1
```

Disabling La1 leaves the other losses' update budgets unchanged. Do not combine
this switch with --la1-samples confident or a non-default La1 temperature.

## 4. Optional softer Gaussian logits

After the three comparisons, test this as a separate, explicitly modified variant:

```python
!python adapt_target.py --source "{SOURCE}" --data "{DATA}" --download --device cuda \
    --epochs 30 --projection-steps 2000 --seeds 0 \
    --la1-samples confident --la1-temperature 10 --eval-every 1 \
    --output /kaggle/working/runs/la1_comparison/confident_T10
```

Temperature divides La1 log-density logits before cross entropy. It affects neither
source Ls1 nor other target losses. T=1 is the unchanged default. Non-unit temperatures
require log-density scoring (automatically selected by the friend preset).

## Saved results

Each output directory contains:

- metrics.json: final-epoch per-seed metrics, support IDs, mean and sample standard
  deviation, and the selected La1 settings.
- seed_0.pt: final model, complete configuration, and training history.
- epoch_metrics_seed_0.json: metrics at epoch 0 (after initialization) and each epoch,
  per-class accuracy, class counts, pseudo-label counts, and confident-set size.
- epoch_metrics_seed_0.csv: the same diagnostic curve in columns suitable for plots.

Replace 0 with the corresponding seed for other trial files. --eval-every 5 reduces
evaluation frequency while retaining epochs 0 and 30; --eval-every 0 disables epoch
scoring. Diagnostics do not alter model weights, BatchNorm statistics, or training
randomness. Evaluation labels never enter optimization. The reported checkpoint is
always the final epoch; there is no best-test-epoch selection.

```python
import json
from pathlib import Path
for variant in ("all", "confident", "no_la1"):
    path = Path("/kaggle/working/runs/la1_comparison") / variant / "metrics.json"
    if path.exists():
        results = json.loads(path.read_text())
        print(variant, {m: results[m] for m in ("ALL", "SHARED", "PRIV")})
        print("supports:", {t["seed"]: t["support_ids"] for t in results["trials"]})
```

Before interpreting differences, check that support IDs match between variants for
each seed. Retain the same source, seed list, preset, batch size, epoch count and
projection steps. Class-averaged metrics and test split remain unchanged. Label
confident-only, temperature scaling, and no-La1 as modified/ablated variants when
comparing against the original paper.
