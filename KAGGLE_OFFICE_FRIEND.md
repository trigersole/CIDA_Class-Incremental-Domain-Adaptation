# Office-31 stabilized preset: Amazon to DSLR

Attach the extracted Office-31 dataset and enable GPU and Internet. Run:

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/CIDA_repo')
def run(*args):
    subprocess.run(list(map(str, args)), check=True)
if (repo / '.git').exists():
    run('git', '-C', repo, 'pull', '--ff-only', 'origin', 'main')
else:
    run('git', 'clone', 'https://github.com/trigersole/CIDA_Class-Incremental-Domain-Adaptation.git', repo)
run(sys.executable, '-m', 'pip', 'install', '-q', '-r', repo / 'requirements.txt')
run(sys.executable, repo / 'run_office_friend.py')
```

The runner locates the dataset, trains one new source checkpoint (one warm-up +
50 source epochs), then runs 60 adaptation epochs for seeds 0-4. A new source is
needed for the full preset because source Gaussian covariance regularization
changes. The old baseline source is not silently reused. Subsequent runner calls
reuse its matching new checkpoint; adaptation starts afresh and overwrites target
results in that directory, so use a new --output for another experiment.

Outputs: /kaggle/working/runs/office_friend_a_d/source.pt and confident/metrics.json,
seed_*.pt, and per-epoch JSON/CSV diagnostics. Save notebook outputs after completion.
Use --seeds 0 for a first diagnostic run; repeat with defaults for five trials.

## What transfers from Digits

- Zero-initialized residual projections, now dimension 256 for Office, with linear output.
- Target BatchNorm recalibration (AdaBN) before adaptation.
- Joint image/support forward passes for La1/La2, sharing training BN statistics.
- Scale-adaptive Gaussian covariance shrinkage in source and target statistics.
- Log-density separability scores.
- At least one confident sample for every nonempty pseudo-class; empty classes
  still have no confident samples.
- Shuffled confident subsets rather than class-balanced replacement sampling.
- Five loader passes of round-robin updates per epoch, matching the Digits preset.
- Confident-only La1 explicitly selected by the runner, fraction 20%, temperature 1.

ResNet-50, the 2048-1024-256-256 feature head, classifier heads, and ImageNet input
normalization remain Office-specific. Do not apply Digits [-1,1] normalization or
replace ResNet with LeNet. All five losses remain enabled. This is a modified
method, not an exact paper reproduction; 50/60 epochs are experiment choices.

The adaptation update budget increases fivefold versus the previous Office run.
A whole-preset comparison cannot attribute gains to one individual change.
Benchmark accuracy must be measured; synthetic integration tests do not predict it.
