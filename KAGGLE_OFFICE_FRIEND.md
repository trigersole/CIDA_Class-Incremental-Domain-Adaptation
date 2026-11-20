# Office-31 density experiment: Amazon to DSLR

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
needed because source Ls1 now uses density scores instead of log-density.
Old log-density checkpoints are rejected for this experiment. Subsequent runner calls
reuse its matching new checkpoint; adaptation starts afresh and overwrites target
results in that directory, so use a new --output for another experiment.

Outputs: /kaggle/working/runs/office_mlp3_density_all/source.pt and target/metrics.json,
seed_*.pt, and per-epoch JSON/CSV diagnostics. Save notebook outputs after completion.
Use --seeds 0 for a first diagnostic run; repeat with defaults for five trials.

## What transfers from Digits

- Office projections use the paper's three 256-to-256 FC layers, each followed by
  ELU, without a skip connection. Both `f_e` and `f_d` return the MLP output.
  They use standard Linear initialization and the existing reconstruction warm-up
  (`--projection-steps 2000`) to learn a near-identity mapping. Digits keeps its
  existing two-layer residual projection with linear output.
- Target BatchNorm recalibration (AdaBN) before adaptation.
- Joint image/support forward passes for La1/La2, sharing training BN statistics.
- Scale-adaptive Gaussian covariance shrinkage in source and target statistics.
- Density separability scores for source Ls1 and target La1. Explicit CLI score
  choices now override the preset; Digits friend still defaults to log-density.
- At least one confident sample for every nonempty pseudo-class; empty classes
  still have no confident samples.
- Shuffled confident subsets rather than class-balanced replacement sampling.
- Five loader passes of round-robin updates per epoch, matching the Digits preset.
- La1 uses all unlabeled targets. La2 and Lc use the confident 20% subset.
  All five target losses are enabled; source Ls1 and negative-training Ls2 are enabled.

ResNet-50, the 2048-1024-256-256 feature head, classifier heads, and ImageNet input
normalization remain Office-specific. Do not apply Digits [-1,1] normalization or
replace ResNet with LeNet. All five losses remain enabled. This is a modified
method, not an exact paper reproduction; 50/60 epochs are experiment choices.

The existing friend-preset update budget remains five loader passes per target
epoch. Compare against the no-skip, all-La1 log-density run at the same budget.
Benchmark accuracy must be measured; synthetic integration tests do not predict it.

## Running the density experiment

Run the notebook cell above after pulling the update. The runner trains source
with `--gaussian-score density`, then adapts with `--gaussian-score density
--la1-samples all`. It uses a new output folder, preserving the earlier results.
Keep the source seed, support seeds, batch size, learning rate and epoch budgets
fixed when comparing with the previous no-skip, all-La1 log-density run.

The projections retain three FC+ELU layers without skips; reconstruction warm-up
remains 2,000 steps. Covariance shrinkage, augmentation, AdaBN and optimizer
assignments are unchanged. This tests the printed density objective and is not
an exact reproduction of every paper setting.

Density logits use the actual Gaussian PDF, not a normalized substitute or a
fallback to log-density. Very small PDFs can produce weak gradients; overflowing
PDFs raise an error. Do not infer useful learning from a finite loss alone.
`source_settings` and `target_settings` logs identify the selected score. Check
`disabled_losses: []` and positive updates for all five target losses.

For a direct CLI adaptation call, an omitted `--gaussian-score` inherits the
source checkpoint's score. Explicitly changing it permits a separate target-only
ablation, but that is not this source-and-target comparison. Old friend source
checkpoints without a score field are interpreted as log-density.
