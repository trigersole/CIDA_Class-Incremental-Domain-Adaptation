# Diagnose the saved Office run without retraining

Attach BOTH the Office-31 image dataset and the saved output of the Office friend
notebook (the output must contain runs/office_friend_a_d/source.pt and
confident/seed_0.pt). Enable GPU and Internet. A notebook .ipynb alone has logs,
not model tensors. Do not run run_office_friend.py for this diagnosis.

```python
from pathlib import Path
import subprocess
import sys

def run(*args):
    subprocess.run(list(map(str, args)), check=True)

repo = Path('/kaggle/working/CIDA_repo')
if (repo / '.git').exists():
    run('git', '-C', repo, 'pull', '--ff-only', 'origin', 'main')
else:
    run('git', 'clone', 'https://github.com/trigersole/CIDA_Class-Incremental-Domain-Adaptation.git', repo)
run(sys.executable, '-m', 'pip', 'install', '-q', '-r', repo / 'requirements.txt')
run(sys.executable, repo / 'debug_office.py', '--seed', 0)
```

The diagnostic searches /kaggle/working/runs and /kaggle/input for exactly one
matching Office friend source/target checkpoint pair. It stops if none or multiple
pairs exist; it never starts source or target training. Resolve ambiguity with
--source /actual/source.pt --target-checkpoint /actual/confident/seed_0.pt.
Likewise, pass --data explicitly if more than one dataset root is attached.

Results: /kaggle/working/office_diagnostics/seed_0.json. Attach that report for
interpretation. Repeat with --seed 1 through 4 to inspect other saved trials.
Input checkpoints are never modified. Initialization replay runs the projection
reconstruction warm-up on a temporary in-memory model only; it does not perform
target adaptation. Source/target image labels only score diagnostics.

## Measurements

- Source training-set accuracy on clean Amazon images: measures fit, not held-out accuracy.
- Copied source with identity projections on clean DSLR images before AdaBN.
- Same model after AdaBN, then after replaying projection warm-up.
- Saved final target model on the same DSLR images, excluding its support IDs.
- Final weights with only BN running buffers restored from the source, as a
  diagnostic counterfactual rather than a proposed model or reported benchmark.

For each stage, SHARED_full_31 scores shared examples against all 31 classes.
SHARED_restricted_20 compares only shared logits. Their difference reveals
competition from private outputs. SHARED_bypass_fd_restricted_20 sends target
features directly to the preserved source head, bypassing the decoder; compare
with SHARED_restricted_20 to inspect decoder effects. Restricted scores are
diagnostics, not valid all-class CIDA performance.

The source-copy stage's shared-restricted decoder and bypass paths are also the
direct source model, because f_t copies f_s and f_d is exact identity initially.
The initially random private head is not trained, so initial PRIV/full-class
scores do not measure learned private recognition.

The tool verifies support IDs, class order, exact source-head preservation and
compares recomputed final metrics with saved metrics. It includes the original
saved epoch-0 metrics when that JSON is attached. Initialization replay can differ
across hardware/library versions; the saved epoch-0 record remains authoritative.
The BN-buffer swap and decoder bypass are counterfactual diagnostics: neither
alone proves the training cause, and target labels must not select a best epoch.
