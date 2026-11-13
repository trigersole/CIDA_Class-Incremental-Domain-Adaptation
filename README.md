# Class-Incremental Domain Adaptation

An independent PyTorch implementation of Kundu et al., [Class-Incremental Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf), ECCV 2020. It implements the single-upgrade one-shot CIDA method for **MNIST â†’ USPS** and **Office-31** and follows the [supplementary algorithms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052-supp.pdf).

## Run

Use Python 3.10 or newer and a compatible PyTorch/torchvision pair. The commands auto-select CUDA when available. Use `--device cpu` to force CPU execution. Run from this directory.

### Digits

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

python train_source.py --download --data data/source --output runs/source.pt
python adapt_target.py --download --data data/target --source runs/source.pt --seeds 0 1 2 3 4
python evaluate.py --data data/target --checkpoint runs/target/seed_0.pt
```

For a quick execution check, pass `--epochs 1 --max-steps 6` to source training and `--epochs 1 --max-steps 10 --projection-steps 20` to adaptation. The step cap limits optimizer updates. Prototype refreshes still use the full corresponding dataset. Drop `--download` when datasets are already cached in the indicated roots.

Source training uses labeled MNIST digits 0â€“4. The source checkpoint contains model weights and Gaussian statistics. Adaptation reads the checkpoint, USPS images and one labeled support image for each of digits 5â€“9. The unlabeled adaptation loader exposes only images and sample IDs. The support IDs are excluded from that loader. Evaluation reports macro accuracy for all ten classes (`ALL`), novel classes (`PRIV`) and shared classes (`SHARED`), along with per-class results and micro accuracy. `runs/target/metrics.json` records seed-level results and aggregate mean and sample standard deviation.

### Office-31 on CUDA

Download and extract [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code). Pass the directory containing `amazon`, `dslr`, and `webcam` as `--data`. Each domain should have 31 class folders containing the images. The loader accepts `amazon/images/<class>/*.jpg` or `amazon/<class>/*.jpg` and the equivalent layouts for DSLR and Webcam.

Start with Amazon â†’ DSLR using the Table 2 class counts:

```bash
python train_source.py --benchmark office31 --domain amazon --data data/office31 --device cuda --output runs/office31/amazon.pt
python adapt_target.py --source runs/office31/amazon.pt --target-domain dslr --data data/office31 --device cuda --seeds 0 1 2 3 4 --output runs/office31/amazon_to_dslr
python evaluate.py --checkpoint runs/office31/amazon_to_dslr/seed_0.pt --data data/office31 --device cuda
```

Use `--office-shared 10` at source training for the Table 3 class counts and give that checkpoint a different output path. Adaptation infers the benchmark and class split from the source checkpoint. Office source training uses ImageNet-pretrained ResNet-50 by default, downloading weights on first use if they are absent from the PyTorch cache. Use `--office-weights none` for an offline smoke test with random initialization. Office defaults to batch size 8 and accepts `--batch-size` to tune memory use. For a quick smoke test use `--epochs 1 --max-steps 6` at source training and `--epochs 1 --max-steps 10 --projection-steps 20` at adaptation.

To run all six one-shot Office-31 transfers with the Table 2 class split, use this Bash loop. It trains one source checkpoint per domain and reuses it for both destinations. Each transfer writes its own `metrics.json` and target checkpoints.

```bash
set -e
for source in amazon dslr webcam; do
  python train_source.py --benchmark office31 --domain "$source" --data data/office31 --device cuda --output "runs/office31/$source.pt"
  for target in amazon dslr webcam; do
    [ "$source" = "$target" ] && continue
    python adapt_target.py --source "runs/office31/$source.pt" --target-domain "$target" --data data/office31 --device cuda --seeds 0 1 2 3 4 --output "runs/office31/${source}_to_${target}"
  done
done
```

Office evaluation uses the transductive target pool with the one-shot supports removed from adaptation and scoring. Every scored image participates in adaptation as an unlabeled example. Source training and target adaptation can also read separate dataset roots containing only their respective domains.

## Debug a Digits run before retraining

Use `--disable-guide-pulling` with `adapt_target.py` for the next controlled ablation. This skips `La2` in its original update slots and retains the other losses' update budgets and confident-batch consumption. The checkpoint config and metrics JSON record the ablation. This is a diagnostic experiment, with no improvement in accuracy assumed in advance.

Use the existing source and target checkpoints to measure where information is lost. The command writes `report.json` after each stage. It reports source training and held-out test accuracy, Gaussian scores and feature gradients, covariance eigenvalues, synthetic negative checks, projection reconstruction error on fixed proxy samples, guide/confidence precision on the adaptation pool, and the saved target model's test accuracy.

```bash
python debug_digits.py --download --device cuda --data /kaggle/working/data/digits --source /kaggle/working/runs/digits/source.pt --target-checkpoint /kaggle/working/runs/digits/mnist_to_usps/seed_0.pt --seed 0 --output /kaggle/working/runs/digits_debug/density.json
```

To inspect the log-density experiment, change both checkpoint paths to their `digits_logdensity` counterparts and use a separate report path. Reproduce the adaptation run's batch size using `--batch-size` if it differs from 64. The diagnostic uses the target checkpoint's protocol and checks that its support IDs match. It compares fixed-step projection warmup with an optional extension, e.g. `--projection-max-steps 5000`, using the same optimizer state and held-out proxy samples. Projection updates affect only a temporary diagnostic model. Existing checkpoints are read only. True target labels are used for diagnostic scoring and never enter an optimization loss. Keep final test results separate from any validation data used to select hyperparameters.

The original density objective produced near-uniform losses in the supplied Kaggle runs. The log-density variant lowered accuracy. These are unresolved reproduction issues. This command gathers evidence for the next correction without changing either objective. The negative report also distinguishes the supplement algorithm's maximum-density-class check from the prose requirement to be outside every class's 3-sigma region.

## Implementation map

| Paper component | Code |
|---|---|
| Supplement Table 2, ResNet-50 and modified LeNet models | `cida/models.py` |
| Office-31 domains, class splits and augmentations | `cida/office.py` |
| Gaussian prototypes, Eqs. 3 and 6, Algorithm 1 negatives | `cida/prototypes.py` |
| Foresighted source training, Algorithm 2 | `cida/training.py` and `train_source.py` |
| Guides, per-class confidence sets, Algorithm 3 | `cida/guides.py` |
| Projection initialization and five alternating target objectives, Algorithm 4 | `cida/training.py` and `adapt_target.py` |
| Joint-logit target evaluation | `cida/metrics.py` and `evaluate.py` |

## Reproduction choices

- **Target split.** `held-out` uses the USPS training split for adaptation and support selection, then evaluates on the USPS test split. `--protocol transductive` adapts and evaluates on the USPS test pool after removing the support examples. The paper gives the class split and one-shot procedure but does not spell out the MNIST/USPS split handling. Record the protocol when comparing with Table 2, which reports 72.5% `ALL` and 71.6% `PRIV` for MNIST â†’ USPS.
- **Architecture.** Digits are resized to RGB 32 Ã— 32. Office uses RGB 224 Ã— 224, ImageNet normalization, the ResNet-50 backbone through average pooling, and the 256-dimensional projection architecture in Supplement Table 2A. The private head has one output per target-private class. DSLR and Webcam use the augmentation families illustrated in Supplement Fig. 2 with configurable experiment settings for batch size and training budget. Support images and epoch statistics use deterministic transforms.
- **Office class ranks.** With `--office-shared 20`, ranks 1â€“20 are shared and 21â€“31 are private. With `--office-shared 10`, ranks 1â€“10 are shared and 21â€“30 are private. The supplement lists private ranks 21â€“31, which has one more class than Table 3's ten private classes. The implementation excludes rank 31 in the 10-shared configuration to match Table 3's class counts. The results JSON records the exact selected class names.
- **Training schedule.** The main paper specifies Adam at `1e-4`, 3Ïƒ negative filtering, a 1:1 ratio of source examples to negative features, and 20% confidence selection. The supplement describes one epoch of source warmup. The code updates both the feature extractor and head in that cross-entropy warmup, then takes the later loss-to-module mappings from the numbered algorithms. The configurable training budgets default to 10 epochs per stage, 1,000 identity-training steps, and batch sizes of 64 for Digits and 8 for Office-31. The paper leaves these budgets unspecified.
- **Numerics.** The default separability objective takes Gaussian densities as logits, as printed in Eqs. 3 and 6. Gaussian calculations use float64 and full covariance with a configurable ridge (`0.1 I` for Digits and `0.14 I` for the higher-dimensional Office features). For a singleton target cluster, use its feature as the mean and the pooled covariance. For an empty one, use the class guide and the pooled covariance. For a confidence set, floor 20% of each cluster size and skip confidence-only updates if all sets are empty. `--gaussian-score log-density` changes the objective and is available for controlled comparisons.
- **Negative features and gradients.** The rejection criterion follows Algorithm 1's maximum-density-class check. The threshold is computed exactly in log space. During adaptation the source head remains frozen, with gradients flowing through it where needed. Each alternating loss owns an independent Adam state for its specified modules.


## Optional modified Digits preset

`--preset friend` adopts the main numerical and initialization changes inspected in
nikita-bachhas/CIDA-reimplementation at commit 03bc2e5. This is an experimental
variant, not a claim of exact paper reproduction or guaranteed higher accuracy.
The baseline remains the default. Office is unchanged; this preset accepts Digits only.

The preset uses inputs normalized to [-1,1], convolutional BatchNorm, LeakyReLU
slope 0.2, residual projections initialized to the identity with linear outputs,
source and target covariance shrinkage (0.9*Sigma + 0.1001*mean(diag(Sigma))*I),
log-density Gaussian logits, and target BatchNorm recalibration before adaptation.
Private guides and target images share a forward pass during La1/La2. Confident
subsets retain at least one member of each nonempty pseudo-class and are shuffled
without class balancing. Each adaptation epoch has five times the baseline update
budget, so each loss receives approximately one target-loader pass of updates.
`--max-steps` caps TOTAL optimizer slots per epoch in both presets.

Train a NEW source: old source weights do not contain BatchNorm or the new
preprocessing. Checkpoints record the preset; adaptation and evaluation restore it
automatically. An explicit incompatible adaptation preset is rejected. Use separate
output directories so previous results are retained.

```bash
python train_source.py --preset friend --download --device cuda \
  --data /kaggle/working/data/digits --epochs 10 \
  --output /kaggle/working/runs/digits_friend/source.pt

python adapt_target.py --download --device cuda \
  --data /kaggle/working/data/digits \
  --source /kaggle/working/runs/digits_friend/source.pt \
  --epochs 30 --projection-steps 2000 --seeds 0 \
  --output /kaggle/working/runs/digits_friend/full

python adapt_target.py --download --device cuda \
  --data /kaggle/working/data/digits \
  --source /kaggle/working/runs/digits_friend/source.pt \
  --epochs 30 --projection-steps 2000 --seeds 0 --disable-separability \
  --output /kaggle/working/runs/digits_friend/no_la1

python evaluate.py --device cuda --data /kaggle/working/data/digits \
  --checkpoint /kaggle/working/runs/digits_friend/full/seed_0.pt
```

Run seed 0 first to compare initialization and forgetting. Then repeat both
adaptation commands with `--seeds 0 1 2 3 4`. Compare macro ALL/SHARED/PRIV and
per-class accuracy in metrics.json; no-La1 is a diagnostic ablation. Both ablation
switches preserve the five-slot schedule and do not increase other losses' budgets.

Remaining differences from the friend's repository: we retain our support-image
selection and exclude supports from the unlabeled pool; we retain guide means for
empty pseudo-classes, existing proxy batch composition, and final partial batches.
Thus seed numbers do not select identical support images across repositories, and
this preset is not a bit-for-bit reproduction of their runs. We also preserve our
paper-aligned Office head rather than copying their missing FC layer.


## Controlled La1 experiments and epoch diagnostics

See [KAGGLE_LA1_EXPERIMENTS.md](KAGGLE_LA1_EXPERIMENTS.md) for commands comparing
all-target La1, confident-only La1, no La1, and optional temperature scaling.
`--la1-samples all` and `--la1-temperature 1` preserve existing loss behavior.
Per-epoch diagnostic JSON/CSV is enabled by default; `--eval-every 0` disables it.
The last epoch remains the reported result. Existing friend-preset source checkpoints
are compatible; no source retraining is needed for this comparison.
