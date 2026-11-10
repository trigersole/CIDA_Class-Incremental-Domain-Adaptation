# Class-Incremental Domain Adaptation

An independent PyTorch implementation of Kundu et al., [Class-Incremental Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf), ECCV 2020. It implements the single-upgrade one-shot setting for **MNIST → USPS** and follows the [supplementary algorithms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052-supp.pdf).

## Run

Use Python 3.10 or newer and a compatible PyTorch/torchvision pair. The commands auto-select CUDA when available. Use `--device cpu` to force CPU execution. Run from this directory.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

python train_source.py --download --data data/source --output runs/source.pt
python adapt_target.py --download --data data/target --source runs/source.pt --seeds 0 1 2 3 4
python evaluate.py --data data/target --checkpoint runs/target/seed_0.pt
```

For a quick execution check, pass `--epochs 1 --max-steps 6` to source training and `--epochs 1 --max-steps 10 --projection-steps 20` to adaptation. The step cap limits optimizer updates. Prototype refreshes still use the full corresponding dataset. Drop `--download` when datasets are already cached in the indicated roots.

Source training uses labeled MNIST digits 0–4. The source checkpoint contains model weights and Gaussian statistics. Adaptation reads the checkpoint, USPS images and one labeled support image for each of digits 5–9. The unlabeled adaptation loader exposes only images and sample IDs. The support IDs are excluded from that loader. Evaluation reports macro accuracy for all ten classes (`ALL`), novel classes (`PRIV`) and shared classes (`SHARED`), along with per-class results and micro accuracy. `runs/target/metrics.json` records seed-level results and aggregate mean and sample standard deviation.

## Implementation map

| Paper component | Code |
|---|---|
| Supplement Table 2B, modified LeNet and heads | `cida/models.py` |
| Gaussian prototypes, Eqs. 3 and 6, Algorithm 1 negatives | `cida/prototypes.py` |
| Foresighted source training, Algorithm 2 | `cida/training.py` and `train_source.py` |
| Guides, per-class confidence sets, Algorithm 3 | `cida/guides.py` |
| Projection initialization and five alternating target objectives, Algorithm 4 | `cida/training.py` and `adapt_target.py` |
| Joint-logit target evaluation | `cida/metrics.py` and `evaluate.py` |

## Reproduction choices

- **Target split.** `held-out` uses the USPS training split for adaptation and support selection, then evaluates on the USPS test split. `--protocol transductive` adapts and evaluates on the USPS test pool after removing the support examples. The paper gives the class split and one-shot procedure but does not spell out the MNIST/USPS split handling. Record the protocol when comparing with Table 2, which reports 72.5% `ALL` and 71.6% `PRIV` for MNIST → USPS.
- **Architecture.** Digits are resized to RGB 32 × 32. Convolution settings match the spatial sizes in Table 2B. The private head emits five logits as specified by the main paper's description and inference equation. Table 2B lists the size of the full target label set for this head.
- **Training schedule.** The main paper specifies Adam at `1e-4`, 3σ negative filtering, a 1:1 ratio of source examples to negative features, and 20% confidence selection. The supplement describes one epoch of source warmup. The code updates both the feature extractor and head in that cross-entropy warmup, then takes the later loss-to-module mappings from the numbered algorithms. The project defaults of 10 training epochs per stage, 1,000 identity-training steps and batch size 64 are configurable. The paper leaves these budgets unspecified.
- **Numerics.** The default separability objective takes Gaussian densities as logits, as printed in Eqs. 3 and 6. Gaussian calculations use float64 and full covariance with a configurable ridge (`0.1 I` by default). For a singleton target cluster, use its feature as the mean and the pooled covariance. For an empty one, use the class guide and the pooled covariance. For a confidence set, floor 20% of each cluster size and skip confidence-only updates if all sets are empty. `--gaussian-score log-density` changes the objective and is available for controlled comparisons.
- **Negative features and gradients.** The rejection criterion follows Algorithm 1's maximum-density-class check. The threshold is computed exactly in log space. During adaptation the source head remains frozen, with gradients flowing through it where needed. Each alternating loss owns an independent Adam state for its specified modules.
