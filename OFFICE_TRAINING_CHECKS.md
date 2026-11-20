# Office training diagnostics

## Feature-extractor learning-rate comparison

After the BN experiment, test slower feature updates with:

```bash
git pull --ff-only
sbatch office_lr_checks.slurm
```

This runs two five-epoch seed-0 trials sequentially in one GPU job:

| Setting | Baseline | Candidate |
| --- | --- | --- |
| `f_t` learning rate | `1e-4` | `1e-5` |
| `f_e`, `f_d`, `g_t` learning rate | `1e-4` | `1e-4` |
| BN running statistics | Updated normally after AdaBN | Same |
| Gaussian objective | Log-density | Same |
| Target losses | All five enabled; La1 uses all targets | Same |

Both trials reuse the saved log-density source, identical support images,
projection warm-up and model initialization. There are no architecture changes
or source retraining. `f_t` uses the smaller rate in Lc, La1 and La2; mixed
optimizers keep the other networks at `1e-4`. Projection warm-up keeps `1e-4`.

Results are in `runs/office_lr_checks_seed0/report.json`, with `baseline/` and
`slower_ft/` subdirectories. Logs are `office-lr-JOBID.out` and `.err`.
The earlier BN report is retained. Alternate source/output/epoch arguments work
with either Slurm launcher, for example:

```bash
sbatch office_lr_checks.slurm --source /path/to/log_density_source.pt --epochs 5
```

Each epoch now reports three kinds of pseudo-label precision:

- `all_clean_eval`: assignments on clean images using eval-mode guide features.
- `confident_statistics`: the actual confident labels and sample IDs selected
  at statistics refresh, for the next training epoch.
- `la1_training_batches`: the pseudo-labels actually used by La1 during the
  completed epoch, including training augmentation and BN behavior. Counts
  represent processed occurrences and may include repeated images.

Each includes micro precision, macro precision over nonempty predicted classes,
per-class precision and assignment counts. Shared/private precision groups by
the **assigned** class; it is different from shared/private classification
accuracy. Always inspect class coverage alongside precision. At epoch 0 there
are no training batches yet, so `la1_training_batches` is null.

Target ground truth is used only by read-only reporting, after pseudo-labels
have been used in their updates. It never enters a training loss, guide,
confidence selection or optimizer. Tests verify the observer leaves model
updates and RNG unchanged.

For a later standalone Office adaptation run, `adapt_target.py --ft-lr 1e-5`
sets this rate while `--lr 1e-4` sets other networks' rates. The flag defaults
to no override, preserving existing commands and Digits behavior. The learning
rate change is a diagnostic modification, not an exact paper reproduction.

## Earlier BN comparison

Run from the campus repository root after pulling the update:

```bash
sbatch office_checks.slurm
```

One GPU job runs all checks sequentially with one GPU, four CPUs, 32G host RAM
and a six-hour limit. Default duration is five target epochs per arm, seed 0,
batch size 8, Adam learning rate 1e-4 and 2,000 projection warm-up steps.

The script reuses `runs/office_residual3_no_la1/source.pt`, the previous Office
friend **log-density** source. The directory name refers to an old target
ablation: source Ls1 and negative-training Ls2 were enabled. The diagnostic
does not retrain or overwrite this checkpoint.

If your source is elsewhere, pass its actual path:

```bash
sbatch office_checks.slurm --source /absolute/path/to/log_density_source.pt
```

Checks:

1. Probe density and log-density Ls1 on real source images, in train and eval BN
   modes. Record score ranges and feature/extractor gradient norms in the actual
   model dtype, without optimizer updates. Restore BN buffers and random state.
2. Measure individual and cycle projection reconstruction errors on a fixed set
   of held-out proxy features. Also score target shared/private predictions and
   the decoder-bypass path before AdaBN, after AdaBN and after warm-up. The
   post-warm-up measurements are final errors, unlike the warm-up average loss.
3. Run the same support seed twice from the same source and initialization:
   normal BN updates versus frozen running statistics after AdaBN. Both use
   log-density, no projection skips and all five losses; La1 uses all targets.
   BN affine parameters remain trainable. Record changed BN buffers and affine
   parameters to verify the intended behavior.

If `runs/office_mlp3_density_all/source.pt` exists, also probe that completed
density source. Alternatively pass `--density-source /path/to/source.pt`.
If the density job stopped before saving its source checkpoint, these tests
still compare both objectives on the available log-density checkpoint; they
cannot recover intermediate weights from the interrupted density run.

Results: `runs/office_checks_seed0/report.json`, with each arm's history and
epoch metrics in `bn_training/` and `bn_frozen/`. Logs: `office-checks-JOBID.out`
and `.err`. Use a new `--output` for a repeat; existing reports are protected.
Target labels only score diagnostics and never train the model or select a
checkpoint. Five epochs are an early-instability check, not a final benchmark.

Kaggle or an interactive GPU allocation can run the Python file directly:

```bash
python diagnose_office_training.py --source /path/to/log_density_source.pt \
    --data /path/to/office31 --output /path/to/office_checks
```

After the diagnostic, a full Office adaptation command can optionally use
`--freeze-bn-stats`. Its default is off; existing training commands and Digits
retain their previous behavior. Do not change loss settings, support seed or
source checkpoint between the two arms of a BN comparison.
