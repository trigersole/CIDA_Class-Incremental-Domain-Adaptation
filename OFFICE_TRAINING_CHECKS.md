# Office training diagnostics

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
