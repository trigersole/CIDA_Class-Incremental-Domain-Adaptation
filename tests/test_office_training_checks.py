"""Checks for actual-dtype gradients and a controlled, all-loss BN comparison."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import diagnose_office_training as checks
from cida.cli import save_checkpoint
from cida.models import SourceModel
from cida.prototypes import GaussianBank, fit_gaussians
from cida.training import TrainingConfig, diagnostic_state, freeze_batchnorm_statistics, target_optimizers
from cida.diagnostics import pseudo_precision_summary
from cida.guides import build_guides


class OfficeTrainingChecksTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_density_probe_detects_lost_gradients_without_mutating_source(self):
        with patch("cida.models.feature_extractor", return_value=nn.Sequential(
                nn.Linear(256, 256), nn.BatchNorm1d(256))):
            source = SourceModel(20, "office31", preset="friend")
        bank = GaussianBank(torch.randn(20, 256, dtype=torch.double) * .1,
                            torch.eye(256, dtype=torch.double).repeat(20, 1, 1))
        data = TensorDataset(torch.randn(8, 256), torch.arange(8))
        before = {key: value.clone() for key, value in source.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        result = checks.source_gradient_probe(source, bank, data,
            TrainingConfig(batch_size=4), torch.device("cpu"), batches=1)
        for mode in ("train", "eval"):
            density = next(r for r in result if r["bn_mode"] == mode and r["objective"] == "density")
            log = next(r for r in result if r["bn_mode"] == mode and r["objective"] == "log-density")
            self.assertAlmostEqual(density["loss"], density["uniform_loss"])
            self.assertEqual(density["extractor_gradient"]["nonzero_elements"], 0)
            self.assertGreater(log["extractor_gradient"]["norm"], 0)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in source.parameters()))

    def test_diagnostic_and_guide_modes_preserve_frozen_bn(self):
        with patch("cida.models.feature_extractor", return_value=nn.Sequential(
                nn.Linear(12, 256), nn.BatchNorm1d(256))):
            source = SourceModel(2, "office31", preset="friend")
        model = checks.AdaptationModel(source, 1)
        model.train()
        freeze_batchnorm_statistics(model.f_t)
        bank = GaussianBank(torch.zeros(2, 256, dtype=torch.double),
                            torch.eye(256, dtype=torch.double).repeat(2, 1, 1))
        build_guides(model, bank, torch.randn(1, 12))
        self.assertTrue(model.f_t.training)
        self.assertFalse(model.f_t[1].training)
        with diagnostic_state(model, torch.device("cpu")):
            model.eval()
            torch.rand(30)
        self.assertTrue(model.f_t.training)
        self.assertFalse(model.f_t[1].training)
        self.assertTrue(model.f_t[1].weight.requires_grad)

    def test_full_script_paired_arms_and_checkpoint_preservation(self):
        self._full_script_paired_arms("bn")

    def test_full_script_learning_rate_comparison_and_precision(self):
        self._full_script_paired_arms("ft-lr")

    def test_only_ft_gets_lower_rate_in_every_optimizer(self):
        with patch("cida.models.feature_extractor", return_value=nn.Sequential(
                nn.Linear(12, 256), nn.BatchNorm1d(256))):
            source = SourceModel(2, "office31", preset="friend")
        model = checks.AdaptationModel(source, 1)
        ft_ids = {id(p) for p in model.f_t.parameters()}
        for name, optimizer in target_optimizers(model, 1e-4, ft_lr=1e-5).items():
            seen_ft = False
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    is_ft = id(parameter) in ft_ids
                    seen_ft |= is_ft
                    self.assertEqual(group["lr"], 1e-5 if is_ft else 1e-4)
            self.assertEqual(seen_ft, name in {"Lc", "La1", "La2"})
        for optimizer in target_optimizers(model, 1e-4).values():
            self.assertTrue(all(group["lr"] == 1e-4 for group in optimizer.param_groups))
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                target_optimizers(model, 1e-4, ft_lr=rate)

    def test_precision_uses_predicted_class_denominators(self):
        result = pseudo_precision_summary(torch.tensor([0, 0, 1, 2]),
                                          torch.tensor([0, 1, 1, 1]), 3, 2)
        self.assertEqual(result["micro_precision"], 50)
        self.assertEqual(result["macro_precision"], 50)
        self.assertEqual(result["per_class_precision"], [50., 100., 0.])
        self.assertAlmostEqual(result["shared_micro_precision"], 200 / 3, places=4)
        empty = pseudo_precision_summary(torch.tensor([], dtype=torch.long),
                                         torch.tensor([], dtype=torch.long), 3, 2)
        self.assertIsNone(empty["micro_precision"])
        self.assertIsNone(empty["macro_precision"])

    def _full_script_paired_arms(self, comparison):
        extractor = lambda *a, **k: nn.Sequential(nn.Linear(12, 256), nn.BatchNorm1d(256))
        with TemporaryDirectory() as temp, patch("cida.models.feature_extractor", side_effect=extractor):
            root = Path(temp)
            source = SourceModel(2, "office31", preset="friend")
            bank = fit_gaussians(torch.randn(16, 256), torch.arange(16) % 2, 2, .1, shrink=.1)
            payload = dict(version=1, stage="source", benchmark="office31", source_domain="amazon",
                           shared_classes=("a", "b"), model=source.state_dict(), prototypes=bank.state_dict(),
                           config=asdict(TrainingConfig(preset="friend", gaussian_score="log-density")))
            source_path, density_path = root / "source.pt", root / "density.pt"
            save_checkpoint(source_path, payload)
            save_checkpoint(density_path, {**payload, "config": {**payload["config"], "gaussian_score": "density"}})
            original = [p.read_bytes() for p in (source_path, density_path)]
            images = torch.randn(12, 12)
            data = TensorDataset(images, torch.arange(12) % 2)
            target = TensorDataset(images[:8], torch.arange(8))
            evaluation = TensorDataset(images[:8], torch.arange(8) % 3)
            output = root / "checks"
            args = ["diagnose_office_training.py", "--source", str(source_path),
                    "--density-source", str(density_path), "--data", str(root),
                    "--output", str(output), "--device", "cpu", "--epochs", "1",
                    "--comparison", comparison,
                    "--batch-size", "4", "--source-batches", "1", "--projection-steps", "2"]
            with patch("sys.argv", args), \
                 patch.object(checks, "office_source", return_value=(data, data, ("a", "b"))), \
                 patch.object(checks, "office_target", return_value=(None, ("c",))), \
                 patch.object(checks, "office_trial", return_value=(target, target, images[8:9], [8], evaluation)):
                checks.main()
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["identical_epoch_zero"])
            self.assertEqual(set(report["source_probes"]), {"log_density_source", "density_source"})
            normal, frozen = (report["arms"][key] for key in (
                ("bn_training", "bn_frozen") if comparison == "bn" else ("baseline", "slower_ft")))
            self.assertEqual(normal["initialization"], frozen["initialization"])
            self.assertTrue(normal["bn_buffers_changed_after_adabn"])
            if comparison == "bn":
                self.assertEqual(frozen["bn_buffers_changed_after_adabn"], [])
            else:
                self.assertTrue(frozen["bn_buffers_changed_after_adabn"])
                self.assertFalse(frozen["freeze_bn_stats"])
                self.assertEqual(normal["ft_lr"], 1e-4)
                self.assertEqual(frozen["ft_lr"], 1e-5)
            self.assertTrue(frozen["bn_affine_parameters_trainable"])
            self.assertTrue(frozen["bn_affine_changed"])
            for arm in (normal, frozen):
                self.assertTrue(arm["all_losses_updated_each_epoch"])
                target_epoch = next(r for r in arm["history"] if r["stage"] == "target")
                self.assertEqual(target_epoch["disabled_losses"], [])
                self.assertEqual(set(target_epoch["updates"]), {"Lr1", "Lr2", "Lc", "La1", "La2"})
                precision = arm["epochs"][-1]["pseudo_label_precision"]
                self.assertTrue(precision["counts_match_statistics"])
                self.assertTrue(precision["confident_assignments_match"])
                self.assertEqual(precision["all_clean_eval"]["count"], 8)
                self.assertEqual(precision["la1_training_batches"]["count"], 8)
            self.assertEqual(original, [p.read_bytes() for p in (source_path, density_path)])


if __name__ == "__main__":
    unittest.main()
