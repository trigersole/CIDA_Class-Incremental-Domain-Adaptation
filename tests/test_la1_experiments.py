import copy
import csv
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from cida.diagnostics import EpochMetrics
from cida.models import AdaptationModel, SourceModel
from cida.prototypes import fit_gaussians, separability_loss
from cida.training import TrainingConfig, loader, target_loss, train_target


class La1ExperimentsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.images = torch.rand(10, 3, 32, 32)
        self.target = TensorDataset(self.images[:8], torch.arange(8))
        self.supports = torch.rand(5, 3, 32, 32)
        self.source = fit_gaussians(torch.randn(30, 100), torch.arange(30) % 5, 5, 0.1)
        self.model = AdaptationModel(SourceModel(5, preset="friend"), 5)
        self.config = TrainingConfig(preset="friend", epochs=1, batch_size=4,
                                     gaussian_score="log-density", max_steps=10)

    def test_temperature_scales_log_density_loss_and_gradient(self):
        x = torch.randn(3, 100, requires_grad=True)
        labels = torch.arange(3)
        actual = separability_loss(x, labels, self.source, "log-density", temperature=10)
        expected = F.cross_entropy(self.source.log_density(x) / 10, labels)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                separability_loss(x, labels, self.source, "log-density", temperature=value)
        with self.assertRaises(ValueError):
            separability_loss(x, labels, self.source, "density", temperature=10)

    def test_confident_la1_uses_selected_images_and_preserves_counts(self):
        stats = SimpleNamespace(selected_ids=torch.tensor([0, 1]), selected_labels=torch.tensor([0, 0]),
                                counts=torch.tensor([8] + [0] * 9),
                                prototypes=fit_gaussians(torch.randn(30, 100), torch.arange(30) % 10, 10, 0.1))
        with patch("cida.training.refresh_statistics", return_value=stats), \
             patch("cida.training.target_loss", wraps=target_loss) as losses:
            history = train_target(self.model, self.target, self.supports, self.source, self.config, 1,
                                   la1_samples="confident", la1_temperature=10)
        self.assertEqual(history[-1]["updates"], dict.fromkeys(("Lr1", "Lr2", "Lc", "La1", "La2"), 2))
        la1_calls = [call for call in losses.call_args_list if call.args[0] == "La1"]
        self.assertEqual(len(la1_calls), 2)
        for call in la1_calls:
            for image in call.args[5]:
                self.assertTrue(any(torch.equal(image, allowed) for allowed in self.images[:2]))
            self.assertEqual(call.kwargs["la1_temperature"], 10)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_diagnostics_preserve_training_rng_and_model_state(self):
        other = copy.deepcopy(self.model)
        torch.manual_seed(100)
        train_target(self.model, self.target, self.supports, self.source, self.config, 1)
        observed = []

        def diagnostic(epoch, model, stats):
            observed.append(epoch)
            torch.rand(50)
            random.random()
            model(self.images[:2])
            return {"ALL": 0.0}

        torch.manual_seed(100)
        train_target(other, self.target, self.supports, self.source, self.config, 1,
                     epoch_callback=diagnostic)
        self.assertEqual(observed, [0, 1])
        for key, expected in self.model.state_dict().items():
            torch.testing.assert_close(other.state_dict()[key], expected, rtol=0, atol=0)

    def test_epoch_json_csv_and_final_epoch_cadence(self):
        dataset = TensorDataset(self.images, torch.arange(10))
        stats = SimpleNamespace(counts=torch.ones(10, dtype=torch.long), selected_ids=torch.tensor([0, 1]))
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            reporter = EpochMetrics(Path(directory), 0, loader(dataset, self.config), torch.device("cpu"),
                                    5, 10, epochs=3, every=2)
            self.model.eval()
            for epoch in range(4):
                reporter(epoch, self.model, stats)
            payload = json.loads((Path(directory) / "epoch_metrics_seed_0.json").read_text())
            self.assertEqual([row["epoch"] for row in payload["epochs"]], [0, 2, 3])
            with (Path(directory) / "epoch_metrics_seed_0.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("accuracy_class_9", rows[0])
            self.assertIn("pseudo_count_class_9", rows[0])
            self.assertEqual(rows[-1]["epoch"], "3")


if __name__ == "__main__":
    unittest.main()
