"""Regression checks for the optional modified Digits training path."""
import unittest

import torch
from torch.utils.data import TensorDataset

from cida.models import SourceModel, AdaptationModel
from cida.prototypes import fit_gaussians
from cida.training import TrainingConfig, train_source, train_target


class FriendPresetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_identity_and_checkpoint_architecture(self):
        model = AdaptationModel(SourceModel(5, preset="friend"), 5)
        x = torch.randn(8, 100)
        torch.testing.assert_close(model.f_d(model.f_e(x)), x, rtol=0, atol=0)
        self.assertTrue(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.f_t.modules()))
        restored = AdaptationModel(SourceModel(5, preset="friend"), 5)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        images = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(model(images), restored(images))

    def test_scale_adaptive_covariance(self):
        x = torch.randn(20, 3, dtype=torch.double)
        bank = fit_gaussians(x, torch.zeros(20, dtype=torch.long), 1, 0.1, shrink=0.1)
        cov = torch.cov(x.T)
        expected = 0.9 * cov + 0.1001 * cov.diagonal().mean() * torch.eye(3)
        torch.testing.assert_close(bank.scale_tril[0] @ bank.scale_tril[0].T, expected)

    def test_source_and_target_training_budget_and_ablation(self):
        config = TrainingConfig(preset="friend", epochs=1, batch_size=4, gaussian_score="log-density")
        images = torch.rand(20, 3, 32, 32) * 2 - 1
        source = SourceModel(5, preset="friend")
        bank, source_history = train_source(source, TensorDataset(images, torch.arange(20) % 5),
                                            config, torch.device("cpu"))
        self.assertEqual(len(source_history), 2)
        target = TensorDataset(images[:8], torch.arange(8))
        model = AdaptationModel(source, 5)
        # Eight targets -> two batches -> ten total updates, two for each loss.
        history = train_target(model, target, images[8:13], bank, config, projection_steps=1)
        self.assertEqual(history[-1]["updates"], dict.fromkeys(("Lr1", "Lr2", "Lc", "La1", "La2"), 2))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        model = AdaptationModel(source, 5)
        history = train_target(model, target, images[8:13], bank, config, projection_steps=1,
                               disable_separability=True)
        self.assertNotIn("La1", history[-1]["updates"])
        self.assertEqual(history[-1]["updates"]["La2"], 2)
        self.assertEqual(history[-1]["disabled_losses"], ["La1"])


if __name__ == "__main__":
    unittest.main()
