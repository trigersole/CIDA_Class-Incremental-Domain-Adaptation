"""Office preset integration: architecture, statistics and all five update paths."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from cida.models import SourceModel, AdaptationModel
from cida.guides import confident_indices
from cida.training import TrainingConfig, train_source, train_target


class OfficeFriendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_real_office_source_compatibility_and_identity(self):
        source = SourceModel(20, "office31", preset="friend")
        self.assertEqual(source.f_s[1].in_features, 2048)
        self.assertEqual(source.f_s[6].out_features, 256)
        model = AdaptationModel(source, 11)
        x = torch.randn(4, 256)
        torch.testing.assert_close(model.f_d(model.f_e(x)), x, rtol=0, atol=0)
        torch.testing.assert_close(model.f_t[1].weight, source.f_s[1].weight)
        torch.testing.assert_close(model.g_s[0].weight, source.g_s[0].weight)
        self.assertFalse(any(p.requires_grad for p in model.g_s.parameters()))
        self.assertEqual(model.joint_logits(x).shape, (4, 31))

    def test_office_training_and_checkpoint_roundtrip(self):
        # Substitute only the expensive CNN; keep 256-D Office heads/projections,
        # BN, Gaussian fitting, source training and every adaptation loss real.
        extractor = lambda *args, **kwargs: nn.Sequential(nn.Linear(12, 256), nn.BatchNorm1d(256))
        with patch("cida.models.feature_extractor", side_effect=extractor):
            source = SourceModel(2, "office31", preset="friend")
            config = TrainingConfig(preset="friend", epochs=1, batch_size=4,
                                    gaussian_score="log-density", max_steps=10)
            images = torch.randn(12, 12)
            bank, _ = train_source(source, TensorDataset(images, torch.arange(12) % 2),
                                   config, torch.device("cpu"))
            model = AdaptationModel(source, 1)
            history = train_target(model, TensorDataset(images[:8], torch.arange(8)),
                                   images[8:9], bank, config, 1, la1_samples="confident")
            self.assertEqual(history[-1]["updates"], dict.fromkeys(("Lr1", "Lr2", "Lc", "La1", "La2"), 2))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            restored = AdaptationModel(source, 1)
            restored.load_state_dict(model.state_dict())
            model.eval()
            restored.eval()
            torch.testing.assert_close(model(images[:2]), restored(images[:2]))
        selected = confident_indices(torch.tensor([0, 1, 1]), torch.tensor([1., 2., 1.]),
                                     2, .2, minimum_one=True)
        self.assertEqual(set(selected.tolist()), {0, 2})


if __name__ == "__main__":
    unittest.main()
