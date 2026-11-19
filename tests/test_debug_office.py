import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import debug_office
from cida.cli import save_checkpoint
from cida.models import SourceModel, AdaptationModel
from cida.prototypes import fit_gaussians
from cida.training import TrainingConfig
from dataclasses import asdict


class DebugOfficeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_separates_decoder_damage_and_private_competition(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_shared = 2
                self.f_t = nn.Identity()
                self.f_d = nn.Identity()
                self.g_s = nn.Linear(2, 3, bias=False)
                with torch.no_grad():
                    self.g_s.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))

            def joint_logits(self, features):
                shared = self.g_s(self.f_d(features))[:, :2]
                return torch.cat([shared, torch.full((len(features), 1), 100.)], 1)

        model = Toy()
        data = TensorDataset(torch.tensor([[2., 0.], [0., 2.], [1., 1.]]), torch.tensor([0, 1, 2]))
        metrics = debug_office.score_paths(model, data, TrainingConfig(), torch.device('cpu'))
        self.assertEqual(metrics['SHARED_restricted_20'], 100.)
        self.assertEqual(metrics['SHARED_full_31'], 0.)
        model.f_d = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.f_d.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
        metrics = debug_office.score_paths(model, data, TrainingConfig(), torch.device('cpu'))
        self.assertEqual(metrics['SHARED_restricted_20'], 0.)
        self.assertEqual(metrics['SHARED_bypass_fd_restricted_20'], 100.)

    def test_bn_counterfactual_changes_buffers_only(self):
        original, target = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
        with torch.no_grad():
            original.running_mean.fill_(4)
            target.weight.fill_(2)
        debug_office.copy_bn_statistics(target, original)
        torch.testing.assert_close(target.running_mean, original.running_mean)
        torch.testing.assert_close(target.weight, torch.full((3,), 2.))

    def test_full_diagnostic_reads_checkpoints_without_modifying_them(self):
        extractor = lambda *args, **kwargs: nn.Sequential(nn.Linear(12, 256), nn.BatchNorm1d(256))
        with TemporaryDirectory(dir=Path.cwd()) as directory, \
                patch('cida.models.feature_extractor', side_effect=extractor):
            directory = Path(directory)
            source = SourceModel(2, 'office31', preset='friend')
            target = AdaptationModel(source, 1)
            features = torch.randn(12, 256)
            bank = fit_gaussians(features, torch.arange(12) % 2, 2, .1, shrink=.1)
            config = {**asdict(TrainingConfig(preset='friend', batch_size=4, gaussian_score='log-density')),
                      'projection_steps': 1}
            source_path, target_path = directory / 'source.pt', directory / 'seed_0.pt'
            save_checkpoint(source_path, dict(version=1, stage='source', benchmark='office31',
                            source_domain='amazon', shared_classes=('a', 'b'), config=config,
                            model=source.state_dict(), prototypes=bank.state_dict()))
            save_checkpoint(target_path, dict(version=1, stage='target', benchmark='office31',
                            source_domain='amazon', target_domain='dslr', shared_classes=('a', 'b'),
                            private_classes=('c',), config=config, model=target.state_dict(), support_ids=[8]))
            original_bytes = [p.read_bytes() for p in (source_path, target_path)]
            images = torch.randn(12, 12)
            clean = TensorDataset(images, torch.arange(12) % 3)
            stats = TensorDataset(images, torch.arange(12))
            source_data = TensorDataset(images, torch.arange(12) % 2)
            report_path = directory / 'report.json'
            with patch.object(debug_office, 'office_source', return_value=(None, source_data, ('a', 'b'))), \
                    patch.object(debug_office, 'office_target', return_value=(None, ('c',))), \
                    patch.object(debug_office, 'office_trial', return_value=(None, stats, images[8:9], [8], clean)), \
                    patch.object(sys, 'argv', ['debug_office.py', '--source', str(source_path),
                        '--target-checkpoint', str(target_path), '--data', str(directory),
                        '--device', 'cpu', '--output', str(report_path)]):
                debug_office.main()
            report = json.loads(report_path.read_text())
            self.assertTrue(report['source_head_preserved_exactly'])
            self.assertEqual(len(report['stages']), 5)
            self.assertEqual(original_bytes, [p.read_bytes() for p in (source_path, target_path)])


if __name__ == '__main__':
    unittest.main()
