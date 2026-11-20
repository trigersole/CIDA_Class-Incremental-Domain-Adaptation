import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import TensorDataset

from cida.data import digits, source_dataset, target_data, target_evaluation
from cida.prototypes import GaussianBank
import train_source
import adapt_target
import evaluate


class Fixture(TensorDataset):
    def __init__(self):
        self.targets = torch.arange(20) % 10
        super().__init__(torch.randn(20, 3, 32, 32), self.targets)


class DigitDomainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_svhn_split_rgb_and_targets(self):
        fake = type('SVHNFixture', (), {'labels': [0, 1, 9]})()
        with patch('torchvision.datasets.SVHN', return_value=fake) as constructor:
            result = digits('svhn', 'data', True, False, 'friend')
            self.assertEqual(result.targets, [0, 1, 9])
            self.assertEqual(constructor.call_args.kwargs['split'], 'train')
            rgb = constructor.call_args.kwargs['transform'](Image.new('RGB', (32, 32), (255, 0, 0)))
            self.assertEqual(tuple(rgb.shape), (3, 32, 32))
            self.assertTrue(torch.all(rgb[0] == 1))
            self.assertTrue(torch.all(rgb[1:] == -1))
            digits('svhn', 'data', False, False)
            self.assertEqual(constructor.call_args.kwargs['split'], 'test')

    def test_domain_routing_and_support_exclusion(self):
        with patch('cida.data.digits', side_effect=lambda *a, **k: Fixture()) as fetch:
            source = source_dataset('data', False, 'friend', 'usps')
            self.assertEqual(fetch.call_args.args[0], 'usps')
            self.assertTrue(all(source[i][1] < 5 for i in range(len(source))))
            target, supports, ids = target_data('data', False, 'held-out', 2, 'friend', 'mnist')
            self.assertEqual(fetch.call_args.args[0], 'mnist')
            self.assertTrue(fetch.call_args.kwargs['train'])
            self.assertTrue(set(ids).isdisjoint(target.indices))
            self.assertEqual([i % 10 for i in ids], [5, 6, 7, 8, 9])
            target_evaluation('data', False, 'held-out', ids, 'friend', 'mnist')
            self.assertFalse(fetch.call_args.kwargs['train'])
            self.assertEqual(fetch.call_args.args[0], 'mnist')

    def test_cli_checkpoint_and_evaluation_domains(self):
        bank = GaussianBank(torch.zeros(5, 100, dtype=torch.double),
                            torch.eye(100, dtype=torch.double).repeat(5, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source_name, target_name in [('mnist', 'usps'), ('usps', 'mnist'), ('svhn', 'mnist')]:
                with self.subTest(source=source_name):
                    source_path = root / (source_name + '.pt')
                    with patch('sys.argv', ['train_source.py', '--domain', source_name, '--preset', 'friend',
                                            '--device', 'cpu', '--output', str(source_path)]), \
                         patch('train_source.source_dataset', return_value=Fixture()) as source_loader, \
                         patch('train_source.train_source', return_value=(bank, [])):
                        train_source.main()
                        self.assertEqual(source_loader.call_args.args[-1], source_name)
                    out = root / source_name
                    with patch('cida.data.digits', side_effect=lambda *a, **k: Fixture()):
                        target = target_data('unused', False, 'held-out', 0, 'friend', target_name)
                    with patch('sys.argv', ['adapt_target.py', '--source', str(source_path),
                                            '--target-domain', target_name, '--device', 'cpu', '--seeds', '0',
                                            '--la1-samples', 'confident', '--output', str(out)]), \
                         patch('adapt_target.target_data', return_value=target) as target_loader, \
                         patch('adapt_target.target_evaluation', return_value=Fixture()), \
                         patch('adapt_target.train_target', return_value=[]), \
                         patch('adapt_target.evaluate', return_value=dict(ALL=1., PRIV=2., SHARED=3.)):
                        adapt_target.main()
                        self.assertEqual(target_loader.call_args.args[-1], target_name)
                    result = json.loads((out / 'metrics.json').read_text())
                    self.assertEqual((result['source_domain'], result['target_domain']), (source_name, target_name))
                    self.assertEqual(result['disabled_losses'], [])
                    with patch('sys.argv', ['evaluate.py', '--checkpoint', str(out / 'seed_0.pt'), '--device', 'cpu']), \
                         patch('evaluate.target_evaluation', return_value=Fixture()) as evaluation_loader, \
                         patch('evaluate.evaluate', return_value={}):
                        evaluate.main()
                        self.assertEqual(evaluation_loader.call_args.args[-1], target_name)


if __name__ == '__main__':
    unittest.main()
