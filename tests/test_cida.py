from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from cida.cli import load_checkpoint, save_checkpoint
from cida.data import PRIVATE, target_data, target_evaluation
from cida.guides import confident_indices, nearest_guides
from cida.models import AdaptationModel, SourceModel
from cida.office import office_classes, office_trial
from cida.prototypes import GaussianBank, fit_gaussians, negative_mask, separability_loss
from cida.training import target_loss, target_optimizers, update
from debug_digits import gaussian_report, prediction_report, projection_report


class CidaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_gaussian_scores_match_paper_cross_entropy(self):
        means = torch.tensor([[0., 0.], [1., 2.]], dtype=torch.double)
        scale = torch.tensor([[[1., 0.], [0.25, 1.]], [[0.7, 0.], [-0.3, 1.2]]], dtype=torch.double)
        bank = GaussianBank(means, scale)
        x = torch.tensor([[0., 1.], [1.5, 2.]], dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1])
        expected_log = torch.distributions.MultivariateNormal(means, scale_tril=scale).log_prob(x[:, None])
        actual = separability_loss(x, labels, bank)
        expected = F.cross_entropy(expected_log.exp(), labels)
        torch.testing.assert_close(bank.log_density(x), expected_log)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])

    def test_negative_sampling_uses_three_sigma_threshold(self):
        means = torch.tensor([[0.], [10.]], dtype=torch.double)
        bank = GaussianBank(means, torch.ones(2, 1, 1, dtype=torch.double))
        candidate = torch.tensor([[0.], [2.99], [3.01], [5.], [10.]])
        self.assertEqual(negative_mask(candidate, bank).tolist(), [False, False, True, True, False])

    def test_sparse_prototypes_and_per_class_confidence(self):
        guides = torch.tensor([[0., 0.], [10., 0.], [20., 0.]])
        x = torch.tensor([[2., 0.], [1., 0.], [0., 0.], [10.5, 0.], [10., 0.]])
        labels, distance = nearest_guides(x, guides)
        self.assertEqual(labels.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(confident_indices(labels, distance, 3, 0.5).tolist(), [2, 4])
        bank = fit_gaussians(x, labels, 3, 0.1, fallback_means=guides)
        torch.testing.assert_close(bank.means[2], guides[2].double())
        self.assertTrue(torch.isfinite(bank.log_density(x)).all())

    def test_each_target_loss_updates_only_its_modules(self):
        model = AdaptationModel(SourceModel(2), num_private=2)
        modules = dict(model.named_children())
        expected = {
            "Lr1": {"f_e", "f_d", "g_t"}, "Lr2": {"f_e", "f_d"},
            "Lc": {"f_t", "g_t"}, "La1": {"f_t"}, "La2": {"f_e", "f_t"},
        }
        source = fit_gaussians(torch.randn(20, 100), torch.arange(20) % 2, 2, 0.1)
        images, supports = torch.rand(4, 3, 32, 32), torch.rand(2, 3, 32, 32)
        labels = torch.arange(4)
        target = fit_gaussians(torch.randn(20, 100), torch.arange(20) % 4, 4, 0.1)
        stats = SimpleNamespace(prototypes=target)
        for name, optimizer in target_optimizers(model, 1e-4).items():
            with self.subTest(loss=name):
                before = {key: [p.detach().clone() for p in module.parameters()]
                          for key, module in modules.items()}
                loss = target_loss(name, model, stats, source, supports, images, labels, "log-density")
                update(model, optimizer, loss)
                changed = {key for key, module in modules.items() if any(
                    not torch.equal(old, new) for old, new in zip(before[key], module.parameters())
                )}
                self.assertEqual(changed, expected[name])
                self.assertTrue(all(p.grad is None for p in model.g_s.parameters()))
        self.assertEqual(model(images).shape, (4, 4))

    def test_one_shot_supports_are_disjoint_and_unlabeled(self):
        class Digits(TensorDataset):
            def __init__(self):
                self.targets = torch.arange(20) % 10
                super().__init__(torch.arange(20).float()[:, None], self.targets)

        with patch("cida.data.digits", side_effect=lambda *args, **kwargs: Digits()) as dataset:
            target, supports, support_ids = target_data("target-only", False, "transductive", 8)
            test = target_evaluation("target-only", False, "transductive", support_ids)
        self.assertTrue(all(call.args[0] == "usps" for call in dataset.call_args_list))
        self.assertEqual(tuple(i % 10 for i in support_ids), PRIVATE)
        self.assertEqual(supports.flatten().int().tolist(), support_ids)
        self.assertTrue(set(support_ids).isdisjoint(target.indices))
        self.assertTrue(set(support_ids).isdisjoint(test.indices))
        self.assertEqual([target[i][1] for i in range(len(target))], list(range(15)))

    def test_source_checkpoint_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            source = SourceModel(5)
            save_checkpoint(path, {"version": 1, "stage": "source", "model": source.state_dict()})
            restored = SourceModel(5)
            restored.load_state_dict(load_checkpoint(path, "source")["model"])
            adapted = AdaptationModel(restored, 5)
            self.assertEqual(adapted(torch.rand(2, 3, 32, 32)).shape, (2, 10))
            for original, copy in zip(source.f_s.parameters(), adapted.f_t.parameters()):
                torch.testing.assert_close(original, copy)
                self.assertNotEqual(original.data_ptr(), copy.data_ptr())

    def test_office_splits_and_support_exclusion(self):
        class OfficeFixture:
            classes = [f"class_{i:02}" for i in range(31)]
            class_to_idx = dict(zip(classes, range(31)))
            targets = [i for i in range(31) for _ in range(2)]

            def __getitem__(self, index):
                return torch.tensor([index]), self.targets[index]

        dataset = OfficeFixture()
        for shared_count, private_count in ((20, 11), (10, 10)):
            with self.subTest(shared_count=shared_count):
                shared, private = office_classes(dataset.classes, shared_count)
                self.assertEqual(len(shared), shared_count)
                self.assertEqual(private, tuple(dataset.classes[20:20 + private_count]))
                train, stats, supports, support_ids, test = office_trial(
                    dataset, "dslr", shared, private, seed=7
                )
                self.assertEqual(len(supports), private_count)
                self.assertEqual(train.indices, stats.indices)
                self.assertEqual(train.indices, test.indices)
                self.assertTrue(set(support_ids).isdisjoint(test.indices))
                self.assertEqual({test[i][1] for i in range(len(test))},
                                 set(range(shared_count + private_count)))

    def test_diagnostic_exposes_negligible_density_gradients(self):
        means = torch.stack((torch.zeros(100), torch.full((100,), 3.))).double()
        bank = GaussianBank(means, torch.eye(100, dtype=torch.double).repeat(2, 1, 1) * 10)
        result = gaussian_report(means.float(), torch.tensor([0, 1]), bank)
        self.assertAlmostEqual(result["gradient_probe"]["density"]["loss"], 0.6931471805599453)
        self.assertLess(result["gradient_probe"]["density"]["feature_gradient_norm"], 1e-100)
        self.assertGreater(result["gradient_probe"]["log-density"]["feature_gradient_norm"], 1e-5)

    def test_diagnostic_projection_error_has_relative_scale(self):
        model = AdaptationModel(SourceModel(5), 5)
        model.f_e = torch.nn.Identity()
        model.f_d = torch.nn.Identity()
        result = projection_report(model, torch.randn(8, 100), torch.arange(8) % 5)
        for key in ("encoder_relative_mse", "decoder_relative_mse", "cycle_relative_mse"):
            self.assertEqual(result[key], 0.)
        self.assertEqual(result["proxy_accuracy_before_projection"],
                         result["proxy_source_head_after_projection"])
        score = prediction_report(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]), 3)
        self.assertEqual(score["per_class"], [100., 50., None])
        self.assertEqual(score["macro_accuracy"], 75.)

    def test_projection_diagnostic_separates_random_private_head(self):
        model = AdaptationModel(SourceModel(5), 5)
        model.f_e = torch.nn.Identity()
        model.f_d = torch.nn.Identity()
        with torch.no_grad():
            for parameter in model.g_s.parameters():
                parameter.zero_()
            model.g_s[-1].bias[0] = 1.
            for parameter in model.g_t.parameters():
                parameter.zero_()
            model.g_t[-1].bias[0] = 100.
        report = projection_report(model, torch.randn(4, 100), torch.zeros(4, dtype=torch.long))
        self.assertEqual(report["proxy_accuracy_before_projection"], 100.)
        self.assertEqual(report["proxy_source_head_after_projection"], 100.)
        self.assertEqual(report["proxy_accuracy_after_projection"], 0.)


if __name__ == "__main__":
    unittest.main()
