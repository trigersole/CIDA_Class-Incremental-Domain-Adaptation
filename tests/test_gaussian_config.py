"""Objective selection must survive preset resolution and source reuse."""
import argparse
import unittest

from cida.cli import (add_training_args, source_gaussian_score, training_config,
                      validate_office_source)


class GaussianConfigTests(unittest.TestCase):
    def config(self, benchmark, *options):
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        return training_config(parser.parse_args(options), benchmark)

    def test_defaults_preserve_digits_and_select_office_density(self):
        self.assertEqual(self.config("digits", "--preset", "friend").gaussian_score, "log-density")
        self.assertEqual(self.config("digits").gaussian_score, "density")
        self.assertEqual(self.config("office31", "--preset", "friend").gaussian_score, "density")

    def test_explicit_objective_overrides_preset(self):
        for benchmark in ("digits", "office31"):
            for score in ("density", "log-density"):
                self.assertEqual(self.config(benchmark, "--preset", "friend", "--gaussian-score", score)
                                 .gaussian_score, score)

    def test_source_objective_is_preserved_and_mismatch_rejected(self):
        checkpoint = dict(stage="source", benchmark="office31", source_domain="amazon",
                          shared_classes=list(range(20)), imagenet_pretrained=True,
                          config=dict(preset="friend", gaussian_score="log-density"))
        self.assertEqual(source_gaussian_score(checkpoint), "log-density")
        with self.assertRaisesRegex(ValueError, "retrain source"):
            validate_office_source(checkpoint, "density")
        checkpoint["config"]["gaussian_score"] = "density"
        validate_office_source(checkpoint, "density")
        self.assertEqual(source_gaussian_score(checkpoint), "density")
        del checkpoint["config"]["gaussian_score"]
        self.assertEqual(source_gaussian_score(checkpoint), "log-density")


if __name__ == "__main__":
    unittest.main()
