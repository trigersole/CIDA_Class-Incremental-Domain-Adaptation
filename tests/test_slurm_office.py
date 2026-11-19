"""Check that separate Slurm trials are combined without losing seed identity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm/summarize_office.py"
SPEC = importlib.util.spec_from_file_location("summarize_office", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SlurmOfficeTests(unittest.TestCase):
    def test_summary_requires_five_matching_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(benchmark="office31", source_domain="amazon",
                            target_domain="dslr", shared_classes=["a"],
                            private_classes=["b"], protocol="transductive",
                            disabled_losses=["La1"], adaptation_settings={"epochs": 60},
                            source_checkpoint="/shared/source.pt")
            for seed in range(5):
                path = root / "target" / f"seed_{seed}" / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({**settings, "trials": [dict(
                    seed=seed, ALL=seed, SHARED=seed + 10, PRIV=seed + 20)]}))
            summary = MODULE.summarize(root)
            self.assertEqual(summary["ALL"]["mean"], 2)
            self.assertAlmostEqual(summary["ALL"]["sample_std"], 2.5 ** 0.5)
            self.assertEqual(len(summary["trials"]), 5)
            result = json.loads(path.read_text())
            result["disabled_losses"] = []
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "settings differ"):
                MODULE.summarize(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                MODULE.summarize(root)


if __name__ == "__main__":
    unittest.main()
