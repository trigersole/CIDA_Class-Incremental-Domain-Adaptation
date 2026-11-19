"""Combine five separate Slurm trial results using sample standard deviation."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    trials = []
    reference = None
    for seed in range(5):
        path = root / "target" / f"seed_{seed}" / "metrics.json"
        result = json.loads(path.read_text())
        signature = {key: result[key] for key in (
            "benchmark", "source_domain", "target_domain", "shared_classes",
            "private_classes", "protocol", "disabled_losses", "adaptation_settings",
            "source_checkpoint",
        )}
        if reference is None:
            reference = signature
        if signature != reference:
            raise ValueError(f"Experiment settings differ: {path}")
        if len(result["trials"]) != 1 or result["trials"][0]["seed"] != seed:
            raise ValueError(f"Expected one trial for seed {seed}: {path}")
        trials.extend(result["trials"])
    summary = {**reference, "trials": trials}
    for metric in ("ALL", "SHARED", "PRIV"):
        scores = [trial[metric] for trial in trials]
        summary[metric] = {"mean": statistics.mean(scores),
                           "sample_std": statistics.stdev(scores)}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output)
    destination = args.output / "metrics.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for metric in ("ALL", "SHARED", "PRIV"):
        print(metric, summary[metric])
    print("Saved:", destination)
