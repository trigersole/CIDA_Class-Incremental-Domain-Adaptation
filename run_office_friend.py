"""Kaggle entry point for a fresh Office source and five adaptation trials."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from cida.cli import load_checkpoint
from cida.office import office_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/runs/office_mlp3_confident"))
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--target-epochs", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Enable a GPU in Kaggle settings.")
    data = args.data
    if data is None:
        roots = sorted({p.parent for p in Path("/kaggle/input").rglob("amazon")
                        if p.is_dir() and (p.parent / "dslr").is_dir()})
        if len(roots) != 1:
            parser.error(f"Expected one extracted Office-31 input; found {roots}. Pass --data explicitly.")
        data = roots[0]
    for domain in ("amazon", "dslr"):
        dataset = office_domain(data, domain)
        print(f"{domain}: {len(dataset)} images, {len(dataset.classes)} classes", flush=True)
    source = args.output / "source.pt"
    target = args.output / "confident"
    repo = Path(__file__).resolve().parent

    def run(script, *options):
        subprocess.run([sys.executable, str(repo / script), *map(str, options)], check=True)

    if source.exists():
        checkpoint = load_checkpoint(source, "source")
        config = checkpoint["config"]
        if not (checkpoint["benchmark"] == "office31" and checkpoint["source_domain"] == "amazon"
                and len(checkpoint["shared_classes"]) == 20 and config["preset"] == "friend"
                and config["epochs"] == args.source_epochs and config["batch_size"] == 8
                and config["lr"] == 1e-4 and config["seed"] == 0
                and checkpoint.get("imagenet_pretrained") is True):
            parser.error("Existing source configuration differs. Use a fresh --output directory.")
        print(f"Reusing matching Office friend source: {source}", flush=True)
    else:
        run("train_source.py", "--benchmark", "office31", "--preset", "friend",
            "--domain", "amazon", "--office-shared", 20, "--office-weights", "imagenet",
            "--data", data, "--device", "cuda", "--batch-size", 8, "--lr", 1e-4,
            "--epochs", args.source_epochs, "--seed", 0, "--output", source)
    run("adapt_target.py", "--source", source, "--target-domain", "dslr",
        "--data", data, "--device", "cuda", "--batch-size", 8, "--lr", 1e-4,
        "--epochs", args.target_epochs, "--projection-steps", 2000,
        "--confidence-fraction", .2, "--la1-samples", "confident", "--eval-every", 1,
        "--output", target, "--seeds", *args.seeds)
    results = json.loads((target / "metrics.json").read_text())
    for metric in ("ALL", "SHARED", "PRIV"):
        print(metric, results[metric])


if __name__ == "__main__":
    main()
