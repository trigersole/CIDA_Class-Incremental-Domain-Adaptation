import argparse
from dataclasses import asdict
from pathlib import Path

from cida.cli import add_training_args, configure_device, save_checkpoint, seed_all, training_config
from cida.data import SHARED, source_dataset
from cida.models import SourceModel
from cida.training import train_source


def main():
    parser = argparse.ArgumentParser(description="CIDA Stage 1: foresighted MNIST source training.")
    add_training_args(parser)
    parser.add_argument("--output", type=Path, default=Path("runs/source.pt"))
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()
    if args.sigma <= 0:
        parser.error("--sigma must be positive")

    config = training_config(args)
    device = configure_device(args)
    seed_all(config.seed)
    dataset = source_dataset(args.data, args.download)
    model = SourceModel(len(SHARED)).to(device)
    prototypes, history = train_source(model, dataset, config, device, args.sigma)
    save_checkpoint(args.output, {
        "version": 1, "stage": "source", "shared_classes": SHARED,
        "model": model.state_dict(), "prototypes": prototypes.state_dict(),
        "config": {**asdict(config), "sigma": args.sigma}, "history": history,
    })
    print(f"Saved source checkpoint to {args.output}")


if __name__ == "__main__":
    main()
