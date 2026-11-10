import argparse
from dataclasses import asdict
from pathlib import Path

from cida.cli import add_training_args, configure_device, save_checkpoint, seed_all, training_config
from cida.data import SHARED, source_dataset
from cida.models import SourceModel
from cida.office import OFFICE_DOMAINS, office_source
from cida.training import train_source


def main():
    parser = argparse.ArgumentParser(description="CIDA Stage 1: foresighted source training.")
    add_training_args(parser)
    parser.add_argument("--benchmark", choices=("digits", "office31"), default="digits")
    parser.add_argument("--domain", choices=OFFICE_DOMAINS, help="Office-31 source domain.")
    parser.add_argument("--office-shared", type=int, choices=(10, 20), default=20)
    parser.add_argument("--office-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--output", type=Path, default=Path("runs/source.pt"))
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()
    if args.sigma <= 0:
        parser.error("--sigma must be positive")
    office = args.benchmark == "office31"
    if office and args.domain is None:
        parser.error("--domain is required for Office-31")
    if not office and args.domain is not None:
        parser.error("--domain applies only to Office-31")
    if office and args.download:
        parser.error("Extract Office-31 under --data. --download is for Digits.")

    config = training_config(args, args.benchmark)
    device = configure_device(args)
    seed_all(config.seed)
    if office:
        dataset, stats_dataset, shared = office_source(args.data, args.domain, args.office_shared)
    else:
        dataset, stats_dataset, shared = source_dataset(args.data, args.download), None, SHARED
    pretrained = office and args.office_weights == "imagenet"
    model = SourceModel(len(shared), args.benchmark, pretrained=pretrained).to(device)
    prototypes, history = train_source(model, dataset, config, device, args.sigma, stats_dataset=stats_dataset)
    save_checkpoint(args.output, {
        "version": 1, "stage": "source", "shared_classes": shared,
        "benchmark": args.benchmark, "source_domain": args.domain, "imagenet_pretrained": pretrained,
        "model": model.state_dict(), "prototypes": prototypes.state_dict(),
        "config": {**asdict(config), "sigma": args.sigma}, "history": history,
    })
    print(f"Saved source checkpoint to {args.output}")


if __name__ == "__main__":
    main()
