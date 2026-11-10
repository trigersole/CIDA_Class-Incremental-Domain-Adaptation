import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

from cida.cli import (
    add_training_args, configure_device, load_checkpoint, positive_int, save_checkpoint,
    save_json, seed_all, training_config,
)
from cida.data import CLASSES, PRIVATE, SHARED, target_data, target_evaluation
from cida.metrics import evaluate
from cida.models import AdaptationModel, SourceModel
from cida.prototypes import GaussianBank
from cida.training import loader, train_target


def main():
    parser = argparse.ArgumentParser(description="CIDA Stage 2: one-shot, source-free USPS adaptation.")
    add_training_args(parser)
    parser.add_argument("--source", type=Path, default=Path("runs/source.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/target"))
    parser.add_argument("--protocol", choices=("held-out", "transductive"), default="held-out")
    parser.add_argument("--projection-steps", type=positive_int, default=1000)
    parser.add_argument("--confidence-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run independent one-shot trials.")
    args = parser.parse_args()
    if not 0 < args.confidence_fraction <= 1:
        parser.error("--confidence-fraction must be in (0, 1]")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must be distinct")

    config = training_config(args)
    device = configure_device(args)
    checkpoint = load_checkpoint(args.source, "source")
    if tuple(checkpoint["shared_classes"]) != SHARED:
        raise ValueError("Source checkpoint must use the Digits shared classes 0–4.")
    source = SourceModel(len(SHARED))
    source.load_state_dict(checkpoint["model"])
    prototypes = GaussianBank(**{key: value.to(device) for key, value in checkpoint["prototypes"].items()})
    results = []
    for seed in seeds:
        trial = replace(config, seed=seed)
        seed_all(seed)
        target, supports, support_ids = target_data(args.data, args.download, args.protocol, seed)
        model = AdaptationModel(source, len(PRIVATE)).to(device)
        history = train_target(
            model, target, supports.to(device), prototypes, trial,
            args.projection_steps, args.confidence_fraction,
        )
        test = target_evaluation(args.data, args.download, args.protocol, support_ids)
        metrics = evaluate(model, loader(test, trial), device, len(SHARED), len(CLASSES))
        result = {"seed": seed, "support_ids": support_ids, **metrics}
        results.append(result)
        save_checkpoint(args.output / f"seed_{seed}.pt", {
            "version": 1, "stage": "target", "shared_classes": SHARED, "private_classes": PRIVATE,
            "model": model.state_dict(), "protocol": args.protocol, "support_ids": support_ids,
            "config": {**asdict(trial), "projection_steps": args.projection_steps,
                       "confidence_fraction": args.confidence_fraction},
            "history": history, "metrics": metrics,
        })
        print(json.dumps(result), flush=True)
    summary = {"protocol": args.protocol, "source_checkpoint": str(args.source), "trials": results}
    for key in ("ALL", "PRIV", "SHARED"):
        scores = [run[key] for run in results]
        summary[key] = {"mean": statistics.mean(scores),
                        "sample_std": statistics.stdev(scores) if len(scores) > 1 else None}
    save_json(args.output / "metrics.json", summary)
    print(json.dumps({key: summary[key] for key in ("ALL", "PRIV", "SHARED")}))


if __name__ == "__main__":
    main()
