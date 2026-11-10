import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

from cida.cli import (
    add_training_args, configure_device, load_checkpoint, positive_int, save_checkpoint,
    save_json, seed_all, training_config,
)
from cida.data import PRIVATE, SHARED, target_data, target_evaluation
from cida.metrics import evaluate
from cida.models import AdaptationModel, SourceModel
from cida.office import OFFICE_DOMAINS, office_target, office_trial
from cida.prototypes import GaussianBank
from cida.training import loader, train_target


def main():
    parser = argparse.ArgumentParser(description="CIDA Stage 2: one-shot, source-free target adaptation.")
    add_training_args(parser)
    parser.add_argument("--source", type=Path, default=Path("runs/source.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/target"))
    parser.add_argument("--target-domain", choices=OFFICE_DOMAINS, help="Office-31 target domain.")
    parser.add_argument("--protocol", choices=("held-out", "transductive"),
                        help="Defaults to held-out for Digits and transductive for Office-31.")
    parser.add_argument("--projection-steps", type=positive_int, default=1000)
    parser.add_argument("--confidence-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run independent one-shot trials.")
    args = parser.parse_args()
    if not 0 < args.confidence_fraction <= 1:
        parser.error("--confidence-fraction must be in (0, 1]")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must be distinct")

    checkpoint = load_checkpoint(args.source, "source")
    benchmark = checkpoint.get("benchmark", "digits")
    office = benchmark == "office31"
    shared = tuple(checkpoint["shared_classes"])
    source_domain = checkpoint.get("source_domain")
    protocol = args.protocol or ("transductive" if office else "held-out")
    if office:
        if args.target_domain is None or args.target_domain == source_domain:
            parser.error("Choose --target-domain distinct from the Office-31 source domain")
        if protocol != "transductive" or args.download:
            parser.error("Office-31 uses extracted folders and transductive evaluation")
        dataset, private = office_target(args.data, args.target_domain, shared)
    else:
        if args.target_domain is not None or shared != SHARED:
            parser.error("Digits uses source labels 0-4 and no --target-domain")
        private = PRIVATE
    config = training_config(args, benchmark)
    device = configure_device(args)
    source = SourceModel(len(shared), benchmark)
    source.load_state_dict(checkpoint["model"])
    prototypes = GaussianBank(**{key: value.to(device) for key, value in checkpoint["prototypes"].items()})
    results = []
    for seed in seeds:
        trial = replace(config, seed=seed)
        seed_all(seed)
        if office:
            target, stats_target, supports, support_ids, test = office_trial(
                dataset, args.target_domain, shared, private, seed
            )
        else:
            target, supports, support_ids = target_data(args.data, args.download, protocol, seed)
            stats_target = None
            test = target_evaluation(args.data, args.download, protocol, support_ids)
        model = AdaptationModel(source, len(private)).to(device)
        history = train_target(
            model, target, supports.to(device), prototypes, trial,
            args.projection_steps, args.confidence_fraction, stats_target=stats_target,
        )
        metrics = evaluate(model, loader(test, trial), device, len(shared), len(shared) + len(private))
        result = {"seed": seed, "support_ids": support_ids, **metrics}
        results.append(result)
        save_checkpoint(args.output / f"seed_{seed}.pt", {
            "version": 1, "stage": "target", "shared_classes": shared, "private_classes": private,
            "benchmark": benchmark, "source_domain": source_domain, "target_domain": args.target_domain,
            "model": model.state_dict(), "protocol": protocol, "support_ids": support_ids,
            "config": {**asdict(trial), "projection_steps": args.projection_steps,
                       "confidence_fraction": args.confidence_fraction},
            "history": history, "metrics": metrics,
        })
        print(json.dumps(result), flush=True)
        del model
    summary = {"benchmark": benchmark, "source_domain": source_domain, "target_domain": args.target_domain,
               "shared_classes": shared, "private_classes": private, "protocol": protocol,
               "source_checkpoint": str(args.source), "trials": results}
    for key in ("ALL", "PRIV", "SHARED"):
        scores = [run[key] for run in results]
        summary[key] = {"mean": statistics.mean(scores),
                        "sample_std": statistics.stdev(scores) if len(scores) > 1 else None}
    save_json(args.output / "metrics.json", summary)
    print(json.dumps({key: summary[key] for key in ("ALL", "PRIV", "SHARED")}))


if __name__ == "__main__":
    main()
