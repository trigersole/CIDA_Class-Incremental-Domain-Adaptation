import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics

from cida.cli import (
    add_training_args, configure_device, load_checkpoint, positive_int, save_checkpoint,
    save_json, seed_all, training_config,
)
from cida.data import PRIVATE, SHARED, target_data, target_evaluation
from cida.diagnostics import EpochMetrics
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
    parser.add_argument("--disable-guide-pulling", action="store_true",
                        help="Ablation: skip La2 while retaining all other update slots.")
    parser.add_argument("--disable-separability", action="store_true",
                        help="Diagnostic: skip La1 while preserving other update slots.")
    parser.add_argument("--la1-samples", choices=("all", "confident"), default="all",
                        help="Draw La1 images from all targets or confident B_t; labels are recomputed.")
    parser.add_argument("--la1-temperature", type=float, default=1.0,
                        help="Divide La1 log-density logits by this positive temperature.")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Diagnostic evaluation interval; 0 disables it. Final reporting always uses last epoch.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run independent one-shot trials.")
    args = parser.parse_args()
    if not 0 < args.confidence_fraction <= 1:
        parser.error("--confidence-fraction must be in (0, 1]")
    if not math.isfinite(args.la1_temperature) or args.la1_temperature <= 0:
        parser.error("--la1-temperature must be finite and positive")
    if args.eval_every < 0:
        parser.error("--eval-every must be nonnegative")
    if args.disable_separability and (args.la1_samples != "all" or args.la1_temperature != 1):
        parser.error("La1 sampling/temperature options do not apply when --disable-separability is set")
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
    saved_preset = checkpoint.get("config", {}).get("preset", "baseline")
    if args.preset is not None and args.preset != saved_preset:
        parser.error("--preset must match the source checkpoint; retrain source for a different preset")
    args.preset = saved_preset
    config = training_config(args, benchmark)
    if args.la1_temperature != 1 and config.gaussian_score != "log-density":
        parser.error("--la1-temperature requires --gaussian-score log-density or the friend preset")
    device = configure_device(args)
    source = SourceModel(len(shared), benchmark, preset=config.preset)
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
            target, supports, support_ids = target_data(args.data, args.download, protocol, seed, config.preset)
            stats_target = None
            test = target_evaluation(args.data, args.download, protocol, support_ids, config.preset)
        model = AdaptationModel(source, len(private)).to(device)
        epoch_metrics = EpochMetrics(args.output, seed, loader(test, trial), device,
                                     len(shared), len(shared) + len(private), trial.epochs, args.eval_every)
        history = train_target(
            model, target, supports.to(device), prototypes, trial,
            args.projection_steps, args.confidence_fraction, stats_target=stats_target,
            disable_guide_pulling=args.disable_guide_pulling, disable_separability=args.disable_separability,
            la1_samples=args.la1_samples, la1_temperature=args.la1_temperature,
            epoch_callback=epoch_metrics,
        )
        metrics = evaluate(model, loader(test, trial), device, len(shared), len(shared) + len(private))
        result = {"seed": seed, "support_ids": support_ids, **metrics}
        results.append(result)
        save_checkpoint(args.output / f"seed_{seed}.pt", {
            "version": 1, "stage": "target", "shared_classes": shared, "private_classes": private,
            "benchmark": benchmark, "source_domain": source_domain, "target_domain": args.target_domain,
            "model": model.state_dict(), "protocol": protocol, "support_ids": support_ids,
            "config": {**asdict(trial), "projection_steps": args.projection_steps,
                       "confidence_fraction": args.confidence_fraction,
                       "disable_guide_pulling": args.disable_guide_pulling,
                       "disable_separability": args.disable_separability,
                       "la1_samples": args.la1_samples, "la1_temperature": args.la1_temperature,
                       "eval_every": args.eval_every},
            "history": history, "metrics": metrics,
        })
        print(json.dumps(result), flush=True)
        del model
    summary = {"benchmark": benchmark, "source_domain": source_domain, "target_domain": args.target_domain,
               "disabled_losses": (["La2"] if args.disable_guide_pulling else []) +
                                  (["La1"] if args.disable_separability else []),
               "shared_classes": shared, "private_classes": private, "protocol": protocol,
               "source_checkpoint": str(args.source), "trials": results}
    summary["adaptation_settings"] = {
        "preset": config.preset, "epochs": config.epochs, "batch_size": config.batch_size,
        "la1_samples": args.la1_samples, "la1_temperature": args.la1_temperature,
        "eval_every": args.eval_every, "reported_epoch": config.epochs,
    }
    for key in ("ALL", "PRIV", "SHARED"):
        scores = [run[key] for run in results]
        summary[key] = {"mean": statistics.mean(scores),
                        "sample_std": statistics.stdev(scores) if len(scores) > 1 else None}
    save_json(args.output / "metrics.json", summary)
    print(json.dumps({key: summary[key] for key in ("ALL", "PRIV", "SHARED")}))


if __name__ == "__main__":
    main()
