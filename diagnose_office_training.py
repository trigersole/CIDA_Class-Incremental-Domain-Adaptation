"""Office source gradients, projection initialization and controlled target experiments."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from cida.cli import (configure_device, load_checkpoint, positive_int, save_json,
                      seed_all, source_gaussian_score)
from cida.diagnostics import EpochMetrics, pseudo_precision_summary
from cida.guides import build_guides, nearest_guides
from cida.models import AdaptationModel, SourceModel
from cida.office import office_source, office_target, office_trial
from cida.prototypes import GaussianBank, separability_loss
from cida.training import (TrainingConfig, diagnostic_state, loader, proxy_batch,
                           squared_distance, train_target)
from debug_digits import projection_report
from debug_office import score_paths, source_accuracy


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def gradient_summary(gradients):
    gradients = [g.detach() for g in gradients if g is not None]
    return {
        "norm": number(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
                if gradients else 0.0,
        "nonzero_elements": sum(int(torch.count_nonzero(g)) for g in gradients),
        "finite": all(bool(torch.isfinite(g).all()) for g in gradients),
    }


def source_gradient_probe(source, bank, dataset, config, device, batches=3):
    """Actual model dtype and parameters; never step an optimizer or retain BN changes."""
    buffers = {name: value.clone() for name, value in source.named_buffers()}
    result = []
    try:
        with diagnostic_state(source, device):
            for batch_index, (images, labels) in enumerate(loader(dataset, config, shuffle=True)):
                if batch_index >= batches:
                    break
                images, labels = images.to(device), labels.to(device)
                for mode in ("eval", "train"):
                    source.train(mode == "train")
                    features = source.f_s(images)
                    scores = bank.log_density(features.detach())
                    common = dict(batch=batch_index, bn_mode=mode, feature_dtype=str(features.dtype),
                                  log_density_min=number(scores.min()), log_density_max=number(scores.max()),
                                  density_max=number(scores.max().exp()),
                                  density_zero_fraction=number((scores.exp() == 0).double().mean()),
                                  gaussian_accuracy=number((scores.argmax(1) == labels).double().mean() * 100),
                                  mean_log_score_spread=number((scores.max(1).values - scores.min(1).values).mean()),
                                  uniform_loss=math.log(source.num_shared))
                    for objective in ("density", "log-density"):
                        entry = {**common, "objective": objective}
                        try:
                            loss = separability_loss(features, labels, bank, score=objective)
                            gradients = torch.autograd.grad(
                                loss, [features, *source.f_s.parameters()], retain_graph=True,
                                allow_unused=True)
                            entry.update(loss=number(loss.detach()),
                                         feature_gradient=gradient_summary(gradients[:1]),
                                         extractor_gradient=gradient_summary(gradients[1:]))
                        except (RuntimeError, FloatingPointError) as error:
                            entry["error"] = str(error)
                        result.append(entry)
                    del features
                    # Train-mode probes must all begin with the checkpoint's BN buffers.
                    with torch.no_grad():
                        for name, value in source.named_buffers():
                            value.copy_(buffers[name])
    finally:
        with torch.no_grad():
            for name, value in source.named_buffers():
                value.copy_(buffers[name])
    return result


def bn_buffers(module):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, layer in module.named_modules()
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in layer.named_buffers(recurse=False)}


def bn_affine(module):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, layer in module.named_modules()
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in layer.named_parameters(recurse=False)}


@torch.inference_mode()
def guide_precision(model, evaluation_loader, bank, supports, stats):
    """Clean/eval assignments and the actual confident subset from statistics refresh."""
    model.eval()
    guides = build_guides(model, bank, supports)
    predictions, truth = [], []
    for images, labels in evaluation_loader:
        assigned, _ = nearest_guides(model.f_t(images.to(supports.device)), guides)
        predictions.append(assigned.cpu())
        truth.append(labels.cpu())
    predictions, truth = torch.cat(predictions), torch.cat(truth)
    if len(truth) != int(stats.counts.sum()):
        raise ValueError("Pseudo-label diagnostics need evaluation images in target-ID order, with supports excluded.")
    selected = stats.selected_ids.cpu().long()
    classes, shared = len(guides), model.num_shared
    return dict(
        all_clean_eval=pseudo_precision_summary(predictions, truth, classes, shared),
        confident_statistics=pseudo_precision_summary(stats.selected_labels, truth[selected], classes, shared),
        confident_assignments_match=bool(torch.equal(predictions[selected], stats.selected_labels.cpu())),
        counts_match_statistics=bool(torch.equal(predictions.bincount(minlength=classes), stats.counts.cpu()))
    ), truth


def run_arm(source, bank, target, stats_target, supports, test, config, projection_steps,
            output, freeze_bn, proxy_samples, proxy_labels, on_result=None, ft_lr=None):
    # Both arms reset RNG before model construction and use the same support images.
    seed_all(config.seed)
    model = AdaptationModel(source, len(supports)).to(supports.device)
    arm = dict(freeze_bn_stats=freeze_bn, gaussian_score="log-density",
               ft_lr=config.lr if ft_lr is None else ft_lr, other_lr=config.lr,
               config=asdict(config), initialization={}, epochs=[])
    initial_buffers, initial_affine = {}, {}
    truth_by_id = None
    observed_la1 = {}

    def initialization(stage, current):
        projection = projection_report(current, proxy_samples, proxy_labels)
        projection.update(
            encoder_mean_squared_l2=float(squared_distance(current.f_e(proxy_samples), proxy_samples)),
            decoder_mean_squared_l2=float(squared_distance(current.f_d(proxy_samples), proxy_samples)))
        arm["initialization"][stage] = {
            "projection": projection,
            "target": score_paths(current, test, config, supports.device),
        }
        if stage == "after_projection_warmup":
            initial_buffers.update(bn_buffers(current.f_t))
            initial_affine.update(bn_affine(current.f_t))
        result = arm["initialization"][stage]
        print(json.dumps(dict(stage=stage, freeze_bn_stats=freeze_bn,
                              ft_lr=arm["ft_lr"],
                              projection={k: v for k, v in result["projection"].items() if "mse" in k or "squared_l2" in k},
                              target={k: v for k, v in result["target"].items() if isinstance(v, (int, float))})),
              flush=True)
        if on_result:
            on_result(arm)

    metrics = EpochMetrics(output, config.seed, loader(test, config), supports.device,
                           source.num_shared, source.num_shared + len(supports), config.epochs)

    def epoch_report(epoch, current, stats):
        nonlocal truth_by_id
        result = metrics(epoch, current, stats)
        precision, truth_by_id = guide_precision(current, loader(test, config), bank, supports, stats)
        precision["statistics_for_next_epoch"] = epoch + 1
        observed = observed_la1.get(epoch, [])
        precision["la1_training_batches"] = (pseudo_precision_summary(
            torch.cat([p for p, _ in observed]), torch.cat([t for _, t in observed]),
            source.num_shared + len(supports), source.num_shared) if observed else None)
        arm["epochs"].append(dict(epoch=epoch, metrics=result,
                                  paths=score_paths(current, test, config, supports.device),
                                  pseudo_label_precision=precision,
                                  pseudo_counts=stats.counts.tolist()))
        print(json.dumps(dict(stage="pseudo_label_precision", epoch=epoch, ft_lr=arm["ft_lr"],
                              clean_all=precision["all_clean_eval"]["micro_precision"],
                              confident=precision["confident_statistics"]["micro_precision"],
                              represented_classes=precision["all_clean_eval"]["represented_classes"],
                              la1_train=(precision["la1_training_batches"]["micro_precision"] if observed else None))),
              flush=True)
        if on_result:
            on_result(arm)
        return result

    def observe_la1(epoch, sample_ids, predictions):
        # Labels are used exclusively for reporting already-used pseudo-labels.
        # Count processed occurrences, including repeats, not unique images.
        if truth_by_id is None:
            raise RuntimeError("Epoch-zero diagnostics must establish target-ID alignment first.")
        observed_la1.setdefault(epoch, []).append((predictions, truth_by_id[sample_ids]))

    history = train_target(model, target, supports, bank, config, projection_steps,
                           stats_target=stats_target, fraction=.2, la1_samples="all",
                           freeze_bn_stats=freeze_bn, initialization_callback=initialization,
                           epoch_callback=epoch_report, ft_lr=ft_lr, la1_observer=observe_la1)
    updates = [row for row in history if row["stage"] == "target"]
    arm["all_losses_updated_each_epoch"] = all(
        not row["disabled_losses"] and all(row["updates"].get(name, 0) > 0
            for name in ("Lr1", "Lr2", "Lc", "La1", "La2")) for row in updates)
    final_buffers, final_affine = bn_buffers(model.f_t), bn_affine(model.f_t)
    arm["bn_buffers_changed_after_adabn"] = [key for key in initial_buffers
        if not torch.equal(initial_buffers[key], final_buffers[key])]
    arm["bn_affine_changed"] = [key for key in initial_affine
        if not torch.equal(initial_affine[key], final_affine[key])]
    arm["bn_affine_parameters_trainable"] = all(
        p.requires_grad for layer in model.f_t.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) for p in layer.parameters())
    arm["final"] = arm["epochs"][-1]["metrics"]
    arm["history"] = history
    save_json(output / "diagnostics.json", arm)
    if not arm["all_losses_updated_each_epoch"]:
        raise RuntimeError("An adaptation loss received no updates; inspect diagnostics.json.")
    if freeze_bn and arm["bn_buffers_changed_after_adabn"]:
        raise RuntimeError("Frozen BN buffers changed; inspect diagnostics.json.")
    return arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/office_residual3_no_la1/source.pt"),
                        help="Existing log-density Office friend source; never retrained or overwritten.")
    parser.add_argument("--density-source", type=Path,
                        help="Optional completed density source to probe as well.")
    parser.add_argument("--data", type=Path, default=Path("data/office31"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparison", choices=("bn", "ft-lr"), default="bn")
    parser.add_argument("--ft-lr", type=float, default=1e-5,
                        help="Candidate f_t learning rate in --comparison ft-lr; others stay at 1e-4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--projection-steps", type=positive_int, default=2000)
    parser.add_argument("--source-batches", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not math.isfinite(args.ft_lr) or args.ft_lr <= 0:
        parser.error("--ft-lr must be finite and positive")
    args.output = args.output or Path(
        f"runs/office_{'lr_checks' if args.comparison == 'ft-lr' else 'checks'}_seed{args.seed}")
    if not args.source.is_file():
        parser.error(f"Missing log-density source: {args.source}. Pass --source with its actual path.")
    report_path = args.output / "report.json"
    if report_path.exists():
        parser.error("This diagnostic output already exists. Choose a fresh --output directory.")
    checkpoint = load_checkpoint(args.source, "source")
    if (checkpoint.get("benchmark") != "office31"
            or checkpoint.get("source_domain") != "amazon"
            or checkpoint.get("config", {}).get("preset") != "friend"
            or source_gaussian_score(checkpoint) != "log-density"):
        parser.error("The A-to-D comparison requires an Amazon Office friend source trained with log-density.")
    device = configure_device(args)
    config = TrainingConfig(preset="friend", gaussian_score="log-density", epochs=args.epochs,
                            batch_size=args.batch_size, lr=1e-4, cov_ridge=.14,
                            workers=args.workers, seed=args.seed)
    source = SourceModel(len(checkpoint["shared_classes"]), "office31", preset="friend").to(device)
    source.load_state_dict(checkpoint["model"])
    bank = GaussianBank(**{key: value.to(device) for key, value in checkpoint["prototypes"].items()})
    clean_source = office_source(args.data, checkpoint["source_domain"], source.num_shared)[1]
    dataset, private = office_target(args.data, "dslr", checkpoint["shared_classes"])
    target, stats_target, supports, ids, test = office_trial(
        dataset, "dslr", checkpoint["shared_classes"], private, args.seed)
    supports = supports.to(device)
    report = dict(source=str(args.source.resolve()), seed=args.seed, support_ids=ids,
                  purpose="Diagnostic comparison; target labels only score predictions; no checkpoint selection.",
                  config=asdict(config), projection_steps=args.projection_steps,
                  comparison=args.comparison,
                  confidence_fraction=.2, source_probe_batches=args.source_batches,
                  source_probes={}, arms={})

    def save():
        save_json(report_path, report)

    seed_all(args.seed)
    report["source_training_accuracy"] = source_accuracy(source, clean_source, config, device)
    report["source_probes"]["log_density_source"] = source_gradient_probe(
        source, bank, clean_source, config, device, args.source_batches)
    density_path = args.density_source
    if density_path is None:
        candidate = Path("runs/office_mlp3_density_all/source.pt")
        density_path = candidate if candidate.is_file() else None
    if density_path is not None:
        density_checkpoint = load_checkpoint(density_path, "source")
        if (density_checkpoint.get("benchmark") != "office31"
                or density_checkpoint.get("config", {}).get("preset") != "friend"
                or density_checkpoint.get("source_domain") != checkpoint["source_domain"]
                or tuple(density_checkpoint["shared_classes"]) != tuple(checkpoint["shared_classes"])
                or source_gaussian_score(density_checkpoint) != "density"):
            parser.error("--density-source must be a matching Office friend density source.")
        other = SourceModel(source.num_shared, "office31", preset="friend").to(device)
        other.load_state_dict(density_checkpoint["model"])
        other_bank = GaussianBank(**{k: v.to(device) for k, v in density_checkpoint["prototypes"].items()})
        report["density_source"] = str(density_path.resolve())
        report["source_probes"]["density_source"] = source_gradient_probe(
            other, other_bank, clean_source, config, device, args.source_batches)
        del other, other_bank
    else:
        report["density_source_note"] = "No completed density checkpoint found; both objectives still probed on the log-density source."
    save()
    for name, records in report["source_probes"].items():
        for record in records:
            print(json.dumps(dict(stage="source_gradient_probe", checkpoint=name, **record)), flush=True)
    # Fixed held-out proxy features, separate from the warm-up draw sequence.
    with diagnostic_state(source, device):
        seed_all(args.seed + 10000)
        proxies, labels = proxy_batch(bank, 256, next(source.parameters()).dtype)
    arms = (("bn_training", False, None), ("bn_frozen", True, None)) if args.comparison == "bn" else (
        ("baseline", False, config.lr), ("slower_ft", False, args.ft_lr))
    for name, frozen, ft_lr in arms:
        def progress(arm):
            report["arms"][name] = arm
            save()
        report["arms"][name] = run_arm(
            source, bank, target, stats_target, supports, test, config, args.projection_steps,
            args.output / name, frozen, proxies, labels, progress, ft_lr=ft_lr)
        save()
    baseline, candidate = (report["arms"][item[0]] for item in arms)
    report["identical_epoch_zero"] = baseline["epochs"][0]["metrics"] == candidate["epochs"][0]["metrics"]
    report["final_difference_candidate_minus_baseline"] = {
        metric: candidate["final"][metric] - baseline["final"][metric]
        for metric in ("ALL", "SHARED", "PRIV")}
    if args.comparison == "bn":
        report["final_difference_frozen_minus_training"] = report["final_difference_candidate_minus_baseline"]
    save()
    if not report["identical_epoch_zero"]:
        raise RuntimeError("The arms did not start with matching epoch-zero predictions; inspect report.json.")
    print("Final comparison:", {name: {k: arm["final"][k] for k in ("ALL", "SHARED", "PRIV")}
                                for name, arm in report["arms"].items()}, flush=True)
    print("Saved diagnostic report:", report_path, flush=True)


if __name__ == "__main__":
    main()
