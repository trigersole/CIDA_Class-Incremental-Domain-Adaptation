"""Diagnose existing MNIST -> USPS checkpoints before another training run."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from cida.cli import (
    add_data_args, configure_device, load_checkpoint, positive_int, save_json, seed_all,
)
from cida.data import PRIVATE, SHARED, digits, source_dataset, target_data, target_evaluation
from cida.guides import build_guides, confident_indices, nearest_guides
from cida.models import AdaptationModel, SourceModel
from cida.prototypes import GaussianBank, fit_gaussians, negative_mask, sample_negatives, separability_loss
from cida.training import TrainingConfig, loader, optimizer_for, proxy_batch, squared_distance, update


def prediction_report(predictions, labels, classes=10):
    predictions, labels = predictions.cpu(), labels.cpu()
    counts = labels.bincount(minlength=classes)
    correct = labels[predictions == labels].bincount(minlength=classes)
    accuracy = [float(100 * right / total) if total else None
                for right, total in zip(correct, counts)]
    present = [value for value in accuracy if value is not None]
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        if prediction < classes:
            confusion[truth, prediction] += 1
    return {"macro_accuracy": sum(present) / len(present), "per_class": accuracy,
            "predicted_counts": predictions.bincount(minlength=classes).tolist(),
            "true_counts": counts.tolist(), "confusion_true_by_predicted": confusion.tolist()}


@torch.no_grad()
def collect(extractor, dataset, config, device):
    extractor.eval()
    features, labels = [], []
    for images, targets in loader(dataset, config):
        features.append(extractor(images.to(device)))
        labels.append(targets.to(device))
    return torch.cat(features), torch.cat(labels)


def gaussian_report(features, labels, bank):
    log_scores = bank.log_density(features)
    probe = features[:256].detach().double().requires_grad_(True)
    gradients = {}
    for score in ("density", "log-density"):
        loss = separability_loss(probe, labels[:len(probe)], bank, score)
        gradient = torch.autograd.grad(loss, probe)[0]
        gradients[score] = {"loss": float(loss.detach()),
                            "feature_gradient_norm": float(gradient.norm()),
                            "finite_gradient": bool(torch.isfinite(gradient).all())}
    covariance = bank.scale_tril @ bank.scale_tril.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    return {
        "gaussian_classifier": prediction_report(log_scores.argmax(1), labels, len(bank.means)),
        "nearest_mean_classifier": prediction_report(
            torch.cdist(features.double(), bank.means).argmin(1), labels, len(bank.means)),
        "feature_norm_mean": float(features.norm(dim=1).mean()),
        "log_density_range": [float(log_scores.min()), float(log_scores.max())],
        "density_zero_fraction": float((log_scores.exp() == 0).double().mean()),
        "mean_log_score_spread": float((log_scores.max(1).values - log_scores.min(1).values).mean()),
        "covariance_eigenvalue_range": [float(eigenvalues.min()), float(eigenvalues.max())],
        "gradient_probe": gradients,
    }


@torch.no_grad()
def projection_report(model, samples, labels):
    model.eval()
    energy = samples.square().sum().clamp_min(torch.finfo(samples.dtype).tiny)
    encoded = model.f_e(samples)
    decoded = model.f_d(samples)
    reconstructed = model.f_d(encoded)
    return {
        "encoder_relative_mse": float((encoded - samples).square().sum() / energy),
        "decoder_relative_mse": float((decoded - samples).square().sum() / energy),
        "cycle_relative_mse": float((reconstructed - samples).square().sum() / energy),
        "proxy_accuracy_before_projection": float(100 * (model.g_s(samples).argmax(1) == labels).float().mean()),
        "proxy_accuracy_after_projection": float(100 * (model.joint_logits(encoded).argmax(1) == labels).float().mean()),
        "proxy_source_head_after_projection": float(
            100 * (model.g_s(reconstructed).argmax(1) == labels).float().mean()),
        "proxy_shared_only_after_projection": float(
            100 * (model.g_s(reconstructed)[:, :model.num_shared].argmax(1) == labels).float().mean()),
        "joint_prediction_note": "The joint metric includes the private head, which is random before adaptation.",
    }


@torch.no_grad()
def target_report(model, source_bank, supports, features, labels, fraction=0.2, ridge=0.1):
    model.eval()
    guides = build_guides(model, source_bank, supports)
    assigned, distances = nearest_guides(features, guides)
    confident = confident_indices(assigned, distances, len(guides), fraction)
    result = {"guide_predictions": prediction_report(assigned, labels),
              "classifier_predictions": prediction_report(model.joint_logits(features).argmax(1), labels),
              "confident_count": len(confident),
              "confident_precision": (float(100 * (assigned[confident] == labels[confident]).float().mean())
                                      if len(confident) else None)}
    shared_mask = labels < model.num_shared
    shared_only, _ = nearest_guides(features[shared_mask], guides[:model.num_shared])
    result["shared_guides_without_private_competition"] = prediction_report(
        shared_only, labels[shared_mask], model.num_shared)
    result["confident_precision_by_assigned_class"] = [
        float(100 * (labels[confident[assigned[confident] == c]] == c).float().mean())
        if bool((assigned[confident] == c).any()) else None
        for c in range(len(guides))
    ]
    bank = fit_gaussians(features, assigned, len(guides), ridge, fallback_means=guides)
    result["assignment_counts"] = assigned.bincount(minlength=len(guides)).tolist()
    result["empty_classes"] = (assigned.bincount(minlength=len(guides)) == 0).nonzero().flatten().tolist()
    # Re-enable autograd for a feature-only probe. Target labels are NEVER
    # used in the loss; pseudo-labels are the supervision in both probes.
    with torch.enable_grad():
        result["pseudo_labeled_gaussians"] = gaussian_report(features, assigned, bank)
    return result


def emit(report, key, value, output):
    report[key] = value
    save_json(output, report)
    print(json.dumps({"stage": key, "result": value}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/digits_debug/report.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=("held-out", "transductive"))
    parser.add_argument("--projection-steps", type=positive_int, default=1000)
    parser.add_argument("--projection-max-steps", type=positive_int,
                        help="Extend ONLY projection training to compare convergence.")
    args = parser.parse_args()
    device = configure_device(args)
    source_checkpoint = load_checkpoint(args.source, "source")
    if source_checkpoint.get("benchmark", "digits") != "digits":
        parser.error("This diagnostic is for MNIST -> USPS checkpoints")
    target_checkpoint = load_checkpoint(args.target_checkpoint, "target") if args.target_checkpoint else None
    if target_checkpoint is not None:
        if target_checkpoint.get("benchmark", "digits") != "digits":
            parser.error("Target checkpoint must be a Digits model")
        if tuple(target_checkpoint["shared_classes"]) != SHARED or tuple(target_checkpoint["private_classes"]) != PRIVATE:
            parser.error("Unexpected target class map")
        if target_checkpoint["config"]["seed"] != args.seed:
            parser.error("--seed must match --target-checkpoint")
    if tuple(source_checkpoint["shared_classes"]) != SHARED:
        parser.error("Unexpected source class map")
    protocol = args.protocol or (target_checkpoint["protocol"] if target_checkpoint else "held-out")
    if target_checkpoint is not None and protocol != target_checkpoint["protocol"]:
        parser.error("--protocol must match --target-checkpoint")
    config = TrainingConfig(batch_size=args.batch_size or 64, workers=args.workers, seed=args.seed)
    source = SourceModel(5).to(device)
    source.load_state_dict(source_checkpoint["model"])
    source.eval()
    bank = GaussianBank(**{name: tensor.to(device) for name, tensor in source_checkpoint["prototypes"].items()})
    training_data = source_dataset(args.data, args.download)
    source_test = digits("mnist", args.data, False, args.download)
    indices = [i for i, label in enumerate(source_test.targets) if int(label) in SHARED]
    source_test = Subset(source_test, indices)
    source_features, source_labels = collect(source.f_s, training_data, config, device)
    test_features, test_labels = collect(source.f_s, source_test, config, device)
    report = {"source_checkpoint": str(args.source), "seed": args.seed, "protocol": protocol,
              "source_training_config": source_checkpoint["config"], "diagnostic_config": asdict(config),
              "target_labels_usage": "diagnostic scoring only; never optimizer inputs"}
    emit(report, "source", {
        "train_classifier": prediction_report(source.g_s(source_features).argmax(1), source_labels, 6),
        "test_classifier": prediction_report(source.g_s(test_features).argmax(1), test_labels, 6),
        "train_prototypes": gaussian_report(source_features, source_labels, bank),
        "test_prototypes": gaussian_report(test_features, test_labels, bank),
    }, args.output)
    seed_all(args.seed)
    target, supports, support_ids = target_data(args.data, args.download, protocol, args.seed)
    supports = supports.to(device)
    if target_checkpoint is not None and support_ids != target_checkpoint["support_ids"]:
        parser.error("Current target data/support IDs differ from the saved run")
    model = AdaptationModel(source, 5).to(device)
    target_labels = torch.as_tensor(target.dataset.targets, device=device)[target.indices]
    target_features, _ = collect(model.f_t, target, config, device)
    report["support_ids"] = support_ids
    with torch.no_grad():
        shared = target_labels < len(SHARED)
        baseline_logits = source.g_s(target_features[shared])
        emit(report, "unadapted_usps_shared_baseline", {
            "source_head_with_unknown": prediction_report(
                baseline_logits.argmax(1), target_labels[shared], len(SHARED) + 1),
            "source_head_shared_only": prediction_report(
                baseline_logits[:, :len(SHARED)].argmax(1), target_labels[shared], len(SHARED)),
            "nearest_source_mean": prediction_report(
                torch.cdist(target_features[shared].double(), bank.means).argmin(1),
                target_labels[shared], len(SHARED)),
        }, args.output)
    # fork_rng keeps probes from changing the projection experiment's RNG.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        fixed_proxy, proxy_labels = proxy_batch(bank, 512, supports.dtype)
        global_bank = fit_gaussians(source_features, torch.zeros_like(source_labels), 1,
                                    source_checkpoint["config"]["cov_ridge"])
        negatives = sample_negatives(global_bank, bank, 512).to(supports)
        scores = bank.log_density(negatives)
        peaks = bank.log_density(bank.means).diagonal()
        inside_any = (scores >= peaks - 4.5).any(1)
        emit(report, "negative_sampling", {
            "accepted_by_current_rule": int(negative_mask(negatives, bank).sum()),
            "inside_another_class_three_sigma_region": int(inside_any.sum()),
            "unknown_prediction_percent": float(100 * (source.g_s(negatives).argmax(1) == 5).float().mean()),
        }, args.output)
    emit(report, "before_projection_warmup", projection_report(model, fixed_proxy, proxy_labels), args.output)
    optimizers = [optimizer_for(1e-4, module) for module in (model.f_e, model.f_d)]
    maximum = args.projection_max_steps or args.projection_steps
    if maximum < args.projection_steps:
        parser.error("--projection-max-steps must be at least --projection-steps")
    for step in range(1, maximum + 1):
        model.train()
        proxies, _ = proxy_batch(bank, config.batch_size, supports.dtype)
        for module, optimizer in zip((model.f_e, model.f_d), optimizers):
            update(model, optimizer, squared_distance(module(proxies), proxies))
        if step % 500 == 0 or step == args.projection_steps or step == maximum:
            emit(report, f"projection_step_{step}", projection_report(model, fixed_proxy, proxy_labels), args.output)
        if step == args.projection_steps or step == maximum:
            emit(report, f"target_initialization_step_{step}",
                 target_report(model, bank, supports, target_features, target_labels,
                               ridge=source_checkpoint["config"]["cov_ridge"]), args.output)
    if target_checkpoint is not None:
        model.load_state_dict(target_checkpoint["model"])
        features, _ = collect(model.f_t, target, config, device)
        emit(report, "saved_target_adaptation_pool", target_report(
            model, bank, supports, features, target_labels,
            fraction=target_checkpoint["config"]["confidence_fraction"],
            ridge=target_checkpoint["config"]["cov_ridge"]), args.output)
        test = target_evaluation(args.data, args.download, protocol, support_ids)
        features, labels = collect(model.f_t, test, config, device)
        emit(report, "saved_target_evaluation", prediction_report(model.joint_logits(features).argmax(1), labels), args.output)
    print(f"Saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
