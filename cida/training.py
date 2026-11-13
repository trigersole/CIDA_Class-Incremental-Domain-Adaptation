from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
import json
import math
import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import ConfidentTarget
from .guides import build_guides, nearest_guides, refresh_statistics
from .prototypes import fit_gaussians, sample_negatives, separability_loss


@dataclass(frozen=True)
class TrainingConfig:
    preset: str = "baseline"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    cov_ridge: float = 0.1
    gaussian_score: str = "density"
    workers: int = 0
    max_steps: int | None = None
    seed: int = 0


def loader(dataset, config, *, shuffle=False, sampler=None, batch_size=None):
    return DataLoader(
        dataset, batch_size=batch_size or config.batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=config.workers,
        generator=torch.Generator().manual_seed(config.seed),
    )


def batches(data_loader, limit):
    return islice(data_loader, limit) if limit is not None else iter(data_loader)


def optimizer_for(lr, *modules):
    return torch.optim.Adam([p for module in modules for p in module.parameters()], lr=lr)


def update(model, optimizer, loss):
    model.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise FloatingPointError("A training loss is non-finite.")
    # Backpropagate only to this optimizer's parameters, while allowing the
    # gradient to pass through intermediate frozen or inactive modules.
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    torch.autograd.backward(loss, inputs=parameters)
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def log_epoch(stage, epoch, totals, steps, **details):
    record = {"stage": stage, "epoch": epoch,
              "loss": {key: totals[key] / steps[key] for key in totals}, **details}
    print(json.dumps(record), flush=True)
    return record


@torch.no_grad()
def source_statistics(model, data_loader, device, ridge):
    model.eval()
    features, labels = [], []
    for images, targets in data_loader:
        features.append(model.f_s(images.to(device)))
        labels.append(targets.to(device))
    features, labels = torch.cat(features), torch.cat(labels)
    prototypes = fit_gaussians(features, labels, model.num_shared, ridge, shrink=0.1 if model.preset == "friend" else None)
    global_prototype = fit_gaussians(features, torch.zeros_like(labels), 1, ridge, shrink=0.1 if model.preset == "friend" else None)
    return prototypes, global_prototype


def train_source(model, dataset, config, device, sigma=3.0, *, stats_dataset=None):
    train_loader = loader(dataset, config, shuffle=True)
    stats_loader = loader(dataset if stats_dataset is None else stats_dataset, config)
    warmup = optimizer_for(config.lr, model.f_s, model.g_s)
    model.train()
    total, count = 0.0, 0
    for images, labels in batches(train_loader, config.max_steps):
        loss = F.cross_entropy(model(images.to(device)), labels.to(device))
        total += update(model, warmup, loss)
        count += 1
    history = [log_epoch("source_warmup", 1, {"CE": total}, {"CE": count})]

    optimizers = {
        "Ls1": optimizer_for(config.lr, model.f_s),
        "Ls2": optimizer_for(config.lr, model.f_s, model.g_s),
    }
    prototypes, global_prototype = source_statistics(model, stats_loader, device, config.cov_ridge)
    order = tuple(optimizers)
    step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, steps = defaultdict(float), defaultdict(int)
        for images, labels in batches(train_loader, config.max_steps):
            name = order[step % len(order)]
            step += 1
            features = model.f_s(images.to(device))
            labels = labels.to(device)
            if name == "Ls1":
                loss = separability_loss(features, labels, prototypes, config.gaussian_score)
            else:
                negatives = sample_negatives(global_prototype, prototypes, len(labels), sigma).to(features)
                unknown = torch.full_like(labels, model.num_shared)
                loss = F.cross_entropy(model.g_s(features), labels)
                loss = loss + F.cross_entropy(model.g_s(negatives), unknown)
            totals[name] += update(model, optimizers[name], loss)
            steps[name] += 1
        history.append(log_epoch("source", epoch, totals, steps))
        prototypes, global_prototype = source_statistics(model, stats_loader, device, config.cov_ridge)
    return prototypes, history


def proxy_batch(prototypes, count, dtype):
    num_classes = len(prototypes.means)
    order = torch.randperm(num_classes, device=prototypes.means.device)
    labels = order[torch.arange(count, device=order.device) % num_classes]
    return prototypes.sample(labels).to(dtype=dtype), labels


def squared_distance(left, right):
    return (left - right).square().sum(dim=1).mean()


def warmup_projections(model, prototypes, config, steps):
    projectors = (model.f_e, model.f_d)
    optimizers = [optimizer_for(config.lr, module) for module in projectors]
    model.train()
    dtype = next(model.parameters()).dtype
    totals = defaultdict(float)
    for _ in range(steps):
        proxies, _ = proxy_batch(prototypes, config.batch_size, dtype)
        for name, module, optimizer in zip(("f_e", "f_d"), projectors, optimizers):
            totals[name] += update(model, optimizer, squared_distance(module(proxies), proxies))
    return log_epoch("projection_warmup", 1, totals, {name: steps for name in totals})


def target_optimizers(model, lr):
    return {
        "Lr1": optimizer_for(lr, model.f_e, model.f_d, model.g_t),
        "Lr2": optimizer_for(lr, model.f_e, model.f_d),
        "Lc": optimizer_for(lr, model.f_t, model.g_t),
        "La1": optimizer_for(lr, model.f_t),
        "La2": optimizer_for(lr, model.f_e, model.f_t),
    }


def confidence_batches(target, stats, config, steps, epoch):
    if len(stats.selected_ids) == 0:
        return iter(())
    dataset = ConfidentTarget(target, stats.selected_ids, stats.selected_labels)
    if config.preset == "friend":
        def repeat():
            while True:
                yield from loader(dataset, config, shuffle=True)
        return repeat()
    class_count = stats.selected_labels.bincount()
    weights = class_count[stats.selected_labels].double().reciprocal()
    sampler = WeightedRandomSampler(
        weights, num_samples=steps * config.batch_size, replacement=True,
        generator=torch.Generator().manual_seed(config.seed + epoch),
    )
    return iter(loader(dataset, config, sampler=sampler))


def target_loss(name, model, stats, source_prototypes, supports, images,
                pseudo_labels=None, gaussian_score="density", la1_temperature=1.0):
    if name in {"Lr1", "Lr2"}:
        proxies, labels = proxy_batch(source_prototypes, len(images), supports.dtype)
        projected = model.f_e(proxies)
        if name == "Lr1":
            return F.cross_entropy(model.joint_logits(projected), labels)
        return squared_distance(model.f_d(projected), proxies)

    if model.preset == "friend" and name in {"La1", "La2"}:
        combined = model.f_t(torch.cat((images, supports)))
        features, private = combined[:len(images)], combined[len(images):]
        guides = torch.cat((model.f_e(source_prototypes.means.to(supports)), private))
    else:
        features = model.f_t(images)
        guides = None
    if name == "Lc":
        return F.cross_entropy(model.joint_logits(features), pseudo_labels)
    if name == "La1":
        with torch.no_grad():
            if guides is None:
                guides = build_guides(model, source_prototypes, supports)
            pseudo_labels, _ = nearest_guides(features, guides)
        return separability_loss(features, pseudo_labels, stats.prototypes, gaussian_score,
                                 temperature=la1_temperature)
    if name == "La2":
        if guides is None:
            guides = build_guides(model, source_prototypes, supports)
        return squared_distance(features, guides[pseudo_labels])
    raise ValueError(f"Unknown target loss {name}.")


def train_target(model, target, supports, source_prototypes, config, projection_steps,
                 fraction=0.2, *, stats_target=None, disable_guide_pulling=False, disable_separability=False,
                 la1_samples="all", la1_temperature=1.0, epoch_callback=None):
    if la1_samples not in {"all", "confident"}:
        raise ValueError("la1_samples must be all or confident")
    if not math.isfinite(la1_temperature) or la1_temperature <= 0:
        raise ValueError("la1_temperature must be finite and positive")
    if la1_temperature != 1 and config.gaussian_score != "log-density":
        raise ValueError("La1 temperature scaling requires log-density scores")
    if config.preset == "friend":
        recalibrate_batchnorm(model.f_t, loader(target if stats_target is None else stats_target, config), supports.device)
    history = [warmup_projections(model, source_prototypes, config, projection_steps)]
    target_loader = loader(target, config, shuffle=True)
    stats_loader = loader(target if stats_target is None else stats_target, config)
    optimizers = target_optimizers(model, config.lr)
    order = tuple(optimizers)
    step = 0
    stats = refresh_statistics(model, stats_loader, source_prototypes, supports, config.cov_ridge, fraction)

    def report(epoch):
        if epoch_callback is None:
            return None
        # Diagnostic evaluation must not change training randomness, BatchNorm
        # statistics, or train/eval modes; target labels never enter a loss.
        modes = {module: module.training for module in model.modules()}
        python_rng = random.getstate()
        devices = [supports.device.index or 0] if supports.is_cuda else []
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                model.eval()
                return epoch_callback(epoch, model, stats)
        finally:
            random.setstate(python_rng)
            for module, mode in modes.items():
                module.training = mode

    initial_metrics = report(0)
    if initial_metrics is not None:
        history.append({"stage": "target_initialization", "epoch": 0, "metrics": initial_metrics})
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, steps = defaultdict(float), defaultdict(int)
        epoch_updates = len(target_loader) * (5 if config.preset == "friend" else 1)
        step_count = min(config.max_steps or epoch_updates, epoch_updates)
        confident = confidence_batches(target, stats, config, step_count, epoch)
        # Separate iterator keeps Lc/La2's draw sequence independent of whether
        # La1 uses all targets or B_t. La1 recomputes nearest-guide labels.
        la1_confident = (confidence_batches(target, stats, config, step_count, epoch + 100000)
                         if la1_samples == "confident" else None)
        def epoch_batches():
            remaining = step_count
            while remaining:
                for batch in target_loader:
                    yield batch
                    remaining -= 1
                    if remaining == 0:
                        return
        for images, _ in epoch_batches():
            name = order[step % len(order)]
            step += 1
            labels = None
            if name in {"Lc", "La2"} or (name == "La1" and la1_samples == "confident"):
                if len(stats.selected_ids) == 0:
                    continue
                images, labels = next(la1_confident if name == "La1" else confident)
                labels = labels.to(supports.device)
            # Retain the five-slot schedule and consume the same confident
            # batch so the other losses keep their baseline update budgets.
            if (name == "La2" and disable_guide_pulling) or (name == "La1" and disable_separability):
                continue
            loss = target_loss(
                name, model, stats, source_prototypes, supports, images.to(supports.device),
                labels, config.gaussian_score, la1_temperature=la1_temperature,
            )
            totals[name] += update(model, optimizers[name], loss)
            steps[name] += 1
        previous_counts = stats.counts.tolist()
        previous_confident_count = len(stats.selected_ids)
        stats = refresh_statistics(model, stats_loader, source_prototypes, supports, config.cov_ridge, fraction)
        metrics = report(epoch)
        history.append(log_epoch(
            "target", epoch, totals, steps, pseudo_label_counts=previous_counts,
            confident_count=previous_confident_count, updates=dict(steps),
            next_pseudo_label_counts=stats.counts.tolist(),
            la1_samples=la1_samples, la1_temperature=la1_temperature,
            **({"metrics": metrics} if metrics is not None else {}),
            disabled_losses=(["La2"] if disable_guide_pulling else []) + (["La1"] if disable_separability else []),
        ))
    return history


@torch.no_grad()
def recalibrate_batchnorm(module, data_loader, device):
    layers = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in layers]
    was_training = module.training
    try:
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
        module.train()
        for images, _ in data_loader:
            module(images.to(device))
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
        module.train(was_training)
