from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import ConfidentTarget
from .guides import build_guides, nearest_guides, refresh_statistics
from .prototypes import fit_gaussians, sample_negatives, separability_loss


@dataclass(frozen=True)
class TrainingConfig:
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
    prototypes = fit_gaussians(features, labels, model.num_shared, ridge)
    global_prototype = fit_gaussians(features, torch.zeros_like(labels), 1, ridge)
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
    class_count = stats.selected_labels.bincount()
    weights = class_count[stats.selected_labels].double().reciprocal()
    sampler = WeightedRandomSampler(
        weights, num_samples=steps * config.batch_size, replacement=True,
        generator=torch.Generator().manual_seed(config.seed + epoch),
    )
    return iter(loader(dataset, config, sampler=sampler))


def target_loss(name, model, stats, source_prototypes, supports, images,
                pseudo_labels=None, gaussian_score="density"):
    if name in {"Lr1", "Lr2"}:
        proxies, labels = proxy_batch(source_prototypes, len(images), supports.dtype)
        projected = model.f_e(proxies)
        if name == "Lr1":
            return F.cross_entropy(model.joint_logits(projected), labels)
        return squared_distance(model.f_d(projected), proxies)

    features = model.f_t(images)
    if name == "Lc":
        return F.cross_entropy(model.joint_logits(features), pseudo_labels)
    if name == "La1":
        with torch.no_grad():
            guides = build_guides(model, source_prototypes, supports)
            pseudo_labels, _ = nearest_guides(features, guides)
        return separability_loss(features, pseudo_labels, stats.prototypes, gaussian_score)
    if name == "La2":
        guides = build_guides(model, source_prototypes, supports)
        return squared_distance(features, guides[pseudo_labels])
    raise ValueError(f"Unknown target loss {name}.")


def train_target(model, target, supports, source_prototypes, config, projection_steps,
                 fraction=0.2, *, stats_target=None):
    history = [warmup_projections(model, source_prototypes, config, projection_steps)]
    target_loader = loader(target, config, shuffle=True)
    stats_loader = loader(target if stats_target is None else stats_target, config)
    optimizers = target_optimizers(model, config.lr)
    order = tuple(optimizers)
    step = 0
    stats = refresh_statistics(model, stats_loader, source_prototypes, supports, config.cov_ridge, fraction)
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, steps = defaultdict(float), defaultdict(int)
        step_count = min(config.max_steps or len(target_loader), len(target_loader))
        confident = confidence_batches(target, stats, config, step_count, epoch)
        for images, _ in batches(target_loader, config.max_steps):
            name = order[step % len(order)]
            step += 1
            labels = None
            if name in {"Lc", "La2"}:
                if len(stats.selected_ids) == 0:
                    continue
                images, labels = next(confident)
                labels = labels.to(supports.device)
            loss = target_loss(
                name, model, stats, source_prototypes, supports, images.to(supports.device),
                labels, config.gaussian_score,
            )
            totals[name] += update(model, optimizers[name], loss)
            steps[name] += 1
        history.append(log_epoch(
            "target", epoch, totals, steps, pseudo_label_counts=stats.counts.tolist(),
            confident_count=len(stats.selected_ids), updates=dict(steps),
        ))
        stats = refresh_statistics(model, stats_loader, source_prototypes, supports, config.cov_ridge, fraction)
    return history
