from dataclasses import dataclass

import torch

from .prototypes import GaussianBank, fit_gaussians


def build_guides(model, source_prototypes, supports):
    shared = model.f_e(source_prototypes.means.to(supports))
    # Use running statistics for Office support images so that each guide
    # is independent of the other support images. Gradients still reach f_t.
    was_training = model.f_t.training
    model.f_t.eval()
    try:
        private = model.f_t(supports)
    finally:
        model.f_t.train(was_training)
    return torch.cat((shared, private), dim=0)


@torch.no_grad()
def nearest_guides(features, guides):
    distance, labels = torch.cdist(features, guides).min(dim=1)
    return labels, distance


@torch.no_grad()
def confident_indices(labels, distances, num_classes, fraction, minimum_one=False):
    selected = []
    for label in range(num_classes):
        indices = (labels == label).nonzero(as_tuple=True)[0]
        ordered = distances[indices].argsort(stable=True)
        count = int(fraction * len(indices))
        if minimum_one and len(indices):
            count = max(1, count)
        selected.append(indices[ordered[:count]])
    return torch.cat(selected)


@dataclass
class TargetStatistics:
    prototypes: GaussianBank
    selected_ids: torch.Tensor
    selected_labels: torch.Tensor
    counts: torch.Tensor


@torch.no_grad()
def refresh_statistics(model, target_loader, source_prototypes, supports, ridge, fraction):
    model.eval()
    guides = build_guides(model, source_prototypes, supports)
    features, ids = [], []
    for images, indices in target_loader:
        features.append(model.f_t(images.to(supports.device)))
        ids.append(indices)
    features = torch.cat(features)
    ids = torch.cat(ids)
    labels, distances = nearest_guides(features, guides)
    count = len(guides)
    modified = model.preset == "friend"
    prototypes = fit_gaussians(features, labels, count, ridge, fallback_means=guides,
                               shrink=0.1 if modified else None)
    selected = confident_indices(labels, distances, count, fraction, minimum_one=modified)
    return TargetStatistics(
        prototypes, ids[selected.cpu()], labels[selected].cpu(), labels.bincount(minlength=count).cpu()
    )
