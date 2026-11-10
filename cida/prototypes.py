from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass
class GaussianBank:
    means: torch.Tensor        # [classes, feature_dim]
    scale_tril: torch.Tensor   # [classes, feature_dim, feature_dim]

    def log_density(self, features):
        delta = features.to(self.means)[:, None, :] - self.means[None, :, :]
        whitened = torch.linalg.solve_triangular(
            self.scale_tril, delta.permute(1, 2, 0), upper=False
        )
        log_peak = (
            -0.5 * self.means.shape[1] * math.log(2 * math.pi)
            - self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=1)
        )
        return log_peak - 0.5 * whitened.square().sum(dim=1).T

    def sample(self, classes):
        means = self.means[classes]
        noise = torch.randn_like(means)
        return means + (self.scale_tril[classes] @ noise.unsqueeze(-1)).squeeze(-1)

    def state_dict(self):
        return {"means": self.means.cpu(), "scale_tril": self.scale_tril.cpu()}


@torch.no_grad()
def fit_gaussians(features, labels, num_classes, ridge, fallback_means=None):
    """Fit empirical full covariance plus ridge I. Sparse classes use pooled covariance."""
    if len(features) < 2 or ridge <= 0:
        raise ValueError("Gaussian fitting needs at least two features and a positive ridge.")
    features = features.double()
    dimension = features.shape[1]
    regularizer = ridge * torch.eye(dimension, dtype=features.dtype, device=features.device)
    pooled = features - features.mean(dim=0)
    pooled_covariance = pooled.T @ pooled / (len(features) - 1)
    means, covariances = [], []
    for label in range(num_classes):
        members = features[labels == label]
        if len(members) == 0 and fallback_means is None:
            raise ValueError(f"Missing source class {label} in prototype fitting.")
        mean = members.mean(dim=0) if len(members) else fallback_means[label].double()
        covariance = pooled_covariance
        if len(members) >= 2:
            centered = members - mean
            covariance = centered.T @ centered / (len(members) - 1)
        means.append(mean)
        covariances.append(covariance + regularizer)
    return GaussianBank(torch.stack(means), torch.linalg.cholesky(torch.stack(covariances)))


def separability_loss(features, labels, prototypes, score="density"):
    log_density = prototypes.log_density(features)
    if score == "log-density":
        return F.cross_entropy(log_density, labels)
    # Shift in log space before exponentiating. This computes density(c) -
    # max_c density(c) with expm1, avoiding cancellation for close scores.
    max_log_density = log_density.max(dim=1, keepdim=True).values
    relative = (log_density - max_log_density).expm1()
    logits = relative * max_log_density.exp()
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Gaussian density loss overflowed. Increase --cov-ridge.")
    return loss


@torch.no_grad()
def negative_mask(features, prototypes, sigma=3.0):
    """Supplement Algorithm 1, comparing with the maximum-density class's threshold."""
    score, best = prototypes.log_density(features).max(dim=1)
    peak = prototypes.log_density(prototypes.means).diagonal()
    # Moving sigma standard deviations along the leading eigenvector reduces
    # log density by sigma^2 / 2, so an eigendecomposition is unnecessary.
    return score < peak[best] - 0.5 * sigma**2


@torch.no_grad()
def sample_negatives(global_prototype, class_prototypes, count, sigma=3.0):
    accepted = []
    remaining = count
    for _ in range(128):
        classes = torch.zeros(max(2 * remaining, 64), dtype=torch.long, device=global_prototype.means.device)
        candidates = global_prototype.sample(classes)
        selected = candidates[negative_mask(candidates, class_prototypes, sigma)][:remaining]
        accepted.append(selected)
        remaining -= len(selected)
        if remaining == 0:
            return torch.cat(accepted)
    raise RuntimeError("The source Gaussian rejection sampler did not fill a negative batch.")
