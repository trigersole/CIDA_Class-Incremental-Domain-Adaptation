import copy

import torch
from torch import nn


FEATURE_DIM = 100


def feature_extractor():
    """Modified LeNet. Spatial sizes follow Supplement Table 2B."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 5), nn.LeakyReLU(),        # 32 -> 28
        nn.Conv2d(64, 64, 5), nn.LeakyReLU(),       # 28 -> 24
        nn.Conv2d(64, 128, 3, 2), nn.LeakyReLU(),   # 24 -> 11
        nn.Conv2d(128, 128, 3, 2), nn.LeakyReLU(),  # 11 -> 5
        nn.Flatten(), nn.Linear(3200, FEATURE_DIM), nn.LeakyReLU(),
    )


def classifier(outputs):
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 32), nn.LeakyReLU(), nn.Linear(32, outputs)
    )


def projector():
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.LeakyReLU(),
        nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.LeakyReLU(),
    )


class SourceModel(nn.Module):
    def __init__(self, num_shared):
        super().__init__()
        self.num_shared = num_shared
        self.f_s = feature_extractor()
        self.g_s = classifier(num_shared + 1)

    def forward(self, images):
        return self.g_s(self.f_s(images))


class AdaptationModel(nn.Module):
    def __init__(self, source, num_private):
        super().__init__()
        self.num_shared = source.num_shared
        self.f_t = copy.deepcopy(source.f_s)
        self.g_s = copy.deepcopy(source.g_s).requires_grad_(False)
        self.f_e = projector()
        self.f_d = projector()
        self.g_t = classifier(num_private)

    def joint_logits(self, features):
        shared = self.g_s(self.f_d(features))[:, :self.num_shared]
        return torch.cat((shared, self.g_t(features)), dim=1)

    def forward(self, images):
        return self.joint_logits(self.f_t(images))
