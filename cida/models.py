import copy

import torch
from torch import nn


FEATURE_DIM = 100
OFFICE_FEATURE_DIM = 256


class FeatureBatchNorm(nn.BatchNorm1d):
    def forward(self, features):
        # Office domains can leave a single image in the last mini-batch.
        if self.training and features.shape[0] == 1:
            return nn.functional.batch_norm(
                features, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, eps=self.eps,
            )
        return super().forward(features)


def feature_extractor(benchmark="digits", pretrained=False):
    """Modified LeNet. Spatial sizes follow Supplement Table 2B."""
    if benchmark == "office31":
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        return nn.Sequential(
            backbone,
            nn.Linear(2048, 1024), nn.ELU(), FeatureBatchNorm(1024),
            nn.Linear(1024, OFFICE_FEATURE_DIM), nn.ELU(),
            nn.Linear(OFFICE_FEATURE_DIM, OFFICE_FEATURE_DIM), nn.ELU(),
            FeatureBatchNorm(OFFICE_FEATURE_DIM),
        )
    if benchmark != "digits":
        raise ValueError(f"Unknown benchmark {benchmark}.")
    return nn.Sequential(
        nn.Conv2d(3, 64, 5), nn.LeakyReLU(),        # 32 -> 28
        nn.Conv2d(64, 64, 5), nn.LeakyReLU(),       # 28 -> 24
        nn.Conv2d(64, 128, 3, 2), nn.LeakyReLU(),   # 24 -> 11
        nn.Conv2d(128, 128, 3, 2), nn.LeakyReLU(),  # 11 -> 5
        nn.Flatten(), nn.Linear(3200, FEATURE_DIM), nn.LeakyReLU(),
    )


def classifier(outputs, benchmark="digits"):
    if benchmark == "office31":
        return nn.Sequential(nn.Linear(OFFICE_FEATURE_DIM, 64), nn.ELU(), nn.Linear(64, outputs))
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 32), nn.LeakyReLU(), nn.Linear(32, outputs)
    )


def projector(benchmark="digits"):
    if benchmark == "office31":
        return nn.Sequential(*(
            layer for _ in range(3)
            for layer in (nn.Linear(OFFICE_FEATURE_DIM, OFFICE_FEATURE_DIM), nn.ELU())
        ))
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.LeakyReLU(),
        nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.LeakyReLU(),
    )


class SourceModel(nn.Module):
    def __init__(self, num_shared, benchmark="digits", pretrained=False):
        super().__init__()
        self.num_shared = num_shared
        self.benchmark = benchmark
        self.f_s = feature_extractor(benchmark, pretrained)
        self.g_s = classifier(num_shared + 1, benchmark)

    def forward(self, images):
        return self.g_s(self.f_s(images))


class AdaptationModel(nn.Module):
    def __init__(self, source, num_private):
        super().__init__()
        self.num_shared = source.num_shared
        self.benchmark = source.benchmark
        self.f_t = copy.deepcopy(source.f_s)
        self.g_s = copy.deepcopy(source.g_s).requires_grad_(False)
        self.f_e = projector(self.benchmark)
        self.f_d = projector(self.benchmark)
        self.g_t = classifier(num_private, self.benchmark)

    def joint_logits(self, features):
        shared = self.g_s(self.f_d(features))[:, :self.num_shared]
        return torch.cat((shared, self.g_t(features)), dim=1)

    def forward(self, images):
        return self.joint_logits(self.f_t(images))
