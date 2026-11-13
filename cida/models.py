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


def feature_extractor(benchmark="digits", pretrained=False, preset="baseline"):
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
    if preset == "friend":
        layers = []
        for cin, cout, kernel, stride in ((3, 64, 5, 1), (64, 64, 5, 1),
                                          (64, 128, 3, 2), (128, 128, 3, 2)):
            layers.extend((nn.Conv2d(cin, cout, kernel, stride), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2)))
        return nn.Sequential(*layers, nn.Flatten(), nn.Linear(3200, FEATURE_DIM), nn.LeakyReLU(0.2))
    return nn.Sequential(
        nn.Conv2d(3, 64, 5), nn.LeakyReLU(),        # 32 -> 28
        nn.Conv2d(64, 64, 5), nn.LeakyReLU(),       # 28 -> 24
        nn.Conv2d(64, 128, 3, 2), nn.LeakyReLU(),   # 24 -> 11
        nn.Conv2d(128, 128, 3, 2), nn.LeakyReLU(),  # 11 -> 5
        nn.Flatten(), nn.Linear(3200, FEATURE_DIM), nn.LeakyReLU(),
    )


def classifier(outputs, benchmark="digits", preset="baseline"):
    if benchmark == "office31":
        return nn.Sequential(nn.Linear(OFFICE_FEATURE_DIM, 64), nn.ELU(), nn.Linear(64, outputs))
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 32), nn.LeakyReLU(0.2 if preset == "friend" else 0.01), nn.Linear(32, outputs)
    )


class ResidualProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(100, 100), nn.LeakyReLU(0.2), nn.Linear(100, 100))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return features + self.net(features)


def projector(benchmark="digits", preset="baseline"):
    if preset == "friend":
        return ResidualProjection()
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
    def __init__(self, num_shared, benchmark="digits", pretrained=False, preset="baseline"):
        super().__init__()
        self.num_shared = num_shared
        self.benchmark = benchmark
        if preset not in {"baseline", "friend"} or (preset == "friend" and benchmark != "digits"):
            raise ValueError("The friend preset currently supports Digits only.")
        self.preset = preset
        self.f_s = feature_extractor(benchmark, pretrained, preset)
        self.g_s = classifier(num_shared + 1, benchmark, preset)

    def forward(self, images):
        return self.g_s(self.f_s(images))


class AdaptationModel(nn.Module):
    def __init__(self, source, num_private):
        super().__init__()
        self.num_shared = source.num_shared
        self.benchmark = source.benchmark
        self.preset = source.preset
        self.f_t = copy.deepcopy(source.f_s)
        self.g_s = copy.deepcopy(source.g_s).requires_grad_(False)
        self.f_e = projector(self.benchmark, self.preset)
        self.f_d = projector(self.benchmark, self.preset)
        self.g_t = classifier(num_private, self.benchmark, self.preset)

    def joint_logits(self, features):
        shared = self.g_s(self.f_d(features))[:, :self.num_shared]
        return torch.cat((shared, self.g_t(features)), dim=1)

    def forward(self, images):
        return self.joint_logits(self.f_t(images))
