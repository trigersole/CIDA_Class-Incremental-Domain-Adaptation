from copy import copy
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from .data import UnlabeledTarget


OFFICE_DOMAINS = ("amazon", "dslr", "webcam")


def swap_rgb_channels(image):
    return image[torch.randperm(3)]


def office_transform(domain, training):
    from torchvision import transforms as T

    if training and domain in {"dslr", "webcam"}:
        spatial = [T.Resize(256), T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                   T.RandomHorizontalFlip(), T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
                   T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
        color = [T.RandomApply([swap_rgb_channels], p=0.25)]
    else:
        spatial = [T.Resize(256), T.CenterCrop(224)]
        color = []
    return T.Compose([*spatial, T.ToTensor(), *color,
                      T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def office_domain(root, domain):
    from torchvision.datasets import ImageFolder

    if domain not in OFFICE_DOMAINS:
        raise ValueError(f"Choose an Office-31 domain from {', '.join(OFFICE_DOMAINS)}.")
    folder = Path(root) / domain
    if (folder / "images").is_dir():
        folder = folder / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"Office-31 domain missing at {folder}. Extract the dataset under --data.")
    dataset = ImageFolder(folder, transform=office_transform(domain, training=False))
    if len(dataset.classes) != 31:
        raise ValueError(f"Expected 31 class folders under {folder}, found {len(dataset.classes)}.")
    return dataset


def office_classes(classes, shared_count):
    # Table 2 specifies 20 + 11. Table 3 specifies 10 + 10, while the
    # supplement lists ranks 21 through 31 (11). Use ranks 21–30 for Table 3.
    if shared_count == 20:
        return tuple(classes[:20]), tuple(classes[20:])
    if shared_count == 10:
        return tuple(classes[:10]), tuple(classes[20:30])
    raise ValueError("Office-31 supports 10 or 20 shared classes.")


class OfficeSubset(Dataset):
    def __init__(self, dataset, classes, transform=None):
        self.dataset = copy(dataset)
        if transform is not None:
            self.dataset.transform = transform
        labels = {dataset.class_to_idx[name]: label for label, name in enumerate(classes)}
        self.indices = [i for i, label in enumerate(dataset.targets) if label in labels]
        self.targets = [labels[dataset.targets[i]] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, self.targets[index]


def office_source(root, domain, shared_count):
    dataset = office_domain(root, domain)
    shared, _ = office_classes(dataset.classes, shared_count)
    return (OfficeSubset(dataset, shared, office_transform(domain, training=True)),
            OfficeSubset(dataset, shared), shared)


def office_target(root, domain, shared):
    dataset = office_domain(root, domain)
    target_shared, private = office_classes(dataset.classes, len(shared))
    if tuple(shared) != target_shared:
        raise ValueError("Office-31 target class order does not match the source checkpoint.")
    return dataset, private


def office_trial(dataset, domain, shared, private, seed):
    classes = tuple(shared) + tuple(private)
    clean = OfficeSubset(dataset, classes)
    augmented = OfficeSubset(dataset, classes, office_transform(domain, training=True))
    rng = torch.Generator().manual_seed(seed)
    support_ids = []
    labels = torch.tensor(clean.targets)
    for label in range(len(shared), len(classes)):
        members = (labels == label).nonzero(as_tuple=True)[0]
        if len(members) < 2:
            raise ValueError(f"Office-31 class {classes[label]} needs separate support and evaluation images.")
        support_ids.append(int(members[torch.randint(len(members), (), generator=rng)]))
    supports = torch.stack([clean[i][0] for i in support_ids])
    excluded = set(support_ids)
    indices = [i for i in range(len(clean)) if i not in excluded]
    return (UnlabeledTarget(augmented, indices), UnlabeledTarget(clean, indices),
            supports, support_ids, Subset(clean, indices))


def office_evaluation(root, domain, shared, private, support_ids):
    dataset, current_private = office_target(root, domain, shared)
    if tuple(private) != current_private:
        raise ValueError("Office-31 target private class order does not match the checkpoint.")
    clean = OfficeSubset(dataset, tuple(shared) + tuple(private))
    excluded = set(support_ids)
    return Subset(clean, [i for i in range(len(clean)) if i not in excluded])
