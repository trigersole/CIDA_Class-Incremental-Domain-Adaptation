from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset


SHARED = (0, 1, 2, 3, 4)
PRIVATE = (5, 6, 7, 8, 9)
CLASSES = SHARED + PRIVATE


def digits(name, root, train, download):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    dataset_type = {"mnist": datasets.MNIST, "usps": datasets.USPS}[name]
    return dataset_type(str(Path(root) / name), train=train, transform=transform, download=download)


def source_dataset(root, download):
    source = digits("mnist", root, train=True, download=download)
    labels = torch.as_tensor(source.targets)
    return Subset(source, torch.isin(labels, torch.tensor(SHARED)).nonzero(as_tuple=True)[0].tolist())


class UnlabeledTarget(Dataset):
    """The training loader receives the image and a stable ID only."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, index


class ConfidentTarget(Dataset):
    def __init__(self, target, indices, pseudo_labels):
        self.target = target
        self.indices = indices.tolist()
        self.pseudo_labels = pseudo_labels.tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.target[self.indices[index]]
        return image, self.pseudo_labels[index]


def target_data(root, download, protocol, seed):
    target = digits("usps", root, train=(protocol == "held-out"), download=download)
    labels = torch.as_tensor(target.targets)
    rng = torch.Generator().manual_seed(seed)
    support_ids = []
    for label in PRIVATE:
        members = (labels == label).nonzero(as_tuple=True)[0]
        support_ids.append(int(members[torch.randint(len(members), (), generator=rng)]))
    supports = torch.stack([target[i][0] for i in support_ids])
    support_set = set(support_ids)
    unlabeled = UnlabeledTarget(target, [i for i in range(len(target)) if i not in support_set])
    return unlabeled, supports, support_ids


def target_evaluation(root, download, protocol, support_ids):
    test = digits("usps", root, train=False, download=download)
    if protocol == "transductive":
        support_set = set(support_ids)
        return Subset(test, [i for i in range(len(test)) if i not in support_set])
    return test
