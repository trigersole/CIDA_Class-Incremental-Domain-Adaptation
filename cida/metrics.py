import torch


@torch.inference_mode()
def evaluate(model, data_loader, device, num_shared, num_classes):
    model.eval()
    totals = torch.zeros(num_classes, dtype=torch.long)
    correct = torch.zeros_like(totals)
    for images, labels in data_loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        totals += labels.bincount(minlength=num_classes)
        correct += labels[labels == predictions].bincount(minlength=num_classes)
    if (totals == 0).any():
        raise ValueError("Every target class must be represented in evaluation.")
    per_class = 100 * correct.double() / totals
    return {
        "ALL": float(per_class.mean()),
        "PRIV": float(per_class[num_shared:].mean()),
        "SHARED": float(per_class[:num_shared].mean()),
        "micro_accuracy": float(100 * correct.sum().double() / totals.sum()),
        "per_class": per_class.tolist(),
        "class_counts": totals.tolist(),
    }
