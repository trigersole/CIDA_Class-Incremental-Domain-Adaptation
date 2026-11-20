"""Read-only epoch metrics. No checkpoint selection or training labels."""
import csv

import torch

from .cli import atomic_write, save_json
from .metrics import evaluate


def pseudo_precision_summary(predictions, truth, num_classes, num_shared):
    """Read-only precision, grouped by assigned class (not by true class)."""
    predictions, truth = predictions.cpu().long(), truth.cpu().long()
    counts = predictions.bincount(minlength=num_classes)
    correct = predictions[predictions == truth].bincount(minlength=num_classes)
    per_class = [float(100 * good / count) if count else None
                 for good, count in zip(correct, counts)]

    def mean(values):
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    def micro(start, end):
        count = int(counts[start:end].sum())
        return float(100 * correct[start:end].sum() / count) if count else None

    return dict(count=len(truth), micro_precision=micro(0, num_classes),
                macro_precision=mean(per_class),
                shared_micro_precision=micro(0, num_shared),
                private_micro_precision=micro(num_shared, num_classes),
                shared_macro_precision=mean(per_class[:num_shared]),
                private_macro_precision=mean(per_class[num_shared:]),
                represented_classes=int((counts > 0).sum()),
                empty_classes=(counts == 0).nonzero().flatten().tolist(),
                macro_averaging="nonempty predicted classes",
                per_class_precision=per_class, predicted_counts=counts.tolist())


class EpochMetrics:
    def __init__(self, output, seed, evaluation_loader, device, num_shared, num_classes,
                 epochs, every=1):
        self.output = output
        self.seed = seed
        self.loader = evaluation_loader
        self.device = device
        self.num_shared = num_shared
        self.num_classes = num_classes
        self.epochs = epochs
        self.every = every
        self.rows = []

    def __call__(self, epoch, model, stats):
        if self.every == 0 or (epoch not in {0, self.epochs} and epoch % self.every):
            return None
        metrics = evaluate(model, self.loader, self.device, self.num_shared, self.num_classes)
        row = {"seed": self.seed, "epoch": epoch, **metrics,
               "pseudo_label_counts": stats.counts.tolist(),
               "confident_count": len(stats.selected_ids)}
        self.rows.append(row)
        save_json(self.output / f"epoch_metrics_seed_{self.seed}.json", {
            "purpose": "diagnostic only; final epoch is always reported",
            "seed": self.seed, "epochs": self.rows,
        })
        scalar_rows = []
        for item in self.rows:
            scalars = {k: item[k] for k in ("seed", "epoch", "ALL", "SHARED", "PRIV",
                                            "micro_accuracy", "confident_count")}
            scalars.update({f"accuracy_class_{c}": acc for c, acc in enumerate(item["per_class"])})
            scalars.update({f"pseudo_count_class_{c}": n for c, n in enumerate(item["pseudo_label_counts"])})
            scalar_rows.append(scalars)

        def write_csv(path):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
                writer.writeheader()
                writer.writerows(scalar_rows)

        atomic_write(self.output / f"epoch_metrics_seed_{self.seed}.csv", write_csv)
        return metrics
