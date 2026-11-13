"""Read-only epoch metrics. No checkpoint selection or training labels."""
import csv

from .cli import atomic_write, save_json
from .metrics import evaluate


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
