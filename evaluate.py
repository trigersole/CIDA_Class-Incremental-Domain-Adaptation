import argparse
import json
from pathlib import Path

from cida.cli import add_data_args, configure_device, default_batch_size, load_checkpoint
from cida.data import target_evaluation
from cida.metrics import evaluate
from cida.models import AdaptationModel, SourceModel
from cida.office import office_evaluation
from cida.training import TrainingConfig, loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adapted CIDA checkpoint.")
    add_data_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = configure_device(args)
    checkpoint = load_checkpoint(args.checkpoint, "target")
    benchmark = checkpoint.get("benchmark", "digits")
    num_shared = len(checkpoint["shared_classes"])
    num_private = len(checkpoint["private_classes"])
    preset = checkpoint.get("config", {}).get("preset", "baseline")
    model = AdaptationModel(SourceModel(num_shared, benchmark, preset=preset), num_private).to(device)
    model.load_state_dict(checkpoint["model"])
    if benchmark == "office31":
        if args.download:
            parser.error("Extract Office-31 under --data. --download is for Digits.")
        test = office_evaluation(args.data, checkpoint["target_domain"], checkpoint["shared_classes"],
                                 checkpoint["private_classes"], checkpoint["support_ids"])
    else:
        test = target_evaluation(args.data, args.download, checkpoint["protocol"], checkpoint["support_ids"], preset)
    config = TrainingConfig(batch_size=args.batch_size or default_batch_size(benchmark), workers=args.workers)
    print(json.dumps(evaluate(model, loader(test, config), device, num_shared, num_shared + num_private), indent=2))


if __name__ == "__main__":
    main()
