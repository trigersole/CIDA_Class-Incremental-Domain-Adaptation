import argparse
import json
from pathlib import Path

from cida.cli import add_data_args, configure_device, load_checkpoint
from cida.data import target_evaluation
from cida.metrics import evaluate
from cida.models import AdaptationModel, SourceModel
from cida.training import TrainingConfig, loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adapted CIDA USPS checkpoint.")
    add_data_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = configure_device(args)
    checkpoint = load_checkpoint(args.checkpoint, "target")
    num_shared = len(checkpoint["shared_classes"])
    num_private = len(checkpoint["private_classes"])
    model = AdaptationModel(SourceModel(num_shared), num_private).to(device)
    model.load_state_dict(checkpoint["model"])
    test = target_evaluation(args.data, args.download, checkpoint["protocol"], checkpoint["support_ids"])
    config = TrainingConfig(batch_size=args.batch_size, workers=args.workers)
    print(json.dumps(evaluate(model, loader(test, config), device, num_shared, num_shared + num_private), indent=2))


if __name__ == "__main__":
    main()
