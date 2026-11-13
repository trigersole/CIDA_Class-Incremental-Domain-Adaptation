from dataclasses import fields
import json
from pathlib import Path
import random
import tempfile

import torch

from .training import TrainingConfig


def positive_int(text):
    from argparse import ArgumentTypeError

    value = int(text)
    if value < 1:
        raise ArgumentTypeError("expected a positive integer")
    return value


def add_data_args(parser):
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download missing Digits datasets.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=positive_int, help="Default: 64 for Digits, 8 for Office-31.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)


def add_training_args(parser):
    add_data_args(parser)
    parser.add_argument("--preset", choices=("baseline", "friend"), default=None,
                        help="friend: modified Digits setup; requires a newly trained source.")
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cov-ridge", type=float, help="Default: 0.1 for Digits, 0.14 for Office-31.")
    parser.add_argument("--gaussian-score", choices=("density", "log-density"), default="density")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=positive_int, help="Limit optimizer steps per epoch for smoke runs.")


def default_batch_size(benchmark):
    return 8 if benchmark == "office31" else 64


def training_config(args, benchmark="digits"):
    values = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    values["preset"] = args.preset or "baseline"
    if values["preset"] == "friend":
        if benchmark != "digits":
            raise ValueError("The friend preset currently supports Digits only.")
        values["gaussian_score"] = "log-density"
    values["batch_size"] = args.batch_size or default_batch_size(benchmark)
    if args.cov_ridge is None:
        values["cov_ridge"] = 0.14 if benchmark == "office31" else 0.1
    if args.lr <= 0 or values["cov_ridge"] <= 0:
        raise ValueError("Learning rate and covariance ridge must be positive.")
    return TrainingConfig(**values)


def configure_device(args):
    if args.workers < 0:
        raise ValueError("Number of data loader workers must be nonnegative.")
    torch.set_num_threads(args.cpu_threads)
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, payload):
    atomic_write(path, lambda temporary: torch.save(payload, temporary))


def load_checkpoint(path, stage):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != stage or checkpoint.get("version") != 1:
        raise ValueError(f"Expected a version-1 {stage} checkpoint at {path}.")
    return checkpoint


def save_json(path, payload):
    atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    ))
