"""Read-only Office checkpoint diagnosis; never retrains or selects a checkpoint."""
import argparse
from dataclasses import fields
from pathlib import Path

import torch

from cida.cli import load_checkpoint, save_json, seed_all
from cida.models import AdaptationModel, SourceModel
from cida.office import office_source, office_target, office_trial
from cida.prototypes import GaussianBank
from cida.training import TrainingConfig, loader, recalibrate_batchnorm, warmup_projections


def macro_accuracy(predictions, labels, classes):
    values = []
    for label in classes:
        mask = labels == label
        if mask.any():
            values.append(float((predictions[mask] == label).double().mean() * 100))
    return sum(values) / len(values) if values else None


@torch.inference_mode()
def score_paths(model, dataset, config, device):
    """Evaluate decoder and decoder-bypass paths on exactly the same clean images."""
    model.eval()
    projected, direct, labels = [], [], []
    for images, targets in loader(dataset, config):
        features = model.f_t(images.to(device))
        projected.append(model.joint_logits(features).cpu())
        direct.append(model.g_s(features)[:, :model.num_shared].cpu())
        labels.append(targets)
    logits, bypass, labels = torch.cat(projected), torch.cat(direct), torch.cat(labels)
    shared = model.num_shared
    full_predictions = logits.argmax(1)
    restricted = logits[:, :shared].argmax(1)
    return {
        "ALL": macro_accuracy(full_predictions, labels, range(logits.shape[1])),
        "SHARED_full_31": macro_accuracy(full_predictions, labels, range(shared)),
        "PRIV_full_31": macro_accuracy(full_predictions, labels, range(shared, logits.shape[1])),
        "SHARED_restricted_20": macro_accuracy(restricted, labels, range(shared)),
        "SHARED_bypass_fd_restricted_20": macro_accuracy(bypass.argmax(1), labels, range(shared)),
        "shared_predicted_private_percent": float(
            (full_predictions[labels < shared] >= shared).double().mean() * 100),
        "per_class_full": [macro_accuracy(full_predictions, labels, [c]) for c in range(logits.shape[1])],
    }


@torch.inference_mode()
def source_accuracy(source, dataset, config, device):
    source.eval()
    logits, labels = [], []
    for images, targets in loader(dataset, config):
        logits.append(source(images.to(device)).cpu())
        labels.append(targets)
    logits, labels = torch.cat(logits), torch.cat(labels)
    classes = range(source.num_shared)
    return {
        "shared_restricted": macro_accuracy(logits[:, :source.num_shared].argmax(1), labels, classes),
        "including_unknown_output": macro_accuracy(logits.argmax(1), labels, classes),
        "purpose": "source training-set fit, not held-out generalization",
    }


def copy_bn_statistics(destination, original):
    """Diagnostic counterfactual: restore source BN buffers without changing weights."""
    count = 0
    originals = dict(original.named_modules())
    with torch.no_grad():
        for name, layer in destination.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                other = originals[name]
                for buffer in ("running_mean", "running_var", "num_batches_tracked"):
                    if getattr(layer, buffer) is not None:
                        getattr(layer, buffer).copy_(getattr(other, buffer))
                count += 1
    return count


def resolve_checkpoints(source, target, seed):
    if (source is None) != (target is None):
        raise ValueError("Pass both --source and --target-checkpoint, or neither.")
    if source is not None:
        return source, target
    candidates = []
    roots = [Path('/kaggle/working/runs'), Path('/kaggle/input')]
    for root in roots:
        for path in root.rglob('source.pt'):
            target_path = path.parent / 'confident' / f'seed_{seed}.pt'
            if not target_path.is_file():
                continue
            checkpoint = load_checkpoint(path, 'source')
            if (checkpoint.get('benchmark') == 'office31'
                    and checkpoint.get('config', {}).get('preset') == 'friend'):
                candidates.append((path, target_path))
    if len(candidates) != 1:
        raise ValueError(
            f'Found {len(candidates)} matching Office runs: {candidates}. '
            'Attach the saved Office notebook OUTPUT containing source.pt and confident/seed_0.pt, '
            'or pass --source and --target-checkpoint explicitly. No training will be started.')
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target-checkpoint', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or Path(f'/kaggle/working/office_diagnostics/seed_{args.seed}.json')
    torch.set_num_threads(4)
    try:
        source_path, target_path = resolve_checkpoints(args.source, args.target_checkpoint, args.seed)
    except ValueError as error:
        parser.error(str(error))
    source_ckpt = load_checkpoint(source_path, 'source')
    target_ckpt = load_checkpoint(target_path, 'target')
    if source_ckpt.get('benchmark') != 'office31' or target_ckpt.get('benchmark') != 'office31':
        parser.error('Both checkpoints must be Office-31 models.')
    for key in ('shared_classes', 'source_domain'):
        if source_ckpt[key] != target_ckpt[key]:
            parser.error(f'Checkpoint mismatch: {key}')
    preset = source_ckpt['config'].get('preset', 'baseline')
    if preset != 'friend' or target_ckpt['config'].get('preset') != preset:
        parser.error('This replay is for the Office friend run; both checkpoints must match.')
    if args.seed != target_ckpt['config']['seed']:
        parser.error('--seed must match the saved target checkpoint.')
    data = args.data
    if data is None:
        roots = sorted({p.parent for p in Path('/kaggle/input').rglob(source_ckpt['source_domain'])
                        if p.is_dir() and (p.parent / target_ckpt['target_domain']).is_dir()})
        if len(roots) != 1:
            parser.error(f'Cannot uniquely locate Office-31 data: {roots}. Pass --data.')
        data = roots[0]
    shared, private = source_ckpt['shared_classes'], target_ckpt['private_classes']
    dataset, actual_private = office_target(data, target_ckpt['target_domain'], shared)
    if tuple(private) != tuple(actual_private):
        parser.error('Private classes do not match dataset ordering.')
    _, stats_target, supports, support_ids, test = office_trial(
        dataset, target_ckpt['target_domain'], shared, private, args.seed)
    if support_ids != target_ckpt['support_ids']:
        parser.error('Support IDs differ from saved run; check dataset and seed.')
    config = TrainingConfig(**{f.name: target_ckpt['config'][f.name]
                              for f in fields(TrainingConfig) if f.name in target_ckpt['config']})
    device = torch.device(args.device)
    source = SourceModel(len(shared), 'office31', preset=preset).to(device)
    source.load_state_dict(source_ckpt['model'])
    source.eval()
    _, clean_source, _ = office_source(data, source_ckpt['source_domain'], len(shared))
    report = {
        'purpose': 'Read-only diagnosis. No checkpoint selection; target labels only score predictions.',
        'source': str(source_path), 'target': str(target_path), 'seed': args.seed,
        'source_domain': source_ckpt['source_domain'], 'target_domain': target_ckpt['target_domain'],
        'config': target_ckpt['config'], 'support_ids': support_ids,
        'source_training_accuracy': source_accuracy(source, clean_source, config, device),
        'stages': {},
    }

    def record(name, model):
        result = score_paths(model, test, config, device)
        report['stages'][name] = result
        save_json(args.output, report)
        print(name, {k: round(v, 2) for k, v in result.items() if isinstance(v, (int, float))}, flush=True)

    seed_all(args.seed)
    model = AdaptationModel(source, len(private)).to(device)
    record('source_copy_identity_before_adabn', model)
    recalibrate_batchnorm(model.f_t, loader(stats_target, config), device)
    record('after_adabn_before_warmup', model)
    bank = GaussianBank(**{k: v.to(device) for k, v in source_ckpt['prototypes'].items()})
    warmup_projections(model, bank, config, target_ckpt['config']['projection_steps'])
    record('after_projection_initialization', model)
    # The source head must truly be the same frozen head used by this trial.
    expected_head = {k: v.cpu() for k, v in source.g_s.state_dict().items()}
    saved_head = {k.removeprefix('g_s.'): v for k, v in target_ckpt['model'].items() if k.startswith('g_s.')}
    if expected_head.keys() != saved_head.keys() or any(
            not torch.equal(expected_head[k], saved_head[k]) for k in expected_head):
        parser.error('Target source-head weights differ from source checkpoint; verify checkpoint pair.')
    model.load_state_dict(target_ckpt['model'])
    record('final_checkpoint', model)
    if target_ckpt.get('metrics'):
        final = report['stages']['final_checkpoint']
        saved = target_ckpt['metrics']
        report['final_metrics_match_saved'] = all(abs(final[name] - saved[key]) < 1e-5
            for name, key in [('ALL', 'ALL'), ('SHARED_full_31', 'SHARED'), ('PRIV_full_31', 'PRIV')])
    # Only counterfactual buffer substitution on an in-memory model; never saved as a trained model.
    report['bn_layers_restored_for_counterfactual'] = copy_bn_statistics(model.f_t, source.f_s)
    record('final_weights_with_source_bn_buffers_counterfactual', model)
    report['saved_final_metrics'] = target_ckpt.get('metrics')
    report['source_head_preserved_exactly'] = True
    # Epoch 0 is authoritative for the original initialization if saved diagnostics exist.
    epoch_file = target_path.parent / f'epoch_metrics_seed_{args.seed}.json'
    if epoch_file.is_file():
        import json
        epochs = json.loads(epoch_file.read_text())['epochs']
        report['original_epoch_zero'] = next((e for e in epochs if e['epoch'] == 0), None)
    save_json(args.output, report)
    print('Source training accuracy:', report['source_training_accuracy'], flush=True)
    print(f'Saved diagnostic report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
