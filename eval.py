"""Deterministic, offline next-token evaluation of TinyStories checkpoints."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
# Required by deterministic CUDA matrix multiplication (before CUDA initializes).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, device):
    # train.py saves tensors and primitive config/metadata; no training import.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {k: checkpoint[k] for k in (
        'config', 'iter_num', 'best_val_loss', 'best_iter', 'train_time', 'tokens_seen'
    ) if k in checkpoint}
    return model, metadata


def batches(data, context_length, batch_size, target_count):
    """Block at s uses inputs [s:s+L], targets [s+1:s+L+1].

    Target blocks never overlap. Context/positions reset per block; adjacent
    inputs share one boundary token. The short tail is separate, without padding.
    """
    full_blocks, tail = divmod(target_count, context_length)
    for first in range(0, full_blocks, batch_size):
        starts = range(first * context_length,
                       min(first + batch_size, full_blocks) * context_length,
                       context_length)
        x = np.stack([data[s:s + context_length] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + context_length + 1] for s in starts]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)
    if tail:
        s = full_blocks * context_length
        yield (torch.tensor(np.array(data[s:s + tail], dtype=np.int64))[None, :],
               torch.tensor(np.array(data[s + 1:s + tail + 1], dtype=np.int64))[None, :])


def perplexity(mean_nll):
    try:
        value = math.exp(mean_nll)
    except OverflowError:
        return None, True
    return (value, False) if math.isfinite(value) else (None, True)


@torch.inference_mode()
def evaluate(model, data, context_length, batch_size, target_count, device, precision):
    model.eval()
    total_nll, token_count = 0.0, 0
    for x, y in batches(data, context_length, batch_size, target_count):
        x, y = x.to(device), y.to(device)
        ctx = (nullcontext() if precision == 'float32' else
               torch.autocast(device_type=device.type, dtype=getattr(torch, precision)))
        with ctx:
            # Passing targets is required: GPT otherwise returns only the last logit.
            logits, _ = model(x, y)
        losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                 y.reshape(-1), ignore_index=-1, reduction='none')
        valid = y.reshape(-1) != -1
        total_nll += losses[valid].double().sum().item()
        token_count += valid.sum().item()
    if not token_count or not math.isfinite(total_nll):
        raise ValueError('Evaluation produced no valid targets or non-finite NLL.')
    mean_nll = total_nll / token_count
    ppl, overflow = perplexity(mean_nll)
    return dict(total_negative_log_likelihood=total_nll,
                mean_cross_entropy_nats=mean_nll, perplexity=ppl,
                perplexity_overflow=overflow, scored_tokens=token_count)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/tinystories'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N')
    parser.add_argument('--precision', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--context-length', type=positive_int, default=256)
    parser.add_argument('--max-blocks', type=positive_int, help='SMOKE TEST: limit prefix target blocks')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    data_path = (args.data_dir / f'{args.split}.bin').resolve(strict=True)
    output = args.output.resolve()
    protected = {checkpoint_path, data_path}
    protected.update((checkpoint_path.parent / name).resolve() for name in
                     ('best.pt', 'final.pt', 'ckpt.pt', 'summary.json', 'log.csv'))
    if (output in protected or output.name in ('summary.json', 'log.csv')
            or output.suffix.lower() != '.json'):
        parser.error('output must be a separate JSON result, not a checkpoint, data or training artifact')
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite explicitly')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('only CPU and CUDA are supported')
    if device.type == 'cpu' and args.precision != 'float32':
        parser.error('CPU evaluation uses float32')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            parser.error('this CUDA device does not support bfloat16')
    # Fixed ordering plus deterministic kernels; no sampled batches or TF32.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if data_path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError('Token file size is not divisible by uint16 item size')
    if data_path.stat().st_size < 4:
        raise ValueError('Need at least two tokens')
    data = np.memmap(data_path, mode='r', dtype=np.uint16)
    model, metadata = load_model(checkpoint_path, device)
    if args.context_length > model.config.block_size:
        parser.error('context length exceeds checkpoint block_size; choose an explicit smaller length')
    targets = len(data) - 1
    if args.max_blocks is not None:
        targets = min(targets, args.max_blocks * args.context_length)
    if int(data[:targets + 1].max()) >= model.config.vocab_size:
        raise ValueError('Token ID exceeds checkpoint vocabulary')
    metrics = evaluate(model, data, args.context_length, args.batch_size,
                       targets, device, args.precision)
    checkpoint_hash, data_hash = sha256(checkpoint_path), sha256(data_path)
    result = dict(
        checkpoint=dict(path=str(checkpoint_path), sha256=checkpoint_hash),
        split=args.split, data=dict(path=str(data_path), sha256=data_hash,
                                   dtype=np.dtype(np.uint16).str, tokens=len(data)),
        metrics=metrics, model_config=asdict(model.config), run_metadata=metadata,
        parameter_count=sum(p.numel() for p in model.parameters()),
        windowing=dict(protocol='non_overlapping_target_blocks_v1',
                       context_length=args.context_length,
                       actual_max_context_length=min(args.context_length, targets),
                       stride=args.context_length, blocks=math.ceil(targets / args.context_length),
                       tail_block_length=targets % args.context_length,
                       target_index_start=1, target_index_end_inclusive=targets,
                       initial_unscored_tokens=1, dropped_tail_tokens=0,
                       limit_excluded_tokens=len(data) - 1 - targets,
                       boundary_policy='reset context and positions per block; cross EOS normally',
                       padding=False, ignore_index=-1),
        smoke_test=args.max_blocks is not None, max_blocks=args.max_blocks,
        device=str(device), device_name=(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'),
        precision=args.precision, parameter_dtype=str(next(model.parameters()).dtype),
        batch_size=args.batch_size, torch_version=str(torch.__version__),
        evaluation_duration_seconds=time.perf_counter() - started,
    )
    # Normalize any non-finite historical training metadata to JSON null.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        return value
    payload = json.dumps(finite_json(result), indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w' if args.overwrite else 'x') as handle:
        handle.write(payload)
    print(f"{'SMOKE TEST' if result['smoke_test'] else 'FULL SPLIT'} {args.split}: "
          f"CE={metrics['mean_cross_entropy_nats']:.6f} nats, "
          f"PPL={metrics['perplexity']}, tokens={metrics['scored_tokens']:,}, "
          f"time={result['evaluation_duration_seconds']:.1f}s\nSaved {output}")


if __name__ == '__main__':
    main()
