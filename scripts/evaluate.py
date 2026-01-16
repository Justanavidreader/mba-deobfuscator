#!/usr/bin/env python3
"""
Evaluation script for MBA Deobfuscator.

Evaluates model on test set with comprehensive metrics:
- Exact match accuracy
- Syntax validity
- Semantic equivalence (execution + Z3)
- Per-depth breakdown
- Inference latency

Usage:
    python scripts/evaluate.py --checkpoint best.pt --test-set data/test.jsonl
    python scripts/evaluate.py --checkpoint best.pt --test-set data/test.jsonl --z3-verify
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.dataset import MBADataset
from src.data.collate import collate_graphs
from src.inference.verify import ThreeTierVerifier
from src.inference.pipeline import InferencePipeline
from src.constants import SOS_IDX, EOS_IDX, MAX_SEQ_LEN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> MBADeobfuscator:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint or use defaults
    config = checkpoint.get('config', {})

    model = MBADeobfuscator(
        encoder_type=config.get('encoder_type', 'gat'),
        hidden_dim=config.get('hidden_dim', 256),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        num_encoder_heads=config.get('num_encoder_heads', 8),
        d_model=config.get('d_model', 512),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_decoder_heads=config.get('num_decoder_heads', 8),
        d_ff=config.get('d_ff', 2048),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


@torch.no_grad()
def greedy_decode(
    model: MBADeobfuscator,
    graph_batch,
    fingerprint: torch.Tensor,
    tokenizer: MBATokenizer,
    max_len: int = MAX_SEQ_LEN,
    device: torch.device = None
) -> List[str]:
    """Greedy decode a batch."""
    if device is None:
        device = fingerprint.device

    memory = model.encode(graph_batch, fingerprint)
    batch_size = memory.shape[0]

    output = torch.full(
        (batch_size, 1), SOS_IDX, dtype=torch.long, device=device
    )

    for _ in range(max_len - 1):
        decoder_out = model.decode(output, memory)

        if isinstance(decoder_out, dict):
            logits = decoder_out['vocab_logits'][:, -1, :]
        else:
            logits = model.vocab_head(decoder_out[:, -1, :])

        next_token = logits.argmax(dim=-1, keepdim=True)
        output = torch.cat([output, next_token], dim=1)

        if (next_token == EOS_IDX).all():
            break

    predictions = []
    for i in range(batch_size):
        tokens = output[i].tolist()
        pred_str = tokenizer.decode(tokens)
        predictions.append(pred_str)

    return predictions


def evaluate_model(
    model: MBADeobfuscator,
    dataloader: DataLoader,
    tokenizer: MBATokenizer,
    verifier: ThreeTierVerifier,
    device: torch.device,
    use_z3: bool = False,
    use_beam_search: bool = False,
    beam_width: int = 10,
) -> Dict:
    """
    Evaluate model on dataset.

    Returns comprehensive metrics including per-depth breakdown.
    """
    model.eval()

    # Metrics
    total_samples = 0
    exact_matches = 0
    syntax_valid = 0
    exec_equiv = 0
    z3_equiv = 0

    # Per-depth metrics
    depth_metrics = defaultdict(lambda: {
        'total': 0, 'exact': 0, 'syntax': 0, 'exec': 0, 'z3': 0
    })

    # Latency tracking
    latencies = []

    # Predictions for analysis
    all_predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        graph_batch = batch['graph_batch'].to(device)
        fingerprint = batch['fingerprint'].to(device)
        depths = batch['depth'].tolist()
        targets = batch['simplified']
        inputs = batch['obfuscated']

        batch_size = len(targets)

        # Decode
        start_time = time.time()

        if use_beam_search:
            # Use inference pipeline with beam search
            pipeline = InferencePipeline(model, tokenizer)
            predictions = []
            for i in range(batch_size):
                result = pipeline.simplify(inputs[i])
                predictions.append(result.simplified)
        else:
            # Greedy decode
            predictions = greedy_decode(
                model, graph_batch, fingerprint, tokenizer, device=device
            )

        decode_time = time.time() - start_time
        latencies.append(decode_time / batch_size)

        # Evaluate each sample
        for i, (pred, target, inp, depth) in enumerate(zip(predictions, targets, inputs, depths)):
            total_samples += 1
            depth_metrics[depth]['total'] += 1

            # Exact match
            pred_norm = pred.replace(' ', '').lower()
            tgt_norm = target.replace(' ', '').lower()
            if pred_norm == tgt_norm:
                exact_matches += 1
                depth_metrics[depth]['exact'] += 1

            # Syntax validity
            try:
                tokens = tokenizer.encode(pred)
                is_syntax_valid = len(tokens) > 2
            except Exception:
                is_syntax_valid = False

            if is_syntax_valid:
                syntax_valid += 1
                depth_metrics[depth]['syntax'] += 1

            # Semantic equivalence
            if is_syntax_valid:
                results = verifier.verify_batch(inp, [pred])
                if results:
                    result = results[0]
                    if result.exec_valid:
                        exec_equiv += 1
                        depth_metrics[depth]['exec'] += 1
                    if use_z3 and result.z3_verified:
                        z3_equiv += 1
                        depth_metrics[depth]['z3'] += 1

            all_predictions.append({
                'input': inp,
                'target': target,
                'prediction': pred,
                'depth': depth,
                'exact_match': pred_norm == tgt_norm,
                'syntax_valid': is_syntax_valid,
            })

    # Compute metrics
    metrics = {
        'total_samples': total_samples,
        'exact_match': exact_matches / max(total_samples, 1),
        'syntax_valid': syntax_valid / max(total_samples, 1),
        'exec_equiv': exec_equiv / max(total_samples, 1),
        'avg_latency_ms': sum(latencies) / len(latencies) * 1000 if latencies else 0,
    }

    if use_z3:
        metrics['z3_equiv'] = z3_equiv / max(total_samples, 1)

    # Per-depth breakdown
    depth_breakdown = {}
    for depth in sorted(depth_metrics.keys()):
        dm = depth_metrics[depth]
        total = max(dm['total'], 1)
        depth_breakdown[depth] = {
            'total': dm['total'],
            'exact_match': dm['exact'] / total,
            'syntax_valid': dm['syntax'] / total,
            'exec_equiv': dm['exec'] / total,
        }
        if use_z3:
            depth_breakdown[depth]['z3_equiv'] = dm['z3'] / total

    metrics['per_depth'] = depth_breakdown

    return metrics, all_predictions


def print_results(metrics: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal samples: {metrics['total_samples']}")
    print(f"\nOverall Metrics:")
    print(f"  Exact Match:    {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"  Syntax Valid:   {metrics['syntax_valid']:.4f} ({metrics['syntax_valid']*100:.2f}%)")
    print(f"  Exec Equiv:     {metrics['exec_equiv']:.4f} ({metrics['exec_equiv']*100:.2f}%)")
    if 'z3_equiv' in metrics:
        print(f"  Z3 Equiv:       {metrics['z3_equiv']:.4f} ({metrics['z3_equiv']*100:.2f}%)")
    print(f"  Avg Latency:    {metrics['avg_latency_ms']:.2f} ms")

    print(f"\nPer-Depth Breakdown:")
    print("-" * 60)
    print(f"{'Depth':>6} | {'Count':>6} | {'Exact':>8} | {'Syntax':>8} | {'Exec':>8}")
    print("-" * 60)

    for depth in sorted(metrics['per_depth'].keys()):
        dm = metrics['per_depth'][depth]
        print(f"{depth:>6} | {dm['total']:>6} | {dm['exact_match']:>8.4f} | {dm['syntax_valid']:>8.4f} | {dm['exec_equiv']:>8.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MBA Deobfuscator')
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-set', type=str, required=True,
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--z3-verify', action='store_true',
        help='Use Z3 for formal verification (slower)'
    )
    parser.add_argument(
        '--beam-search', action='store_true',
        help='Use beam search instead of greedy decode'
    )
    parser.add_argument(
        '--beam-width', type=int, default=10,
        help='Beam width for beam search'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Load test set
    test_dataset = MBADataset(
        data_path=args.test_set,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_graphs,
    )

    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Create verifier
    verifier = ThreeTierVerifier(tokenizer)

    # Evaluate
    metrics, predictions = evaluate_model(
        model, test_loader, tokenizer, verifier, device,
        use_z3=args.z3_verify,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
    )

    # Print results
    print_results(metrics)

    # Save detailed results
    if args.output:
        output_data = {
            'metrics': metrics,
            'predictions': predictions,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved detailed results to {args.output}")


if __name__ == '__main__':
    main()
