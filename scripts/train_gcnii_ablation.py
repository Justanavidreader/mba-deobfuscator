#!/usr/bin/env python3
"""
GCNII Over-Smoothing Mitigation Ablation Study.

Trains HGT encoder with and without GCNII techniques, comparing:
- Baseline: HGT with standard architecture
- GCNII: HGT with initial residuals + identity mapping

Evaluates on depth buckets [2-4, 5-7, 8-10, 11-14] to measure
over-smoothing mitigation effectiveness on deep expressions.

Usage:
    # Train baseline
    python scripts/train_gcnii_ablation.py --mode baseline --run-id 1

    # Train GCNII variant
    python scripts/train_gcnii_ablation.py --mode gcnii --run-id 1

    # Run full comparison (both models, 3 trials each)
    python scripts/train_gcnii_ablation.py --mode full --num-trials 3

    # Evaluate and compare saved checkpoints
    python scripts/train_gcnii_ablation.py --mode evaluate --baseline-ckpt path1 --gcnii-ckpt path2
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
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
from src.training.phase2_trainer import Phase2Trainer
from src.constants import (
    ABLATION_DEPTH_BUCKETS,
    GCNII_ALPHA,
    GCNII_LAMBDA,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_baseline_config(base_config_path: str, run_id: int) -> Dict[str, Any]:
    """Create configuration for baseline HGT (no GCNII)."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for baseline
    config['model']['encoder_type'] = 'hgt'
    config['model']['edge_type_mode'] = 'optimized'
    config['model']['hidden_dim'] = 256
    config['model']['num_encoder_layers'] = 12
    config['model']['num_encoder_heads'] = 16

    # GCNII disabled
    config['model']['use_initial_residual'] = False
    config['model']['use_identity_mapping'] = False

    # Training config
    config['training']['learning_rate'] = 3e-4
    config['training']['batch_size'] = 16

    # Checkpoint directory
    config['checkpoint']['dir'] = f'checkpoints/gcnii_ablation/baseline_run{run_id}'
    config['logging']['log_dir'] = f'logs/gcnii_ablation/baseline_run{run_id}'

    return config


def create_gcnii_config(base_config_path: str, run_id: int) -> Dict[str, Any]:
    """Create configuration for GCNII-HGT."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for GCNII
    config['model']['encoder_type'] = 'hgt'
    config['model']['edge_type_mode'] = 'optimized'
    config['model']['hidden_dim'] = 256
    config['model']['num_encoder_layers'] = 12
    config['model']['num_encoder_heads'] = 16

    # GCNII enabled
    config['model']['use_initial_residual'] = True
    config['model']['use_identity_mapping'] = True
    config['model']['gcnii_alpha'] = GCNII_ALPHA
    config['model']['gcnii_lambda'] = GCNII_LAMBDA

    # Training config
    config['training']['learning_rate'] = 3e-4
    config['training']['batch_size'] = 16

    # Checkpoint directory
    config['checkpoint']['dir'] = f'checkpoints/gcnii_ablation/gcnii_run{run_id}'
    config['logging']['log_dir'] = f'logs/gcnii_ablation/gcnii_run{run_id}'

    return config


def create_model_from_config(config: Dict[str, Any], device: torch.device) -> MBADeobfuscator:
    """Create MBADeobfuscator model from configuration."""
    model_cfg = config['model']

    model = MBADeobfuscator(
        encoder_type=model_cfg['encoder_type'],
        hidden_dim=model_cfg['hidden_dim'],
        num_encoder_layers=model_cfg['num_encoder_layers'],
        num_encoder_heads=model_cfg['num_encoder_heads'],
        encoder_dropout=model_cfg.get('encoder_dropout', 0.1),
        d_model=model_cfg.get('d_model', 512),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 6),
        num_decoder_heads=model_cfg.get('num_decoder_heads', 8),
        d_ff=model_cfg.get('d_ff', 2048),
        decoder_dropout=model_cfg.get('decoder_dropout', 0.1),
        use_initial_residual=model_cfg.get('use_initial_residual', False),
        use_identity_mapping=model_cfg.get('use_identity_mapping', False),
        gcnii_alpha=model_cfg.get('gcnii_alpha', GCNII_ALPHA),
        gcnii_lambda=model_cfg.get('gcnii_lambda', GCNII_LAMBDA),
        edge_type_mode=model_cfg.get('edge_type_mode', 'optimized'),
    )

    return model.to(device)


def train_model(
    config: Dict[str, Any],
    train_path: str,
    val_path: str,
    device: torch.device,
    quick_mode: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Train a single model configuration.

    Args:
        config: Model and training configuration
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        device: Training device
        quick_mode: If True, reduce epochs for fast testing

    Returns:
        Tuple of (checkpoint_path, final_metrics)
    """
    logger.info(f"Training model: {config['checkpoint']['dir']}")

    # Create model
    model = create_model_from_config(config, device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Override curriculum for quick mode
    if quick_mode:
        config['training']['curriculum_stages'] = [
            {'max_depth': 2, 'epochs': 2, 'target': 0.95},
            {'max_depth': 5, 'epochs': 2, 'target': 0.90},
            {'max_depth': 10, 'epochs': 2, 'target': 0.80},
            {'max_depth': 14, 'epochs': 2, 'target': 0.70},
        ]
        logger.info("Quick mode: reduced epochs to 2 per stage")

    # Create trainer
    trainer = Phase2Trainer(
        model=model,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        train_path=train_path,
        val_path=val_path,
        config=config['training'],
        device=device,
        checkpoint_dir=config['checkpoint']['dir'],
    )

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config['logging']['log_dir'])

    # Train
    start_time = time.time()
    history = trainer.fit(
        eval_interval=config['training'].get('eval_interval', 1),
        save_interval=config['training'].get('save_interval', 5),
        metric_for_best='exact_match',
        higher_is_better=True,
    )
    train_time = time.time() - start_time

    trainer.close()

    # Get best checkpoint path
    checkpoint_dir = Path(config['checkpoint']['dir'])
    best_checkpoint = checkpoint_dir / 'phase2_best.pt'

    final_metrics = {
        'train_time_hours': train_time / 3600,
        'final_val_metrics': history['val'][-1] if history['val'] else {},
        'history': history,
    }

    logger.info(f"Training complete. Time: {train_time/3600:.2f}h")
    logger.info(f"Best checkpoint: {best_checkpoint}")

    return str(best_checkpoint), final_metrics


@torch.no_grad()
def evaluate_on_depth_buckets(
    model: MBADeobfuscator,
    test_path: str,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on depth buckets.

    Args:
        model: Trained model
        test_path: Path to test JSONL
        tokenizer: MBATokenizer
        fingerprint: SemanticFingerprint
        device: Device
        batch_size: Batch size

    Returns:
        Dict mapping depth bucket to metrics
    """
    model.eval()

    # Load full test dataset
    test_dataset = MBADataset(
        data_path=test_path,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
    )

    # Group samples by depth bucket
    bucket_samples = defaultdict(list)
    for idx, sample in enumerate(test_dataset.data):
        depth = sample['depth']
        for min_d, max_d in ABLATION_DEPTH_BUCKETS:
            if min_d <= depth <= max_d:
                bucket_samples[f"{min_d}-{max_d}"].append(idx)
                break

    results = {}

    for bucket_name, indices in bucket_samples.items():
        logger.info(f"Evaluating bucket {bucket_name} ({len(indices)} samples)...")

        # Create subset dataset
        subset_data = [test_dataset.data[i] for i in indices]

        # Create temporary dataset
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for sample in subset_data:
                f.write(json.dumps(sample) + '\n')
            temp_path = f.name

        # Load subset
        subset_dataset = MBADataset(
            data_path=temp_path,
            tokenizer=tokenizer,
            fingerprint=fingerprint,
        )

        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graphs,
        )

        # Evaluate
        total = 0
        exact_matches = 0
        syntax_valid = 0

        for batch in tqdm(subset_loader, desc=f"Bucket {bucket_name}", leave=False):
            graph_batch = batch['graph_batch'].to(device)
            fingerprint_batch = batch['fingerprint'].to(device)
            targets = batch['simplified']

            # Greedy decode
            predictions = greedy_decode_batch(
                model, graph_batch, fingerprint_batch, tokenizer, device
            )

            for pred, target in zip(predictions, targets):
                total += 1

                # Exact match
                pred_norm = pred.replace(' ', '').lower()
                tgt_norm = target.replace(' ', '').lower()
                if pred_norm == tgt_norm:
                    exact_matches += 1

                # Syntax validity
                try:
                    tokens = tokenizer.encode(pred)
                    if len(tokens) > 2:
                        syntax_valid += 1
                except Exception:
                    pass

        results[bucket_name] = {
            'total': total,
            'exact_match': exact_matches / max(total, 1),
            'syntax_valid': syntax_valid / max(total, 1),
        }

        logger.info(
            f"Bucket {bucket_name}: "
            f"exact_match={results[bucket_name]['exact_match']:.4f}, "
            f"syntax_valid={results[bucket_name]['syntax_valid']:.4f}"
        )

        # Clean up temp file
        Path(temp_path).unlink()

    return results


def greedy_decode_batch(
    model: MBADeobfuscator,
    graph_batch,
    fingerprint: torch.Tensor,
    tokenizer: MBATokenizer,
    device: torch.device,
    max_len: int = 64,
) -> List[str]:
    """Greedy decode a batch of expressions."""
    from src.constants import SOS_IDX, EOS_IDX

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


def load_checkpoint_and_evaluate(
    checkpoint_path: str,
    test_path: str,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Load checkpoint and evaluate on depth buckets."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model
    model_cfg = config if 'model' not in config else config['model']
    model = MBADeobfuscator(
        encoder_type=model_cfg.get('encoder_type', 'hgt'),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_encoder_layers=model_cfg.get('num_encoder_layers', 12),
        num_encoder_heads=model_cfg.get('num_encoder_heads', 16),
        d_model=model_cfg.get('d_model', 512),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 6),
        num_decoder_heads=model_cfg.get('num_decoder_heads', 8),
        use_initial_residual=model_cfg.get('use_initial_residual', False),
        use_identity_mapping=model_cfg.get('use_identity_mapping', False),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Evaluate
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    results = evaluate_on_depth_buckets(
        model, test_path, tokenizer, fingerprint, device
    )

    return results


def compare_results(
    baseline_results: Dict[str, Dict[str, float]],
    gcnii_results: Dict[str, Dict[str, float]],
) -> None:
    """Print comparison table of baseline vs GCNII results."""
    print("\n" + "=" * 80)
    print("GCNII ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\n{'Depth Bucket':<15} | {'Baseline Acc':<15} | {'GCNII Acc':<15} | {'Improvement':<15}")
    print("-" * 80)

    for bucket in sorted(baseline_results.keys(), key=lambda x: int(x.split('-')[0])):
        baseline_acc = baseline_results[bucket]['exact_match']
        gcnii_acc = gcnii_results[bucket]['exact_match']
        improvement = gcnii_acc - baseline_acc

        print(
            f"{bucket:<15} | "
            f"{baseline_acc:>14.4f} | "
            f"{gcnii_acc:>14.4f} | "
            f"{improvement:>+14.4f}"
        )

    print("=" * 80)

    # Summary statistics
    all_baseline = [r['exact_match'] for r in baseline_results.values()]
    all_gcnii = [r['exact_match'] for r in gcnii_results.values()]

    print(f"\nOverall Average:")
    print(f"  Baseline: {np.mean(all_baseline):.4f}")
    print(f"  GCNII:    {np.mean(all_gcnii):.4f}")
    print(f"  Improvement: {np.mean(all_gcnii) - np.mean(all_baseline):+.4f}")

    # Deep bucket (11-14) analysis
    deep_bucket = '11-14'
    if deep_bucket in baseline_results:
        print(f"\nDeep Expressions (depth {deep_bucket}):")
        baseline_deep = baseline_results[deep_bucket]['exact_match']
        gcnii_deep = gcnii_results[deep_bucket]['exact_match']
        improvement_deep = gcnii_deep - baseline_deep
        print(f"  Baseline: {baseline_deep:.4f}")
        print(f"  GCNII:    {gcnii_deep:.4f}")
        print(f"  Improvement: {improvement_deep:+.4f} ({improvement_deep/baseline_deep*100:+.1f}%)")


def generate_small_dataset(output_dir: str, samples_per_depth: int = 100):
    """
    Generate small synthetic dataset for quick validation.

    Creates train/val/test splits with balanced depth distribution.
    """
    logger.info(f"Generating synthetic dataset: {samples_per_depth} samples per depth bucket")

    import random
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_data import generate_mba_pair

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define depth ranges
    depths = list(range(2, 15))

    # Generate samples
    all_samples = []
    for target_depth in depths:
        for _ in range(samples_per_depth):
            # Generate MBA pair
            obfuscated, simplified, actual_depth = generate_mba_pair(target_depth)

            all_samples.append({
                'obfuscated': obfuscated,
                'simplified': simplified,
                'depth': actual_depth,
            })

    # Shuffle
    random.shuffle(all_samples)

    # Split: 70% train, 15% val, 15% test
    n = len(all_samples)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        'train.jsonl': all_samples[:train_end],
        'val.jsonl': all_samples[train_end:val_end],
        'test.jsonl': all_samples[val_end:],
    }

    for filename, samples in splits.items():
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Written {len(samples)} samples to {filepath}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='GCNII Ablation Study')
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['baseline', 'gcnii', 'full', 'evaluate', 'generate-data'],
        help='Execution mode'
    )
    parser.add_argument(
        '--run-id', type=int, default=1,
        help='Run ID for checkpoint directories'
    )
    parser.add_argument(
        '--num-trials', type=int, default=1,
        help='Number of trials for full mode'
    )
    parser.add_argument(
        '--base-config', type=str, default='configs/phase2.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--train-data', type=str, default='data/train.jsonl',
        help='Training data path'
    )
    parser.add_argument(
        '--val-data', type=str, default='data/val.jsonl',
        help='Validation data path'
    )
    parser.add_argument(
        '--test-data', type=str, default='data/test.jsonl',
        help='Test data path'
    )
    parser.add_argument(
        '--baseline-ckpt', type=str, default=None,
        help='Baseline checkpoint for evaluation mode'
    )
    parser.add_argument(
        '--gcnii-ckpt', type=str, default=None,
        help='GCNII checkpoint for evaluation mode'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--quick-mode', action='store_true',
        help='Quick mode: 2 epochs per stage for testing'
    )
    parser.add_argument(
        '--generate-samples', type=int, default=100,
        help='Samples per depth for synthetic data generation'
    )
    parser.add_argument(
        '--output', type=str, default='results/gcnii_ablation.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Execute mode
    if args.mode == 'generate-data':
        logger.info("Generating synthetic dataset...")
        data_dir = generate_small_dataset('data', args.generate_samples)
        logger.info(f"Dataset generated in {data_dir}")
        return

    elif args.mode == 'baseline':
        logger.info(f"Training baseline HGT (run {args.run_id})...")
        config = create_baseline_config(args.base_config, args.run_id)
        checkpoint_path, metrics = train_model(
            config, args.train_data, args.val_data, device, args.quick_mode
        )
        logger.info(f"Baseline training complete: {checkpoint_path}")

        # Evaluate on test set
        results = load_checkpoint_and_evaluate(checkpoint_path, args.test_data, device)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'run_id': args.run_id,
                'mode': 'baseline',
                'checkpoint': checkpoint_path,
                'training_metrics': metrics,
                'test_results': results,
            }, f, indent=2)

    elif args.mode == 'gcnii':
        logger.info(f"Training GCNII-HGT (run {args.run_id})...")
        config = create_gcnii_config(args.base_config, args.run_id)
        checkpoint_path, metrics = train_model(
            config, args.train_data, args.val_data, device, args.quick_mode
        )
        logger.info(f"GCNII training complete: {checkpoint_path}")

        # Evaluate on test set
        results = load_checkpoint_and_evaluate(checkpoint_path, args.test_data, device)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'run_id': args.run_id,
                'mode': 'gcnii',
                'checkpoint': checkpoint_path,
                'training_metrics': metrics,
                'test_results': results,
            }, f, indent=2)

    elif args.mode == 'full':
        logger.info(f"Running full ablation study ({args.num_trials} trials)...")

        all_results = {
            'baseline': [],
            'gcnii': [],
        }

        for trial in range(1, args.num_trials + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRIAL {trial}/{args.num_trials}")
            logger.info(f"{'='*60}\n")

            # Train baseline
            logger.info(f"Training baseline (trial {trial})...")
            baseline_config = create_baseline_config(args.base_config, trial)
            baseline_ckpt, baseline_metrics = train_model(
                baseline_config, args.train_data, args.val_data, device, args.quick_mode
            )
            baseline_results = load_checkpoint_and_evaluate(
                baseline_ckpt, args.test_data, device
            )
            all_results['baseline'].append({
                'trial': trial,
                'checkpoint': baseline_ckpt,
                'metrics': baseline_metrics,
                'test_results': baseline_results,
            })

            # Train GCNII
            logger.info(f"Training GCNII (trial {trial})...")
            gcnii_config = create_gcnii_config(args.base_config, trial)
            gcnii_ckpt, gcnii_metrics = train_model(
                gcnii_config, args.train_data, args.val_data, device, args.quick_mode
            )
            gcnii_results = load_checkpoint_and_evaluate(
                gcnii_ckpt, args.test_data, device
            )
            all_results['gcnii'].append({
                'trial': trial,
                'checkpoint': gcnii_ckpt,
                'metrics': gcnii_metrics,
                'test_results': gcnii_results,
            })

            # Compare trial results
            logger.info(f"\nTrial {trial} Results:")
            compare_results(baseline_results, gcnii_results)

        # Aggregate across trials
        logger.info("\n" + "="*80)
        logger.info(f"AGGREGATE RESULTS ({args.num_trials} trials)")
        logger.info("="*80)

        # Compute means and std devs
        baseline_agg = aggregate_results([r['test_results'] for r in all_results['baseline']])
        gcnii_agg = aggregate_results([r['test_results'] for r in all_results['gcnii']])

        print_aggregate_comparison(baseline_agg, gcnii_agg)

        # Save all results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'num_trials': args.num_trials,
                'all_results': all_results,
                'aggregate': {
                    'baseline': baseline_agg,
                    'gcnii': gcnii_agg,
                },
            }, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    elif args.mode == 'evaluate':
        if not args.baseline_ckpt or not args.gcnii_ckpt:
            logger.error("--baseline-ckpt and --gcnii-ckpt required for evaluate mode")
            return

        logger.info("Evaluating saved checkpoints...")
        baseline_results = load_checkpoint_and_evaluate(
            args.baseline_ckpt, args.test_data, device
        )
        gcnii_results = load_checkpoint_and_evaluate(
            args.gcnii_ckpt, args.test_data, device
        )

        compare_results(baseline_results, gcnii_results)

        # Save comparison
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'baseline_checkpoint': args.baseline_ckpt,
                'gcnii_checkpoint': args.gcnii_ckpt,
                'baseline_results': baseline_results,
                'gcnii_results': gcnii_results,
            }, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")


def aggregate_results(results_list: List[Dict]) -> Dict:
    """Aggregate results across multiple trials."""
    # Collect metrics by bucket
    bucket_metrics = defaultdict(lambda: defaultdict(list))

    for results in results_list:
        for bucket, metrics in results.items():
            for metric_name, value in metrics.items():
                if metric_name != 'total':
                    bucket_metrics[bucket][metric_name].append(value)

    # Compute mean and std
    aggregated = {}
    for bucket, metrics in bucket_metrics.items():
        aggregated[bucket] = {}
        for metric_name, values in metrics.items():
            aggregated[bucket][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }

    return aggregated


def print_aggregate_comparison(baseline_agg: Dict, gcnii_agg: Dict) -> None:
    """Print aggregate comparison with confidence intervals."""
    print(f"\n{'Depth Bucket':<15} | {'Baseline (mean±std)':<25} | {'GCNII (mean±std)':<25} | {'Improvement':<15}")
    print("-" * 95)

    for bucket in sorted(baseline_agg.keys(), key=lambda x: int(x.split('-')[0])):
        baseline_acc = baseline_agg[bucket]['exact_match']
        gcnii_acc = gcnii_agg[bucket]['exact_match']

        baseline_str = f"{baseline_acc['mean']:.4f} ± {baseline_acc['std']:.4f}"
        gcnii_str = f"{gcnii_acc['mean']:.4f} ± {gcnii_acc['std']:.4f}"
        improvement = gcnii_acc['mean'] - baseline_acc['mean']

        print(
            f"{bucket:<15} | "
            f"{baseline_str:<25} | "
            f"{gcnii_str:<25} | "
            f"{improvement:>+14.4f}"
        )

    print("=" * 95)


if __name__ == '__main__':
    main()
