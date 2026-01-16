#!/usr/bin/env python3
"""
Main training script for MBA Deobfuscator.

Supports all training phases:
- Phase 1: Contrastive pretraining (InfoNCE + MaskLM)
- Phase 1b: GMN training with frozen encoder
- Phase 1c: End-to-end GMN fine-tuning
- Phase 2: Supervised learning with curriculum
- Phase 3: RL fine-tuning with PPO

Usage:
    python scripts/train.py --phase 1 --config configs/phase1.yaml
    python scripts/train.py --phase 1b --config configs/phase1b_gmn.yaml
    python scripts/train.py --phase 1c --config configs/phase1c_gmn_finetune.yaml
    python scripts/train.py --phase 2 --config configs/phase2.yaml
    python scripts/train.py --phase 3 --config configs/phase3.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.dataset import ContrastiveDataset, MBADataset
from src.data.collate import collate_contrastive, collate_graphs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> MBADeobfuscator:
    """Create MBADeobfuscator model from config."""
    model_cfg = config.get('model', {})

    model = MBADeobfuscator(
        encoder_type=model_cfg.get('encoder_type', 'gat'),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_encoder_layers=model_cfg.get('num_encoder_layers', 4),
        num_encoder_heads=model_cfg.get('num_encoder_heads', 8),
        encoder_dropout=model_cfg.get('encoder_dropout', 0.1),
        d_model=model_cfg.get('d_model', 512),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 6),
        num_decoder_heads=model_cfg.get('num_decoder_heads', 8),
        d_ff=model_cfg.get('d_ff', 2048),
        decoder_dropout=model_cfg.get('decoder_dropout', 0.1),
    )

    return model.to(device)


def train_phase1(config: dict, device: torch.device):
    """Run Phase 1: Contrastive pretraining."""
    from src.training.phase1_trainer import Phase1Trainer
    from torch.utils.data import DataLoader

    logger.info("=== Phase 1: Contrastive Pretraining ===")

    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Load data
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    train_dataset = ContrastiveDataset(
        data_path=data_cfg['train_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=data_cfg.get('max_depth'),
    )

    val_dataset = ContrastiveDataset(
        data_path=data_cfg['val_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=data_cfg.get('max_depth'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 64),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_contrastive,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 64),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_contrastive,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation samples")

    # Create trainer
    checkpoint_cfg = config.get('checkpoint', {})
    trainer = Phase1Trainer(
        model=model,
        config=training_cfg,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase1'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_cfg.get('num_epochs', 20),
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 5),
        metric_for_best='sim_gap',  # Maximize positive-negative similarity gap
        higher_is_better=True,
    )

    trainer.close()
    logger.info("Phase 1 training complete!")

    return history


def _load_gmn_model(config: dict, device: torch.device):
    """Load GMN model from Phase 1a checkpoint."""
    from src.models.gmn import HGTWithGMN, GATWithGMN

    model_cfg = config.get('model', {})
    gmn_type = model_cfg.get('gmn_type', 'hgt_gmn')
    gmn_config = model_cfg.get('gmn_config', {})
    phase1a_path = config.get('checkpoint', {}).get('load_phase1a')

    if not phase1a_path:
        raise ValueError("Phase 1b requires Phase 1a checkpoint (checkpoint.load_phase1a)")

    if gmn_type == 'hgt_gmn':
        model = HGTWithGMN(hgt_checkpoint_path=phase1a_path, gmn_config=gmn_config)
    elif gmn_type == 'gat_gmn':
        model = GATWithGMN(gat_checkpoint_path=phase1a_path, gmn_config=gmn_config)
    else:
        raise ValueError(f"Unknown gmn_type: {gmn_type}")

    return model.to(device)


def _create_negative_sampler(config: dict):
    """Create negative sampler from config."""
    import json
    from src.training.negative_sampler import NegativeSampler

    data_cfg = config.get('data', {})
    sampler_cfg = data_cfg.get('negative_sampler', {})

    with open(data_cfg['train_path'], 'r') as f:
        full_dataset = [json.loads(line) for line in f if line.strip()]

    return NegativeSampler(
        dataset=full_dataset,
        z3_timeout_ms=sampler_cfg.get('z3_timeout_ms', 500),
        cache_size=sampler_cfg.get('cache_size', 10000),
        num_workers=sampler_cfg.get('num_workers', 4),
    )


def train_phase1b(config: dict, device: torch.device):
    """Run Phase 1b: GMN training with frozen encoder."""
    from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer
    from src.data.dataset import GMNDataset
    from src.models.gmn.batch_collator import GMNBatchCollator
    from torch.utils.data import DataLoader

    logger.info("=== Phase 1b: GMN Training (Frozen Encoder) ===")

    # Load model
    model = _load_gmn_model(config, device)
    logger.info(f"GMN model: {sum(p.numel() for p in model.parameters()):,} params, frozen={model.is_encoder_frozen}")

    # Create negative sampler
    negative_sampler = _create_negative_sampler(config)

    # Create datasets
    data_cfg = config.get('data', {})
    sampler_cfg = data_cfg.get('negative_sampler', {})

    train_dataset = GMNDataset(
        data_path=data_cfg['train_path'],
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.0),
        max_depth=data_cfg.get('max_depth'),
    )

    val_dataset = GMNDataset(
        data_path=data_cfg['val_path'],
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.0),
        max_depth=data_cfg.get('max_depth'),
    )

    # Create dataloaders
    training_cfg = config.get('training', {})
    collator = GMNBatchCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 32),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Create trainer
    checkpoint_cfg = config.get('checkpoint', {})

    trainer = Phase1bGMNTrainer(
        model=model,
        config=training_cfg,
        negative_sampler=negative_sampler,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase1b_gmn'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_cfg.get('num_epochs', 15),
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 3),
        metric_for_best='separation_gap',
        higher_is_better=True,
    )

    logger.info(f"Negative sampler stats: {negative_sampler.stats}")
    trainer.close()
    logger.info("Phase 1b GMN training complete!")

    return history


def train_phase1c(config: dict, device: torch.device):
    """Run Phase 1c: End-to-end GMN fine-tuning."""
    from src.training.phase1c_gmn_trainer import Phase1cGMNTrainer
    from src.data.dataset import GMNDataset
    from src.models.gmn.batch_collator import GMNBatchCollator
    from src.models.gmn import HGTWithGMN, GATWithGMN
    from torch.utils.data import DataLoader

    logger.info("=== Phase 1c: End-to-End GMN Fine-Tuning ===")

    # Load Phase 1b checkpoint
    checkpoint_cfg = config.get('checkpoint', {})
    phase1b_path = checkpoint_cfg.get('load_phase1b')

    if not phase1b_path:
        raise ValueError("Phase 1c requires Phase 1b checkpoint (checkpoint.load_phase1b)")

    # Load checkpoint and reconstruct model
    ckpt = torch.load(phase1b_path, map_location=device)

    model_cfg = config.get('model', {})
    gmn_type = model_cfg.get('gmn_type', 'hgt_gmn')
    gmn_config = model_cfg.get('gmn_config', {})

    # For Phase 1c, we need to create the model differently since encoder should be unfrozen
    # We'll load from the checkpoint directly
    if gmn_type == 'hgt_gmn':
        # Create with frozen=False for Phase 1c
        gmn_config_unfrozen = dict(gmn_config)
        gmn_config_unfrozen['freeze_encoder'] = False
        model = HGTWithGMN(hgt_checkpoint_path=None, gmn_config=gmn_config_unfrozen,
                          hgt_encoder=None)
    elif gmn_type == 'gat_gmn':
        gmn_config_unfrozen = dict(gmn_config)
        gmn_config_unfrozen['freeze_encoder'] = False
        model = GATWithGMN(gat_checkpoint_path=None, gmn_config=gmn_config_unfrozen,
                          gat_encoder=None)

    # Actually, simpler approach: load from Phase 1b path which has complete model
    # Re-create using the same path as Phase 1a but with freeze_encoder=False
    phase1a_path = ckpt.get('hgt_checkpoint_path') or checkpoint_cfg.get('load_phase1a')
    gmn_config_unfrozen = dict(gmn_config)
    gmn_config_unfrozen['freeze_encoder'] = False

    if gmn_type == 'hgt_gmn':
        model = HGTWithGMN(hgt_checkpoint_path=phase1a_path, gmn_config=gmn_config_unfrozen)
    else:
        model = GATWithGMN(gat_checkpoint_path=phase1a_path, gmn_config=gmn_config_unfrozen)

    # Load Phase 1b weights
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)

    logger.info(f"GMN model: {sum(p.numel() for p in model.parameters()):,} params")

    # Create negative sampler
    negative_sampler = _create_negative_sampler(config)

    # Create datasets
    data_cfg = config.get('data', {})
    sampler_cfg = data_cfg.get('negative_sampler', {})

    train_dataset = GMNDataset(
        data_path=data_cfg['train_path'],
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.5),
        max_depth=data_cfg.get('max_depth'),
    )

    val_dataset = GMNDataset(
        data_path=data_cfg['val_path'],
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.5),
        max_depth=data_cfg.get('max_depth'),
    )

    # Create dataloaders
    training_cfg = config.get('training', {})
    collator = GMNBatchCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Create trainer
    trainer = Phase1cGMNTrainer(
        model=model,
        config=training_cfg,
        negative_sampler=negative_sampler,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase1c_gmn_finetune'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_cfg.get('num_epochs', 5),
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 1),
        metric_for_best='separation_gap',
        higher_is_better=True,
    )

    logger.info(f"Negative sampler stats: {negative_sampler.stats}")
    trainer.close()
    logger.info("Phase 1c GMN fine-tuning complete!")

    return history


def train_phase2(config: dict, device: torch.device):
    """Run Phase 2: Supervised training with curriculum."""
    from src.training.phase2_trainer import Phase2Trainer

    logger.info("=== Phase 2: Supervised Training with Curriculum ===")

    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load Phase 1 checkpoint if specified
    checkpoint_cfg = config.get('checkpoint', {})
    if checkpoint_cfg.get('load_phase1'):
        logger.info(f"Loading Phase 1 checkpoint: {checkpoint_cfg['load_phase1']}")
        ckpt = torch.load(checkpoint_cfg['load_phase1'], map_location=device)
        # Load only encoder weights
        model_state = model.state_dict()
        pretrained_state = {
            k: v for k, v in ckpt['model_state_dict'].items()
            if k in model_state and 'encoder' in k
        }
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        logger.info(f"Loaded {len(pretrained_state)} pretrained encoder weights")

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Get configs
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    # Create trainer
    trainer = Phase2Trainer(
        model=model,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        train_path=data_cfg['train_path'],
        val_path=data_cfg['val_path'],
        config=training_cfg,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase2'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train with curriculum
    history = trainer.fit(
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 5),
        metric_for_best='exact_match',
        higher_is_better=True,
    )

    trainer.close()
    logger.info("Phase 2 training complete!")

    return history


def train_phase3(config: dict, device: torch.device):
    """Run Phase 3: RL fine-tuning with PPO."""
    from src.training.phase3_trainer import Phase3Trainer
    from torch.utils.data import DataLoader

    logger.info("=== Phase 3: RL Fine-tuning with PPO ===")

    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load Phase 2 checkpoint
    checkpoint_cfg = config.get('checkpoint', {})
    if checkpoint_cfg.get('load_phase2'):
        logger.info(f"Loading Phase 2 checkpoint: {checkpoint_cfg['load_phase2']}")
        ckpt = torch.load(checkpoint_cfg['load_phase2'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded Phase 2 model weights")
    else:
        logger.warning("No Phase 2 checkpoint specified! Training from scratch.")

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Get configs
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    # Create dataset and dataloader
    train_dataset = MBADataset(
        data_path=data_cfg['train_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=data_cfg.get('max_depth'),
    )

    val_dataset = MBADataset(
        data_path=data_cfg['val_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=data_cfg.get('max_depth'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation samples")

    # Create trainer
    trainer = Phase3Trainer(
        model=model,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        config=training_cfg,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase3'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_cfg.get('num_epochs', 10),
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 2),
        metric_for_best='equiv_rate',
        higher_is_better=True,
    )

    trainer.close()
    logger.info("Phase 3 training complete!")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train MBA Deobfuscator')
    parser.add_argument(
        '--phase', type=str, required=True, choices=['1', '1b', '1c', '2', '3'],
        help='Training phase (1=contrastive, 1b=GMN frozen, 1c=GMN finetune, 2=supervised, 3=RL)'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Run appropriate phase
    if args.phase == '1':
        train_phase1(config, device)
    elif args.phase == '1b':
        train_phase1b(config, device)
    elif args.phase == '1c':
        train_phase1c(config, device)
    elif args.phase == '2':
        train_phase2(config, device)
    elif args.phase == '3':
        train_phase3(config, device)


if __name__ == '__main__':
    main()
