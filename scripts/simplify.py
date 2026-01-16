#!/usr/bin/env python3
"""
CLI tool for simplifying MBA expressions.

Usage:
    python scripts/simplify.py --expr "(x & y) + (x ^ y)"
    python scripts/simplify.py --expr "(x & y) + (x ^ y)" --mode beam
    python scripts/simplify.py --expr "(x & y) + (x ^ y)" --mode htps --verify
    python scripts/simplify.py --file expressions.txt --output results.jsonl

Modes:
    - greedy: Fast greedy decoding (default)
    - beam: Beam search with diversity
    - htps: HyperTree Proof Search for hard expressions
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.ast_parser import expr_to_graph
from src.inference.pipeline import InferencePipeline
from src.inference.beam_search import BeamSearchDecoder
from src.inference.htps import MinimalHTPS
from src.inference.verify import ThreeTierVerifier
from src.constants import SOS_IDX, EOS_IDX, MAX_SEQ_LEN

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> MBADeobfuscator:
    """Load model from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return model


@torch.no_grad()
def simplify_greedy(
    model: MBADeobfuscator,
    expr: str,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: torch.device,
) -> str:
    """Simplify using greedy decoding."""
    # Build graph
    graph = expr_to_graph(expr)
    graph = graph.to(device)

    # Compute fingerprint
    fp = fingerprint.compute(expr)
    fp = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0)

    # Encode
    memory = model.encode(graph, fp)

    # Greedy decode
    output = torch.full((1, 1), SOS_IDX, dtype=torch.long, device=device)

    for _ in range(MAX_SEQ_LEN - 1):
        decoder_out = model.decode(output, memory)

        if isinstance(decoder_out, dict):
            logits = decoder_out['vocab_logits'][:, -1, :]
        else:
            logits = model.vocab_head(decoder_out[:, -1, :])

        next_token = logits.argmax(dim=-1, keepdim=True)
        output = torch.cat([output, next_token], dim=1)

        if next_token.item() == EOS_IDX:
            break

    return tokenizer.decode(output[0].tolist())


def simplify_beam(
    model: MBADeobfuscator,
    expr: str,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: torch.device,
    beam_width: int = 10,
    num_candidates: int = 5,
) -> list:
    """Simplify using beam search."""
    decoder = BeamSearchDecoder(
        model=model,
        tokenizer=tokenizer,
        beam_width=beam_width,
    )

    # Build graph
    graph = expr_to_graph(expr)
    graph = graph.to(device)

    # Compute fingerprint
    fp = fingerprint.compute(expr)
    fp = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0)

    # Encode
    memory = model.encode(graph, fp)

    # Beam search
    candidates = decoder.decode(memory, num_return=num_candidates)

    return [tokenizer.decode(c) for c in candidates]


def simplify_htps(
    model: MBADeobfuscator,
    expr: str,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: torch.device,
    budget: int = 500,
) -> Optional[str]:
    """Simplify using HyperTree Proof Search."""
    htps = MinimalHTPS(
        model=model,
        tokenizer=tokenizer,
        budget=budget,
    )

    result = htps.search(expr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Simplify MBA expressions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single expression with greedy decode
    python scripts/simplify.py --expr "(x & y) + (x ^ y)"

    # With beam search and verification
    python scripts/simplify.py --expr "(x & y) + (x ^ y)" --mode beam --verify

    # Process file
    python scripts/simplify.py --file exprs.txt --output results.jsonl
        """
    )

    parser.add_argument(
        '--expr', '-e', type=str,
        help='Expression to simplify'
    )
    parser.add_argument(
        '--file', '-f', type=str,
        help='File with expressions (one per line)'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output file for results (JSONL)'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, default='checkpoints/phase2/phase2_best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--mode', '-m', type=str, default='greedy',
        choices=['greedy', 'beam', 'htps'],
        help='Decoding mode'
    )
    parser.add_argument(
        '--beam-width', type=int, default=10,
        help='Beam width for beam search'
    )
    parser.add_argument(
        '--htps-budget', type=int, default=500,
        help='Search budget for HTPS'
    )
    parser.add_argument(
        '--verify', '-v', action='store_true',
        help='Verify equivalence of result'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    if not args.expr and not args.file:
        parser.error("Must provide --expr or --file")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.quiet:
        logger.info(f"Using device: {device}")

    # Load model
    if not args.quiet:
        logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    # Create tokenizer and fingerprint
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Create verifier if needed
    verifier = ThreeTierVerifier(tokenizer) if args.verify else None

    # Process expressions
    expressions = []
    if args.expr:
        expressions = [args.expr]
    elif args.file:
        with open(args.file, 'r') as f:
            expressions = [line.strip() for line in f if line.strip()]

    results = []

    for expr in expressions:
        start_time = time.time()

        # Simplify
        if args.mode == 'greedy':
            simplified = simplify_greedy(model, expr, tokenizer, fingerprint, device)
            candidates = [simplified]
        elif args.mode == 'beam':
            candidates = simplify_beam(
                model, expr, tokenizer, fingerprint, device,
                beam_width=args.beam_width
            )
            simplified = candidates[0] if candidates else expr
        elif args.mode == 'htps':
            simplified = simplify_htps(
                model, expr, tokenizer, fingerprint, device,
                budget=args.htps_budget
            )
            if simplified is None:
                simplified = expr
            candidates = [simplified]

        elapsed = time.time() - start_time

        # Verify if requested
        verified = None
        if args.verify and verifier:
            results_v = verifier.verify_batch(expr, [simplified])
            if results_v:
                verified = results_v[0].exec_valid or results_v[0].z3_verified

        result = {
            'input': expr,
            'simplified': simplified,
            'mode': args.mode,
            'elapsed_ms': elapsed * 1000,
        }

        if args.mode == 'beam':
            result['candidates'] = candidates

        if verified is not None:
            result['verified'] = verified

        results.append(result)

        # Print result
        if not args.quiet:
            print(f"\nInput:      {expr}")
            print(f"Simplified: {simplified}")
            if args.mode == 'beam' and len(candidates) > 1:
                print(f"Candidates: {candidates[:5]}")
            if verified is not None:
                print(f"Verified:   {'✓' if verified else '✗'}")
            print(f"Time:       {elapsed*1000:.2f} ms")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        if not args.quiet:
            logger.info(f"\nSaved {len(results)} results to {args.output}")


if __name__ == '__main__':
    main()
