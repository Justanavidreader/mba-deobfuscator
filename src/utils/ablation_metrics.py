"""
Metrics collection for ablation study.

Addresses issues from quality review:
- Validates list lengths before zip() to prevent silent truncation
- Handles edge cases in metric computation
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EncoderMetrics:
    """Per-encoder metrics for ablation comparison."""

    encoder_name: str
    run_id: int
    depth_bucket: str  # "2-4", "5-7", etc.

    # Accuracy metrics
    exact_match: float  # Token-level exact match
    z3_equivalence: float  # Z3 equivalence rate
    syntax_valid: float  # Syntactically valid outputs

    # Simplification quality
    simplification_ratio: float  # avg(output_len / input_len)
    avg_output_depth: float
    identity_rate: float  # How often output == input

    # Performance
    inference_latency_ms: float  # Mean per-sample latency
    parameter_count: int
    training_time_hours: float

    # Sample counts
    num_samples: int
    num_correct: int


class AblationMetricsCollector:
    """Collects and aggregates metrics across depth buckets."""

    def __init__(self, depth_buckets: List[Tuple[int, int]]):
        """
        Initialize collector.

        Args:
            depth_buckets: List of (min_depth, max_depth) tuples
        """
        self.depth_buckets = depth_buckets
        self.results: List[EncoderMetrics] = []

    def bucket_name(self, depth: int) -> str:
        """Map depth to bucket name."""
        for min_d, max_d in self.depth_buckets:
            if min_d <= depth <= max_d:
                return f"{min_d}-{max_d}"
        return "unknown"

    def collect(
        self,
        encoder_name: str,
        run_id: int,
        predictions: List[str],
        targets: List[str],
        inputs: List[str],
        depths: List[int],
        latencies: List[float],
        encoder_params: int,
        training_time_hours: float,
    ) -> None:
        """
        Collect metrics from evaluation run.

        Args:
            encoder_name: Encoder architecture name
            run_id: Run number (1-5)
            predictions: Model outputs
            targets: Ground truth simplified expressions
            inputs: Input obfuscated expressions
            depths: Expression depth for each sample
            latencies: Per-sample inference latency (seconds)
            encoder_params: Total encoder parameters
            training_time_hours: Total training time

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        # Critical fix: validate list lengths before processing
        lengths = {
            "predictions": len(predictions),
            "targets": len(targets),
            "inputs": len(inputs),
            "depths": len(depths),
            "latencies": len(latencies),
        }

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"Input lists have mismatched lengths: {lengths}. "
                "All lists must have the same number of elements."
            )

        num_samples = len(predictions)
        if num_samples == 0:
            logger.warning(
                f"No samples provided for {encoder_name} run {run_id}. Skipping."
            )
            return

        # Initialize bucket data
        bucket_data: Dict[str, Dict[str, List]] = {}
        for min_d, max_d in self.depth_buckets:
            bucket = f"{min_d}-{max_d}"
            bucket_data[bucket] = {
                "preds": [],
                "targets": [],
                "inputs": [],
                "latencies": [],
                "depths": [],
            }

        # Group by depth bucket
        for i in range(num_samples):
            pred = predictions[i]
            target = targets[i]
            inp = inputs[i]
            depth = depths[i]
            lat = latencies[i]

            bucket = self.bucket_name(depth)
            if bucket not in bucket_data:
                # Create bucket for unknown depths
                bucket_data[bucket] = {
                    "preds": [],
                    "targets": [],
                    "inputs": [],
                    "latencies": [],
                    "depths": [],
                }

            bucket_data[bucket]["preds"].append(pred)
            bucket_data[bucket]["targets"].append(target)
            bucket_data[bucket]["inputs"].append(inp)
            bucket_data[bucket]["latencies"].append(lat)
            bucket_data[bucket]["depths"].append(depth)

        # Compute metrics per bucket
        for bucket, data in bucket_data.items():
            if not data["preds"]:
                continue

            metrics = self._compute_bucket_metrics(
                encoder_name=encoder_name,
                run_id=run_id,
                bucket=bucket,
                preds=data["preds"],
                targets=data["targets"],
                inputs=data["inputs"],
                latencies=data["latencies"],
                encoder_params=encoder_params,
                training_time_hours=training_time_hours,
            )
            self.results.append(metrics)

    def _compute_bucket_metrics(
        self,
        encoder_name: str,
        run_id: int,
        bucket: str,
        preds: List[str],
        targets: List[str],
        inputs: List[str],
        latencies: List[float],
        encoder_params: int,
        training_time_hours: float,
    ) -> EncoderMetrics:
        """Compute metrics for a single depth bucket."""
        num_samples = len(preds)
        exact_match_count = 0
        z3_equiv_count = 0
        syntax_valid_count = 0
        identity_count = 0
        simplification_ratios: List[float] = []
        output_depths: List[int] = []

        # Lazy import to avoid circular dependencies
        try:
            from src.utils.expr_eval import compute_depth, parse_expr
            from src.utils.z3_interface import verify_equivalence

            has_z3 = True
        except ImportError:
            has_z3 = False
            logger.warning("Z3 or expr_eval not available, skipping equivalence checks")

        for i in range(num_samples):
            pred = preds[i]
            target = targets[i]
            inp = inputs[i]

            # Exact match
            if pred.strip() == target.strip():
                exact_match_count += 1

            # Identity check
            if pred.strip() == inp.strip():
                identity_count += 1

            # Syntax validity and Z3 equivalence
            if has_z3:
                try:
                    pred_ast = parse_expr(pred)
                    syntax_valid_count += 1

                    # Z3 equivalence (only if syntax valid)
                    try:
                        if verify_equivalence(pred, target, timeout_ms=1000):
                            z3_equiv_count += 1
                    except Exception as e:
                        logger.debug(f"Z3 verification failed: {e}")

                    # Simplification ratio
                    input_len = max(len(inp), 1)
                    ratio = len(pred) / input_len
                    simplification_ratios.append(ratio)

                    # Output depth
                    try:
                        output_depths.append(compute_depth(pred_ast))
                    except Exception:
                        pass

                except Exception:
                    # Syntax error
                    pass
            else:
                # Fallback: assume valid and use string length ratio
                syntax_valid_count += 1
                input_len = max(len(inp), 1)
                simplification_ratios.append(len(pred) / input_len)

        # Compute averages safely
        avg_simp_ratio = float(np.mean(simplification_ratios)) if simplification_ratios else 0.0
        avg_depth = float(np.mean(output_depths)) if output_depths else 0.0
        avg_latency = float(np.mean(latencies)) * 1000 if latencies else 0.0  # Convert to ms

        return EncoderMetrics(
            encoder_name=encoder_name,
            run_id=run_id,
            depth_bucket=bucket,
            exact_match=exact_match_count / num_samples,
            z3_equivalence=z3_equiv_count / num_samples,
            syntax_valid=syntax_valid_count / num_samples,
            simplification_ratio=avg_simp_ratio,
            avg_output_depth=avg_depth,
            identity_rate=identity_count / num_samples,
            inference_latency_ms=avg_latency,
            parameter_count=encoder_params,
            training_time_hours=training_time_hours,
            num_samples=num_samples,
            num_correct=z3_equiv_count,
        )

    def save_results(self, output_path: str) -> None:
        """Save results to JSON."""
        results_dict = [asdict(r) for r in self.results]
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Saved {len(self.results)} metric records to {output_path}")

    def load_results(self, input_path: str) -> None:
        """Load results from JSON."""
        with open(input_path, "r") as f:
            results_dict = json.load(f)

        self.results = [EncoderMetrics(**r) for r in results_dict]
        logger.info(f"Loaded {len(self.results)} metric records from {input_path}")

    def aggregate_by_encoder(self) -> Dict[str, Dict[str, Dict]]:
        """
        Aggregate metrics across runs for each encoder.

        Returns:
            {encoder_bucket_key: {metric_name: {mean, std, runs: [values]}}}
        """
        from collections import defaultdict

        encoder_results: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for result in self.results:
            enc = result.encoder_name
            bucket = result.depth_bucket
            key = f"{enc}_{bucket}"

            for field in [
                "exact_match",
                "z3_equivalence",
                "simplification_ratio",
                "inference_latency_ms",
                "identity_rate",
            ]:
                value = getattr(result, field)
                encoder_results[key][field].append(value)

        # Compute mean/std
        aggregated: Dict[str, Dict[str, Dict]] = {}
        for key, metrics in encoder_results.items():
            aggregated[key] = {}
            for metric, values in metrics.items():
                if values:
                    aggregated[key][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "runs": values,
                    }
                else:
                    aggregated[key][metric] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "runs": [],
                    }

        return aggregated

    def summary_table(self) -> str:
        """Generate a summary table for all encoders."""
        aggregated = self.aggregate_by_encoder()

        lines = ["Encoder Ablation Study Results", "=" * 60]
        lines.append(
            f"{'Encoder_Bucket':<30} {'Z3 Acc':<12} {'Exact':<12} {'Simp Ratio':<12}"
        )
        lines.append("-" * 60)

        for key in sorted(aggregated.keys()):
            metrics = aggregated[key]
            z3_acc = metrics.get("z3_equivalence", {})
            exact = metrics.get("exact_match", {})
            simp = metrics.get("simplification_ratio", {})

            z3_str = f"{z3_acc.get('mean', 0):.3f}±{z3_acc.get('std', 0):.3f}"
            exact_str = f"{exact.get('mean', 0):.3f}±{exact.get('std', 0):.3f}"
            simp_str = f"{simp.get('mean', 0):.3f}±{simp.get('std', 0):.3f}"

            lines.append(f"{key:<30} {z3_str:<12} {exact_str:<12} {simp_str:<12}")

        return "\n".join(lines)
