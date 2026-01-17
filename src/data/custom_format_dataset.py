"""
Dataset loader for custom text format with manual fingerprint/AST computation.

Handles files with sections like:
    #obfuscated, groundtruth
    expr1, simplified1
    expr2, simplified2

    #linear,groundtruth,poly
    linear1, gt1, poly1
    linear2, gt2, poly2

Each expression is parsed to AST, converted to graph, and fingerprint is computed.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.ast_parser import expr_to_graph, expr_to_ast_depth
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.constants import USE_DAG_FEATURES, FINGERPRINT_DIM

logger = logging.getLogger(__name__)


class CustomFormatDataset(Dataset):
    """
    Dataset for custom text format with comma-separated expressions.

    File format:
        #section_name, field1, field2, ...
        expr1_field1, expr1_field2, ...
        expr2_field1, expr2_field2, ...

        #another_section
        ...

    Example:
        #obfuscated, groundtruth
        2*(z^(x|(~y|z))) - (~(x&y)&(x^(y^z))), x + y
        -3*~(x|(y|z)) + ~(x|(~y|z)), x + y

        #linear,groundtruth,poly
        -1*y+1*~(x|y)+1*(x&y), 1*~y-1*(x^y), -8*~y*(x&y)+...

    For each row, the first column is treated as "obfuscated" and the second
    column as "simplified" (ground truth). Additional columns are ignored.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        max_depth: Optional[int] = None,
        use_dag_features: bool = USE_DAG_FEATURES,
        edge_type_mode: str = "optimized",
        skip_invalid: bool = True,
    ):
        """
        Initialize dataset from custom text format.

        Args:
            data_path: Path to text file
            tokenizer: MBATokenizer instance
            fingerprint: SemanticFingerprint instance
            max_depth: Filter samples with depth > max_depth (if provided)
            use_dag_features: Compute DAG positional features
            edge_type_mode: "legacy" (6-type) or "optimized" (8-type)
            skip_invalid: Skip samples that fail parsing (vs raising error)
        """
        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.max_depth = max_depth
        self.use_dag_features = use_dag_features
        self.edge_type_mode = edge_type_mode
        self.skip_invalid = skip_invalid

        logger.info(f"Loading custom format dataset from {data_path}")
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples")

    def _load_data(self, data_path: str) -> List[Dict]:
        """
        Load and parse custom text format.

        Returns:
            List of dictionaries with keys: obfuscated, simplified, depth,
                                            graph_data, fingerprint
        """
        data = []
        current_section = None
        header_fields = []
        invalid_count = 0

        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Section header
                if line.startswith('#'):
                    # Parse header: #field1,field2,field3
                    header_text = line[1:].strip()
                    header_fields = [field.strip() for field in header_text.split(',')]
                    current_section = '_'.join(header_fields)
                    logger.debug(f"Entering section: {current_section} with fields {header_fields}")
                    continue

                # Data row
                try:
                    # Parse comma-separated expressions
                    parts = [p.strip() for p in line.split(',')]

                    if len(parts) < 2:
                        logger.warning(
                            f"Line {line_num}: Expected at least 2 fields, got {len(parts)}. Skipping."
                        )
                        invalid_count += 1
                        if not self.skip_invalid:
                            raise ValueError(f"Invalid row at line {line_num}")
                        continue

                    # First column = obfuscated, second = simplified
                    obfuscated_expr = parts[0]
                    simplified_expr = parts[1]

                    # For 3-column format (linear, groundtruth, poly/nonpoly),
                    # use first column as obfuscated
                    # The third column can be stored as metadata if needed

                    # Compute depth
                    depth = expr_to_ast_depth(obfuscated_expr)

                    # Filter by depth if specified
                    if self.max_depth is not None and depth > self.max_depth:
                        continue

                    # Build graph from obfuscated expression
                    graph_data = expr_to_graph(
                        obfuscated_expr,
                        use_dag_features=self.use_dag_features,
                        edge_type_mode=self.edge_type_mode,
                    )

                    # Compute fingerprint for obfuscated expression
                    fp_raw = self.fingerprint.compute(obfuscated_expr)  # 448 dims
                    fp = self._strip_derivatives(fp_raw)  # 416 dims

                    # Store sample
                    sample = {
                        'obfuscated': obfuscated_expr,
                        'simplified': simplified_expr,
                        'depth': depth,
                        'graph_data': graph_data,
                        'fingerprint': fp,
                        'section': current_section,
                    }

                    # Store third column if present (e.g., poly expression)
                    if len(parts) >= 3:
                        sample['additional'] = parts[2]

                    data.append(sample)

                except Exception as e:
                    logger.warning(f"Line {line_num}: Failed to parse '{line[:50]}...': {e}")
                    invalid_count += 1
                    if not self.skip_invalid:
                        raise

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid samples")

        return data

    def _strip_derivatives(self, fp: np.ndarray) -> np.ndarray:
        """
        Strip derivative features from fingerprint (448 -> 416 dims).

        Fingerprint layout:
            0-31: Symbolic (32)
            32-287: Corner (256)
            288-351: Random (64)
            352-383: Derivative (32) <- REMOVED
            384-447: Truth table (64)

        Args:
            fp: Full fingerprint (448 dims)

        Returns:
            Fingerprint without derivatives (416 dims)
        """
        if len(fp) == FINGERPRINT_DIM:
            return fp  # Already stripped

        if len(fp) == 448:
            # Strip derivatives: [0:352] + [384:448]
            return np.concatenate([fp[:352], fp[384:]])

        raise ValueError(f"Invalid fingerprint dimension: {len(fp)}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample at index.

        Returns:
            {
                'obfuscated': str,
                'simplified': str,
                'depth': int,
                'graph_data': Data,  # PyTorch Geometric Data
                'fingerprint': np.ndarray,  # (416,)
                'obfuscated_tokens': torch.Tensor,  # Tokenized obfuscated
                'simplified_tokens': torch.Tensor,  # Tokenized simplified
                'section': str,  # Section name from header
                'additional': str (optional),  # Third column if present
            }
        """
        sample = self.data[idx]

        # Tokenize expressions
        obfuscated_tokens = torch.tensor(
            self.tokenizer.encode(sample['obfuscated']),
            dtype=torch.long
        )
        simplified_tokens = torch.tensor(
            self.tokenizer.encode(sample['simplified']),
            dtype=torch.long
        )

        return {
            'obfuscated': sample['obfuscated'],
            'simplified': sample['simplified'],
            'depth': sample['depth'],
            'graph_data': sample['graph_data'],
            'fingerprint': torch.tensor(sample['fingerprint'], dtype=torch.float32),
            'obfuscated_tokens': obfuscated_tokens,
            'simplified_tokens': simplified_tokens,
            'section': sample['section'],
            'additional': sample.get('additional'),
        }

    def get_depth_distribution(self) -> Dict[int, int]:
        """
        Get distribution of samples by depth.

        Returns:
            {depth: count}
        """
        from collections import Counter
        return dict(Counter(sample['depth'] for sample in self.data))

    def get_section_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples by section.

        Returns:
            {section_name: count}
        """
        from collections import Counter
        return dict(Counter(sample['section'] for sample in self.data))

    def filter_by_section(self, section_name: str) -> 'CustomFormatDataset':
        """
        Create a new dataset with only samples from specified section.

        Args:
            section_name: Section name (e.g., "obfuscated_groundtruth")

        Returns:
            New CustomFormatDataset instance with filtered data
        """
        filtered = CustomFormatDataset.__new__(CustomFormatDataset)
        filtered.tokenizer = self.tokenizer
        filtered.fingerprint = self.fingerprint
        filtered.max_depth = self.max_depth
        filtered.use_dag_features = self.use_dag_features
        filtered.edge_type_mode = self.edge_type_mode
        filtered.skip_invalid = self.skip_invalid

        filtered.data = [s for s in self.data if s.get('section') == section_name]

        logger.info(f"Filtered to {len(filtered.data)} samples from section '{section_name}'")
        return filtered


# Example usage and testing
if __name__ == '__main__':
    # Example: Load dataset
    from src.data.tokenizer import MBATokenizer
    from src.data.fingerprint import SemanticFingerprint

    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    dataset = CustomFormatDataset(
        data_path="path/to/custom_format.txt",
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=14,
    )

    print(f"Loaded {len(dataset)} samples")
    print(f"Depth distribution: {dataset.get_depth_distribution()}")
    print(f"Section distribution: {dataset.get_section_distribution()}")

    # Get first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Obfuscated: {sample['obfuscated'][:50]}...")
    print(f"  Simplified: {sample['simplified']}")
    print(f"  Depth: {sample['depth']}")
    print(f"  Section: {sample['section']}")
    print(f"  Graph nodes: {sample['graph_data'].num_nodes}")
    print(f"  Fingerprint shape: {sample['fingerprint'].shape}")
