"""
PyTorch Dataset classes for MBA deobfuscation.

Provides datasets for supervised and contrastive learning.
Includes ScaledMBADataset with subexpression sharing for 360M model.
Includes GMNDataset for Graph Matching Network training.
"""

import json
import hashlib
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.ast_parser import expr_to_graph
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.augmentation import VariableAugmentationMixin
from src.constants import (
    EDGE_TYPES, VAR_AUGMENT_ENABLED, VAR_AUGMENT_PROB, USE_DAG_FEATURES,
    FINGERPRINT_DIM,
)
from src.models.edge_types import EdgeType, NodeType, NODE_TYPE_MAP, LEGACY_SKIP_TYPES, convert_legacy_node_types

if TYPE_CHECKING:
    from src.training.negative_sampler import NegativeSampler


# =============================================================================
# C++ Generator Compatibility Helpers
# =============================================================================

def _normalize_item(item: Dict) -> Dict:
    """
    Normalize field names from C++ generator format to internal format.

    C++ format: obfuscated_expr, ground_truth_expr, ast_v2
    Internal:   obfuscated, simplified, ast

    Uses falsy check to handle empty strings correctly.
    """
    normalized = item.copy()

    # Expression fields - use falsy check to handle empty strings
    if not normalized.get('obfuscated'):
        normalized['obfuscated'] = normalized.get('obfuscated_expr')
    if not normalized.get('simplified'):
        normalized['simplified'] = normalized.get('ground_truth_expr')

    # AST field (C++ uses ast_v2)
    if 'ast' not in normalized and 'ast_v2' in normalized:
        normalized['ast'] = normalized['ast_v2']

    return normalized


def _validate_precomputed_fingerprint(fp: List[float]) -> np.ndarray:
    """
    Validate pre-computed fingerprint dimension and numeric values.

    Args:
        fp: Pre-computed fingerprint from C++ generator

    Returns:
        Validated fingerprint as numpy array

    Raises:
        ValueError: If dimension mismatch or NaN/inf detected
    """
    if len(fp) != FINGERPRINT_DIM:
        raise ValueError(
            f"Pre-computed fingerprint has {len(fp)} dims, expected {FINGERPRINT_DIM}"
        )

    fp_array = np.array(fp, dtype=np.float32)
    if not np.all(np.isfinite(fp_array)):
        raise ValueError(
            f"Pre-computed fingerprint contains NaN or inf values"
        )

    return fp_array


class MBADataset(Dataset, VariableAugmentationMixin):
    """Dataset for supervised seq2seq training."""

    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        max_depth: Optional[int] = None,
        node_type_schema: Optional[str] = None,
        augment_variables: bool = VAR_AUGMENT_ENABLED,
        augment_prob: float = VAR_AUGMENT_PROB,
        augment_seed: Optional[int] = None,
        use_dag_features: bool = USE_DAG_FEATURES,
    ):
        """
        Load dataset from JSONL file.

        Args:
            data_path: Path to JSONL file with fields: obfuscated, simplified, depth
            tokenizer: MBATokenizer instance
            fingerprint: SemanticFingerprint instance
            max_depth: Optional depth filter for curriculum learning
            node_type_schema: Node type ID format. REQUIRED - must be "legacy" or "current".
                Set to "legacy" for datasets generated before 2026-01-15.
                Set to "current" for datasets with schema_version >= 2.
            augment_variables: Enable variable permutation augmentation
            augment_prob: Probability of applying permutation (0.0-1.0)
            augment_seed: Random seed for reproducibility (None for random)
            use_dag_features: Compute DAG positional features for graphs

        Raises:
            ValueError: If node_type_schema is None or invalid
        """
        # Validate node_type_schema (REQUIRED parameter)
        if node_type_schema is None:
            raise ValueError(
                "node_type_schema is REQUIRED. Specify 'legacy' or 'current'.\n"
                "  - Use 'legacy' for datasets generated before 2026-01-15\n"
                "  - Use 'current' for datasets with schema_version >= 2\n"
                "  - Check dataset JSON for 'schema_version' field to confirm"
            )
        if node_type_schema not in ("legacy", "current"):
            raise ValueError(f"node_type_schema must be 'legacy' or 'current', got: {node_type_schema}")

        self.node_type_schema = node_type_schema
        self._schema_validated = False

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.max_depth = max_depth
        self.use_dag_features = use_dag_features

        # Variable permutation augmentation
        self._init_augmentation(augment_variables, augment_prob, augment_seed)

        # Load data
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and filter data from JSONL file. Supports C++ generator format."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Normalize field names (supports C++ generator format)
                item = _normalize_item(item)

                # Validate required fields
                if not item.get('obfuscated') or not item.get('simplified'):
                    continue

                # Apply depth filter if specified
                if self.max_depth is not None:
                    depth = item.get('depth', 0)
                    if depth > self.max_depth:
                        continue

                data.append(item)

        if not data:
            raise ValueError(f"No valid data loaded from {data_path}")

        return data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def _validate_node_type_schema(self, node_types: torch.Tensor) -> None:
        """
        Validate node type IDs are in expected range after loading first batch.

        Args:
            node_types: [num_nodes] tensor with node type IDs

        Raises:
            ValueError: If node types are out of expected range (indicates schema mismatch)
        """
        if node_types.numel() == 0:
            return

        min_id = node_types.min().item()
        max_id = node_types.max().item()

        # Both legacy and current schemas use IDs in [0-9]
        if min_id < 0 or max_id > 9:
            raise ValueError(
                f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}].\n"
                f"Dataset schema validation failed. Check:\n"
                f"  1. Dataset file is not corrupted\n"
                f"  2. node_type_schema parameter matches dataset format\n"
                f"  3. Dataset JSON contains 'schema_version' field"
            )

        # Distribution check to catch obvious mismatches
        type_counts = torch.bincount(node_types.view(-1), minlength=10)

        if self.node_type_schema == "legacy":
            # Legacy: operators (0-7) should be common, terminals (8-9) for VAR/CONST
            operator_count = type_counts[0:8].sum().item()
            terminal_count = type_counts[8:10].sum().item()
            if operator_count == 0 or terminal_count == 0:
                raise ValueError(
                    f"Unusual node type distribution for legacy schema. "
                    f"Operators (0-7): {operator_count}, Terminals (8-9): {terminal_count}. "
                    f"Verify dataset was generated with legacy schema."
                )
        elif self.node_type_schema == "current":
            # Current: terminals (0-1) for VAR/CONST, operators (2-9) for ops
            terminal_count = type_counts[0:2].sum().item()
            operator_count = type_counts[2:10].sum().item()
            if terminal_count == 0 or operator_count == 0:
                raise ValueError(
                    f"Unusual node type distribution for current schema. "
                    f"Terminals (0-1): {terminal_count}, Operators (2-9): {operator_count}. "
                    f"Verify dataset has schema_version >= 2."
                )

    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item.

        Returns:
            graph: PyG Data object for encoder
            fingerprint: torch.Tensor [FINGERPRINT_DIM]
            target_ids: torch.Tensor [seq_len] target token IDs
            source_tokens: torch.Tensor for copy mechanism
            depth: int for curriculum weighting
            obfuscated: str (original expression)
            simplified: str (target expression)
        """
        item = self.data[idx]

        obfuscated = item['obfuscated']
        simplified = item['simplified']
        depth = item.get('depth', 0)

        # Apply variable permutation BEFORE any processing
        obfuscated, simplified = self._apply_augmentation(obfuscated, simplified)

        # Convert obfuscated expression to graph
        graph = expr_to_graph(obfuscated, use_dag_features=self.use_dag_features)

        # Validate schema on first item
        if not self._schema_validated and hasattr(graph, 'x'):
            node_types = graph.x.argmax(dim=-1) if graph.x.dim() > 1 else graph.x
            self._validate_node_type_schema(node_types)
            self._schema_validated = True

        # Convert node types if legacy schema
        if self.node_type_schema == "legacy" and hasattr(graph, 'x'):
            if graph.x.dim() == 1:
                # Node type IDs directly
                graph.x = convert_legacy_node_types(graph.x)
            elif graph.x.dim() == 2:
                # One-hot encoded - convert via argmax then back
                node_types = graph.x.argmax(dim=-1)
                converted = convert_legacy_node_types(node_types)
                # Reconstruct one-hot
                new_x = torch.zeros_like(graph.x)
                new_x.scatter_(1, converted.unsqueeze(1), 1)
                graph.x = new_x

        # Load pre-computed fingerprint if available AND augmentation disabled.
        # CRITICAL: Pre-computed fingerprints were computed on original expressions.
        # Variable augmentation changes expression semantics, invalidating the fingerprint.
        can_use_precomputed = (
            'fingerprint' in item
            and 'flat' in item['fingerprint']
            and not self.augment_variables
        )

        if can_use_precomputed:
            fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
            fp_tensor = torch.from_numpy(fp_array)
        else:
            fp = self.fingerprint.compute(obfuscated)
            fp_tensor = torch.from_numpy(fp).float()

        # Tokenize target (simplified expression)
        target_ids = self.tokenizer.encode(simplified, add_special=True)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        # Get source tokens for copy mechanism
        source_tokens = self.tokenizer.get_source_tokens(obfuscated)
        source_tensor = torch.tensor(source_tokens, dtype=torch.long)

        return {
            'graph': graph,
            'fingerprint': fp_tensor,
            'target_ids': target_tensor,
            'source_tokens': source_tensor,
            'depth': depth,
            'obfuscated': obfuscated,
            'simplified': simplified,
        }


class ContrastiveDataset(Dataset, VariableAugmentationMixin):
    """Dataset for Phase 1 contrastive pretraining."""

    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        max_depth: Optional[int] = None,
        node_type_schema: Optional[str] = None,
        augment_variables: bool = VAR_AUGMENT_ENABLED,
        augment_prob: float = VAR_AUGMENT_PROB,
        use_dag_features: bool = USE_DAG_FEATURES,
    ):
        """
        Load dataset for contrastive learning.

        Each item contains an obfuscated and simplified expression pair.
        The model learns to map equivalent expressions to similar embeddings.

        For contrastive learning, different permutations are applied to anchor
        and positive views to learn permutation invariance.

        Args:
            data_path: Path to JSONL file
            tokenizer: MBATokenizer instance
            fingerprint: SemanticFingerprint instance
            max_depth: Optional depth filter
            node_type_schema: Node type ID format. REQUIRED - must be "legacy" or "current".
                Set to "legacy" for datasets generated before 2026-01-15.
                Set to "current" for datasets with schema_version >= 2.
            augment_variables: Enable variable permutation augmentation
            augment_prob: Probability of applying permutation
            use_dag_features: Compute DAG positional features for graphs

        Raises:
            ValueError: If node_type_schema is None or invalid
        """
        # Validate node_type_schema (REQUIRED parameter)
        if node_type_schema is None:
            raise ValueError(
                "node_type_schema is REQUIRED. Specify 'legacy' or 'current'.\n"
                "  - Use 'legacy' for datasets generated before 2026-01-15\n"
                "  - Use 'current' for datasets with schema_version >= 2\n"
                "  - Check dataset JSON for 'schema_version' field to confirm"
            )
        if node_type_schema not in ("legacy", "current"):
            raise ValueError(f"node_type_schema must be 'legacy' or 'current', got: {node_type_schema}")

        self.node_type_schema = node_type_schema
        self._schema_validated = False

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.max_depth = max_depth
        self.use_dag_features = use_dag_features

        # Variable permutation - no seed (want different permutations each call)
        self._init_augmentation(augment_variables, augment_prob, augment_seed=None)

        # Load data (same format as supervised)
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and filter data from JSONL file. Supports C++ generator format."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Normalize field names (supports C++ generator format)
                item = _normalize_item(item)

                # Validate required fields
                if not item.get('obfuscated') or not item.get('simplified'):
                    continue

                # Apply depth filter if specified
                if self.max_depth is not None:
                    depth = item.get('depth', 0)
                    if depth > self.max_depth:
                        continue

                data.append(item)

        if not data:
            raise ValueError(f"No valid data loaded from {data_path}")

        return data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def _validate_node_type_schema(self, node_types: torch.Tensor) -> None:
        """Validate node type IDs are in expected range."""
        if node_types.numel() == 0:
            return

        min_id = node_types.min().item()
        max_id = node_types.max().item()

        if min_id < 0 or max_id > 9:
            raise ValueError(
                f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}].\n"
                f"Dataset schema validation failed."
            )

    def _convert_graph_node_types(self, graph: Data) -> Data:
        """Convert graph node types if using legacy schema."""
        if self.node_type_schema == "legacy" and hasattr(graph, 'x'):
            if graph.x.dim() == 1:
                graph.x = convert_legacy_node_types(graph.x)
            elif graph.x.dim() == 2:
                node_types = graph.x.argmax(dim=-1)
                converted = convert_legacy_node_types(node_types)
                new_x = torch.zeros_like(graph.x)
                new_x.scatter_(1, converted.unsqueeze(1), 1)
                graph.x = new_x
        return graph

    def __getitem__(self, idx: int) -> Dict:
        """
        Get contrastive pair.

        For contrastive learning, DIFFERENT permutations are applied to
        anchor and positive views to learn permutation invariance.

        Returns:
            obf_graph: PyG Data for obfuscated expression (anchor)
            simp_graph: PyG Data for simplified expression (positive)
            obf_fingerprint: torch.Tensor [FINGERPRINT_DIM]
            simp_fingerprint: torch.Tensor [FINGERPRINT_DIM]
            label: int (index for positive pair matching)
            obfuscated: str (permuted)
            simplified: str (permuted differently)
        """
        item = self.data[idx]

        obfuscated = item['obfuscated']
        simplified = item['simplified']

        # Apply DIFFERENT permutations to anchor and positive for invariance learning
        # Each call to permuter uses its RNG, producing different results
        if self.permuter is not None:
            obfuscated, _ = self.permuter(obfuscated, obfuscated)
            simplified, _ = self.permuter(simplified, simplified)

        # Convert both expressions to graphs
        obf_graph = expr_to_graph(obfuscated, use_dag_features=self.use_dag_features)
        simp_graph = expr_to_graph(simplified, use_dag_features=self.use_dag_features)

        # Validate schema on first item
        if not self._schema_validated and hasattr(obf_graph, 'x'):
            node_types = obf_graph.x.argmax(dim=-1) if obf_graph.x.dim() > 1 else obf_graph.x
            self._validate_node_type_schema(node_types)
            self._schema_validated = True

        # Convert node types if legacy schema
        obf_graph = self._convert_graph_node_types(obf_graph)
        simp_graph = self._convert_graph_node_types(simp_graph)

        # Load pre-computed fingerprint if available AND augmentation disabled.
        # CRITICAL: Pre-computed fingerprints were computed on original expressions.
        # ContrastiveDataset applies DIFFERENT permutations to anchor/positive,
        # so pre-computed fingerprints are ONLY valid when augmentation is disabled.
        can_use_precomputed = (
            'fingerprint' in item
            and 'flat' in item['fingerprint']
            and not self.augment_variables
        )

        if can_use_precomputed:
            fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
            obf_fp = torch.from_numpy(fp_array)
        else:
            obf_fp = torch.from_numpy(self.fingerprint.compute(obfuscated)).float()

        # Simplified always computed (not pre-stored in C++ format)
        simp_fp = torch.from_numpy(self.fingerprint.compute(simplified)).float()

        return {
            'obf_graph': obf_graph,
            'simp_graph': simp_graph,
            'obf_fingerprint': obf_fp,
            'simp_fingerprint': simp_fp,
            'label': idx,  # Used for positive pair matching in InfoNCE
            'obfuscated': obfuscated,
            'simplified': simplified,
        }


class ScaledMBADataset(Dataset, VariableAugmentationMixin):
    """
    Dataset for scaled model with subexpression sharing and pre-computed features.

    Supports JSON Schema v6 with:
    - boolean_domain_only: Conditioning signal for domain-specific rules
    - fingerprint.flat: Pre-flattened 448-dim vector
    - complexity_score: Pre-computed complexity score
    - ast: Pre-built AST with nodes and edges
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: Optional[SemanticFingerprint] = None,
        max_depth: Optional[int] = None,
        node_type_schema: Optional[str] = None,
        use_subexpr_sharing: bool = True,
        augment_variables: bool = VAR_AUGMENT_ENABLED,
        augment_prob: float = VAR_AUGMENT_PROB,
        augment_seed: Optional[int] = None,
        use_dag_features: bool = USE_DAG_FEATURES,
    ):
        """
        Load dataset for scaled model training.

        Args:
            data_path: Path to JSONL file with v6 schema
            tokenizer: MBATokenizer instance
            fingerprint: Optional SemanticFingerprint (used if pre-computed not available)
            max_depth: Optional depth filter for curriculum learning
            node_type_schema: Node type ID format. REQUIRED - must be "legacy" or "current".
                Set to "legacy" for datasets generated before 2026-01-15.
                Set to "current" for datasets with schema_version >= 2.
            use_subexpr_sharing: Enable subexpression sharing (default True)
            augment_variables: Enable variable permutation augmentation
            augment_prob: Probability of applying permutation (0.0-1.0)
            augment_seed: Random seed for reproducibility (None for random)
            use_dag_features: Compute DAG positional features for graphs

        Raises:
            ValueError: If node_type_schema is None or invalid
        """
        # Validate node_type_schema (REQUIRED parameter)
        if node_type_schema is None:
            raise ValueError(
                "node_type_schema is REQUIRED. Specify 'legacy' or 'current'.\n"
                "  - Use 'legacy' for datasets generated before 2026-01-15\n"
                "  - Use 'current' for datasets with schema_version >= 2\n"
                "  - Check dataset JSON for 'schema_version' field to confirm"
            )
        if node_type_schema not in ("legacy", "current"):
            raise ValueError(f"node_type_schema must be 'legacy' or 'current', got: {node_type_schema}")

        self.node_type_schema = node_type_schema
        self._schema_validated = False

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.max_depth = max_depth
        self.use_subexpr_sharing = use_subexpr_sharing
        self.use_dag_features = use_dag_features

        # Variable permutation augmentation
        self._init_augmentation(augment_variables, augment_prob, augment_seed)

        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and filter data from JSONL file. Supports C++ generator format."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Normalize field names (supports C++ generator format)
                item = _normalize_item(item)

                # Validate required fields
                if not item.get('obfuscated') or not item.get('simplified'):
                    continue

                # Apply depth filter if specified
                if self.max_depth is not None:
                    depth = item.get('depth', 0)
                    if depth > self.max_depth:
                        continue

                data.append(item)

        if not data:
            raise ValueError(f"No valid data loaded from {data_path}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def _is_v2_ast(self, ast_data: Dict) -> bool:
        """Check if AST uses C++ generator v2 format.

        C++ generator MUST set explicit 'version': 2 field.
        Do NOT use uses_subexpr_sharing check - that field indicates whether
        sharing was enabled, not whether the AST uses v2 format.
        """
        return ast_data.get('version') == 2

    def _compute_subtree_hash(self, node_id: int, nodes_by_id: Dict,
                               edges_by_src: Dict) -> str:
        """Compute hash of subtree for subexpression sharing."""
        node = nodes_by_id[node_id]
        node_type = node.get('type', '')
        node_value = node.get('value', '')

        # Collect children (only CHILD_LEFT, CHILD_RIGHT from legacy edge types)
        children = []
        for edge in edges_by_src.get(node_id, []):
            edge_type = edge.get('type', -1)
            if edge_type in [EDGE_TYPES['CHILD_LEFT'], EDGE_TYPES['CHILD_RIGHT']]:
                children.append((edge_type, edge['dst']))
        children.sort()

        # Recursively hash children
        child_hashes = []
        for edge_type, child_id in children:
            child_hash = self._compute_subtree_hash(child_id, nodes_by_id, edges_by_src)
            child_hashes.append(f"{edge_type}:{child_hash}")

        subtree_str = f"{node_type}|{node_value}|{','.join(child_hashes)}"
        # Use full MD5 hash to avoid collision risk
        return hashlib.md5(subtree_str.encode()).hexdigest()

    def _build_optimized_graph(self, ast_data: Dict):
        """
        Convert AST to optimized graph with subexpression sharing.

        Returns:
            nodes: List of node dicts with 'type_id' field
            edges: List of (src, dst, edge_type) tuples
        """
        nodes = ast_data.get('nodes', [])
        edges = ast_data.get('edges', [])

        if not nodes:
            return [], []

        # Build lookup structures
        nodes_by_id = {n['id']: n for n in nodes}
        edges_by_src = defaultdict(list)
        for e in edges:
            edges_by_src[e['src']].append(e)

        # Apply subexpression sharing
        if self.use_subexpr_sharing:
            subtree_hash_to_id = {}
            node_mapping = {}  # old_id -> new_id
            new_nodes = []

            # Process nodes (reverse for bottom-up)
            for node in reversed(nodes):
                node_id = node['id']
                h = self._compute_subtree_hash(node_id, nodes_by_id, edges_by_src)

                if h in subtree_hash_to_id:
                    # Merge with existing node
                    node_mapping[node_id] = subtree_hash_to_id[h]
                else:
                    # New unique subtree
                    new_id = len(new_nodes)
                    subtree_hash_to_id[h] = new_id
                    node_mapping[node_id] = new_id

                    # Add type_id for heterogeneous graph
                    node_copy = node.copy()
                    node_copy['type_id'] = NODE_TYPE_MAP.get(node.get('type', ''), 0)
                    new_nodes.append(node_copy)
        else:
            new_nodes = []
            node_mapping = {}
            for i, node in enumerate(nodes):
                node_copy = node.copy()
                node_copy['type_id'] = NODE_TYPE_MAP.get(node.get('type', ''), 0)
                new_nodes.append(node_copy)
                node_mapping[node['id']] = i

        # Build optimized edges with new types and inverses
        new_edges = []
        seen_edges = set()

        if self._is_v2_ast(ast_data):
            # V2: Edges already include inverses, load directly
            # Track invalid references to catch data corruption
            total_edges = len(edges)
            skipped_edges = 0

            for edge in edges:
                src = node_mapping.get(edge['src'])
                dst = node_mapping.get(edge['dst'])
                if src is None or dst is None:
                    skipped_edges += 1
                    continue
                edge_type = edge.get('type', 0)
                edge_tuple = (src, dst, edge_type)
                if edge_tuple not in seen_edges:
                    new_edges.append(edge_tuple)
                    seen_edges.add(edge_tuple)

            # Validate edge integrity - catch C++ generator bugs
            if total_edges > 0:
                skip_rate = skipped_edges / total_edges
                if skip_rate > 0.1:  # More than 10% invalid
                    raise ValueError(
                        f"AST v2 has {skip_rate*100:.1f}% invalid edge references "
                        f"({skipped_edges}/{total_edges}). Data corruption detected."
                    )
                elif skipped_edges > 0:
                    warnings.warn(
                        f"AST v2 skipped {skipped_edges}/{total_edges} edges with invalid references"
                    )
        else:
            # Legacy: Generate inverse edges
            for edge in edges:
                src = node_mapping.get(edge['src'])
                dst = node_mapping.get(edge['dst'])
                if src is None or dst is None:
                    continue

                edge_type_idx = edge.get('type', -1)

                # Skip legacy types that we're replacing
                if edge_type_idx in LEGACY_SKIP_TYPES:
                    continue

                # Map legacy edge types to optimized types with inverses
                if edge_type_idx == EDGE_TYPES['CHILD_LEFT']:
                    fwd = (src, dst, int(EdgeType.LEFT_OPERAND))
                    inv = (dst, src, int(EdgeType.LEFT_OPERAND_INV))
                elif edge_type_idx == EDGE_TYPES['CHILD_RIGHT']:
                    fwd = (src, dst, int(EdgeType.RIGHT_OPERAND))
                    inv = (dst, src, int(EdgeType.RIGHT_OPERAND_INV))
                else:
                    continue

                if fwd not in seen_edges:
                    new_edges.append(fwd)
                    seen_edges.add(fwd)
                if inv not in seen_edges:
                    new_edges.append(inv)
                    seen_edges.add(inv)

        # Add DOMAIN_BRIDGE edges at bool<->arith boundaries (legacy format only)
        for i, node in enumerate(new_nodes):
            if NodeType.is_boolean(node.get('type_id', -1)):
                for edge in new_edges:
                    if edge[0] == i and edge[2] in [int(EdgeType.LEFT_OPERAND),
                                                     int(EdgeType.RIGHT_OPERAND)]:
                        child_idx = edge[1]
                        if child_idx < len(new_nodes):
                            child_type = new_nodes[child_idx].get('type_id', -1)
                            if NodeType.is_arithmetic(child_type):
                                bridge = (i, child_idx, int(EdgeType.DOMAIN_BRIDGE_DOWN))
                                if bridge not in seen_edges:
                                    new_edges.append(bridge)
                                    seen_edges.add(bridge)

        return new_nodes, new_edges

    def _validate_node_type_schema(self, node_types: torch.Tensor) -> None:
        """Validate node type IDs are in expected range."""
        if node_types.numel() == 0:
            return

        min_id = node_types.min().item()
        max_id = node_types.max().item()

        if min_id < 0 or max_id > 9:
            raise ValueError(
                f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}].\n"
                f"Dataset schema validation failed."
            )

    def _convert_graph_node_types(self, graph: Data) -> Data:
        """Convert graph node types if using legacy schema."""
        if self.node_type_schema == "legacy" and hasattr(graph, 'x'):
            if graph.x.dim() == 1:
                graph.x = convert_legacy_node_types(graph.x)
            elif graph.x.dim() == 2:
                node_types = graph.x.argmax(dim=-1)
                converted = convert_legacy_node_types(node_types)
                new_x = torch.zeros_like(graph.x)
                new_x.scatter_(1, converted.unsqueeze(1), 1)
                graph.x = new_x
        return graph

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item with optimized graph and pre-computed features."""
        item = self.data[idx]

        obfuscated = item['obfuscated']
        simplified = item['simplified']
        depth = item.get('depth', 0)

        # Apply variable permutation BEFORE any processing
        # Note: This invalidates pre-computed AST, so we rebuild the graph
        obfuscated, simplified = self._apply_augmentation(obfuscated, simplified)

        # Pre-computed features from JSON schema v6
        complexity_score = item.get('complexity_score', 0.0)
        boolean_domain_only = item.get('boolean_domain_only', False)

        # Build graph from pre-computed AST if available
        # Note: If augmentation changed variable names, pre-computed AST is stale
        # We detect this by checking if augmentation is enabled
        if 'ast' in item and not self.augment_variables:
            nodes, edges = self._build_optimized_graph(item['ast'])

            if nodes:
                node_types = [n.get('type_id', 0) for n in nodes]
                node_tensor = torch.tensor(node_types, dtype=torch.long)

                if edges:
                    edge_src = [e[0] for e in edges]
                    edge_dst = [e[1] for e in edges]
                    edge_types = [e[2] for e in edges]
                    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                    edge_type = torch.tensor(edge_types, dtype=torch.long)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_type = torch.zeros(0, dtype=torch.long)

                from torch_geometric.data import Data
                graph = Data(x=node_tensor, edge_index=edge_index, edge_type=edge_type)

                # Compute DAG features for pre-built graph
                if self.use_dag_features:
                    from src.data.dag_features import compute_dag_positional_features
                    dag_pos = compute_dag_positional_features(
                        num_nodes=len(nodes),
                        edge_index=edge_index,
                        node_types=node_tensor,
                    )
                    graph.dag_pos = dag_pos
            else:
                graph = expr_to_graph(obfuscated, use_dag_features=self.use_dag_features)
        else:
            graph = expr_to_graph(obfuscated, use_dag_features=self.use_dag_features)

        # Validate schema on first item
        if not self._schema_validated and hasattr(graph, 'x'):
            node_types = graph.x.argmax(dim=-1) if graph.x.dim() > 1 else graph.x
            self._validate_node_type_schema(node_types)
            self._schema_validated = True

        # Convert node types if legacy schema
        graph = self._convert_graph_node_types(graph)

        # Load pre-computed fingerprint if available AND augmentation disabled.
        # CRITICAL: Pre-computed fingerprints were computed on original expressions.
        # Variable augmentation changes expression semantics, invalidating the fingerprint.
        # Pattern matches AST protection above (line ~838).
        can_use_precomputed = (
            'fingerprint' in item
            and 'flat' in item['fingerprint']
            and not self.augment_variables
        )

        if can_use_precomputed:
            fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
            fp_tensor = torch.from_numpy(fp_array)
        elif self.fingerprint is not None:
            fp = self.fingerprint.compute(obfuscated)
            fp_tensor = torch.from_numpy(fp).float()
        else:
            raise ValueError("No fingerprint available and no fingerprint computer provided")

        # Tokenize target
        target_ids = self.tokenizer.encode(simplified, add_special=True)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        # Source tokens for copy mechanism
        source_tokens = self.tokenizer.get_source_tokens(obfuscated)
        source_tensor = torch.tensor(source_tokens, dtype=torch.long)

        return {
            'graph': graph,
            'fingerprint': fp_tensor,
            'target_ids': target_tensor,
            'source_tokens': source_tensor,
            'boolean_domain_only': torch.tensor(boolean_domain_only, dtype=torch.bool),
            'complexity_score': torch.tensor(complexity_score, dtype=torch.float32),
            'depth': depth,
            'obfuscated': obfuscated,
            'simplified': simplified,
        }


class GMNDataset(Dataset):
    """
    Dataset for GMN training with negative sampling.

    Provides (graph1, graph2, label) tuples where:
        - label=1: graph1 and graph2 are equivalent expressions
        - label=0: graph1 and graph2 are non-equivalent
    """

    def __init__(
        self,
        data_path: str,
        negative_sampler: 'NegativeSampler',
        negative_ratio: float = 1.0,
        max_depth: Optional[int] = None,
        use_dag_features: bool = USE_DAG_FEATURES,
        max_graph_size: int = 100,
    ):
        """
        Initialize GMN dataset.

        Args:
            data_path: Path to JSONL file with positive pairs
            negative_sampler: NegativeSampler for generating negatives
            negative_ratio: Negatives per positive (default: 1.0, balanced dataset)
            max_depth: Optional depth filter
            use_dag_features: Compute DAG positional features for graphs
            max_graph_size: Maximum nodes per graph (filter larger graphs)
        """
        self.negative_sampler = negative_sampler
        self.negative_ratio = negative_ratio
        self.max_depth = max_depth
        self.use_dag_features = use_dag_features
        self.max_graph_size = max_graph_size

        # Load positive pairs
        self.positive_pairs = self._load_data(data_path)

        # Build dataset with positives and negatives
        self.samples = self._build_samples()

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and filter data from JSONL file."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Validate required fields
                if 'obfuscated' not in item or 'simplified' not in item:
                    continue

                # Apply depth filter if specified
                if self.max_depth is not None:
                    depth = item.get('depth', 0)
                    if depth > self.max_depth:
                        continue

                data.append(item)

        if not data:
            raise ValueError(f"No valid data loaded from {data_path}")

        return data

    def _build_samples(self) -> List[Tuple[str, str, int]]:
        """
        Build list of (expr1, expr2, label) samples.

        Includes all positive pairs and sampled negatives.
        """
        samples = []

        for item in self.positive_pairs:
            obf = item['obfuscated']
            simp = item['simplified']

            # Add positive pair
            samples.append((obf, simp, 1))

            # Sample negatives
            num_negatives = int(self.negative_ratio)
            fractional = self.negative_ratio - num_negatives
            if random.random() < fractional:
                num_negatives += 1

            for _ in range(num_negatives):
                _, negative_expr, _ = self.negative_sampler.sample_negative(obf, simp)
                samples.append((obf, negative_expr, 0))

        # Shuffle samples
        random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, int]:
        """
        Get dataset item.

        Returns:
            Tuple of (graph1, graph2, label):
                - graph1: PyG Data (first expression graph)
                - graph2: PyG Data (second expression graph)
                - label: 1 if equivalent, 0 if not
        """
        expr1, expr2, label = self.samples[idx]

        # Convert expressions to graphs
        graph1 = expr_to_graph(expr1, use_dag_features=self.use_dag_features)
        graph2 = expr_to_graph(expr2, use_dag_features=self.use_dag_features)

        return graph1, graph2, label

    def resample_negatives(self):
        """Resample negative pairs (call between epochs for diversity)."""
        self.samples = self._build_samples()
