"""
Base encoder interface for ablation study.

All encoders must implement this interface for fair comparison.
Addresses critical issues:
- Edge type handling: validates edge_type availability when required
- Consistent forward signature across all encoders
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all encoder architectures.

    Ensures consistent interface across:
    - GAT+JKNet (homogeneous, no edge types)
    - GGNN (heterogeneous, requires edge types)
    - HGT (heterogeneous, requires edge types)
    - RGCN (heterogeneous, requires edge types)
    - Transformer-only (sequence, no edge types)
    - Hybrid GREAT (mixed, no edge types)
    """

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__()
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self) -> int:
        """Output embedding dimension."""
        return self._hidden_dim

    @property
    @abstractmethod
    def requires_edge_types(self) -> bool:
        """
        Whether this encoder uses edge type information.

        If True, forward() will raise ValueError when edge_type is None.
        """
        pass

    @property
    @abstractmethod
    def requires_node_features(self) -> bool:
        """
        True if encoder expects [total_nodes, node_dim] features.
        False if encoder expects [total_nodes] node type IDs (embeds internally).
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode graph to node embeddings.

        Args:
            x: Node features
               - [total_nodes, node_dim] if requires_node_features is True
               - [total_nodes] node type IDs if requires_node_features is False
            edge_index: [2, num_edges] edge indices
            batch: [total_nodes] batch assignment
            edge_type: [num_edges] edge type indices (required if requires_edge_types)
            dag_pos: [total_nodes, 4] DAG positional features (optional)
                     Columns: [depth, subtree_size, in_degree, is_shared]

        Returns:
            [total_nodes, hidden_dim] node embeddings

        Raises:
            ValueError: If requires_edge_types is True but edge_type is None
        """
        # Critical fix: validate edge_type before forwarding
        if self.requires_edge_types and edge_type is None:
            raise ValueError(
                f"{self.__class__.__name__} requires edge_type but got None. "
                "Ensure your dataset provides edge_type for heterogeneous encoders."
            )

        return self._forward_impl(x, edge_index, batch, edge_type, dag_pos)

    @abstractmethod
    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        dag_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementation-specific forward pass.

        Subclasses implement this instead of forward() to ensure
        edge_type validation always runs first.

        Args:
            dag_pos: [total_nodes, 4] DAG positional features (optional)
                     Columns: [depth, subtree_size, in_degree, is_shared]
                     Encoders can ignore if USE_DAG_FEATURES is False.
        """
        pass

    def parameter_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _compute_gcnii_beta(
        self, layer_idx: int, gcnii_lambda: float, use_identity_mapping: bool
    ) -> float:
        """
        Compute GCNII identity mapping strength for layer l.

        Beta decreases with depth, making deep layers act more like identity:
        - Early layers: β ≈ 0.6-0.7 (more transformation)
        - Deep layers: β ≈ 0.09-0.12 (mostly identity)

        Formula: β = log(λ / (l+1) + 1)

        Args:
            layer_idx: Current layer index (0-indexed)
            gcnii_lambda: Lambda parameter for decay rate
            use_identity_mapping: Whether identity mapping is enabled

        Returns:
            Beta value in [0, 1]. Returns 1.0 if identity mapping disabled.
        """
        if not use_identity_mapping:
            return 1.0  # No identity mapping, full transformation
        return math.log(gcnii_lambda / (layer_idx + 1) + 1)

    def _validate_checkpoint_compatibility(
        self, checkpoint_metadata: Dict[str, Any]
    ) -> None:
        """
        Validate that GCNII configuration matches checkpoint.

        Prevents silent failures when loading checkpoints trained with different
        GCNII configurations (e.g., loading GCNII-enabled checkpoint into
        GCNII-disabled model).

        Args:
            checkpoint_metadata: Metadata dict from checkpoint state_dict

        Raises:
            ValueError: If GCNII configuration mismatch detected
        """
        # Only validate if checkpoint has GCNII metadata
        if "_gcnii_config" not in checkpoint_metadata:
            return  # Old checkpoint without GCNII, allow loading

        ckpt_config = checkpoint_metadata["_gcnii_config"]

        # Check if current model has GCNII attributes
        has_gcnii = hasattr(self, "use_initial_residual") and hasattr(
            self, "use_identity_mapping"
        )

        if not has_gcnii:
            # Current model doesn't support GCNII but checkpoint does
            if ckpt_config.get("use_initial_residual") or ckpt_config.get(
                "use_identity_mapping"
            ):
                raise ValueError(
                    f"Checkpoint trained with GCNII enabled, but current model "
                    f"({self.__class__.__name__}) does not support GCNII. "
                    "Cannot load checkpoint."
                )
            return

        # Both have GCNII, validate configuration matches
        current_use_residual = getattr(self, "use_initial_residual", False)
        current_use_identity = getattr(self, "use_identity_mapping", False)

        ckpt_use_residual = ckpt_config.get("use_initial_residual", False)
        ckpt_use_identity = ckpt_config.get("use_identity_mapping", False)

        if current_use_residual != ckpt_use_residual:
            raise ValueError(
                f"GCNII configuration mismatch: "
                f"checkpoint use_initial_residual={ckpt_use_residual}, "
                f"model use_initial_residual={current_use_residual}. "
                "GCNII configuration must match for checkpoint compatibility."
            )

        if current_use_identity != ckpt_use_identity:
            raise ValueError(
                f"GCNII configuration mismatch: "
                f"checkpoint use_identity_mapping={ckpt_use_identity}, "
                f"model use_identity_mapping={current_use_identity}. "
                "GCNII configuration must match for checkpoint compatibility."
            )

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Override state_dict to include GCNII metadata.

        This metadata enables checkpoint validation on load, preventing
        silent failures from configuration mismatches.
        """
        sd = super().state_dict(*args, **kwargs)

        # Add GCNII metadata if model supports it
        if hasattr(self, "use_initial_residual") and hasattr(
            self, "use_identity_mapping"
        ):
            sd["_metadata"] = {
                "_gcnii_config": {
                    "use_initial_residual": self.use_initial_residual,
                    "use_identity_mapping": self.use_identity_mapping,
                    "gcnii_alpha": getattr(self, "gcnii_alpha", None),
                    "gcnii_lambda": getattr(self, "gcnii_lambda", None),
                }
            }

        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Override load_state_dict to validate GCNII compatibility.

        Ensures checkpoint and model have matching GCNII configurations
        before loading weights.
        """
        # Extract and validate metadata
        if "_metadata" in state_dict:
            self._validate_checkpoint_compatibility(state_dict["_metadata"])
            # Remove metadata before passing to parent (not a model parameter)
            state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}

        return super().load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"requires_edge_types={self.requires_edge_types}, "
            f"params={self.parameter_count():,})"
        )
