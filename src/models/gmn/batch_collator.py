"""
Batch collation utilities for GMN training.

Handles variable-size graph pairs with proper batching and masking.
"""

from typing import List, Tuple

import torch
from torch_geometric.data import Batch, Data


class GMNBatchCollator:
    """
    Collate function for batching graph pairs with GMN.

    Handles variable-size graphs by:
      1. Packing graphs into PyG Batch (standard)
      2. Creating attention masks for cross-attention
      3. Tracking pair indices for loss computation
    """

    def __call__(
        self,
        batch_list: List[Tuple[Data, Data, int]]
    ) -> Tuple[Batch, Batch, torch.Tensor, torch.Tensor]:
        """
        Collate batch of graph pairs.

        Args:
            batch_list: List of (graph1, graph2, label) tuples
              - graph1: PyG Data (e.g., obfuscated expression)
              - graph2: PyG Data (e.g., simplified expression)
              - label: 1 if equivalent, 0 if not

        Returns:
            graph1_batch: PyG Batch containing all graph1s
            graph2_batch: PyG Batch containing all graph2s
            labels: [batch_size] tensor of labels
            pair_indices: [batch_size, 2] tensor mapping batch indices
        """
        graphs1, graphs2, labels = zip(*batch_list)

        # Standard PyG batching (handles variable sizes automatically)
        graph1_batch = Batch.from_data_list(list(graphs1))
        graph2_batch = Batch.from_data_list(list(graphs2))

        labels = torch.tensor(labels, dtype=torch.float32)

        # Pair indices: [i, i] means graph1[i] pairs with graph2[i]
        batch_size = len(batch_list)
        pair_indices = torch.arange(batch_size).unsqueeze(-1).repeat(1, 2)

        return graph1_batch, graph2_batch, labels, pair_indices


class GMNTripletCollator:
    """
    Collate function for triplet-based GMN training.

    Handles (anchor, positive, negative) triplets for triplet loss.
    """

    def __call__(
        self,
        batch_list: List[Tuple[Data, Data, Data]]
    ) -> Tuple[Batch, Batch, Batch]:
        """
        Collate batch of graph triplets.

        Args:
            batch_list: List of (anchor, positive, negative) tuples
              - anchor: PyG Data (obfuscated expression)
              - positive: PyG Data (equivalent simplified)
              - negative: PyG Data (non-equivalent expression)

        Returns:
            anchor_batch: PyG Batch containing all anchors
            positive_batch: PyG Batch containing all positives
            negative_batch: PyG Batch containing all negatives
        """
        anchors, positives, negatives = zip(*batch_list)

        anchor_batch = Batch.from_data_list(list(anchors))
        positive_batch = Batch.from_data_list(list(positives))
        negative_batch = Batch.from_data_list(list(negatives))

        return anchor_batch, positive_batch, negative_batch


def create_cross_attention_mask(
    batch1: torch.Tensor,
    batch2: torch.Tensor
) -> torch.Tensor:
    """
    Create attention mask for batched cross-attention.

    Prevents attention across different graph pairs in the batch.

    Args:
        batch1: [N1] batch indices for graph 1 nodes
        batch2: [N2] batch indices for graph 2 nodes

    Returns:
        mask: [N1, N2] boolean mask (True = allow attention)
    """
    # Each node in batch1[i] can only attend to nodes in batch2[i]
    # Shape: [N1, 1] == [1, N2] -> [N1, N2]
    mask = batch1.unsqueeze(-1) == batch2.unsqueeze(0)
    return mask
