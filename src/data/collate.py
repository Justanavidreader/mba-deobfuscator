"""
Batch collation functions for DataLoader.

Handles batching of graphs and sequences with proper padding.
"""

from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

from src.constants import PAD_IDX


def collate_graphs(batch: List[Dict]) -> Dict:
    """
    Collate function for MBADataset.

    Batches PyG graphs using Batch.from_data_list().
    Pads sequences to max length in batch.

    Args:
        batch: List of dataset items from MBADataset

    Returns:
        Collated batch dictionary with:
            - graph_batch: Batched PyG Data
            - fingerprint: [batch_size, FINGERPRINT_DIM]
            - target_ids: [batch_size, max_seq_len] (padded)
            - target_lengths: [batch_size] (original lengths)
            - source_tokens: [batch_size, max_src_len] (padded)
            - source_lengths: [batch_size]
            - depth: [batch_size]
            - obfuscated: List[str]
            - simplified: List[str]
    """
    # Extract components
    graphs = [item['graph'] for item in batch]
    fingerprints = [item['fingerprint'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    source_tokens = [item['source_tokens'] for item in batch]
    depths = [item['depth'] for item in batch]
    obfuscated = [item['obfuscated'] for item in batch]
    simplified = [item['simplified'] for item in batch]

    # Batch graphs using PyG
    graph_batch = Batch.from_data_list(graphs)

    # Stack fingerprints
    fingerprint_tensor = torch.stack(fingerprints, dim=0)

    # Pad target sequences
    target_lengths = torch.tensor([len(seq) for seq in target_ids], dtype=torch.long)
    target_padded = pad_sequence(
        target_ids, batch_first=True, padding_value=PAD_IDX
    )

    # Pad source token sequences
    source_lengths = torch.tensor([len(seq) for seq in source_tokens], dtype=torch.long)
    source_padded = pad_sequence(
        source_tokens, batch_first=True, padding_value=PAD_IDX
    )

    # Depths
    depth_tensor = torch.tensor(depths, dtype=torch.long)

    return {
        'graph_batch': graph_batch,
        'fingerprint': fingerprint_tensor,
        'target_ids': target_padded,
        'target_lengths': target_lengths,
        'source_tokens': source_padded,
        'source_lengths': source_lengths,
        'depth': depth_tensor,
        'obfuscated': obfuscated,
        'simplified': simplified,
    }


def collate_contrastive(batch: List[Dict]) -> Dict:
    """
    Collate function for ContrastiveDataset.

    Args:
        batch: List of dataset items from ContrastiveDataset

    Returns:
        Collated batch dictionary with:
            - obf_graph_batch: Batched PyG Data for obfuscated expressions
            - simp_graph_batch: Batched PyG Data for simplified expressions
            - obf_fingerprint: [batch_size, FINGERPRINT_DIM]
            - simp_fingerprint: [batch_size, FINGERPRINT_DIM]
            - labels: [batch_size] (for positive pair matching)
            - obfuscated: List[str]
            - simplified: List[str]
    """
    # Extract components
    obf_graphs = [item['obf_graph'] for item in batch]
    simp_graphs = [item['simp_graph'] for item in batch]
    obf_fingerprints = [item['obf_fingerprint'] for item in batch]
    simp_fingerprints = [item['simp_fingerprint'] for item in batch]
    labels = [item['label'] for item in batch]
    obfuscated = [item['obfuscated'] for item in batch]
    simplified = [item['simplified'] for item in batch]

    # Batch graphs
    obf_graph_batch = Batch.from_data_list(obf_graphs)
    simp_graph_batch = Batch.from_data_list(simp_graphs)

    # Stack fingerprints
    obf_fp_tensor = torch.stack(obf_fingerprints, dim=0)
    simp_fp_tensor = torch.stack(simp_fingerprints, dim=0)

    # Labels
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        'obf_graph_batch': obf_graph_batch,
        'simp_graph_batch': simp_graph_batch,
        'obf_fingerprint': obf_fp_tensor,
        'simp_fingerprint': simp_fp_tensor,
        'labels': label_tensor,
        'obfuscated': obfuscated,
        'simplified': simplified,
    }


def collate_custom_format(batch: List[Dict]) -> Dict:
    """
    Collate function for CustomFormatDataset.

    Compatible with custom text format datasets that have manually computed
    fingerprints and AST graphs.

    Args:
        batch: List of dataset items from CustomFormatDataset

    Returns:
        Collated batch dictionary with:
            - graph_batch: Batched PyG Data
            - fingerprint: [batch_size, FINGERPRINT_DIM]
            - obfuscated_tokens: [batch_size, max_obf_len] (padded)
            - obfuscated_lengths: [batch_size]
            - simplified_tokens: [batch_size, max_simp_len] (padded)
            - simplified_lengths: [batch_size]
            - depth: [batch_size]
            - obfuscated: List[str]
            - simplified: List[str]
            - section: List[str]
            - additional: List[str or None] (optional third column)
    """
    # Extract components
    graphs = [item['graph_data'] for item in batch]
    fingerprints = [item['fingerprint'] for item in batch]
    obfuscated_tokens = [item['obfuscated_tokens'] for item in batch]
    simplified_tokens = [item['simplified_tokens'] for item in batch]
    depths = [item['depth'] for item in batch]
    obfuscated = [item['obfuscated'] for item in batch]
    simplified = [item['simplified'] for item in batch]
    sections = [item['section'] for item in batch]
    additional = [item['additional'] for item in batch]

    # Batch graphs using PyG
    graph_batch = Batch.from_data_list(graphs)

    # Stack fingerprints
    fingerprint_tensor = torch.stack(fingerprints, dim=0)

    # Pad obfuscated token sequences
    obfuscated_lengths = torch.tensor([len(seq) for seq in obfuscated_tokens], dtype=torch.long)
    obfuscated_padded = pad_sequence(
        obfuscated_tokens, batch_first=True, padding_value=PAD_IDX
    )

    # Pad simplified token sequences
    simplified_lengths = torch.tensor([len(seq) for seq in simplified_tokens], dtype=torch.long)
    simplified_padded = pad_sequence(
        simplified_tokens, batch_first=True, padding_value=PAD_IDX
    )

    # Depths
    depth_tensor = torch.tensor(depths, dtype=torch.long)

    return {
        'graph_batch': graph_batch,
        'fingerprint': fingerprint_tensor,
        'obfuscated_tokens': obfuscated_padded,
        'obfuscated_lengths': obfuscated_lengths,
        'simplified_tokens': simplified_padded,
        'simplified_lengths': simplified_lengths,
        'depth': depth_tensor,
        'obfuscated': obfuscated,
        'simplified': simplified,
        'section': sections,
        'additional': additional,
    }
