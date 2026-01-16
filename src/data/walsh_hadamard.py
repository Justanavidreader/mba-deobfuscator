"""
Walsh-Hadamard Transform for MBA expression analysis.

The Walsh-Hadamard Transform (WHT) is the Fourier transform for Boolean functions.
It reveals algebraic structure that neural networks otherwise must learn through
extended training (the "grokking" phenomenon).

Key insight from research:
- Networks trained on modular arithmetic discover Fourier representations
- By providing Walsh coefficients explicitly, we shortcut this discovery
- Linear functions have sparse Walsh spectra (exactly 2 non-zero coefficients)
- Nonlinearity = distance from nearest linear function = f(Walsh coefficients)

References:
- Power et al. "Grokking: Generalization Beyond Overfitting" (2022)
- Nayebi et al. "A Mechanistic Interpretability Analysis of Grokking" (2023)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


@lru_cache(maxsize=8)
def _get_hadamard_matrix(n: int) -> np.ndarray:
    """
    Generate n x n Hadamard matrix using Sylvester construction.

    H_1 = [[1]]
    H_2 = [[1,  1],
           [1, -1]]
    H_2n = [[H_n,  H_n],
            [H_n, -H_n]]

    Args:
        n: Size (must be power of 2)

    Returns:
        n x n Hadamard matrix (unnormalized)
    """
    if n == 1:
        return np.array([[1]], dtype=np.float32)

    if n & (n - 1) != 0:
        raise ValueError(f"n must be power of 2, got {n}")

    h_half = _get_hadamard_matrix(n // 2)
    return np.block([
        [h_half, h_half],
        [h_half, -h_half]
    ])


def walsh_hadamard_transform(truth_table: torch.Tensor) -> torch.Tensor:
    """
    Compute Walsh-Hadamard Transform of a truth table.

    For a Boolean function f: {0,1}^n -> {0,1}, the Walsh coefficient W_f(a) is:
        W_f(a) = Σ_{x ∈ {0,1}^n} (-1)^(f(x) ⊕ <a,x>)

    where <a,x> is the inner product mod 2.

    Properties:
    - W_f(0) = 2^n - 2*wt(f) where wt(f) = number of 1s in truth table
    - For linear f(x) = <a,x>: W_f(a) = ±2^n, all others = 0
    - Parseval: Σ W_f(a)^2 = 2^(2n)

    Args:
        truth_table: [2^n] tensor of 0/1 values (or [batch, 2^n])

    Returns:
        [2^n] Walsh coefficients (or [batch, 2^n])

    Raises:
        ValueError: If truth table size is not power of 2 or not 64
    """
    is_batched = truth_table.dim() == 2
    if not is_batched:
        truth_table = truth_table.unsqueeze(0)

    batch_size, n = truth_table.shape

    # Validate size is power of 2 BEFORE recursive matrix generation
    if n & (n - 1) != 0:
        raise ValueError(
            f"Truth table size must be power of 2, got {n}. "
            f"Expected 64 (2^6) for MBA 6-variable truth tables."
        )
    if n != 64:
        raise ValueError(
            f"Walsh-Hadamard transform expects 64-entry truth tables, got {n}. "
            f"SemanticHGT is designed for 6-variable (2^6=64) MBA expressions."
        )

    # Convert {0,1} to {1,-1} for Walsh transform
    # f(x) = 0 -> +1, f(x) = 1 -> -1
    f_polar = 1 - 2 * truth_table.float()  # [batch, 2^n]

    # Get Hadamard matrix
    H = torch.from_numpy(_get_hadamard_matrix(n)).to(truth_table.device)

    # Walsh coefficients = H @ f_polar^T
    # [2^n, 2^n] @ [2^n, batch] -> [2^n, batch] -> [batch, 2^n]
    walsh = torch.mm(H, f_polar.t()).t()

    if not is_batched:
        walsh = walsh.squeeze(0)

    return walsh


def _extract_key_coefficients(
    walsh: torch.Tensor,
    n: int,
    num_vars: int,
) -> torch.Tensor:
    """
    Extract key Walsh coefficients (8d).

    Args:
        walsh: [batch, 64] Walsh coefficients
        n: Truth table size (64)
        num_vars: Number of variables (6)

    Returns:
        [batch, 8] key Walsh coefficients normalized by n
    """
    features = []

    # W(0): overall bias
    features.append(walsh[:, 0:1] / n)  # Normalize by 2^n

    # W(2^i) for i=0..5: single-variable linear coefficients
    for i in range(num_vars):
        idx = 1 << i
        features.append(walsh[:, idx:idx+1] / n)

    # W(2^n - 1): parity of all variables
    features.append(walsh[:, n-1:n] / n)

    return torch.cat(features, dim=1)  # [batch, 8]


def _compute_spectral_statistics(walsh: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute spectral statistics from Walsh spectrum (5d).

    Args:
        walsh: [batch, 64] Walsh coefficients
        n: Truth table size (64)

    Returns:
        [batch, 5] spectral statistics
    """
    walsh_abs = walsh.abs()
    walsh_sq = walsh ** 2

    features = []

    # Sparsity: fraction of coefficients below threshold
    threshold = 0.1 * n  # 10% of max possible
    sparsity = (walsh_abs < threshold).float().mean(dim=1, keepdim=True)
    features.append(sparsity)

    # Max magnitude (normalized)
    max_mag = walsh_abs.max(dim=1, keepdim=True)[0] / n
    features.append(max_mag)

    # L1 norm (normalized)
    l1_norm = walsh_abs.sum(dim=1, keepdim=True) / (n * n)
    features.append(l1_norm)

    # L2 norm (Parseval: sum of squares = 2^2n for Boolean functions)
    l2_norm = walsh_sq.sum(dim=1, keepdim=True).sqrt() / (n * n)
    features.append(l2_norm)

    # Spectral entropy
    walsh_sq_norm = walsh_sq / (walsh_sq.sum(dim=1, keepdim=True) + 1e-10)
    entropy = -(walsh_sq_norm * (walsh_sq_norm + 1e-10).log()).sum(dim=1, keepdim=True)
    entropy = entropy / np.log(n)  # Normalize to [0, 1]
    features.append(entropy)

    return torch.cat(features, dim=1)  # [batch, 5]


def _compute_linearity_indicators(
    walsh: torch.Tensor,
    n: int,
    num_vars: int,
) -> torch.Tensor:
    """
    Compute linearity indicators from Walsh spectrum (4d).

    Args:
        walsh: [batch, 64] Walsh coefficients
        n: Truth table size (64)
        num_vars: Number of variables (6)

    Returns:
        [batch, 4] linearity indicators
    """
    walsh_abs = walsh.abs()
    threshold = 0.1 * n

    features = []

    # Is linear? Linear functions have 1-2 large Walsh coefficients
    # Balanced linear functions (like f=x_i) have 1 large coeff
    # Unbalanced linear functions (like f=x_i XOR 1) have 2 large coeffs
    large_coeffs = (walsh_abs > 0.5 * n).float().sum(dim=1, keepdim=True)
    is_linear = ((large_coeffs >= 1) & (large_coeffs <= 2)).float()
    features.append(is_linear)

    # Nonlinearity score: N(f) = 2^(n-1) - max|W_f(a)|/2
    # Higher = more nonlinear, normalized to [0, 1]
    # Clamp handles bent functions where formula gives negative before scaling
    nonlinearity = 0.5 - walsh_abs.max(dim=1, keepdim=True)[0] / (2 * n)
    nonlinearity = nonlinearity.clamp(0, 0.5) * 2  # Scale to [0, 1]
    features.append(nonlinearity)

    # Algebraic degree estimate from spectral support
    # Degree d function has W(a) = 0 for |a| > d (Hamming weight)
    hamming_weights = torch.tensor(
        [bin(i).count('1') for i in range(n)],
        device=walsh.device,
        dtype=torch.float
    )
    # Weighted average of Hamming weights where Walsh is large
    large_mask = (walsh_abs > threshold).float()
    degree_estimate = (large_mask * hamming_weights).sum(dim=1, keepdim=True) / \
                     (large_mask.sum(dim=1, keepdim=True) + 1e-10)
    degree_estimate = degree_estimate / num_vars  # Normalize to [0, 1]
    features.append(degree_estimate)

    # Bent function indicator: all |W(a)| = 2^(n/2)
    # Bent functions achieve maximum nonlinearity
    expected_bent_mag = 2 ** (num_vars / 2)  # = 8 for n=6
    bent_deviation = ((walsh_abs - expected_bent_mag).abs() / expected_bent_mag).mean(
        dim=1, keepdim=True
    )
    is_bent = (bent_deviation < 0.1).float()
    features.append(is_bent)

    return torch.cat(features, dim=1)  # [batch, 4]


def compute_walsh_features(truth_table: torch.Tensor) -> torch.Tensor:
    """
    Extract meaningful features from Walsh spectrum.

    Features (17 dimensions):
    1. Normalized Walsh coefficients at key positions (8d)
       - W(0): bias/balance
       - W(2^i) for i=0..5: single-variable linear terms
       - W(2^n - 1): all-variables parity
    2. Spectral statistics (5d)
       - Sparsity: fraction of near-zero coefficients
       - Max magnitude (normalized)
       - L1 norm (normalized)
       - L2 norm (normalized)
       - Entropy of |W|^2 distribution
    3. Linearity indicators (4d)
       - Is linear? (exactly 2 large coefficients)
       - Nonlinearity score: max correlation with linear functions
       - Algebraic degree estimate from spectral support
       - Bent function indicator (constant |W| magnitude)

    Args:
        truth_table: [64] or [batch, 64] truth table (2^6 = 64 for 6 vars)

    Returns:
        [17] or [batch, 17] feature tensor
    """
    is_batched = truth_table.dim() == 2
    if not is_batched:
        truth_table = truth_table.unsqueeze(0)

    batch_size = truth_table.size(0)
    n = truth_table.size(1)  # Should be 64 = 2^6
    num_vars = 6  # log2(64)

    # Compute Walsh transform
    walsh = walsh_hadamard_transform(truth_table)  # [batch, 64]

    # Extract feature groups
    key_coeffs = _extract_key_coefficients(walsh, n, num_vars)
    spectral_stats = _compute_spectral_statistics(walsh, n)
    linearity_indicators = _compute_linearity_indicators(walsh, n, num_vars)

    # Concatenate all features
    result = torch.cat([key_coeffs, spectral_stats, linearity_indicators], dim=1)

    if not is_batched:
        result = result.squeeze(0)

    return result


def compute_extended_walsh_features(
    truth_table: torch.Tensor,
    include_raw_spectrum: bool = False,
    top_k: int = 16,
) -> torch.Tensor:
    """
    Compute extended Walsh features for Semantic HGT.

    Args:
        truth_table: [64] or [batch, 64] truth table
        include_raw_spectrum: Include top-k raw Walsh coefficients
        top_k: Number of largest coefficients to include

    Returns:
        [17 + top_k] or [batch, 17 + top_k] if include_raw_spectrum
        [17] or [batch, 17] otherwise
    """
    base_features = compute_walsh_features(truth_table)  # [batch, 17] or [17]

    if not include_raw_spectrum:
        return base_features

    is_batched = truth_table.dim() == 2
    if not is_batched:
        truth_table = truth_table.unsqueeze(0)

    # Get top-k Walsh coefficients by magnitude (using topk for efficiency)
    walsh = walsh_hadamard_transform(truth_table)
    n = truth_table.size(1)

    # Use torch.topk instead of full sort for memory efficiency
    top_k_coeffs, _ = torch.topk(walsh.abs(), k=top_k, dim=1, largest=True)
    top_k_coeffs = top_k_coeffs / n

    if not is_batched:
        base_features = base_features.unsqueeze(0)

    result = torch.cat([base_features, top_k_coeffs], dim=1)

    if not is_batched:
        result = result.squeeze(0)

    return result


class WalshSpectrumEncoder(nn.Module):
    """
    Neural network module to encode Walsh spectrum features.

    Transforms raw Walsh features into embeddings suitable for
    injection into the HGT encoder.
    """

    def __init__(
        self,
        input_dim: int = 17,  # Base Walsh features
        hidden_dim: int = 64,
        output_dim: int = 64,
        include_raw_spectrum: bool = True,
        top_k: int = 16,
    ):
        super().__init__()
        self.include_raw_spectrum = include_raw_spectrum
        self.top_k = top_k

        actual_input_dim = input_dim + (top_k if include_raw_spectrum else 0)

        self.encoder = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, truth_table: torch.Tensor) -> torch.Tensor:
        """
        Encode truth table via Walsh features.

        Args:
            truth_table: [batch, 64] truth table values

        Returns:
            [batch, output_dim] Walsh spectrum embeddings
        """
        walsh_features = compute_extended_walsh_features(
            truth_table,
            include_raw_spectrum=self.include_raw_spectrum,
            top_k=self.top_k,
        )
        return self.encoder(walsh_features)
