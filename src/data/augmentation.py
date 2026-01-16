"""
Variable permutation augmentation for MBA expressions.

Prevents the model from memorizing variable name patterns by randomly
permuting variable names during training while keeping obfuscated/simplified
pairs consistent.
"""

import re
import random
from typing import Tuple, List, Dict, Optional


class VariablePermuter:
    """
    Randomly permutes variable names in MBA expressions.

    Ensures both input and output expressions use consistent permutation,
    preventing the model from memorizing variable name patterns.
    """

    # Canonical variable names (have dedicated tokens in vocab)
    CANONICAL_VARS = [f'x{i}' for i in range(8)]  # x0, x1, ..., x7

    # Pattern to match variables: single letter optionally followed by single digit
    # Matches: x, y, z, a, b, c, x0, x1, etc.
    # Does NOT match: x10, abc, 0x (hex prefix)
    VAR_PATTERN = re.compile(r'\b([a-zA-Z]\d?)\b')

    def __init__(
        self,
        permute_prob: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize variable permuter.

        Args:
            permute_prob: Probability of applying permutation (0.0-1.0)
            seed: Random seed for reproducibility (None for random)
        """
        if not 0.0 <= permute_prob <= 1.0:
            raise ValueError(f"permute_prob must be in [0, 1], got {permute_prob}")

        self.permute_prob = permute_prob

        # Worker-aware seeding for multi-worker DataLoader
        # Without this, all workers with same seed produce identical permutations
        if seed is not None:
            try:
                import torch.utils.data
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    seed = seed + worker_info.id
            except ImportError:
                pass  # torch not available, use seed as-is

        self.rng = random.Random(seed)

    def extract_variables(self, expr: str) -> List[str]:
        """
        Extract unique variables from expression in order of first appearance.

        Args:
            expr: MBA expression string

        Returns:
            List of variable names in order of first appearance
        """
        seen = set()
        variables = []
        for match in self.VAR_PATTERN.finditer(expr):
            var = match.group(1)
            if var not in seen:
                seen.add(var)
                variables.append(var)
        return variables

    def generate_permutation(self, variables: List[str]) -> Dict[str, str]:
        """
        Generate a random mapping from original variables to canonical names.

        Args:
            variables: List of original variable names

        Returns:
            Dict mapping original names to permuted canonical names
        """
        if len(variables) > len(self.CANONICAL_VARS):
            raise ValueError(
                f"Expression has {len(variables)} variables, "
                f"but only {len(self.CANONICAL_VARS)} canonical names available"
            )

        # Shuffle canonical names and assign
        shuffled = self.CANONICAL_VARS[:len(variables)].copy()
        self.rng.shuffle(shuffled)

        return dict(zip(variables, shuffled))

    def apply_permutation(self, expr: str, mapping: Dict[str, str]) -> str:
        """
        Apply variable permutation to expression.

        Uses word boundary matching to avoid partial replacements.
        Processes longer variable names first to avoid conflicts (e.g., x1 before x).

        Args:
            expr: Original expression
            mapping: Variable name mapping

        Returns:
            Expression with permuted variable names
        """
        if not mapping:
            return expr

        # Sort by length descending to replace longer names first
        # This prevents 'x' from matching in 'x1'
        sorted_vars = sorted(mapping.keys(), key=len, reverse=True)

        result = expr
        # Use temporary placeholders to avoid replacement conflicts
        placeholders = {var: f'__VAR_{i}__' for i, var in enumerate(sorted_vars)}

        # First pass: replace with placeholders
        for var in sorted_vars:
            pattern = rf'\b{re.escape(var)}\b'
            result = re.sub(pattern, placeholders[var], result)

        # Second pass: replace placeholders with final names
        for var in sorted_vars:
            result = result.replace(placeholders[var], mapping[var])

        return result

    def __call__(
        self,
        obfuscated: str,
        simplified: str,
    ) -> Tuple[str, str]:
        """
        Permute variables in both expressions consistently.

        Args:
            obfuscated: Obfuscated MBA expression
            simplified: Simplified MBA expression

        Returns:
            Tuple of (permuted_obfuscated, permuted_simplified)
        """
        # Skip permutation with probability (1 - permute_prob)
        if self.rng.random() > self.permute_prob:
            return obfuscated, simplified

        # Extract variables from both expressions (union)
        vars_obf = set(self.extract_variables(obfuscated))
        vars_simp = set(self.extract_variables(simplified))
        all_vars = list(vars_obf | vars_simp)

        # Sort for deterministic ordering before shuffling
        all_vars.sort()

        if not all_vars:
            return obfuscated, simplified

        # Generate and apply permutation
        mapping = self.generate_permutation(all_vars)

        permuted_obf = self.apply_permutation(obfuscated, mapping)
        permuted_simp = self.apply_permutation(simplified, mapping)

        return permuted_obf, permuted_simp


class VariableAugmentationMixin:
    """
    Mixin for variable permutation augmentation.

    Provides shared initialization and application logic for all dataset classes.
    """

    def _init_augmentation(
        self,
        augment_variables: bool = True,
        augment_prob: float = 0.8,
        augment_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize variable permutation augmentation.

        Args:
            augment_variables: Whether to enable augmentation
            augment_prob: Probability of applying permutation
            augment_seed: Random seed (None for random)
        """
        self.augment_variables = augment_variables
        self.permuter: Optional[VariablePermuter] = None
        if augment_variables:
            self.permuter = VariablePermuter(
                permute_prob=augment_prob,
                seed=augment_seed,
            )

    def _apply_augmentation(
        self,
        obfuscated: str,
        simplified: str,
    ) -> Tuple[str, str]:
        """
        Apply variable permutation to expression pair.

        Args:
            obfuscated: Obfuscated expression
            simplified: Simplified expression

        Returns:
            Tuple of (permuted_obfuscated, permuted_simplified)
        """
        if self.permuter is not None:
            return self.permuter(obfuscated, simplified)
        return obfuscated, simplified
