"""
Tokenizer for MBA expressions.

Handles conversion between expression strings and token ID sequences.
"""

import json
import re
from typing import Dict, List, Optional

from src.constants import (
    SPECIAL_TOKENS,
    OPERATORS,
    PARENS,
    MAX_VARS,
    MAX_CONST,
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
)


class MBATokenizer:
    """Tokenizer for MBA expressions."""

    def __init__(self):
        """Initialize tokenizer and build vocabulary."""
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """
        Build vocabulary: special tokens, operators, parens, variables, constants.

        Vocabulary structure:
        - 0-4: Special tokens (<pad>, <sos>, <eos>, <unk>, <sep>)
        - 5-11: Operators (+, -, *, &, |, ^, ~)
        - 12-13: Parentheses ((, ))
        - 14-21: Variables (x0-x7)
        - 22-277: Constants (0-255)
        - 278-299: Reserved for future use
        """
        current_idx = 0

        # Special tokens
        for token, idx in SPECIAL_TOKENS.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
            current_idx = max(current_idx, idx + 1)

        # Operators
        for op in OPERATORS:
            self.token2id[op] = current_idx
            self.id2token[current_idx] = op
            current_idx += 1

        # Parentheses
        for paren in PARENS:
            self.token2id[paren] = current_idx
            self.id2token[current_idx] = paren
            current_idx += 1

        # Variables (x0 through x7)
        for i in range(MAX_VARS):
            var_name = f'x{i}'
            self.token2id[var_name] = current_idx
            self.id2token[current_idx] = var_name
            current_idx += 1

        # Also support single-letter variables (x, y, z, a, b, c, d, e)
        for var in ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e']:
            if var not in self.token2id:
                self.token2id[var] = current_idx
                self.id2token[current_idx] = var
                current_idx += 1

        # Constants (0-255)
        for i in range(MAX_CONST):
            const_str = str(i)
            self.token2id[const_str] = current_idx
            self.id2token[current_idx] = const_str
            current_idx += 1

        # Verify vocabulary size
        assert current_idx <= VOCAB_SIZE, f"Vocabulary size {current_idx} exceeds limit {VOCAB_SIZE}"

    def tokenize(self, expr: str) -> List[str]:
        """
        Split expression into tokens.

        Args:
            expr: Expression string (e.g., "(x & y) + 1")

        Returns:
            List of token strings

        Example:
            >>> tokenizer.tokenize("(x & y) + 1")
            ['(', 'x', '&', 'y', ')', '+', '1']
        """
        # Normalize whitespace
        expr = ' '.join(expr.split())

        # Add spaces around operators and parentheses
        for op in OPERATORS + PARENS:
            expr = expr.replace(op, f' {op} ')

        # Split and filter
        tokens = [t.strip() for t in expr.split() if t.strip()]

        return tokens

    def encode(self, expr: str, add_special: bool = True) -> List[int]:
        """
        Convert expression to token IDs.

        Args:
            expr: Expression string
            add_special: If True, add <sos> and <eos> tokens

        Returns:
            List of token IDs

        Example:
            >>> tokenizer.encode("x + 1")
            [1, 22, 5, 23, 2]  # <sos>, x, +, 1, <eos>
        """
        tokens = self.tokenize(expr)
        ids = []

        if add_special:
            ids.append(SOS_IDX)

        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                # Unknown token - try to handle constants outside 0-255
                if token.isdigit():
                    # Large constant - use <unk> or map to closest
                    ids.append(UNK_IDX)
                elif token.startswith('x') and token[1:].isdigit():
                    # Variable like x9, x10 - map to <unk>
                    ids.append(UNK_IDX)
                else:
                    ids.append(UNK_IDX)

        if add_special:
            ids.append(EOS_IDX)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to expression string.

        Args:
            ids: List of token IDs
            skip_special: If True, skip special tokens (<sos>, <eos>, <pad>)

        Returns:
            Expression string

        Example:
            >>> tokenizer.decode([1, 22, 5, 23, 2])
            "x + 1"
        """
        tokens = []

        for token_id in ids:
            if skip_special and token_id in (PAD_IDX, SOS_IDX, EOS_IDX):
                continue

            if token_id in self.id2token:
                tokens.append(self.id2token[token_id])
            else:
                tokens.append('<unk>')

        # Join tokens with proper spacing
        result = ' '.join(tokens)

        # Clean up spacing around parentheses
        result = result.replace('( ', '(')
        result = result.replace(' )', ')')

        return result

    def get_source_tokens(self, expr: str) -> List[int]:
        """
        Get token IDs for copy mechanism (no special tokens).

        Args:
            expr: Expression string

        Returns:
            List of token IDs without <sos>/<eos>

        Example:
            >>> tokenizer.get_source_tokens("x + 1")
            [22, 5, 23]  # x, +, 1
        """
        return self.encode(expr, add_special=False)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return PAD_IDX

    @property
    def sos_token_id(self) -> int:
        """Get start-of-sequence token ID."""
        return SOS_IDX

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return EOS_IDX

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return UNK_IDX

    def save(self, path: str):
        """
        Save tokenizer vocabulary to file.

        Args:
            path: Path to save vocabulary JSON
        """
        with open(path, 'w') as f:
            json.dump({
                'token2id': self.token2id,
                'id2token': {str(k): v for k, v in self.id2token.items()},
            }, f, indent=2)

    def load(self, path: str):
        """
        Load tokenizer vocabulary from file.

        Args:
            path: Path to vocabulary JSON
        """
        with open(path, 'r') as f:
            data = json.load(f)
            self.token2id = data['token2id']
            self.id2token = {int(k): v for k, v in data['id2token'].items()}
