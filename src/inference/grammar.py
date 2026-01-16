"""
Grammar-constrained decoding for MBA expressions.

Uses Lark parser to ensure syntactically valid token sequences during generation.
"""

from typing import List, Set, Dict, Optional, Tuple
import torch
from lark import Lark, LarkError

from src.constants import (
    OPERATORS, PARENS, MAX_VARS, SOS_IDX, EOS_IDX, PAD_IDX,
    VOCAB_SIZE, NODE_TYPES
)
from src.data.tokenizer import MBATokenizer


class GrammarConstraint:
    """
    Grammar-based token masking for MBA expressions.
    Precomputes valid token sets for each parser state.
    """

    # Termination conditions to prevent infinite cache building
    MAX_STATES = 5000
    MAX_DEPTH = 20

    def __init__(self, tokenizer: MBATokenizer):
        """
        Initialize grammar constraint.

        Args:
            tokenizer: MBATokenizer instance for token-ID mapping
        """
        self.tokenizer = tokenizer
        self.grammar = self._build_mba_grammar()
        self.parser = Lark(self.grammar, start='expr', parser='lalr')

        # Precomputed cache: parser_state → valid token IDs
        self.valid_tokens_cache: Dict[str, Set[int]] = {}
        self._build_token_cache()

    def _build_mba_grammar(self) -> str:
        """
        Build Lark grammar for MBA expressions.

        Returns:
            Grammar string in Lark EBNF format

        Grammar rules with operator precedence:
            expr   : term (("+"|"-") term)*
            term   : factor (("*") factor)*
            factor : bitwise (("&"|"|"|"^") bitwise)*
            bitwise: unary
            unary  : ("~"|"-") unary | primary
            primary: VAR | CONST | "(" expr ")"
        """
        # Build variable alternatives (x, y, z, x0-x7)
        var_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e']
        var_names += [f'x{i}' for i in range(MAX_VARS)]
        var_alternatives = ' | '.join(f'"{v}"' for v in var_names)

        grammar = f'''
        ?start: expr

        ?expr: term (("+" | "-") term)*

        ?term: factor ("*" factor)*

        ?factor: bitwise (("&" | "|" | "^") bitwise)*

        ?bitwise: unary

        ?unary: ("~" | "-") unary
              | primary

        ?primary: VAR
                | CONST
                | "(" expr ")"

        VAR: {var_alternatives}

        CONST: /[0-9]+/

        %import common.WS
        %ignore WS
        '''

        return grammar

    def _build_token_cache(self) -> None:
        """
        Precompute valid tokens for each grammar state.
        Explores reachable parser states and caches allowed tokens.

        Algorithm:
        1. Start with empty string
        2. For each parser state, try all tokens
        3. If token is valid (doesn't cause syntax error), cache it
        4. Continue DFS until MAX_STATES or MAX_DEPTH reached
        """
        visited_states = set()

        def canonicalize(tokens: List[int]) -> str:
            """Create canonical state key from token sequence."""
            # Remove special tokens
            filtered = [t for t in tokens if t not in (SOS_IDX, EOS_IDX, PAD_IDX)]
            return ','.join(map(str, filtered))

        def explore_state(partial_tokens: List[int], depth: int):
            """Recursively explore grammar states."""
            if len(visited_states) >= self.MAX_STATES or depth >= self.MAX_DEPTH:
                return

            state_key = canonicalize(partial_tokens)
            if state_key in visited_states:
                return

            visited_states.add(state_key)

            # Try all tokens from vocabulary
            valid_tokens = set()

            for token_id in range(VOCAB_SIZE):
                if token_id not in self.tokenizer.id2token:
                    continue

                token_str = self.tokenizer.id2token[token_id]

                # Skip special tokens during exploration
                if token_id in (PAD_IDX, SOS_IDX, UNK_IDX := 3):
                    continue

                # Build test expression
                test_tokens = [self.tokenizer.id2token[t] for t in partial_tokens
                             if t in self.tokenizer.id2token and t not in (SOS_IDX, EOS_IDX, PAD_IDX)]
                test_tokens.append(token_str)
                test_expr = ' '.join(test_tokens)

                # Try to parse
                try:
                    # Allow partial expressions by catching specific errors
                    try:
                        self.parser.parse(test_expr)
                        # Complete expression - mark EOS as valid
                        valid_tokens.add(token_id)
                        valid_tokens.add(EOS_IDX)
                    except LarkError:
                        # Try adding closing parens to see if it could complete
                        # If it's a partial valid expression, the token is valid
                        # This is approximate - check if parser state is reasonable
                        paren_depth = test_expr.count('(') - test_expr.count(')')
                        if paren_depth >= 0:
                            # Could potentially be valid partial
                            valid_tokens.add(token_id)
                except:
                    pass

            # Cache valid tokens for this state
            if valid_tokens:
                self.valid_tokens_cache[state_key] = valid_tokens

            # Expand high-value states (short sequences)
            if depth < 5 and len(partial_tokens) < 10:
                for token_id in list(valid_tokens)[:10]:  # Limit branching
                    if token_id != EOS_IDX:
                        explore_state(partial_tokens + [token_id], depth + 1)

        # Start exploration from initial state
        explore_state([SOS_IDX], 0)

    def get_valid_tokens(self, partial_tokens: List[int]) -> Set[int]:
        """
        Get valid next tokens given partial output sequence.

        Args:
            partial_tokens: List of token IDs generated so far

        Returns:
            Set of valid token IDs that can follow

        Example:
            >>> constraint.get_valid_tokens([SOS_IDX, tokenizer.token2id['x']])
            {tokenizer.token2id['+'], tokenizer.token2id['&'], EOS_IDX, ...}
        """
        # Canonicalize state
        state_key = ','.join(str(t) for t in partial_tokens
                           if t not in (SOS_IDX, EOS_IDX, PAD_IDX))

        # Check cache
        if state_key in self.valid_tokens_cache:
            return self.valid_tokens_cache[state_key]

        # Cache miss - compute on-the-fly
        valid_tokens = self._compute_valid_tokens(partial_tokens)
        self.valid_tokens_cache[state_key] = valid_tokens
        return valid_tokens

    def _compute_valid_tokens(self, partial_tokens: List[int]) -> Set[int]:
        """Compute valid tokens for uncached state."""
        valid_tokens = set()

        # Build partial expression
        expr_tokens = []
        for token_id in partial_tokens:
            if token_id in self.tokenizer.id2token and token_id not in (SOS_IDX, EOS_IDX, PAD_IDX):
                expr_tokens.append(self.tokenizer.id2token[token_id])

        partial_expr = ' '.join(expr_tokens)

        # Try each vocabulary token
        for token_id in range(VOCAB_SIZE):
            if token_id not in self.tokenizer.id2token:
                continue

            if token_id in (PAD_IDX, SOS_IDX):
                continue

            token_str = self.tokenizer.id2token[token_id]
            test_expr = partial_expr + (' ' if partial_expr else '') + token_str

            try:
                # Try parsing complete expression
                self.parser.parse(test_expr)
                valid_tokens.add(token_id)
                valid_tokens.add(EOS_IDX)
            except:
                # Try with added closing parens (partial expression)
                paren_depth = test_expr.count('(') - test_expr.count(')')
                if paren_depth >= 0:
                    valid_tokens.add(token_id)

        return valid_tokens

    def mask_logits(self, logits: torch.Tensor, valid_tokens: Set[int]) -> torch.Tensor:
        """
        Apply grammar constraint mask to logits.

        Args:
            logits: [vocab_size] or [batch, vocab_size] raw logits
            valid_tokens: Set of valid token IDs

        Returns:
            Masked logits with -inf for invalid tokens
        """
        # Create mask
        mask = torch.full_like(logits, float('-inf'))

        # Set valid positions to 0 (no penalty)
        valid_indices = torch.tensor(list(valid_tokens), dtype=torch.long, device=logits.device)

        if logits.dim() == 1:
            mask[valid_indices] = 0.0
        else:
            mask[:, valid_indices] = 0.0

        return logits + mask

    def is_complete(self, tokens: List[int]) -> bool:
        """
        Check if token sequence forms a complete valid expression.

        Args:
            tokens: List of token IDs (including SOS/EOS if present)

        Returns:
            True if sequence can be parsed successfully
        """
        # Build expression string
        expr_tokens = []
        for token_id in tokens:
            if token_id in self.tokenizer.id2token and token_id not in (SOS_IDX, EOS_IDX, PAD_IDX):
                expr_tokens.append(self.tokenizer.id2token[token_id])

        if not expr_tokens:
            return False

        expr = ' '.join(expr_tokens)

        # Try parsing
        try:
            self.parser.parse(expr)
            return True
        except:
            return False

    def parse_check(self, expr_str: str) -> Tuple[bool, Optional[str]]:
        """
        Syntax check for expression string.

        Args:
            expr_str: Expression string (e.g., "x & y + 1")

        Returns:
            (is_valid, error_message)
            - (True, None) if valid
            - (False, error_msg) if invalid
        """
        try:
            self.parser.parse(expr_str)
            return (True, None)
        except LarkError as e:
            return (False, str(e))
        except Exception as e:
            return (False, f"Parse error: {str(e)}")
