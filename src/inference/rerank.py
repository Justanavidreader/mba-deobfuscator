"""
Candidate reranking by simplicity metrics and model confidence.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.data.tokenizer import MBATokenizer
from src.data.ast_parser import parse_to_ast, get_ast_depth
from src.inference.verify import VerificationResult


@dataclass
class RankingFeatures:
    """
    Features for candidate ranking.
    """
    candidate: str                  # Expression string
    token_length: int               # Number of tokens
    ast_depth: int                  # Maximum AST depth
    num_operators: int              # Operator count
    num_variables: int              # Variable count
    num_constants: int              # Constant count
    model_confidence: float         # Model's log probability
    verification_level: int         # 0=none, 1=syntax, 2=exec, 3=z3
    counterexample_found: bool      # Failed verification flag


class CandidateReranker:
    """
    Rerank verified candidates by simplicity and confidence.
    """

    def __init__(
        self,
        tokenizer: MBATokenizer,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize reranker.

        Args:
            tokenizer: MBATokenizer instance
            weights: Optional feature weights dict
                     Default: {
                         'token_length': -1.0,     # Prefer shorter
                         'ast_depth': -0.5,        # Prefer shallower
                         'num_operators': -0.3,    # Prefer fewer ops
                         'model_confidence': 2.0,  # Trust model
                         'verification_level': 5.0 # Strongly prefer verified
                     }
        """
        self.tokenizer = tokenizer
        self.weights = weights or self._default_weights()

    def _default_weights(self) -> Dict[str, float]:
        """
        Default feature weights for ranking.

        Returns:
            Dictionary of feature name → weight
        """
        return {
            'token_length': -1.0,
            'ast_depth': -0.5,
            'num_operators': -0.3,
            'num_variables': 0.0,      # Neutral (should match input)
            'num_constants': -0.2,      # Slightly prefer fewer constants
            'model_confidence': 2.0,
            'verification_level': 5.0,
        }

    def extract_features(
        self,
        candidate: str,
        model_score: float,
        verification: VerificationResult
    ) -> RankingFeatures:
        """
        Extract ranking features from candidate.

        Args:
            candidate: Expression string
            model_score: Model's log probability for this candidate
            verification: VerificationResult from three-tier verification

        Returns:
            RankingFeatures dataclass
        """
        # Token length
        tokens = self.tokenizer.tokenize(candidate)
        token_length = len(tokens)

        # AST metrics
        ast_depth, num_operators, num_variables, num_constants = self._get_ast_metrics(candidate)

        # Verification level
        if verification.z3_verified:
            verification_level = 3
        elif verification.exec_valid:
            verification_level = 2
        elif verification.syntax_valid:
            verification_level = 1
        else:
            verification_level = 0

        return RankingFeatures(
            candidate=candidate,
            token_length=token_length,
            ast_depth=ast_depth,
            num_operators=num_operators,
            num_variables=num_variables,
            num_constants=num_constants,
            model_confidence=model_score,
            verification_level=verification_level,
            counterexample_found=(verification.counterexample is not None)
        )

    def compute_score(self, features: RankingFeatures) -> float:
        """
        Compute ranking score from features.

        Args:
            features: RankingFeatures instance

        Returns:
            Scalar ranking score (higher is better)

        Formula:
            score = Σ(weight_i * normalize(feature_i))

        Normalization:
            - token_length: / 64 (MAX_SEQ_LEN)
            - ast_depth: / 16 (MAX_OUTPUT_DEPTH)
            - num_operators: / 20 (typical max)
            - model_confidence: already in log prob range
            - verification_level: {0, 1, 2, 3} directly
        """
        # Filter candidates with counterexamples
        if features.counterexample_found:
            return float('-inf')

        score = 0.0

        # Normalize and weight features
        score += self.weights['token_length'] * (features.token_length / 64.0)
        score += self.weights['ast_depth'] * (features.ast_depth / 16.0)
        score += self.weights['num_operators'] * (features.num_operators / 20.0)
        score += self.weights['num_variables'] * (features.num_variables / 8.0)
        score += self.weights['num_constants'] * (features.num_constants / 10.0)
        score += self.weights['model_confidence'] * features.model_confidence
        score += self.weights['verification_level'] * features.verification_level

        return score

    def rerank(
        self,
        candidates: List[str],
        model_scores: List[float],
        verifications: List[VerificationResult]
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates by combined score.

        Args:
            candidates: List of candidate expressions
            model_scores: Model log probabilities for each candidate
            verifications: VerificationResults for each candidate

        Returns:
            List of (candidate, score) tuples sorted by descending score
        """
        ranked = []

        for candidate, model_score, verification in zip(candidates, model_scores, verifications):
            features = self.extract_features(candidate, model_score, verification)
            score = self.compute_score(features)
            ranked.append((candidate, score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Filter out -inf scores (counterexamples)
        ranked = [(c, s) for c, s in ranked if s != float('-inf')]

        return ranked

    def _get_ast_metrics(self, expr: str) -> Tuple[int, int, int, int]:
        """
        Compute AST metrics for expression.

        Args:
            expr: Expression string

        Returns:
            (depth, num_operators, num_variables, num_constants)
        """
        try:
            ast = parse_to_ast(expr)

            # Depth
            depth = get_ast_depth(ast)

            # Count node types
            num_operators = 0
            num_variables = 0
            num_constants = 0

            def count_nodes(node):
                nonlocal num_operators, num_variables, num_constants
                if node.type == 'VAR':
                    num_variables += 1
                elif node.type == 'CONST':
                    num_constants += 1
                elif node.is_binary() or node.is_unary():
                    num_operators += 1

                for child in node.children:
                    count_nodes(child)

            count_nodes(ast)

            return (depth, num_operators, num_variables, num_constants)
        except:
            # Parse failed - return defaults
            return (0, 0, 0, 0)
