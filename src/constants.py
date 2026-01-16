"""
Shared constants for MBA Deobfuscator.

All modules MUST import dimension constants from here to ensure consistency.
These values are derived from docs/ML_PIPELINE.md specifications.
"""

from typing import Dict, List, Tuple

# =============================================================================
# AST NODE REPRESENTATION
# =============================================================================

NODE_TYPES: Dict[str, int] = {
    'VAR': 0,      # Variables: x, y, z, ...
    'CONST': 1,    # Constants: 0, 1, 2, ...
    'ADD': 2,      # +
    'SUB': 3,      # -
    'MUL': 4,      # *
    'AND': 5,      # &
    'OR': 6,       # |
    'XOR': 7,      # ^
    'NOT': 8,      # ~
    'NEG': 9,      # unary -
}

NODE_TYPE_TO_STR: Dict[int, str] = {v: k for k, v in NODE_TYPES.items()}

NUM_NODE_TYPES: int = len(NODE_TYPES)
NODE_DIM: int = 32  # Node feature dimension (one-hot + positional info)

# =============================================================================
# EDGE TYPES (for GGNN)
# =============================================================================

EDGE_TYPES: Dict[str, int] = {
    'CHILD_LEFT': 0,   # parent -> left operand
    'CHILD_RIGHT': 1,  # parent -> right operand
    'PARENT': 2,       # child -> parent
    'SIBLING_NEXT': 3, # left -> right sibling
    'SIBLING_PREV': 4, # right -> left sibling
    'SAME_VAR': 5,     # links all uses of same variable
}

NUM_EDGE_TYPES: int = len(EDGE_TYPES)

# =============================================================================
# TOKENIZER / VOCABULARY
# =============================================================================

SPECIAL_TOKENS: Dict[str, int] = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    '<unk>': 3,
    '<sep>': 4,
}

PAD_IDX: int = 0
SOS_IDX: int = 1
EOS_IDX: int = 2
UNK_IDX: int = 3

OPERATORS: List[str] = ['+', '-', '*', '&', '|', '^', '~']
PARENS: List[str] = ['(', ')']

MAX_VARS: int = 8        # x0 through x7
MAX_SEQ_LEN: int = 64    # Maximum output sequence length
MAX_CONST: int = 256     # Constants 0-255 get dedicated tokens

# Approximate vocab size: 5 special + 7 ops + 2 parens + 8 vars + 256 consts = ~278
VOCAB_SIZE: int = 300    # Padded for safety

# =============================================================================
# SEMANTIC FINGERPRINT
# =============================================================================

SYMBOLIC_DIM: int = 32        # Symbolic features
CORNER_DIM: int = 256         # 4 widths x 64 corner cases
RANDOM_DIM: int = 64          # 4 widths x 16 hash values
DERIVATIVE_DIM: int = 32      # 4 widths x 8 derivative orders
TRUTH_TABLE_DIM: int = 64     # 2^6 entries for 6 variables

FINGERPRINT_DIM: int = SYMBOLIC_DIM + CORNER_DIM + RANDOM_DIM + DERIVATIVE_DIM + TRUTH_TABLE_DIM
assert FINGERPRINT_DIM == 448, f"Fingerprint dimension mismatch: {FINGERPRINT_DIM}"

TRUTH_TABLE_VARS: int = 6     # Number of variables for truth table

# Bit widths for evaluation
BIT_WIDTHS: List[int] = [8, 16, 32, 64]

# Corner case values (per width, computed at runtime)
def get_corner_values(width: int) -> List[int]:
    """Get corner case values for a given bit width."""
    max_val = (1 << width) - 1
    half = 1 << (width - 1)
    corners = [
        0, 1, 2, 3,                           # Small values
        max_val, max_val - 1, max_val - 2,    # Near max
        half, half - 1, half + 1,             # Around midpoint
        0xAA & max_val, 0x55 & max_val,       # Alternating bits
    ]
    # Add powers of 2
    for i in range(width):
        val = 1 << i
        if val <= max_val and val not in corners:
            corners.append(val)
        val_minus_1 = (1 << i) - 1
        if val_minus_1 not in corners:
            corners.append(val_minus_1)
    return corners[:64]  # Limit to 64 per width

# =============================================================================
# MODEL DIMENSIONS
# =============================================================================

# Encoder (GNN)
HIDDEN_DIM: int = 256         # GNN hidden dimension
NUM_ENCODER_LAYERS: int = 4   # GAT layers
NUM_ENCODER_HEADS: int = 8    # GAT attention heads
ENCODER_DROPOUT: float = 0.1

# GGNN specific
GGNN_TIMESTEPS: int = 8       # Message passing iterations

# Decoder (Transformer)
D_MODEL: int = 512            # Transformer model dimension
NUM_DECODER_LAYERS: int = 6   # Transformer decoder layers
NUM_DECODER_HEADS: int = 8    # Transformer attention heads
D_FF: int = 2048              # Feed-forward dimension
DECODER_DROPOUT: float = 0.1

# Output heads
MAX_OUTPUT_LENGTH: int = 64   # For ComplexityHead
MAX_OUTPUT_DEPTH: int = 16    # For ComplexityHead

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Phase 1: Contrastive
INFONCE_TEMPERATURE: float = 0.07
MASKLM_MASK_RATIO: float = 0.15
MASKLM_WEIGHT: float = 0.5

# Phase 2: Supervised
CE_WEIGHT: float = 1.0
COMPLEXITY_WEIGHT: float = 0.1
COPY_WEIGHT: float = 0.1

# Curriculum stages
CURRICULUM_STAGES: List[Dict] = [
    {'max_depth': 2, 'epochs': 10, 'target': 0.95},
    {'max_depth': 5, 'epochs': 15, 'target': 0.90},
    {'max_depth': 10, 'epochs': 15, 'target': 0.80},
    {'max_depth': 14, 'epochs': 10, 'target': 0.70},
]

# Self-paced learning
SELF_PACED_LAMBDA_INIT: float = 0.5
SELF_PACED_LAMBDA_GROWTH: float = 1.1

# Phase 3: RL
PPO_EPSILON: float = 0.2
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01

# Reward function
REWARD_EQUIV_BONUS: float = 10.0
REWARD_LEN_PENALTY: float = 0.1
REWARD_DEPTH_PENALTY: float = 0.2
REWARD_IDENTITY_PENALTY: float = 5.0
REWARD_SYNTAX_ERROR_PENALTY: float = 5.0
REWARD_SIMPLIFICATION_BONUS: float = 2.0
REWARD_IDENTITY_THRESHOLD: float = 0.9

# =============================================================================
# INFERENCE PARAMETERS
# =============================================================================

BEAM_WIDTH: int = 50
BEAM_DIVERSITY_GROUPS: int = 4
BEAM_DIVERSITY_PENALTY: float = 0.5
BEAM_TEMPERATURE: float = 0.7

HTPS_BUDGET: int = 500
HTPS_DEPTH_THRESHOLD: int = 10
HTPS_UCB_CONSTANT: float = 1.414

# Verification
EXEC_TEST_SAMPLES: int = 100
Z3_TIMEOUT_MS: int = 1000
Z3_TOP_K: int = 10

# =============================================================================
# HTPS TACTICS
# =============================================================================

HTPS_TACTICS: List[str] = [
    'identity_xor_self',    # x ^ x -> 0
    'identity_and_not',     # x & ~x -> 0
    'identity_or_not',      # x | ~x -> -1
    'mba_and_xor',          # (x&y)+(x^y) -> x|y
    'constant_fold',        # 3 + 5 -> 8
    'simplify_subexpr',     # Recurse on subexpression
]

# =============================================================================
# SCALED MODEL DIMENSIONS (360M parameters)
# =============================================================================
# For use with 12M sample dataset (7.2B tokens, Chinchilla-optimal)

# Scaled Encoder (HGT/RGCN) - ~60M params
SCALED_HIDDEN_DIM: int = 768
SCALED_NUM_ENCODER_LAYERS: int = 12
SCALED_NUM_ENCODER_HEADS: int = 16
SCALED_ENCODER_DROPOUT: float = 0.1

# Scaled Decoder (Transformer) - ~302M params
# 8 layers × ~37.7M/layer = ~302M
SCALED_D_MODEL: int = 1536
SCALED_NUM_DECODER_LAYERS: int = 8
SCALED_NUM_DECODER_HEADS: int = 24
SCALED_D_FF: int = 6144
SCALED_DECODER_DROPOUT: float = 0.1

# Scaled sequence length (for depth 14 expressions)
SCALED_MAX_SEQ_LEN: int = 2048

# Optimized edge/node types
NUM_OPTIMIZED_EDGE_TYPES: int = 8   # LEFT/RIGHT/UNARY_OPERAND + inverses + DOMAIN_BRIDGE_DOWN/UP
NUM_NODE_TYPES_HETEROGENEOUS: int = 10  # ADD,SUB,MUL,NEG,AND,OR,XOR,NOT,VAR,CONST

# Scaled curriculum (1.5× epochs for larger model)
SCALED_CURRICULUM_STAGES: List[Dict] = [
    {'max_depth': 2, 'epochs': 15, 'target': 0.95},
    {'max_depth': 5, 'epochs': 23, 'target': 0.90},
    {'max_depth': 10, 'epochs': 23, 'target': 0.85},
    {'max_depth': 14, 'epochs': 15, 'target': 0.80},
]

# =============================================================================
# ABLATION STUDY PARAMETERS
# =============================================================================

# Depth buckets for per-depth analysis in ablation studies
ABLATION_DEPTH_BUCKETS: List[Tuple[int, int]] = [
    (2, 4),    # Easy expressions
    (5, 7),    # Medium expressions
    (8, 10),   # Hard expressions
    (11, 14),  # Very hard expressions
]

# Number of runs per encoder for statistical significance
ABLATION_NUM_RUNS: int = 5

# Significance level for statistical tests
ABLATION_SIGNIFICANCE_LEVEL: float = 0.05
