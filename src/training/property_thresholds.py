"""
Threshold constants for fingerprint-based property detection.

Thresholds are tuned for 70-80% accuracy based on:
    - Walsh-Hadamard spectral analysis
    - Operator ratio statistics
    - Corner evaluation variance
    - Truth table entropy

Calibration: Run scripts/calibrate_property_thresholds.py on validation set
"""

PROPERTY_THRESHOLDS = {
    # LINEAR detection
    'linear_degree_max': 1.2,           # Walsh degree estimate (normalized by 6)
    'linear_variance_max': 100.0,       # Corner variance threshold

    # QUADRATIC detection
    'quadratic_degree_min': 1.5,        # Minimum degree
    'quadratic_degree_max': 2.5,        # Maximum degree

    # CONST_CONTRIB detection
    'const_variance_min': 50.0,         # Corner variance for constants

    # BOOLEAN_ONLY detection
    'boolean_only_ratio': 0.95,         # Boolean ops / total ops

    # ARITHMETIC_ONLY detection
    'arithmetic_only_ratio': 0.95,      # Arithmetic ops / total ops

    # COMPLEMENTARY detection
    'complementary_cancel_ratio': 0.6,  # XOR cancellation ratio

    # MASKED detection
    'masked_cv_max': 0.3,               # Coefficient of variation

    # MIXED_DOMAIN detection
    'mixed_domain_min_ratio': 0.2,      # Minimum ratio for each domain
}


def get_default_thresholds():
    """
    Get default threshold dictionary.

    Returns:
        Dictionary of threshold values for property detection
    """
    return PROPERTY_THRESHOLDS.copy()


def update_thresholds(new_thresholds: dict):
    """
    Update thresholds dynamically (for calibration).

    Args:
        new_thresholds: Dictionary of threshold updates

    Example:
        update_thresholds({'linear_degree_max': 1.5, 'boolean_only_ratio': 0.90})
    """
    global PROPERTY_THRESHOLDS
    PROPERTY_THRESHOLDS.update(new_thresholds)
