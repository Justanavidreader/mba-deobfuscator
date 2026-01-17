#!/usr/bin/env python3
"""
Test script for pybind11 C++ integration.

Demonstrates:
- Checking C++ availability
- Using both Python and C++ implementations
- Automatic fallback behavior
"""

from src.data.fingerprint import (
    SemanticFingerprint,
    has_cpp_acceleration,
    get_implementation_info
)
import numpy as np
import json


def main():
    print("=" * 70)
    print("pybind11 C++ Integration Test")
    print("=" * 70)
    print()

    # Check implementation availability
    print("Implementation Info:")
    info = get_implementation_info()
    print(json.dumps(info, indent=2))
    print()

    if has_cpp_acceleration():
        print("✓ C++ acceleration AVAILABLE")
    else:
        print("✗ C++ acceleration NOT available (using Python)")
    print()

    # Create fingerprint computers
    print("-" * 70)
    print("Creating Fingerprint Computers")
    print("-" * 70)

    # Default (tries C++ if available)
    fp_default = SemanticFingerprint()
    print(f"Default:     use_cpp={fp_default.use_cpp}")

    # Explicitly request Python
    fp_python = SemanticFingerprint(use_cpp=False)
    print(f"Python only: use_cpp={fp_python.use_cpp}")

    # Explicitly request C++ (may warn if unavailable)
    fp_cpp_req = SemanticFingerprint(use_cpp=True)
    print(f"C++ request: use_cpp={fp_cpp_req.use_cpp}")
    print()

    # Test fingerprint computation
    print("-" * 70)
    print("Computing Fingerprints")
    print("-" * 70)

    test_expressions = [
        "(x0+x1)",
        "((x0|x1)+(x0&x1))",
        "(x0&(x1^x2))",
    ]

    for expr in test_expressions:
        print(f"\nExpression: {expr}")

        # Compute with default
        fp = fp_default.compute(expr)
        print(f"  Dimensions: {fp.shape}")
        print(f"  Dtype: {fp.dtype}")
        print(f"  First 5 values: {fp[:5]}")
        print(f"  Implementation: {'C++' if fp_default.use_cpp else 'Python'}")

        # Verify shape and type
        assert fp.shape == (448,), f"Expected shape (448,), got {fp.shape}"
        assert fp.dtype == np.float64, f"Expected float64, got {fp.dtype}"

    print()
    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
