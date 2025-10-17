"""
Utility functions for solver-related operations.
"""
import numpy as np


def compute_relative_error(a, b, eps=1e-12):
    """
    Compute relative error between two arrays using mean values as denominator.

    This is a common operation across multiple solvers when comparing solutions.

    Args:
        a: First array
        b: Second array
        eps: Small epsilon to avoid division by zero (default: 1e-12)

    Returns:
        Array of relative errors: |a - b| / (0.5 * (mean(|a|) + mean(|b|)) + eps)
    """
    mean_a = np.mean(np.abs(a))
    mean_b = np.mean(np.abs(b))
    denom = 0.5 * (mean_a + mean_b) + eps
    return np.abs(a - b) / denom


def format_param_for_path(value):
    """
    Format parameter values for clean folder/file names.

    Args:
        value: Parameter value (float, int, or other)

    Returns:
        str: Cleanly formatted string suitable for file paths
    """
    if isinstance(value, float):
        if value >= 1e-3 and value < 1e3:
            # Use fixed point for reasonable range, remove trailing zeros
            return f"{value:.6g}".rstrip("0").rstrip(".")
        else:
            # Use scientific notation for very small/large values
            return f"{value:.2e}"
    else:
        return str(value)
