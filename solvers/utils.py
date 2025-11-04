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


def compute_nrmse(field_low, field_high, eps=1e-12):
    """
    Compute Normalized Root Mean Square Error (NRMSE) between two fields.

    Uses the higher resolution field's mean and standard deviation for normalization.
    This avoids issues with small denominators in relative error.

    Algorithm:
    1. Normalize both fields using high-res statistics: (field - mean_high) / std_high
    2. Compute RMSE on normalized difference

    Args:
        field_low: Lower resolution or less accurate field (1D array)
        field_high: Higher resolution or ground truth field (1D array)
        eps: Small epsilon to avoid division by zero (default: 1e-12)

    Returns:
        float: NRMSE value (scalar)
    """
    # Use higher resolution field statistics for normalization
    mean_high = np.mean(field_high)
    std_high = np.std(field_high) + eps  # Add eps to avoid division by zero

    # Normalize both fields
    field_low_norm = (field_low - mean_high) / std_high
    field_high_norm = (field_high - mean_high) / std_high

    # Compute RMSE on normalized fields
    diff = field_low_norm - field_high_norm
    rmse = np.sqrt(np.mean(diff ** 2))

    return rmse


def compute_nrmse_maxabs(field_low, field_high, eps=1e-12):
    """
    Compute Normalized Root Mean Square Error (NRMSE) using max absolute value normalization.

    Uses the higher resolution field's maximum absolute value for normalization.
    This metric is scale-independent and bounded by the data range.

    Algorithm:
    1. Compute RMSE: sqrt(mean((field_low - field_high)^2))
    2. Normalize by max absolute value of high-res field

    Args:
        field_low: Lower resolution or less accurate field (1D array)
        field_high: Higher resolution or ground truth field (1D array)
        eps: Small epsilon to avoid division by zero (default: 1e-12)

    Returns:
        float: NRMSE value normalized by max absolute value (scalar)
    """
    # Compute RMSE
    diff = field_low - field_high
    rmse = np.sqrt(np.mean(diff ** 2))

    # Normalize by max absolute value of high-res field
    max_abs_high = np.max(np.abs(field_high)) + eps  # Add eps to avoid division by zero

    return rmse / max_abs_high


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
            # Use fixed point for reasonable range, remove trailing zeros only after decimal point
            formatted = f"{value:.6g}"
            # Only strip trailing zeros if there's a decimal point
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
        else:
            # Use scientific notation for very small/large values
            return f"{value:.2e}"
    else:
        return str(value)
