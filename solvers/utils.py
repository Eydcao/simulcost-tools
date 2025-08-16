"""
Utility functions for solver-related operations.
"""


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
