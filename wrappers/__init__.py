"""
Wrapper functions for various PDE solvers.

This package uses lazy loading - wrappers are imported on-demand to minimize
startup time and avoid loading unnecessary dependencies (like Taichi).

IMPORTANT: Do NOT add top-level imports here, as they will break lazy loading.
Wrappers are loaded dynamically when needed by their respective tool_call modules.
"""
