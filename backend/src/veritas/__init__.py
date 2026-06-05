"""Veritas — deepfake & AI-content detection backend.

The package is intentionally split so that the *data preparation* core
(:mod:`veritas.data`) depends only on the Python standard library.  The heavier
machine-learning, inference and serving layers live behind optional dependency
groups (``ml``, ``api``) and are imported lazily by the relevant subpackages.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
