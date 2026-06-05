"""Deterministic seeding for reproducible training.

Seeds every RNG the project touches (Python, NumPy, PyTorch CPU/CUDA) and
enables deterministic algorithms so that, given the same data split and seed,
training and evaluation are reproducible.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 1337, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is part of the ml extra
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuDNN determinism + a fixed workspace config make conv/matmul
            # kernels reproducible at a small performance cost.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - older torch
                pass
    except ImportError:  # pragma: no cover - torch is part of the ml extra
        pass

    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - exercised by DataLoader
    """`worker_init_fn` for DataLoader so each worker is deterministically seeded."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


__all__ = ["seed_everything", "seed_worker"]
