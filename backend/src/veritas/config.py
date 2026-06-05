"""Project-wide configuration.

Kept dependency-free (standard library only) so the data-preparation pipeline
runs on a bare Python interpreter.  Values can be overridden via environment
variables (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository layout ---------------------------------------------------------
# ``config.py`` lives at ``backend/src/veritas/config.py``; the repo root is
# four parents up.
PACKAGE_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Paths:
    """Canonical on-disk locations for data and artifacts."""

    data_root: Path = field(default_factory=lambda: _env_path("VERITAS_DATA_ROOT", REPO_ROOT / "data"))

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def models(self) -> Path:
        return self.data_root / "models"

    def modality_processed(self, modality: str) -> Path:
        return self.processed / modality


# Default split fractions.  Test + val carved out of the whole; train is the
# remainder.  Chosen to give a substantial training majority while keeping
# evaluation sets large enough to be meaningful.
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15

# Global reproducibility seed.  Every RNG in the project derives from this so
# that data splits and training runs are deterministic.
DEFAULT_SEED = int(os.environ.get("VERITAS_SEED", "1337"))

PATHS = Paths()

__all__ = [
    "PATHS",
    "Paths",
    "REPO_ROOT",
    "BACKEND_ROOT",
    "PACKAGE_ROOT",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
    "DEFAULT_SEED",
]
