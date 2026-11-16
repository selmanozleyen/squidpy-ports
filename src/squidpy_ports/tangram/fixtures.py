"""Fixture definitions the Tangram reference bundle is generated from.

The arrays are stored in the bundle too, so squidpy's tests read the inputs from the bundle
rather than rebuilding them. It depends on **numpy only**.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = [
    "EPOCHS",
    "GENES",
    "LEARNING_RATE",
    "N_QUERY",
    "N_REF",
    "RANDOM_STATE",
    "SEED",
    "SETTINGS",
    "Data",
    "checksum",
    "make_data",
]

SEED = 0
GENES = [f"g{i}" for i in range(10)]
#: The query measures the first 7 genes; the other 3 are imputed.
N_MEASURED = 7
N_TYPES = 3
N_REF = 60
N_QUERY = 40

#: Seeds upstream's initial mapping. Not 0: upstream seeds only ``if self.random_state:``.
RANDOM_STATE = 1
LEARNING_RATE = 0.1
#: ``density_prior`` per run, upstream's default first. Upstream weighs the density term 1 whenever a
#: prior is given (mapping_utils.py:214), so these are the same three choices as squidpy's.
SETTINGS = ("rna_count_based", "uniform", None)
#: Every setting is run for each of these epoch counts, from the same initial mapping.
EPOCHS = (1, 10, 100, 300)


class Data(NamedTuple):
    """Reference and query expression, and the true query values of the unmeasured genes."""

    ref: np.ndarray
    query: np.ndarray
    truth: np.ndarray

    @property
    def measured(self) -> list[str]:
        """Genes the query measures."""
        return GENES[:N_MEASURED]

    @property
    def unmeasured(self) -> list[str]:
        """Genes the query lacks, which are imputed."""
        return GENES[N_MEASURED:]


def make_data() -> Data:
    """A reference of 3 cell types, and a query of noisy copies of its cells.

    Identical to ``_data()`` in squidpy's ``tests/experimental/test_impute_tangram.py``.
    """
    rng = np.random.default_rng(SEED)
    profiles = rng.gamma(2.0, size=(N_TYPES, len(GENES)))
    ref = profiles[rng.integers(N_TYPES, size=N_REF)] * rng.uniform(0.8, 1.2, size=(N_REF, len(GENES)))
    query = ref[rng.integers(N_REF, size=N_QUERY)] * rng.uniform(0.9, 1.1, size=(N_QUERY, len(GENES)))
    return Data(ref=ref, query=query[:, :N_MEASURED], truth=query[:, N_MEASURED:])


def checksum() -> str:
    """SHA-256 of this file, so the bundle can prove which definitions it came from."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
