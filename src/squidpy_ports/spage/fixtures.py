"""Fixture definitions the SpaGE reference bundle is generated from.

The arrays are stored in the bundle too, so squidpy's tests read the inputs from the bundle
rather than rebuilding them. It depends on **numpy only**.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["GENES", "N_PV", "N_QUERY", "N_REF", "SEED", "SETTINGS", "Data", "checksum", "make_data"]

SEED = 0
GENES = [f"g{i}" for i in range(12)]
#: The query measures the first 8 genes; the other 4 are imputed.
N_MEASURED = 8
RANK = 3
NOISE = 0.1
N_REF = 300
N_QUERY = 150

N_PV = 3
#: ``(cosine_threshold, n_neighbors)`` per run. ``(0.3, 50)`` is what upstream hard-codes. 0.99 drops
#: a principal vector; 300 makes every reference cell a neighbor, so the ones at cosine distance
#: >= 1 are dropped and neighbor ties cannot matter.
SETTINGS = [(t, k) for t in (0.3, 0.99) for k in (20, 50, 300)]


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
    """Sample reference and query from one rank-3 model, clipped at 0 like expression.

    Identical to ``_data()`` in squidpy's ``tests/experimental/test_impute_spage.py``.
    """
    rng = np.random.default_rng(SEED)
    loadings = rng.normal(size=(RANK, len(GENES)))

    def sample(n: int) -> np.ndarray:
        x = rng.normal(size=(n, RANK)) @ loadings + NOISE * rng.normal(size=(n, len(GENES)))
        return np.maximum(x, 0)

    query = sample(N_QUERY)
    ref = sample(N_REF)
    return Data(ref=ref, query=query[:, :N_MEASURED], truth=query[:, N_MEASURED:])


def checksum() -> str:
    """SHA-256 of this file, so the bundle can prove which definitions it came from."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
