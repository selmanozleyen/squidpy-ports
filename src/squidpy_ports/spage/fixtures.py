"""Fixture definitions the SpaGE reference bundle is generated from.

The arrays are stored in the bundle too, so squidpy's tests read the inputs from the bundle
rather than rebuilding them. It depends on **numpy only**.
"""

from __future__ import annotations

import numpy as np

from squidpy_ports._common import Imputation
from squidpy_ports._common import checksum as _checksum

__all__ = ["GENES", "N_PV", "N_QUERY", "N_REF", "SEED", "SETTINGS", "checksum", "make_data"]

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


def make_data() -> Imputation:
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
    return Imputation(ref, query[:, :N_MEASURED], query[:, N_MEASURED:], GENES)


def checksum() -> str:
    """SHA-256 of this file, so the bundle can prove which definitions it came from."""
    return _checksum(__file__)
