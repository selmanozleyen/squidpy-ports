"""Load the vendored upstream SpaGE and run it without modifying it.

Upstream is imported *by path* from ``vendor/SpaGE``: it has no ``setup.py`` and is not on
PyPI, and ``main.py`` imports its siblings as ``SpaGE.principal_vectors``.
"""

from __future__ import annotations

from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from squidpy_ports._common import ROOT, import_vendored, verify_pin

if TYPE_CHECKING:
    from squidpy_ports._common import Imputation

__all__ = ["UPSTREAM_SHA", "UPSTREAM_URL", "VENDOR", "load", "run"]

#: The single upstream commit this package is a reference for. Upstream has no tags.
UPSTREAM_SHA = "edd0fb7086eca791c6589020893eb4f034b195c7"
UPSTREAM_URL = "https://github.com/tabdelaal/SpaGE"
#: sha256 of each file under ``SpaGE/`` at :data:`UPSTREAM_SHA`.
SHA256 = {
    "main.py": "32030e441e37a5075c2682cf8b5074e9c2fc094818a4ad9b9352bfb27a253f3b",
    "principal_vectors.py": "47f857a5d502782bc5e033091caee6c40b424a113af9ff841f62e26183a4b8fd",
    "dimensionality_reduction.py": "47f5bb301685005d3854807f0cfc8a696c2988a8fdaebf75a97988bd60fe0e1c",
}
#: What ``SpaGE()`` hard-codes (main.py:78 and :84).
COSINE_THRESHOLD = 0.3
N_NEIGHBORS = 50


#: The vendored checkout.
VENDOR = ROOT / "vendor" / "SpaGE"


@cache
def load() -> ModuleType:
    """Import the vendored ``SpaGE.main`` module, asserting the pin (see :func:`.verify_pin`)."""
    verify_pin(VENDOR / "SpaGE", SHA256, UPSTREAM_SHA)
    return import_vendored(VENDOR, "SpaGE.main")


def run(
    spage: ModuleType,
    data: Imputation,
    *,
    n_pv: int,
    cosine_threshold: float = COSINE_THRESHOLD,
    n_neighbors: int = N_NEIGHBORS,
) -> dict[str, Any]:
    """Impute ``data``'s unmeasured genes with upstream ``SpaGE()``, the two settings it hard-codes swapped in.

    ``SpaGE()`` takes neither a cosine threshold nor a neighbor count, so both are injected by
    replacing names in ``SpaGE.main``'s namespace for the duration of the call, the same way
    :func:`squidpy_ports.stalign.upstream.lddmm_with_grads` observes STalign:

    - ``PVComputation`` is subclassed so that ``fit`` records the cosine similarities and, for a
      threshold other than 0.3, replaces the diagonal ``SpaGE()`` thresholds at 0.3 with 1 where it
      exceeds ``cosine_threshold`` and 0 elsewhere. Nothing else reads that matrix.
    - ``NearestNeighbors`` is wrapped to swap ``n_neighbors`` and to record the query it is asked
      about; the number of principal vectors used is the width of the projection it is fitted on.

    At the hard-coded settings the result is the same as calling ``SpaGE()`` directly.

    Identical reference cells are at exactly the same distance from every query cell. Where such a tie
    straddles the ``n_neighbors``-th neighbor, which cell is kept is up to the search backend, so
    another implementation may legitimately differ on that row. Those rows are returned as
    ``tied_rows``: the query cells whose ``n_neighbors``-th and next distance are equal.

    Returns
    -------
    ``imputed`` (query cells x unmeasured genes), and the observed ``cosine`` diagonal, ``n_pv_used``
    and ``tied_rows``.
    """
    import pandas as pd

    ref = pd.DataFrame(data.ref, columns=data.genes)
    query = pd.DataFrame(data.query, columns=data.measured)
    captured: dict[str, Any] = {}
    real_pv, real_nn = spage.PVComputation, spage.NearestNeighbors

    class PVComputation(real_pv):
        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)
            cosine = np.diag(self.cosine_similarity_matrix_).copy()
            captured["cosine"] = cosine
            if cosine_threshold != COSINE_THRESHOLD:
                self.cosine_similarity_matrix_ = np.diag((cosine > cosine_threshold).astype(float))
            return self

    def nearest_neighbors(**kwargs):
        if kwargs.get("n_neighbors") != N_NEIGHBORS:
            raise RuntimeError(f"SpaGE() now builds NearestNeighbors({kwargs}); re-check upstream.run")
        captured["nn"] = Recording(**{**kwargs, "n_neighbors": n_neighbors})
        return captured["nn"]

    class Recording(real_nn):
        def kneighbors(self, X=None, *args, **kwargs):
            captured["query"] = X
            return super().kneighbors(X, *args, **kwargs)

    spage.PVComputation, spage.NearestNeighbors = PVComputation, nearest_neighbors
    try:
        imputed = spage.SpaGE(query, ref, n_pv, genes_to_predict=data.unmeasured)
    finally:
        spage.PVComputation, spage.NearestNeighbors = real_pv, real_nn

    nn = captured["nn"]
    tied_rows = np.array([], dtype=int)
    if n_neighbors < nn.n_samples_fit_:
        dist = nn.kneighbors(captured["query"], n_neighbors + 1, return_distance=True)[0]
        tied_rows = np.flatnonzero(dist[:, n_neighbors - 1] == dist[:, n_neighbors])
    return {
        "imputed": imputed[data.unmeasured].to_numpy(),
        "cosine": captured["cosine"],
        "n_pv_used": nn.n_features_in_,
        "tied_rows": tied_rows,
    }
