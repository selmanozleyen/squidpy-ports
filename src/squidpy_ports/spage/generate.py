"""Emit the reference bundle squidpy's SpaGE reference tests compare against.

Run with::

    hatch run spage:generate --out spage_reference.zarr.zip

One AnnData in a zipped zarr store, the same format as the STalign bundle: the inputs in
``uns["inputs"]``, the fixture parameters in ``uns["fixtures"]``, and one section per
``(cosine_threshold, n_neighbors)`` in :data:`.fixtures.SETTINGS`, each with its provenance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from squidpy_ports._common import THREADS, generator_main, provenance

from . import fixtures as F
from . import upstream

__all__ = ["build", "main", "section_name"]

#: Versions recorded in the provenance.
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn")


def section_name(cosine_threshold: float, n_neighbors: int) -> str:
    """Bundle key of one run, e.g. ``cosine30_k20`` (no dots: they are zarr path noise)."""
    return f"cosine{round(cosine_threshold * 100)}_k{n_neighbors}"


def build() -> dict[str, Any]:
    """Run upstream on every setting and return the bundle's ``uns``."""
    spage = upstream.load()
    data = F.make_data()
    thresholds, neighbors = zip(*F.SETTINGS, strict=True)
    uns = data.uns(F, N_PV=F.N_PV, COSINE_THRESHOLDS=np.asarray(thresholds), N_NEIGHBORS=np.asarray(neighbors))
    for cosine_threshold, n_neighbors in F.SETTINGS:
        params = {"cosine_threshold": cosine_threshold, "n_neighbors": n_neighbors}
        uns[section_name(**params)] = {
            **upstream.run(spage, data, n_pv=F.N_PV, **params),
            **params,
            "provenance": provenance(upstream, F, PACKAGES, **params),
        }
    return uns


def main(argv: list[str] | None = None) -> int:
    """Generate the whole bundle."""
    return generator_main(build, THREADS, argv, __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
