"""Emit the reference bundle squidpy's Tangram reference tests compare against.

Run with::

    hatch run tangram:generate --out tangram_reference.zarr.zip

One AnnData in a zipped zarr store, the same format as the STalign bundle: the inputs in
``uns["inputs"]``, the fixture parameters in ``uns["fixtures"]``, and one section per
``density_prior`` in :data:`.fixtures.SETTINGS` and epoch count in
:data:`.fixtures.EPOCHS`, each with its provenance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from squidpy_ports._common import THREADS, generator_main, provenance

from . import fixtures as F
from . import upstream

__all__ = ["build", "main", "section_name"]

#: Versions recorded in the provenance.
PACKAGES = ("numpy", "pandas", "scipy", "torch", "anndata", "scanpy")


def section_name(density_prior: str | None, num_epochs: int) -> str:
    """Bundle key of one run, e.g. ``rna_count_based_epochs300`` or ``none_epochs1``."""
    return f"{density_prior or 'none'}_epochs{num_epochs}"


def build() -> dict[str, Any]:
    """Run upstream on every setting and return the bundle's ``uns``."""
    import torch

    torch.set_num_threads(1)
    tg = upstream.load()
    data = F.make_data()

    uns = data.uns(
        F,
        RANDOM_STATE=F.RANDOM_STATE,
        LEARNING_RATE=F.LEARNING_RATE,
        SETTINGS=np.asarray([prior or "none" for prior in F.SETTINGS]),
        EPOCHS=np.asarray(F.EPOCHS),
    )
    for density_prior in F.SETTINGS:
        for num_epochs in F.EPOCHS:
            out = upstream.run(
                tg,
                data,
                density_prior=density_prior,
                num_epochs=num_epochs,
                learning_rate=F.LEARNING_RATE,
                random_state=F.RANDOM_STATE,
            )
            # one initial mapping and gene order for all runs, stored once
            for key in ("initial_logits", "training_genes"):
                value = out.pop(key)
                np.testing.assert_array_equal(uns["inputs"].setdefault(key, value), value)
            params = {"density_prior": density_prior or "none", "num_epochs": num_epochs}
            uns[section_name(density_prior, num_epochs)] = {
                **out,
                **params,
                "provenance": provenance(upstream, F, PACKAGES, **params),
            }
    return uns


def main(argv: list[str] | None = None) -> int:
    """Generate the whole bundle."""
    # upstream orders the training genes by iterating a set of strings
    return generator_main(build, {**THREADS, "PYTHONHASHSEED": "0"}, argv, __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
