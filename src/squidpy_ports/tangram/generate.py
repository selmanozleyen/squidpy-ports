"""Emit the reference bundle squidpy's Tangram reference tests compare against.

Run with::

    hatch run tangram:generate --out tangram_reference.zarr.zip

One AnnData in a zipped zarr store, the same format as the STalign bundle: the inputs in
``uns["inputs"]``, the fixture parameters in ``uns["fixtures"]``, and one section per
``density_prior`` in :data:`.fixtures.SETTINGS` and epoch count in
:data:`.fixtures.EPOCHS`, each with its provenance.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from squidpy_ports._bundle import provenance, write_bundle

from . import fixtures as F
from . import upstream

__all__ = ["build", "main", "section_name"]

#: Versions recorded in the provenance.
PACKAGES = ("numpy", "scipy", "torch", "anndata", "scanpy")


def section_name(density_prior: str | None, num_epochs: int) -> str:
    """Bundle key of one run, e.g. ``rna_count_based_epochs300`` or ``none_epochs1``."""
    return f"{density_prior or 'none'}_epochs{num_epochs}"


def build() -> dict[str, Any]:
    """Run upstream on every setting and return the bundle's ``uns``."""
    import torch

    torch.set_num_threads(1)
    tg = upstream.load()
    data = F.make_data()

    uns: dict[str, Any] = {
        "fixtures": {
            "checksum": F.checksum(),
            "SEED": F.SEED,
            "RANDOM_STATE": F.RANDOM_STATE,
            "LEARNING_RATE": F.LEARNING_RATE,
            "SETTINGS": np.asarray([prior or "none" for prior in F.SETTINGS]),
            "EPOCHS": np.asarray(F.EPOCHS),
            "genes": np.asarray(F.GENES),
            "measured": np.asarray(data.measured),
            "unmeasured": np.asarray(data.unmeasured),
        },
        "inputs": {"ref": data.ref, "query": data.query, "truth": data.truth},
    }
    for density_prior in F.SETTINGS:
        for num_epochs in F.EPOCHS:
            out = upstream.run(
                tg,
                data.ref,
                data.query,
                F.GENES,
                data.measured,
                density_prior=density_prior,
                num_epochs=num_epochs,
                learning_rate=F.LEARNING_RATE,
                random_state=F.RANDOM_STATE,
            )
            # one initial mapping and gene order for all runs, stored once
            for key in ("initial_logits", "training_genes"):
                value = out.pop(key)
                np.testing.assert_array_equal(uns["inputs"].setdefault(key, value), value)
            uns[section_name(density_prior, num_epochs)] = {
                **out,
                "imputed": out["imputed"][:, len(data.measured) :],
                "density_prior": density_prior or "none",
                "num_epochs": num_epochs,
                "provenance": provenance(
                    upstream, F, PACKAGES, density_prior=density_prior or "none", num_epochs=num_epochs
                ),
            }
    return uns


def main(argv: list[str] | None = None) -> int:
    """Generate the whole bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="the .zarr.zip store to write")
    args = parser.parse_args(argv)

    # threaded BLAS reduction order, and upstream's training genes, ordered by iterating a set of
    # strings, would otherwise vary run to run
    for var, value in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}.items():
        if os.environ.get(var) != value:
            raise RuntimeError(f"{var} must be {value}. Use `hatch run tangram:generate`.")
    write_bundle(build(), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
