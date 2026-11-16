"""Emit the reference bundle squidpy's SpaGE reference tests compare against.

Run with::

    hatch run generate:spage --out spage_reference.zarr.zip

One AnnData in a zipped zarr store, the same format as the STalign bundle: the inputs in
``uns["inputs"]``, the fixture parameters in ``uns["fixtures"]``, and one section per
``(cosine_threshold, n_neighbors)`` in :data:`.fixtures.SETTINGS`, each with its provenance.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squidpy_ports._bundle import write_bundle

from . import fixtures as F
from . import upstream

__all__ = ["build", "main", "section_name"]


def section_name(cosine_threshold: float, n_neighbors: int) -> str:
    """Bundle key of one run, e.g. ``cosine30_k20`` (no dots: they are zarr path noise)."""
    return f"cosine{round(cosine_threshold * 100)}_k{n_neighbors}"


def _provenance(**extra: Any) -> dict[str, Any]:
    """The provenance stamped into every section, so the outputs stay falsifiable."""
    root = Path(__file__).resolve().parents[3]
    try:
        ports_commit = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:  # pragma: no cover - not a checkout
        ports_commit = "unknown"
    return {
        "upstream_sha": upstream.UPSTREAM_SHA,
        "upstream_url": upstream.UPSTREAM_URL,
        "ports_commit": ports_commit,
        "fixtures_checksum": F.checksum(),
        **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn")},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def build() -> dict[str, Any]:
    """Run upstream on every setting and return the bundle's ``uns``."""
    spage = upstream.load()
    data = F.make_data()
    ref = pd.DataFrame(data.ref, columns=F.GENES)
    query = pd.DataFrame(data.query, columns=data.measured)

    uns: dict[str, Any] = {
        "fixtures": {
            "checksum": F.checksum(),
            "SEED": F.SEED,
            "N_PV": F.N_PV,
            "SETTINGS": np.asarray(F.SETTINGS),
            "genes": np.asarray(F.GENES),
            "measured": np.asarray(data.measured),
            "unmeasured": np.asarray(data.unmeasured),
        },
        "inputs": {"ref": data.ref, "query": data.query, "truth": data.truth},
    }
    for cosine_threshold, n_neighbors in F.SETTINGS:
        imputed, observed = upstream.run(
            spage,
            ref,
            query,
            data.unmeasured,
            n_pv=F.N_PV,
            cosine_threshold=cosine_threshold,
            n_neighbors=n_neighbors,
        )
        uns[section_name(cosine_threshold, n_neighbors)] = {
            "imputed": imputed[data.unmeasured].to_numpy(),
            "n_pv_used": observed["n_pv_used"],
            "cosine": observed["cosine"],
            "tied_rows": observed["tied_rows"],
            "cosine_threshold": cosine_threshold,
            "n_neighbors": n_neighbors,
            "provenance": _provenance(cosine_threshold=cosine_threshold, n_neighbors=n_neighbors),
        }
    return uns


def main(argv: list[str] | None = None) -> int:
    """Generate the whole bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="the .zarr.zip store to write")
    args = parser.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(var) != "1":
            raise RuntimeError(f"{var} must be 1 so BLAS reduction order is fixed. Use `hatch run generate:spage`.")
    write_bundle(build(), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
