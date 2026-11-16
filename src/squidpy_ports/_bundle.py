"""Write a reference bundle: an AnnData whose ``uns`` holds everything, in a zipped zarr store."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

__all__ = ["write_bundle"]


def write_bundle(uns: dict[str, Any], out: Path) -> None:
    """Write ``uns`` into an otherwise empty AnnData, stored as a zipped zarr store at ``out``."""
    import anndata as ad
    from zarr.storage import MemoryStore

    # written in memory and zipped once: anndata rewrites zarr.json while writing, which a
    # ZipStore can only append (duplicate members), and a directory on a case-insensitive
    # filesystem merges keys such as `A` and `a`
    store: dict = {}
    ad.AnnData(uns=uns).write_zarr(MemoryStore(store_dict=store))
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(store):
            zf.writestr(key, store[key].to_bytes())
