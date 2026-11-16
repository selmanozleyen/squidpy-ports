"""Write a reference bundle: an AnnData whose ``uns`` holds everything, in a zipped zarr store."""

from __future__ import annotations

import platform
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["provenance", "write_bundle"]


def provenance(upstream: ModuleType, fixtures: ModuleType, packages: tuple[str, ...], **extra: Any) -> dict[str, Any]:
    """The provenance stamped into every section, so the outputs stay falsifiable."""
    root = Path(__file__).resolve().parents[2]
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
        "fixtures_checksum": fixtures.checksum(),
        **{pkg: version(pkg) for pkg in packages},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # no argv: the --out path would make the bundle depend on where it was generated
        "generated_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


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
