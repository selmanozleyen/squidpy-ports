"""Reference outputs for squidpy's ``squidpy.experimental.tl.impute_spage``.

The upstream implementation is vendored at ``vendor/SpaGE`` and pinned to a single commit.
This package runs it, unmodified, and writes the numbers squidpy's SpaGE reference tests
compare against.
"""

from . import fixtures, upstream

__all__ = ["fixtures", "upstream"]
