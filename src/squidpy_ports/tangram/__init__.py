"""Reference outputs for squidpy's ``squidpy.experimental.tl.impute_tangram``.

The upstream implementation is vendored at ``vendor/Tangram`` and pinned to a single commit.
This package runs it, unmodified, and writes the numbers squidpy's Tangram reference tests
compare against.
"""

from . import fixtures, upstream

__all__ = ["fixtures", "upstream"]
