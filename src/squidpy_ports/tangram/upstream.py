"""Load the vendored upstream Tangram and run it without modifying it.

Upstream is imported *by path* from ``vendor/Tangram``, like the other upstreams here, so the
reference is the pinned commit rather than whichever ``tangram-sc`` release resolves.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

__all__ = ["UPSTREAM_SHA", "UPSTREAM_URL", "load", "run", "vendor_root"]

#: The single upstream commit this package is a reference for.
UPSTREAM_SHA = "4c68995a418f41dc8caef567598c4d9b47781a13"
UPSTREAM_URL = "https://github.com/broadinstitute/Tangram"
#: sha256 of each file under ``tangram/`` at :data:`UPSTREAM_SHA`, so the pin is checked on the code
#: that runs, with or without git metadata.
SHA256 = {
    "__init__.py": "c01b2b1a2faa95b40c9c61ab9ff63881a8e4b8b86c11df4b66983be3ed1f1edc",
    "_version.py": "6b0cdd3e99141c22071f64c23a24084f04f900ef010206bf1c1e5d99ea189eb9",
    "mapping_optimizer.py": "eb6837258289f97919b7b7f597a20d1ffc984a71baa96d60bee5eb1445eebcf1",
    "mapping_parameter_tuning.py": "9ef409cf2f815e2fc6fed6215c66cbb27ab1a1cdbd4de0ae5a32b3a24eeac119",
    "mapping_utils.py": "3aaddf186006bdeed336bb013a2bfc152417872c9e1ebd9b7480a7708bdacdd1",
    "plot_utils.py": "fd10fd5a9bee359797a54ca4d353c27515648b3caa58486b45d3a40ba32bf7a3",
    "spatial_weights.py": "4d09a9e2d9d7bb04918f32c6ace625d83b251273f04766e7490014d7a7915c4b",
    "utils.py": "e40316bea7e486e5bfae431fdc1207e1e38ec6119c6ce3381df74f9e1a0907ac",
}


def vendor_root() -> Path:
    """Path to the vendored upstream checkout."""
    return Path(__file__).resolve().parents[3] / "vendor" / "Tangram"


@cache
def load() -> ModuleType:
    """Import the vendored ``tangram`` package, asserting the pin.

    Raises
    ------
    FileNotFoundError
        If the submodule is not checked out.
    RuntimeError
        If a file under ``tangram/`` is not the one at :data:`UPSTREAM_SHA`.
    """
    root = vendor_root()
    pkg = root / "tangram"
    if not (pkg / "__init__.py").is_file():
        raise FileNotFoundError(f"vendored Tangram missing at {pkg}; run `git submodule update --init`")
    for name, expected in SHA256.items():
        digest = hashlib.sha256((pkg / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"vendored tangram/{name} has sha256 {digest}, not the file at {UPSTREAM_SHA}. Re-pin "
                f"deliberately in upstream.py and regenerate the bundle."
            )

    import matplotlib

    matplotlib.use("Agg")  # `import tangram` imports its plotting module

    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("tangram")
    finally:
        sys.path.remove(str(root))
    if Path(module.__file__).resolve() != (pkg / "__init__.py").resolve():
        raise RuntimeError(f"imported tangram from {module.__file__}, not from {pkg}")
    return module


def run(
    tg: ModuleType,
    ref: np.ndarray,
    query: np.ndarray,
    genes: list[str],
    measured: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Map ``ref`` onto ``query`` with upstream's public API, in ``mode="cells"``, and project all genes.

    This is the tutorial's ``pp_adatas`` -> ``map_cells_to_space`` -> ``project_genes``, with
    ``kwargs`` passed to ``map_cells_to_space``. Two inputs are handed over in the form upstream still
    accepts on current dependencies:

    - The reference as a :class:`scipy.sparse.csr_matrix`: for a dense one upstream calls
      ``ndarray.toarray()`` (mapping_utils.py:262) and fails.
    - A named ``density_prior`` as the values ``pp_adatas`` stored for it in ``obs``, in a list. Given
      the name, upstream passes that ``obs`` column, a :class:`pandas.Series`, to ``torch.tensor``
      (mapping_optimizer.py:116), which current torch rejects; given the documented ndarray, it compares
      it to a string (mapping_utils.py:287), which current numpy rejects. A list reaches ``torch.tensor``
      the same way, as float32.

    The initial mapping is created inside ``Mapper.__init__`` and never returned, so it is observed
    by subclassing ``Mapper`` in ``tangram.mapping_optimizer`` for the duration of the call; it is
    what another implementation needs to start from to reproduce the trajectory.

    Returns
    -------
    ``imputed`` (query cells x ``genes``), the mapping ``probs`` (reference x query cells),
    ``initial_logits``, the ``training_genes`` in the order upstream used them, the per-epoch
    ``main_loss`` (mean cosine similarity) and ``total_loss``, and, with a prior, the ``density`` and
    the per-epoch ``kl_reg``.
    """
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    mo = sys.modules["tangram.mapping_optimizer"]
    captured: dict[str, Any] = {}
    real_mapper = mo.Mapper

    class Mapper(real_mapper):
        def train(self, *args, **kw):
            captured["initial_logits"] = self.M.detach().numpy().copy()
            return super().train(*args, **kw)

    adata_sc = ad.AnnData(csr_matrix(ref), var=pd.DataFrame(index=genes))
    adata_sp = ad.AnnData(query, var=pd.DataFrame(index=measured))
    tg.pp_adatas(adata_sc, adata_sp)
    if isinstance(prior := kwargs.setdefault("density_prior", "rna_count_based"), str):  # upstream's default
        kwargs["density_prior"] = adata_sp.obs[f"{prior}_density"].tolist()

    mo.Mapper = Mapper
    try:
        adata_map = tg.map_cells_to_space(adata_sc, adata_sp, mode="cells", device="cpu", verbose=False, **kwargs)
    finally:
        mo.Mapper = real_mapper
    projected = tg.project_genes(adata_map, adata_sc)

    history = adata_map.uns["training_history"]
    return {
        "imputed": np.asarray(projected[:, [g.lower() for g in genes]].X),
        "probs": np.asarray(adata_map.X),
        "initial_logits": captured["initial_logits"],
        "training_genes": np.asarray(adata_sc.uns["training_genes"]),
        "main_loss": np.asarray(history["main_loss"], dtype=float),
        "total_loss": np.asarray(history["total_loss"], dtype=float),
        # without a prior upstream records `kl_reg` as NaN
        **(
            {}
            if kwargs.get("density_prior") is None
            else {"density": np.asarray(kwargs["density_prior"]), "kl_reg": np.asarray(history["kl_reg"], dtype=float)}
        ),
    }
