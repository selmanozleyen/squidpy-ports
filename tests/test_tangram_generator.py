"""Guards on the Tangram reference: the pin, and what is observed of upstream's mapping."""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest

from squidpy_ports._common import differing, read_bundle, write_bundle
from squidpy_ports.tangram import fixtures as F
from squidpy_ports.tangram import generate, upstream


def _run(tangram, **kwargs):
    return upstream.run(tangram, F.make_data(), num_epochs=1, random_state=F.RANDOM_STATE, **kwargs)


@pytest.mark.parametrize("density_prior", F.SETTINGS)
def test_observes_the_initial_mapping(tangram, density_prior):
    """The recorded start is upstream's seeded draw, and the genes are projected through the mapping."""
    mapper = sys.modules["tangram.mapping_optimizer"].Mapper

    out = _run(tangram, density_prior=density_prior)

    assert sys.modules["tangram.mapping_optimizer"].Mapper is mapper  # restored
    np.random.seed(F.RANDOM_STATE)  # the legacy global RNG is what upstream seeds
    expected = np.random.normal(0, 1, (F.N_REF, F.N_QUERY)).astype(np.float32)
    np.testing.assert_array_equal(out["initial_logits"], expected)
    np.testing.assert_allclose(out["imputed"], out["probs"].T @ F.make_data().ref[:, F.N_MEASURED :], rtol=1e-6)


def test_density_prior_is_weighted(tangram):
    """Upstream turns the density term on whenever a prior is given (mapping_utils.py:214)."""
    loss = {prior: _run(tangram, density_prior=prior)["total_loss"][0] for prior in F.SETTINGS}

    assert loss["rna_count_based"] != loss[None]
    assert loss["uniform"] != loss[None]
    # `run` falls back to the same prior as upstream
    assert inspect.signature(tangram.map_cells_to_space).parameters["density_prior"].default == "rna_count_based"
    assert _run(tangram)["total_loss"][0] == loss["rna_count_based"]


def test_bundle_round_trips(tangram, tmp_path):
    uns = generate.build()
    write_bundle(uns, tmp_path / "bundle.zarr.zip")

    read = read_bundle(tmp_path / "bundle.zarr.zip")

    assert differing(uns, read) == []
    assert sorted(read["inputs"]["training_genes"]) == F.GENES[: F.N_MEASURED]
    for prior in F.SETTINGS:
        for num_epochs in F.EPOCHS:
            section = read[generate.section_name(prior, num_epochs)]
            assert section["imputed"].shape == (F.N_QUERY, len(F.GENES) - F.N_MEASURED)
            assert section["main_loss"].shape == (num_epochs,)
            assert ("density" in section) == (prior is not None)
            assert section["provenance"]["upstream_sha"] == upstream.UPSTREAM_SHA
