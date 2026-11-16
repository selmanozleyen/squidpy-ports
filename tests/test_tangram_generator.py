"""Guards on the Tangram reference: the pin, the fixtures, and what is observed of upstream's mapping."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from zarr.storage import ZipStore

from squidpy_ports._bundle import write_bundle
from squidpy_ports.tangram import fixtures as F
from squidpy_ports.tangram import generate, upstream


@pytest.fixture(scope="session")
def tangram():
    """The vendored upstream ``tangram`` package, or a skip if it is not checked out."""
    try:
        return upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_vendored_upstream_is_pinned(tangram):
    assert callable(tangram.map_cells_to_space)


def test_data_is_deterministic():
    first, second = F.make_data(), F.make_data()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.ref.shape == (F.N_REF, len(F.GENES))
    assert first.query.shape == (F.N_QUERY, len(first.measured))
    assert first.truth.shape == (F.N_QUERY, len(first.unmeasured))


@pytest.mark.parametrize("density_prior", F.SETTINGS)
def test_observes_the_initial_mapping(tangram, density_prior):
    """The recorded start is upstream's seeded draw, and the genes are projected through the mapping."""
    data = F.make_data()

    out = upstream.run(
        tangram, data.ref, data.query, F.GENES, data.measured,
        density_prior=density_prior, num_epochs=1, random_state=F.RANDOM_STATE,
    )  # fmt: skip

    np.random.seed(F.RANDOM_STATE)  # the legacy global RNG is what upstream seeds
    expected = np.random.normal(0, 1, (F.N_REF, F.N_QUERY)).astype(np.float32)
    np.testing.assert_array_equal(out["initial_logits"], expected)
    np.testing.assert_allclose(out["imputed"], out["probs"].T @ data.ref, rtol=1e-6)


def test_density_prior_is_weighted(tangram):
    """Upstream turns the density term on whenever a prior is given (mapping_utils.py:214)."""
    data = F.make_data()

    def first_loss(**kwargs) -> float:
        out = upstream.run(
            tangram, data.ref, data.query, F.GENES, data.measured, num_epochs=1, random_state=F.RANDOM_STATE, **kwargs
        )
        return out["total_loss"][0]

    loss = {prior: first_loss(density_prior=prior) for prior in F.SETTINGS}

    assert loss["rna_count_based"] != loss[None]
    assert loss["uniform"] != loss[None]
    assert first_loss() == loss["rna_count_based"]  # upstream's default prior


def test_bundle_round_trips(tmp_path):
    uns = generate.build()
    write_bundle(uns, tmp_path / "bundle.zarr.zip")

    with ZipStore(tmp_path / "bundle.zarr.zip", mode="r") as store:
        read = ad.read_zarr(store).uns

    assert read["fixtures"]["checksum"] == F.checksum()
    np.testing.assert_array_equal(read["inputs"]["truth"], F.make_data().truth)
    assert sorted(read["inputs"]["training_genes"]) == F.make_data().measured
    for prior in F.SETTINGS:
        for num_epochs in F.EPOCHS:
            section = read[generate.section_name(prior, num_epochs)]
            np.testing.assert_array_equal(section["probs"], uns[generate.section_name(prior, num_epochs)]["probs"])
            assert section["imputed"].shape == (F.N_QUERY, len(F.GENES) - F.N_MEASURED)
            assert section["main_loss"].shape == (num_epochs,)
            assert ("density" in section) == (prior is not None)
            assert section["provenance"]["upstream_sha"] == upstream.UPSTREAM_SHA
