"""Guards on the SpaGE reference: the pin, the fixtures, and the settings injected into ``SpaGE()``."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from zarr.storage import ZipStore

from squidpy_ports._bundle import write_bundle
from squidpy_ports.spage import fixtures as F
from squidpy_ports.spage import generate, upstream


@pytest.fixture(scope="session")
def spage():
    """The vendored upstream ``SpaGE.main`` module, or a skip if it is not checked out."""
    try:
        return upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _frames(data: F.Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(data.ref, columns=F.GENES), pd.DataFrame(data.query, columns=data.measured)


def test_vendored_upstream_is_pinned(spage):
    assert callable(spage.SpaGE)


def test_data_is_deterministic():
    first, second = F.make_data(), F.make_data()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.ref.shape == (F.N_REF, len(F.GENES))
    assert first.query.shape == (F.N_QUERY, len(first.measured))
    assert first.truth.shape == (F.N_QUERY, len(first.unmeasured))
    assert (first.ref == 0).any()
    assert (first.ref >= 0).all()


def test_hard_coded_settings_leave_spage_unchanged(spage):
    data = F.make_data()
    ref, query = _frames(data)

    imputed, observed = upstream.run(spage, ref, query, data.unmeasured, n_pv=F.N_PV)

    pd.testing.assert_frame_equal(imputed, spage.SpaGE(query, ref, F.N_PV, genes_to_predict=data.unmeasured))
    assert observed["n_pv_used"] == F.N_PV


@pytest.mark.parametrize(("cosine_threshold", "n_pv_used"), [(0.3, 3), (0.99, 2)])
def test_cosine_threshold_is_applied(spage, cosine_threshold, n_pv_used):
    data = F.make_data()

    _, observed = upstream.run(spage, *_frames(data), data.unmeasured, n_pv=F.N_PV, cosine_threshold=cosine_threshold)

    assert observed["n_pv_used"] == n_pv_used == np.sum(observed["cosine"] > cosine_threshold)


def test_untied_rows_do_not_depend_on_reference_order(spage):
    """Only ``tied_rows`` may depend on how the search backend breaks ties."""
    data = F.make_data()
    ref, query = _frames(data)
    kwargs = {"n_pv": F.N_PV, "n_neighbors": 50}

    imputed, observed = upstream.run(spage, ref, query, data.unmeasured, **kwargs)
    reversed_, _ = upstream.run(spage, ref.iloc[::-1].reset_index(drop=True), query, data.unmeasured, **kwargs)

    assert observed["tied_rows"].size > 0
    untied = np.setdiff1d(np.arange(F.N_QUERY), observed["tied_rows"])
    np.testing.assert_allclose(reversed_.to_numpy()[untied], imputed.to_numpy()[untied], rtol=1e-12)


def test_bundle_round_trips(tmp_path):
    uns = generate.build()
    write_bundle(uns, tmp_path / "bundle.zarr.zip")

    with ZipStore(tmp_path / "bundle.zarr.zip", mode="r") as store:
        read = ad.read_zarr(store).uns

    assert read["fixtures"]["checksum"] == F.checksum()
    np.testing.assert_array_equal(read["inputs"]["truth"], F.make_data().truth)
    for cosine_threshold, n_neighbors in F.SETTINGS:
        section = read[generate.section_name(cosine_threshold, n_neighbors)]
        np.testing.assert_array_equal(
            section["imputed"], uns[generate.section_name(cosine_threshold, n_neighbors)]["imputed"]
        )
        assert section["n_neighbors"] == n_neighbors
        assert section["provenance"]["upstream_sha"] == upstream.UPSTREAM_SHA
