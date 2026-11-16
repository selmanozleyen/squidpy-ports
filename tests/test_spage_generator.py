"""Guards on the SpaGE reference: the pin, and the settings injected into ``SpaGE()``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from squidpy_ports._common import differing, read_bundle, write_bundle
from squidpy_ports.spage import fixtures as F
from squidpy_ports.spage import generate, upstream


def test_hard_coded_settings_leave_spage_unchanged(spage):
    data = F.make_data()
    ref, query = pd.DataFrame(data.ref, columns=data.genes), pd.DataFrame(data.query, columns=data.measured)

    out = upstream.run(spage, data, n_pv=F.N_PV)

    expected = spage.SpaGE(query, ref, F.N_PV, genes_to_predict=data.unmeasured)
    np.testing.assert_array_equal(out["imputed"], expected.to_numpy())
    assert out["n_pv_used"] == F.N_PV


@pytest.mark.parametrize(("cosine_threshold", "n_pv_used"), [(0.3, 3), (0.99, 2)])
def test_cosine_threshold_is_applied(spage, cosine_threshold, n_pv_used):
    out = upstream.run(spage, F.make_data(), n_pv=F.N_PV, cosine_threshold=cosine_threshold)

    assert out["n_pv_used"] == n_pv_used == np.sum(out["cosine"] > cosine_threshold)


def test_n_neighbors_is_applied(spage):
    data = F.make_data()

    default = upstream.run(spage, data, n_pv=F.N_PV)

    np.testing.assert_array_equal(upstream.run(spage, data, n_pv=F.N_PV, n_neighbors=50)["imputed"], default["imputed"])
    for k in (20, 300):
        assert not np.allclose(upstream.run(spage, data, n_pv=F.N_PV, n_neighbors=k)["imputed"], default["imputed"])


def test_untied_rows_do_not_depend_on_reference_order(spage):
    """Only ``tied_rows`` may depend on how the search backend breaks ties."""
    data = F.make_data()

    out = upstream.run(spage, data, n_pv=F.N_PV)
    reversed_ = upstream.run(spage, data._replace(ref=data.ref[::-1]), n_pv=F.N_PV)

    assert out["tied_rows"].size > 0
    untied = np.setdiff1d(np.arange(F.N_QUERY), out["tied_rows"])
    np.testing.assert_allclose(reversed_["imputed"][untied], out["imputed"][untied], rtol=1e-12)


def test_bundle_round_trips(spage, tmp_path):
    uns = generate.build()
    write_bundle(uns, tmp_path / "bundle.zarr.zip")

    read = read_bundle(tmp_path / "bundle.zarr.zip")

    assert differing(uns, read) == []
    for cosine_threshold, n_neighbors in F.SETTINGS:
        section = read[generate.section_name(cosine_threshold, n_neighbors)]
        assert section["imputed"].shape == (F.N_QUERY, len(F.GENES) - F.N_MEASURED)
        assert section["n_neighbors"] == n_neighbors
        assert section["provenance"]["upstream_sha"] == upstream.UPSTREAM_SHA
