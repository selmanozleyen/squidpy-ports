"""Guards on what every method shares; needs the base install only."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy_ports import _common
from squidpy_ports.spage import fixtures as spage_fixtures
from squidpy_ports.tangram import fixtures as tangram_fixtures


def test_bundle_round_trips(tmp_path):
    uns = {
        "fixtures": {"checksum": "abc", "genes": np.array(["g0", "g1"])},
        "run": {"x": np.arange(6.0).reshape(2, 3), "n": 3, "name": "none", "provenance": {"python": "3"}},
    }

    _common.write_bundle(uns, tmp_path / "bundle.zarr.zip")
    _common.write_bundle(uns, tmp_path / "bundle.zarr.zip")  # overwrites rather than appends

    read = _common.read_bundle(tmp_path / "bundle.zarr.zip")
    assert _common.differing(uns, read) == []
    assert read["fixtures"]["checksum"] == "abc"
    assert read["run"]["provenance"] == {"python": "3"}


def test_differing_ignores_provenance_only():
    a = {"run": {"x": np.zeros(2), "provenance": {"generated_utc": "1"}}}
    b = {"run": {"x": np.zeros(2), "provenance": {"generated_utc": "2"}}}
    assert _common.differing(a, b) == []

    b["run"]["x"] = np.ones(2)
    b["run"]["y"] = 1
    b["fixtures"] = {"params": {"a": np.ones(1)}}
    a["fixtures"] = {"params": {"a": np.zeros(1)}}
    assert _common.differing(a, b) == ["fixtures/params/a", "run/x", "run/y"]


def test_verify_pin(tmp_path):
    (tmp_path / "up.py").write_text("x = 1\n")
    digest = _common.checksum(tmp_path / "up.py")

    _common.verify_pin(tmp_path, {"up.py": digest}, "sha")
    with pytest.raises(RuntimeError, match="re-pin|Re-pin"):
        _common.verify_pin(tmp_path, {"up.py": "0" * 64}, "sha")
    with pytest.raises(FileNotFoundError):
        _common.verify_pin(tmp_path, {"missing.py": digest}, "sha")


def test_generator_main_checks_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS must be 1"):
        _common.generator_main(dict, {"OMP_NUM_THREADS": "1"}, ["--out", str(tmp_path / "b.zarr.zip")], "")

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    _common.generator_main(
        lambda: {"run": {"x": np.ones(1)}}, {"OMP_NUM_THREADS": "1"}, ["--out", str(tmp_path / "b.zarr.zip")], ""
    )
    assert _common.read_bundle(tmp_path / "b.zarr.zip")["run"]["x"].tolist() == [1.0]


@pytest.mark.parametrize("F", [spage_fixtures, tangram_fixtures], ids=["spage", "tangram"])
def test_imputation_fixtures_are_deterministic(F):
    first, second = F.make_data(), F.make_data()

    for a, b in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.ref.shape == (F.N_REF, len(F.GENES))
    assert first.query.shape == (F.N_QUERY, F.N_MEASURED)
    assert first.truth.shape == (F.N_QUERY, len(F.GENES) - F.N_MEASURED)
    assert first.measured + first.unmeasured == F.GENES
    assert (first.ref >= 0).all()
    uns = first.uns(F)
    assert uns["fixtures"]["checksum"] == F.checksum()
    np.testing.assert_array_equal(uns["inputs"]["truth"], first.truth)
