import pytest

from squidpy_ports.stalign import upstream


@pytest.fixture(scope="session")
def stalign():
    """The vendored upstream STalign module, or a skip if it is not checked out."""
    try:
        return upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def spage():
    """The vendored upstream ``SpaGE.main`` module, or a skip if it is not checked out."""
    from squidpy_ports.spage import upstream as spage_upstream

    try:
        return spage_upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
