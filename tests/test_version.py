"""Version-string tests (L2)."""
import magicquant


def test_version_not_stale():
    assert magicquant.__version__ != "0.1.0"


def test_version_matches_metadata_when_installed():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        return
    try:
        meta = version("magicquant")
    except PackageNotFoundError:  # pragma: no cover - not installed
        return
    assert magicquant.__version__ == meta
