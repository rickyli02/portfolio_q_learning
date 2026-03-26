"""Unit tests for src/utils/paths.py."""
import tempfile
from pathlib import Path

from src.utils.paths import REPO_ROOT, make_run_dir


def test_repo_root_exists():
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "src").exists()


def test_make_run_dir_creates_directory():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = make_run_dir("test_exp", base_dir=Path(tmp))
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert "run_" in run_dir.name
        assert run_dir.parent.name == "test_exp"


def test_make_run_dir_unique():
    # Microsecond timestamps make same-second collision extremely unlikely
    # without requiring any sleep.
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dirs = {make_run_dir("exp", base_dir=base) for _ in range(10)}
        assert len(dirs) == 10
