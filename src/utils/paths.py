"""Output directory creation and timestamped path helpers."""
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"


def make_run_dir(experiment_name: str, base_dir: Path = OUTPUTS_DIR) -> Path:
    """Create and return a unique timestamped run directory.

    Structure: base_dir / experiment_name / run_YYYYMMDD_HHMMSS_ffffff

    Microsecond precision prevents collisions between runs launched within
    the same second.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / experiment_name / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
