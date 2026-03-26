"""Smoke tests: verify all src packages import cleanly."""


def test_src_package():
    import src  # noqa: F401


def test_utils_submodules():
    from src.utils import device, io, logging, paths, seed  # noqa: F401


def test_domain_packages():
    from src import (  # noqa: F401
        algos,
        backtest,
        data,
        envs,
        eval,
        features,
        models,
        train,
    )


def test_seed_runs():
    from src.utils.seed import set_seed
    set_seed(0)


def test_device_runs():
    from src.utils.device import get_device
    d = get_device()
    assert d is not None
