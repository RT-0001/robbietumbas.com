import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def cfg(root):
    from dimcomp.config import resolve
    return resolve(root / "specs" / "00034966.json")


@pytest.fixture(scope="session")
def ctx(cfg):
    from dimcomp.pipeline import load_context
    return load_context(cfg)  # uses the committed fits/00034966.json
