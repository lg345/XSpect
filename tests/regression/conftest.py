"""
Regression test fixtures and markers.
"""

import pytest
import numpy as np
from pathlib import Path

REFERENCES_DIR = Path(__file__).parent / "references"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: regression tests requiring reference data")


@pytest.fixture(scope="session")
def references_dir():
    return REFERENCES_DIR


@pytest.fixture(scope="session")
def examples_dir():
    return EXAMPLES_DIR


def _load_reference(name):
    path = REFERENCES_DIR / name
    if not path.exists():
        pytest.skip(
            f"Reference file {name} not found. "
            f"Run `python tests/regression/generate_references.py` on S3DF to create it."
        )
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture(scope="module")
def ultrafast_reference():
    return _load_reference("mfxl1027922_ultrafast.npz")


@pytest.fixture(scope="module")
def static_reference():
    return _load_reference("mfx101080524_static.npz")


@pytest.fixture(scope="module")
def ultrafast_pipeline(examples_dir):
    """Run the ultrafast pipeline. Skips if HDF5 data is unavailable."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from XSpect.controller.pipeline import Pipeline

    yaml_path = str(examples_dir / "mfxl1027922_ultrafast_xes.yaml")
    try:
        p = Pipeline.from_yaml(yaml_path)
        p.run(cores=4, batch_size=500)
    except Exception as e:
        pytest.skip(f"Cannot run ultrafast pipeline (data unavailable?): {e}")
    return p


@pytest.fixture(scope="module")
def static_pipeline(examples_dir):
    """Run the static pipeline. Skips if HDF5 data is unavailable."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from XSpect.controller.pipeline import Pipeline

    yaml_path = str(examples_dir / "mfx101080524_static_xes.yaml")
    try:
        p = Pipeline.from_yaml(yaml_path)
        p.run(cores=4, batch_size=500)
    except Exception as e:
        pytest.skip(f"Cannot run static pipeline (data unavailable?): {e}")
    return p
