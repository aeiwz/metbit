import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import fastapi  # noqa: F401
except ImportError:  # GUI extras not installed: skip this folder
    collect_ignore_glob = ["test_*.py"]

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GUI" / "server"))

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    ppm = np.linspace(10, 0, 40)
    X = rng.normal(size=(12, 40))
    X[6:, 5] += 3  # group B carries an extra peak
    df = pd.DataFrame(X, columns=[f"{v:.4f}" for v in ppm])
    df.insert(0, "Group", ["A"] * 6 + ["B"] * 6)
    df.insert(0, "Sample", [f"s{i}" for i in range(12)])
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)
