"""Shared fixtures. The tests use small synthetic frames and temporary directories,
so none of them needs the competition data or the (uncommitted) prediction arrays."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def raw_frames():
    """Tiny train/test frames with the raw-data schema after native encoding."""
    rng = np.random.default_rng(0)

    def make(n, with_target):
        df = pd.DataFrame({
            "id": np.arange(n),
            "tenure": rng.integers(1, 73, n),
            "MonthlyCharges": rng.uniform(18, 120, n).round(2),
            "Contract": pd.Categorical(rng.choice(["Month-to-month", "One year", "Two year"], n)),
            "PaymentMethod": pd.Categorical(rng.choice(["Electronic check", "Mailed check"], n)),
            "InternetService": pd.Categorical(rng.choice(["DSL", "Fiber optic", "No"], n)),
        })
        df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
        if with_target:
            df["Churn"] = rng.integers(0, 2, n)
        return df

    return make(200, True), make(80, False)
