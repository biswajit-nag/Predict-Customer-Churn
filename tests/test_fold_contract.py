"""The shared-folds contract that makes OOF vectors from different models stackable."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "experiments" / "cv_folds_seed42.csv.gz"
RAW = ROOT / "data" / "train.csv"


@pytest.fixture(scope="module")
def folds():
    return pd.read_csv(FOLDS)


def test_fold_file_is_a_partition(folds):
    assert folds["id"].is_unique
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    sizes = folds["fold"].value_counts()
    assert sizes.max() - sizes.min() <= 1          # StratifiedKFold sizes differ by <= 1


@pytest.mark.skipif(not RAW.exists(), reason="competition data not downloaded")
def test_fold_file_matches_stratified_kfold_seed_42(folds):
    train = pd.read_csv(RAW, usecols=["id", "Churn"])
    assert (train["id"].to_numpy() == folds["id"].to_numpy()).all()
    expected = pd.Series(-1, index=train.index)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (_, va) in enumerate(cv.split(train, train["Churn"])):
        expected.iloc[va] = k
    assert (expected.to_numpy() == folds["fold"].to_numpy()).all()
