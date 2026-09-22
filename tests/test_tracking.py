import numpy as np
import pandas as pd
import pytest

import src.tracking as tracking


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point the tracking module at an empty temporary experiments directory."""
    monkeypatch.setattr(tracking, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(tracking, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def _artifacts(n=10):
    rng = np.random.default_rng(0)
    return {
        "params": {"a": 1}, "metrics": {"fold_roc_aucs": [0.9]},
        "oof_proba": rng.random(n), "test_proba_folds": rng.random((5, n)),
        "test_proba_mean": rng.random(n), "feature_importance": None,
        "environment_text": "", "git_diff": "", "notes": "note", "models": None,
    }


def test_save_and_load_round_trip_sorted_by_auc(ledger):
    tracking.save_run({"run_id": "r1", "oof_roc_auc": 0.90}, _artifacts(), "r1")
    tracking.save_run({"run_id": "r2", "oof_roc_auc": 0.95}, _artifacts(), "r2")
    runs = tracking.load_runs()
    assert list(runs["run_id"]) == ["r2", "r1"]
    saved = ledger / "runs" / "r1"
    for name in ("params.json", "metrics.json", "oof_proba.npy", "test_proba_mean.npy",
                 "notes.md", "git_diff.patch", "environment.txt"):
        assert (saved / name).exists(), name
    np.testing.assert_array_equal(np.load(saved / "oof_proba.npy"), _artifacts()["oof_proba"])


def test_new_column_triggers_outer_join_not_misalignment(ledger):
    tracking.save_run({"run_id": "r1", "oof_roc_auc": 0.90}, _artifacts(), "r1")
    tracking.save_run({"run_id": "r2", "oof_roc_auc": 0.91, "lb_private": 0.905},
                      _artifacts(), "r2")
    runs = pd.read_csv(ledger / "runs.csv").set_index("run_id")
    assert list(runs.columns) == ["oof_roc_auc", "lb_private"]
    assert np.isnan(runs.loc["r1", "lb_private"])
    assert runs.loc["r2", "lb_private"] == 0.905
    assert runs.loc["r1", "oof_roc_auc"] == 0.90


def test_delete_run_removes_row_and_directory(ledger):
    tracking.save_run({"run_id": "r1", "oof_roc_auc": 0.90}, _artifacts(), "r1")
    tracking.delete_run("r1")
    assert tracking.load_runs().empty
    assert not (ledger / "runs" / "r1").exists()


def test_train_parquet_for_resolves_label_then_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "DATA_DIR", tmp_path)
    (tmp_path / "train_df_fe_v4_native.parquet").touch()
    (tmp_path / "train_df_native.parquet").touch()
    assert tracking.train_parquet_for("fe_v4_native").name == "train_df_fe_v4_native.parquet"
    assert tracking.train_parquet_for("fe_v0_native").name == "train_df_native.parquet"


def test_params_hash_ignores_key_order():
    assert tracking.params_hash({"a": 1, "b": 2}) == tracking.params_hash({"b": 2, "a": 1})
    assert tracking.params_hash({"a": 1}) != tracking.params_hash({"a": 2})
