import pandas as pd
import pytest

import src.features as features


def test_fe_v4_adds_three_columns_and_keeps_rows(raw_frames):
    train, _ = raw_frames
    out = features.engineer_features(train, "fe_v4_native")
    assert len(out) == len(train)
    assert list(out["id"]) == list(train["id"])          # row order preserved
    added = [c for c in out.columns if c not in train.columns]
    assert added == ["AverageMonthly", "contract_x_payment", "contract_x_internet"]
    assert isinstance(out["contract_x_payment"].dtype, pd.CategoricalDtype)
    assert out["Churn"].equals(train["Churn"])


def test_average_monthly_guards_zero_tenure(raw_frames):
    train = raw_frames[0].copy()
    train.loc[0, "tenure"] = 0
    out = features.engineer_features(train, "fe_v4_native")
    assert out["AverageMonthly"].notna().all()
    assert out.loc[0, "AverageMonthly"] == train.loc[0, "TotalCharges"]


def test_input_frame_not_mutated(raw_frames):
    train = raw_frames[0]
    before = train.copy()
    features.engineer_features(train, "fe_v4_native")
    pd.testing.assert_frame_equal(train, before)


def test_unknown_version_raises(raw_frames):
    with pytest.raises(ValueError, match="unknown data_version"):
        features.engineer_features(raw_frames[0], "fe_v99")


def _v1(df):
    return df.assign(extra=1)


def _v2(df):
    return df.assign(extra=2)


def test_cache_is_rebuilt_when_feature_code_changes(raw_frames, tmp_path, monkeypatch):
    monkeypatch.setattr(features, "prepare_data", lambda encoding: raw_frames)
    monkeypatch.setitem(features.FEATURE_SETS, "fe_test", ("native", _v1))
    train, _ = features.build_features("fe_test", data_dir=tmp_path)
    assert (train["extra"] == 1).all()

    # Same label, same code: served from the cache without rebuilding.
    def fail(encoding):
        raise AssertionError("cache was rebuilt")
    monkeypatch.setattr(features, "prepare_data", fail)
    train, _ = features.build_features("fe_test", data_dir=tmp_path)
    assert (train["extra"] == 1).all()

    # Same label, different code: the stale parquet must not be served.
    monkeypatch.setattr(features, "prepare_data", lambda encoding: raw_frames)
    monkeypatch.setitem(features.FEATURE_SETS, "fe_test", ("native", _v2))
    train, _ = features.build_features("fe_test", data_dir=tmp_path)
    assert (train["extra"] == 2).all()
