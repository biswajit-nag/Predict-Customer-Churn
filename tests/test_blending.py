import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import src.blending as bl

N = 400


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """A synthetic ledger of full-length runs: base models, a blend and a level-2 run."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, N)
    runs_dir = tmp_path / "runs"
    rows = []

    def add(run_id, tag, model_class, data_version, signal):
        oof = 1 / (1 + np.exp(-(signal * y + rng.normal(0, 1, N))))
        d = runs_dir / run_id
        d.mkdir(parents=True)
        np.save(d / "oof_proba.npy", oof)
        np.save(d / "test_proba_mean.npy", rng.random(50))
        rows.append(dict(run_id=run_id, tag=tag, model_class=model_class, status="success",
                         data_version=data_version, oof_roc_auc=roc_auc_score(y, oof)))

    add("a1", "lgbm", "LGBMClassifier", "fe_v4_native", 2.0)
    add("a2", "lgbm", "LGBMClassifier", "fe_v3_native", 1.8)
    add("a3", "xgb", "XGBClassifier", "fe_v4_native", 1.9)
    add("a4", "lr", "Pipeline", "fe_v0", 1.0)
    add("b1", "blend", "LogitStackLR", "blend_v1", 2.2)
    add("s1", "lgbm-stack", "LGBMClassifier", "fe_v5_stack", 2.1)
    pd.DataFrame(rows).to_csv(tmp_path / "runs.csv", index=False)

    monkeypatch.setattr(bl, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(bl, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(bl, "CANON_N", N)
    return y, tmp_path


def test_blends_are_never_blend_members(pool):
    y, _ = pool
    oof, test, meta, _ = bl.load_aligned_runs(y, exclude_versions=())
    assert "blend_v1" not in set(meta["data_version"])
    assert oof.shape == (N, len(meta)) and test.shape[1] == len(meta)


def test_level2_runs_can_be_excluded(pool):
    y, _ = pool
    _, _, meta, _ = bl.load_aligned_runs(y, exclude_versions=("fe_v5_stack",))
    assert "fe_v5_stack" not in set(meta["data_version"])


def test_shared_tags_get_unique_labels(pool):
    y, tmp = pool
    runs = pd.read_csv(tmp / "runs.csv")
    extra = runs[runs.run_id == "a2"].assign(run_id="a5")     # same tag AND data_version
    (tmp / "runs" / "a5").mkdir()
    for f in ("oof_proba.npy", "test_proba_mean.npy"):
        (tmp / "runs" / "a5" / f).write_bytes((tmp / "runs" / "a2" / f).read_bytes())
    pd.concat([runs, extra]).to_csv(tmp / "runs.csv", index=False)
    oof, _, meta, _ = bl.load_aligned_runs(y)
    assert meta["label"].is_unique and oof.columns.is_unique
    assert {"lgbm", "lgbm|fe_v3_native"} <= set(meta["label"])


def test_alignment_guard_rejects_reordered_oof(pool):
    y, tmp = pool
    path = tmp / "runs" / "a3" / "oof_proba.npy"
    np.save(path, np.load(path)[::-1])                            # rows reordered
    with pytest.raises(AssertionError, match="not row-aligned"):
        bl.load_aligned_runs(y)


def test_curate_pool_one_per_engine(pool):
    y, _ = pool
    _, _, meta, _ = bl.load_aligned_runs(y, exclude_versions=("fe_v5_stack",))
    curated = bl.curate_pool(meta)
    engines = meta.set_index("label").loc[curated, "engine"]
    assert engines.is_unique
    assert set(engines) == {"LightGBM", "XGBoost", "Logistic"}
    assert "lgbm" in curated                    # the stronger LightGBM run wins its slot


def test_logit_stack_and_fold_scores(pool):
    y, _ = pool
    oof, test, _, _ = bl.load_aligned_runs(y, exclude_versions=("fe_v5_stack",))
    folds = np.arange(N) % 5
    p, t, coef = bl.logit_stack(oof, test, list(oof.columns), y, folds, C=1.0)
    assert p.shape == (N,) and t.shape == (50,) and len(coef) == oof.shape[1]
    assert np.all((p > 0) & (p < 1))
    per_fold, mean, std = bl.fold_aucs(p, y, folds)
    assert len(per_fold) == 5 and 0.5 < mean <= 1 and std >= 0
