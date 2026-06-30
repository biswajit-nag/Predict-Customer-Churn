"""Complete the unfinished fe_v4_native Optuna studies locally, then finalize.

Three studies did not finish on Kaggle (12 h limit / a joblib crash); their
partial SQLite DBs live in experiments/optuna_studies/. This script resumes a
study (CPU models only), tops it up to its trial budget, then runs the top-K
trials through a 5-fold OUTER CV and saves the best by outer OOF ROC-AUC — the
same finalize logic as the Kaggle notebooks, but local (kaggle_run=False), so
the OOF vector is row-aligned with the canonical seed-42 folds and stackable.

  lgbm     : resume -> 150 trials, finalize (CPU, parallel trials)
  ebm      : resume -> 80 trials, finalize (CPU, sequential; tolerant of crashes)
  catboost : NO top-up (GPU-tuned; no local GPU). Finalize the existing study
             as-is with the outer eval forced to task_type=CPU. Each fold's
             fitted model is CACHED to --cache-dir so the (slow, ~13 min/fold)
             run resumes across interruptions (e.g. the machine sleeping).

Usage:  python scripts/finalize_local_studies.py {lgbm|ebm|catboost} [--topk 3]
                 [--dry-run] [--cache-dir DIR]
"""
import argparse
import hashlib
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.tracking import DATA_DIR
from src.cv import run_cv_experiment, save_experiment
from src.tuning import (build_study, finalize_topk, lgbm_tuning_objective,
                        ebm_tuning_objective, catboost_tuning_objective)

FE_VERSION = "fe_v4_native"
SEARCH = "narrow"
STUDY_DIR = str(ROOT / "experiments" / "optuna_studies")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = {"lgbm": 150, "ebm": 80, "catboost": None}   # None -> finalize as-is
PARENT = {
    "lgbm": "20260610-183508-48286b",
    "ebm": "20260612-202507-e65b68",
    "catboost": "20260611-051031-5d6bc5",
}
ITER_KEY = {"lgbm": "n_estimators", "ebm": None, "catboost": "iterations"}
NB = {
    "lgbm": "predict-customer-churn-optuna-lgbm-cpu-fe_v4.ipynb",
    "ebm": "predict-customer-churn-optuna-ebm-cpu-fe_v4.ipynb",
    "catboost": "predict-customer-churn-optuna-catboost-gpu-fe_v4.ipynb",
}


def make_cached_catboost_cls(cache_dir):
    """CatBoostClassifier whose fit() caches the fitted model per training-set
    (keyed by the train-row index). A completed fold reloads in seconds, so the
    finalize run is resumable across interruptions. Class is named
    'CatBoostClassifier' so the ledger's model_class column stays consistent."""
    from catboost import CatBoostClassifier as _CB
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    class CatBoostClassifier(_CB):  # noqa: N801  (intentional: keep ledger name)
        def fit(self, X, y, **kw):
            key = hashlib.md5(np.ascontiguousarray(
                np.asarray(X.index.values, dtype=np.int64)).tobytes()).hexdigest()[:16]
            path = Path(cache_dir) / f"cat_fold_{key}.cbm"
            if path.exists():
                self.load_model(str(path))
                print(f"    [cache] loaded fold model {path.name}")
                return self
            super().fit(X, y, **kw)
            self.save_model(str(path))
            print(f"    [cache] saved fold model {path.name}")
            return self

    return CatBoostClassifier


def load_data():
    train_df = pd.read_parquet(DATA_DIR / f"train_df_{FE_VERSION}.parquet")
    test_df = pd.read_parquet(DATA_DIR / f"test_df_{FE_VERSION}.parquet")
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y = train_df[feats], train_df["Churn"]
    X_test = test_df[feats]
    fp = ROOT / "experiments" / "cv_folds_seed42.csv.gz"
    folds_ref = pd.read_csv(fp)
    assert (folds_ref["id"].to_numpy() == train_df["id"].to_numpy()).all(), "row id order != folds"
    chk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    here = np.full(len(train_df), -1)
    for f, (_, va) in enumerate(chk.split(np.zeros(len(train_df)), train_df["Churn"])):
        here[va] = f
    assert (folds_ref["fold"].to_numpy() == here).all(), "reconstructed folds != canonical"
    print(f"fold-contract OK | X {X.shape}  X_test {X_test.shape}  features {len(feats)}")
    return train_df, test_df, X, y, X_test, feats


def fail_stale_running(study):
    storage = study._storage
    n = 0
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
            n += 1
        except Exception as e:
            print(f"  (could not fail stale trial {t.number}: {e})")
    if n:
        print(f"  marked {n} stale RUNNING trial(s) as FAIL (clean TPE resume)")


def trial_counts(study):
    from collections import Counter
    return dict(Counter(str(t.state).split(".")[-1] for t in study.trials))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["lgbm", "ebm", "catboost"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / ".catboost_finalize_cache"))
    args = ap.parse_args()
    model = args.model

    train_df, test_df, X, y, X_test, feats = load_data()
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    study_name = f"{model}-{FE_VERSION}"
    study = build_study(study_name, storage_dir=STUDY_DIR)
    print(f"\n[{study_name}] loaded: {len(study.trials)} trials {trial_counts(study)}; "
          f"best so far {study.best_value:.6f}")

    if model == "lgbm":
        from lightgbm import LGBMClassifier
        objective = lgbm_tuning_objective(X, y, inner_cv=inner_cv, search=SEARCH,
                                          device="cpu", model_n_jobs=1)
        LGBM_FIXED = dict(subsample_freq=1, verbose=-1)
        model_factory = lambda p: LGBMClassifier(**p, **LGBM_FIXED)
        opt_kwargs = dict(n_jobs=os.cpu_count() or 4)
        device_note = "CPU"
    elif model == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        ftypes = ["nominal" if isinstance(X[c].dtype, pd.CategoricalDtype) else "continuous"
                  for c in feats]
        objective = ebm_tuning_objective(X, y, feature_types=ftypes, inner_cv=inner_cv, search=SEARCH)
        model_factory = lambda p: ExplainableBoostingClassifier(**p, feature_types=ftypes,
                                                                n_jobs=-1, random_state=42)
        opt_kwargs = dict(catch=(Exception,))
        device_note = "CPU"
    else:  # catboost — finalize as-is on CPU, with per-fold model caching for resumability
        cat_features = [c for c in feats if str(X[c].dtype) == "category"]
        objective = None
        CachedCat = make_cached_catboost_cls(args.cache_dir)
        CAT_FIXED = dict(verbose=0, random_seed=42)   # no task_type -> CPU
        model_factory = lambda p: CachedCat(**p, **CAT_FIXED, cat_features=cat_features)
        device_note = "CPU (params from GPU-tuned study)"
        print(f"  catboost fold cache: {args.cache_dir}")

    target = TARGETS[model]
    if target is not None:
        fail_stale_running(study)
        remaining = max(0, target - len(study.trials))
        print(f"  budget {target}; running {remaining} more trial(s)...")
        if remaining:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False, **opt_kwargs)
        print(f"  after top-up: {len(study.trials)} trials {trial_counts(study)}; "
              f"best {study.best_value:.6f} (trial {study.best_trial.number})")
    else:
        print(f"  catboost: finalizing existing {len(study.trials)} trials as-is (no top-up)")

    n_total = len(study.trials)
    candidates = finalize_topk(study, k=args.topk, iter_param=ITER_KEY[model])
    print(f"\n  top-{len(candidates)} by inner-CV ROC-AUC:")
    for c in candidates:
        extra = f"  {ITER_KEY[model]}={c['params'].get(ITER_KEY[model])}" if ITER_KEY[model] else ""
        print(f"    trial {c['trial_number']:>3}  inner {c['inner_roc_auc']:.6f}{extra}")
    if args.dry_run:
        print("\n  --dry-run: stopping before outer eval/save")
        return

    BASE_TAG = f"{model}-optuna-{FE_VERSION}"
    base_notes = (
        f"{model.upper()} on {FE_VERSION}, finalized LOCALLY ({device_note}). Params from the "
        f"{SEARCH}-search Optuna study (3-fold inner CV, ROC-AUC; multivariate TPE + MedianPruner; "
        f"early stopping decides tree count; SQLite-persisted; {n_total} trials), warm-started from "
        f"run {PARENT[model]}; partial Kaggle study resumed/finalized locally. Final params chosen "
        f"among top-{len(candidates)} trials by OUTER 5-fold OOF ROC-AUC. Notebook: kaggle/{NB[model]}."
    )
    results = []
    for i, cand in enumerate(candidates):
        rc = {
            "model_factory": model_factory,
            "metric": accuracy_score, "metric_name": "accuracy",
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "parent_run_id": PARENT[model], "save_models": False,
            "data_version": FE_VERSION, "params": cand["params"],
            "tag": f"{BASE_TAG}-top{i + 1}",
            "notes": base_notes + (f" Candidate {i + 1}/{len(candidates)} by inner-CV "
                                   f"(trial {cand['trial_number']}, inner {cand['inner_roc_auc']:.6f})."),
        }
        print(f"\n=== candidate {i + 1}/{len(candidates)} (inner {cand['inner_roc_auc']:.6f}) ===")
        res = run_cv_experiment(rc, X, y, X_test, feats, kaggle_run=False)
        results.append({"i": i, "inner": cand["inner_roc_auc"], "outer": res["oof_roc_auc"],
                        "res": res, "rc": rc})

    results.sort(key=lambda d: d["outer"], reverse=True)
    print("\n--- inner vs outer ROC-AUC (anti-overfit guard) ---")
    for d in results:
        print(f"  {d['rc']['tag']}: inner {d['inner']:.6f}   outer {d['outer']:.6f}")
    best = results[0]
    print(f"\nBEST by outer OOF: {best['rc']['tag']}  outer={best['outer']:.6f}")
    run_id = save_experiment(best["res"])
    print(f"\nSAVED {model}: run_id={run_id}  outer_oof={best['outer']:.6f}")


if __name__ == "__main__":
    main()
