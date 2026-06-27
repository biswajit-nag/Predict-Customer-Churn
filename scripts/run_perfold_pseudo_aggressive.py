"""Per-fold OOF encoding + 100% pseudo-labels, aggressive tuning (LGBM & XGB).

Combines the per-fold dataset (fe_v6_perfold = fe_v4_native + 30 per-fold OOF columns)
with true pseudo-labelling: ALL ~254k test rows are appended to each fold's training set
carrying a hard pseudo-label (rank-mean blend > 0.5). Versus the earlier stack+pseudo
runs (fe_v5_stack), each appended test row now carries **30 per-fold prediction columns**
(5x the 6 collapsed OOF columns).

The point of interest: real train rows have only 1/5 per-fold columns filled (NaN for the
other 4), but the appended test rows have all 5 filled — so the model now *sees the
"all-5-filled" pattern during training*, potentially mitigating the missingness-pattern
train/test asymmetry that depressed the per-fold-only runs' test LB. OOF is still scored on
real rows only (leakage-free). Caveat: the pseudo-label is ~the mean of a row's own per-fold
columns, a strong shortcut on the pseudo rows — read the result with that in mind.

Run: .venv\\Scripts\\python.exe scripts\\run_perfold_pseudo_aggressive.py [lgbm xgb]
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.cv import run_cv_experiment, save_experiment
from src.stacking import build_perfold_dataset
from src.tracking import RUNS_DIR
# Reuse: aggressive tuner / model builder, the per-fold spec, the pseudo blend pool.
from run_stacked_aggressive import FIXED_EXTRA, OUTER_CV, make_model, tune
from run_stacked_experiments import PSEUDO_BLEND_RUNS
from run_perfold_aggressive import DATA_VERSION, FOLD_RUNS

PLAN = {
    "lgbm": dict(n_trials=60, tune_subsample=None, tag="lgbm-perfold-pl100-aggr-fe_v6",
                 parent="20260610-183508-48286b"),
    "xgb":  dict(n_trials=60, tune_subsample=None, tag="xgb-perfold-pl100-aggr-fe_v6",
                 parent="20260610-200857-74b63f"),
}


def main() -> None:
    train_df, test_df = build_perfold_dataset("fe_v4_native", FOLD_RUNS, DATA_VERSION)
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y, X_test = train_df[feats], train_df["Churn"], test_df[feats]
    cat_features = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]
    n_pf = sum("_f" in c and c.rsplit("_f", 1)[-1].isdigit() for c in feats)

    # 100% pseudo-labels: every test row, hard label from the rank-mean blend.
    tp = np.column_stack([np.load(RUNS_DIR / r / "test_proba_mean.npy")
                          for r in PSEUDO_BLEND_RUNS])
    blend = np.column_stack([rankdata(tp[:, j]) for j in range(tp.shape[1])]).mean(axis=1)
    blend = (blend - blend.min()) / (blend.max() - blend.min())
    X_pseudo = X_test.reset_index(drop=True)          # all test rows, per-fold cols 5/5 filled
    y_pseudo = (blend > 0.5).astype(int)
    print(f"fe_v6_perfold + 100% pseudo: {len(feats)} features ({n_pf} per-fold), "
          f"+{len(X_pseudo):,} test rows ({y_pseudo.mean():.3f} positive)\n", flush=True)

    only = set(sys.argv[1:])
    for name, cfg in PLAN.items():
        if only and name not in only:
            continue
        print(f"=== {cfg['tag']} ===", flush=True)
        try:
            best = tune(name, X, y, cat_features, cfg["n_trials"], cfg["tune_subsample"])
            params = {**best, **FIXED_EXTRA[name]}
            run_config = {
                "model_factory": (lambda p, _n=name: make_model(_n, p, cat_features)),
                "params": params, "cv": OUTER_CV, "tag": cfg["tag"],
                "notes": f"AGGRESSIVE per-fold + 100% pseudo: {name} on {DATA_VERSION} "
                         f"(fe_v4_native + {n_pf} per-fold OOF columns) with ALL test rows "
                         f"appended to training folds as pseudo-labels (hard label = rank-mean "
                         f"blend > 0.5); each carries 30 per-fold columns (5x the collapsed "
                         f"fe_v5_stack pseudo runs). Appended test rows are 5/5-filled vs 1/5 "
                         f"for real rows, so the model sees the all-filled pattern in training. "
                         f"Unconstrained Optuna, {cfg['n_trials']} trials, full-data inner CV. "
                         f"OOF on real rows only. scripts/run_perfold_pseudo_aggressive.py.",
                "parent_run_id": cfg["parent"], "save_models": False,
                "data_version": DATA_VERSION,
            }
            result = run_cv_experiment(run_config, X, y, X_test, feats,
                                       X_pseudo=X_pseudo, y_pseudo=y_pseudo)
            save_experiment(result, submit=True)
            print(f"  saved + submitted: OOF ROC-AUC {result['oof_roc_auc']:.6f}\n", flush=True)
        except Exception as e:
            print(f"  ERROR on {name}: {type(e).__name__}: {e}\n", flush=True)


if __name__ == "__main__":
    main()
