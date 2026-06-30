"""Feature-masking experiments on the per-fold dataset (controlled).

Tests the two label-free ways to address the per-fold missingness asymmetry
(train rows 1/5 filled, test rows 5/5 filled), holding everything else fixed:

  - maskbag  : test-time fold-view masking + bagging (PerFoldMaskingClassifier,
               test_mask_bag=True) — every prediction (OOF + test) presented in 5
               masked views averaged, so every forward pass is train-like in density.
  - drop50   : train-time per-fold dropout (train_dropout=0.5) — present per-fold
               values randomly NaN-ed at fit; prediction unmasked.

Crucially, each run REUSES the exact tuned hyperparameters of the matching
per-fold aggressive run (no pseudo), so the ONLY variable changed is the masking.
That makes the comparison to lgbm/xgb-perfold-aggr-fe_v6 clean.

Run: .venv\\Scripts\\python.exe scripts\\run_perfold_masking.py [maskbag drop50 lgbm xgb]
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.cv import run_cv_experiment, save_experiment
from src.stacking import PerFoldMaskingClassifier, build_perfold_dataset
from src.tracking import RUNS_DIR
from run_perfold_aggressive import DATA_VERSION, FOLD_RUNS

OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Source of tuned hyperparameters = the per-fold aggressive (no-pseudo) runs.
PARAM_RUN = {"lgbm": "20260626-223318-c2cbf6", "xgb": "20260626-234321-3aaedc"}

EXPERIMENTS = [
    ("lgbm", dict(test_mask_bag=True),  "lgbm-perfold-maskbag-fe_v6"),
    ("xgb",  dict(test_mask_bag=True),  "xgb-perfold-maskbag-fe_v6"),
    ("lgbm", dict(train_dropout=0.5),   "lgbm-perfold-drop50-fe_v6"),
    ("xgb",  dict(train_dropout=0.5),   "xgb-perfold-drop50-fe_v6"),
]


def make_base(name, params):
    """Rebuild the tuned base model from its saved get_params() dict."""
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    from xgboost import XGBClassifier
    return XGBClassifier(**params)


def main() -> None:
    train_df, test_df = build_perfold_dataset("fe_v4_native", FOLD_RUNS, DATA_VERSION)
    feats = [c for c in train_df.columns if c not in ("id", "Churn")]
    X, y, X_test = train_df[feats], train_df["Churn"], test_df[feats]
    perfold = [c for c in feats if "_f" in c and c.rsplit("_f", 1)[-1].isdigit()]
    print(f"fe_v6_perfold: {len(feats)} features, {len(perfold)} per-fold.\n", flush=True)

    only = set(sys.argv[1:])
    for name, opts, tag in EXPERIMENTS:
        variant = "maskbag" if opts.get("test_mask_bag") else "drop50"
        if only and not ({name, variant, tag} & only):
            continue
        params = json.load(open(RUNS_DIR / PARAM_RUN[name] / "params.json"))
        print(f"=== {tag} ===", flush=True)
        try:
            factory = (lambda p, _n=name, _o=opts:
                       PerFoldMaskingClassifier(lambda: make_base(_n, p), perfold,
                                                params=p, **_o))
            if variant == "maskbag":
                detail = ("test-time fold-view masking + bagging: OOF and test each predicted "
                          "as the mean of 5 masked views (view k keeps *_f{k}, NaNs the rest), "
                          "so every forward pass has the train-like 1/5 density.")
            else:
                detail = ("train-time per-fold dropout 0.5: present per-fold values randomly "
                          "NaN-ed at fit; prediction unmasked.")
            run_config = {
                "model_factory": factory, "params": params, "cv": OUTER, "tag": tag,
                "notes": f"Controlled masking test: {name} on {DATA_VERSION}, REUSING the tuned "
                         f"hyperparameters of {PARAM_RUN[name]} (per-fold aggressive, no pseudo) "
                         f"so masking is the only change. {detail} "
                         f"scripts/run_perfold_masking.py.",
                "parent_run_id": PARAM_RUN[name], "save_models": False,
                "data_version": DATA_VERSION,
            }
            result = run_cv_experiment(run_config, X, y, X_test, feats)
            save_experiment(result, submit=True)
            print(f"  saved + submitted: OOF ROC-AUC {result['oof_roc_auc']:.6f}\n", flush=True)
        except Exception as e:
            print(f"  ERROR on {tag}: {type(e).__name__}: {e}\n", flush=True)


if __name__ == "__main__":
    main()
