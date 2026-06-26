"""Preview blending value of the neural runs: OOF correlations + stack AUCs.

Uses only already-logged runs. The logistic stack is leakage-free: meta-features
are logit-transformed OOF probabilities (the standard scale for stacking
probabilistic base models — raw probabilities double-squash through the
sigmoid), and the stack itself is evaluated OOF over the canonical folds from
cv_folds_seed42.csv.gz. Each run's OOF row alignment is asserted against its
logged score before use (same check as scripts/check_oof_alignment.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tracking import DATA_DIR, PROJECT_ROOT, RUNS_DIR, load_runs  # noqa: E402

RUNS = {
    "lgbm-min3":      "20260610-183508-48286b",
    "xgb-min3":       "20260610-200857-74b63f",
    "catboost-min3":  "20260611-051031-5d6bc5",
    "tabm-fe_v0":     "20260601-022539-10bd38",
    "realmlp-fe_v0":  "20260603-021557-d716c1",
    "tabicl-subfold": "20260612-031158-5adf7a",
}

y = pd.read_parquet(DATA_DIR / "train_df_native.parquet", columns=["Churn"])["Churn"].to_numpy()
folds = pd.read_csv(PROJECT_ROOT / "experiments" / "cv_folds_seed42.csv.gz")["fold"].to_numpy()
runs_csv = load_runs().set_index("run_id")

oof = {}
print("Individual OOF ROC-AUC (each asserted against the logged value):")
for name, rid in RUNS.items():
    p = np.load(RUNS_DIR / rid / "oof_proba.npy")
    auc = roc_auc_score(y, p)
    assert abs(auc - runs_csv.loc[rid, "oof_roc_auc"]) < 1e-9, f"{name}: OOF not row-aligned!"
    oof[name] = p
    print(f"  {name:15s} {auc:.6f}")

names = list(RUNS)
P = np.column_stack([oof[n] for n in names])
print("\nPearson correlation of OOF probabilities:")
corr = np.corrcoef(P, rowvar=False)
print("  " + " ".join(f"{n:>15s}" for n in names))
for i, n in enumerate(names):
    print(f"  {n:15s} " + " ".join(f"{corr[i, j]:15.4f}" for j in range(len(names))))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def stack_auc(cols: list[str]) -> float:
    """OOF AUC of a logistic stack on logit-probabilities over the canonical folds."""
    X = np.column_stack([logit(oof[c]) for c in cols])
    stacked = np.zeros(len(y))
    for k in range(5):
        tr, va = folds != k, folds == k
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        stacked[va] = clf.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, stacked)


print("\nBlend preview (logistic stack on logit-probs, OOF over canonical folds):")
gbdt = ["lgbm-min3", "xgb-min3", "catboost-min3"]
print(f"  best single GBDT                   {roc_auc_score(y, oof['lgbm-min3']):.6f}")
print(f"  equal-weight mean of 3 GBDTs       {roc_auc_score(y, P[:, :3].mean(axis=1)):.6f}")
print(f"  stack: 3 GBDTs                     {stack_auc(gbdt):.6f}")
print(f"  stack: 3 GBDTs + TabM (fe_v0)      {stack_auc(gbdt + ['tabm-fe_v0']):.6f}")
print(f"  stack: 3 GBDTs + RealMLP (fe_v0)   {stack_auc(gbdt + ['realmlp-fe_v0']):.6f}")
print(f"  stack: 3 GBDTs + TabICL (subfold)  {stack_auc(gbdt + ['tabicl-subfold']):.6f}")
print(f"  stack: 3 GBDTs + both fe_v0 NNs    {stack_auc(gbdt + ['tabm-fe_v0', 'realmlp-fe_v0']):.6f}")
print(f"  stack: 3 GBDTs + all 3 neural      {stack_auc(gbdt + ['tabm-fe_v0', 'realmlp-fe_v0', 'tabicl-subfold']):.6f}")
print(f"  stack: all 6                       {stack_auc(names):.6f}")
