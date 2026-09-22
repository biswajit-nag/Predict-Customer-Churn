# Feature engineering

How features are built, versioned and kept leakage-free, and what the feature
experiments found.

## Versioned feature sets

Every feature set has a label, its `data_version`, which is recorded on each run
in `experiments/runs.csv` and names the cached files the run trained on:

```
data/processed/train_df_{data_version}.parquet
data/processed/test_df_{data_version}.parquet
```

The feature code lives in [`src/features.py`](../src/features.py). Each version is a
function registered in `FEATURE_SETS`; `build_features(data_version)` returns the
train/test frames, building and caching them on first use:

```python
from src.features import build_features

train_df, test_df = build_features("fe_v4_native")
```

**Cache safety.** A cache keyed only on the label can silently serve stale data:
edit the function, forget to change the label, and the old parquet is loaded as if
nothing had changed. `build_features` stores a hash of the feature function's source
code next to the parquets and rebuilds them when the hash no longer matches. The
label still changes whenever a change is meant to produce a new, separately logged
feature set — never reuse a label.

**Reproducing a past run's data.** Each ledger row carries `data_hash`, a SHA-256
fingerprint of the exact training parquet. Rebuilding `fe_v4_native` from the raw
CSVs reproduces the recorded hash byte for byte (demonstrated in
`02_Experiments.ipynb`). For versions whose code predates `src/features.py`, the
row's `git_hash` and the run's `git_diff.patch` recover the code that built them.

## Stateless vs stateful features

The split decides *where* a transform may run.

**Stateless (row-wise)** — each row's output depends only on that row:
ratios, differences, logs, indicators, crosses of two categorical columns. These are
computed once on the full train and test frames, before any cross-validation split,
without leaking anything across folds. Every feature in `src/features.py` is of this
kind.

**Stateful** — anything fitted on data: target or frequency encoding, scaling,
imputation with learned values, binning with learned edges. Fitting these on the
full training set before the split leaks validation-fold statistics into training
and inflates out-of-fold scores. They must be fitted inside the fold loop on the
training rows only. The clean way to do that is to make the transform part of the
model, so `run_cv_experiment` refits it per fold:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

model_factory = lambda params: make_pipeline(
    make_column_transformer(
        (TargetEncoder(), ["contract_x_payment", "contract_x_internet"]),
        remainder=StandardScaler()),
    LogisticRegression(**params))
```

The target-encoded logistic-regression run (`lr-targetenc-crosses-native`) used this
pattern, with one-hot encoding for the remaining categorical columns.

**Categorical encoding is structural, not fitted.** `src/data.py` produces two
encodings of the raw data — one-hot columns for linear models, and native pandas
`category` columns that LightGBM, XGBoost and CatBoost consume directly. Both depend
only on the column names and the fixed set of category levels, so train and test are
encoded independently with identical schemas and no fitted encoder.

## What the feature experiments found

Best run per feature set (OOF ROC-AUC over the shared 5-fold CV, and the private
leaderboard score):

| `data_version` | Features added to the 19 raw columns | Best OOF | Private LB |
|---|---|---|---|
| `fe_v0_native` | none (native categoricals) | 0.91629 (EBM) | 0.91485 |
| `fe_v0` | none (40 one-hot columns) | 0.91655 (XGBoost) | 0.91525 |
| `fe_v1` | `TotalDiscount` | 0.91656 (LightGBM) | 0.91525 |
| `fe_v2_native` | `TotalDiscount`, `PriceRatio`, `AverageMonthly` | 0.91651 (LightGBM) | 0.91528 |
| `fe_v3_native` | fe_v2 + four categorical crosses: `Contract × PaymentMethod`, `Contract × InternetService`, a combined service-subscription profile, `tenure band × Contract` | 0.91652 (CatBoost) | 0.91521 |
| `fe_nocross5_native` | fe_v3 without the two high-cardinality crosses | 0.91656 (XGBoost) | 0.91513 |
| **`fe_v4_native`** | **`AverageMonthly`, `Contract × PaymentMethod`, `Contract × InternetService`** | **0.91681** (LightGBM) | **0.91558** |

Where `AverageMonthly = TotalCharges / tenure` (what the customer has actually paid
per month), `TotalDiscount = TotalCharges − MonthlyCharges × tenure` and
`PriceRatio = MonthlyCharges / AverageMonthly`.

Three things to take from the table:

1. **The differences are small.** Every version lands within about 0.0003 OOF of the
   others, less than the ~0.0009 fold-to-fold standard deviation. Feature engineering
   was not where this problem's score came from; the untuned baseline already
   reached 0.9150 (`01_EDA.ipynb`).
2. **Fewer features won.** The smallest engineered set, `fe_v4_native` ("min3"),
   scored best on both OOF and the private leaderboard, though by margins inside the
   noise of point 1. The
   high-cardinality crosses and the redundant charge ratios added variance rather
   than signal.
3. **The reason is the structure of the target.** The max-depth probe in
   `01_EDA.ipynb` shows that the score peaks at depth-3 trees: the target is main
   effects plus a few low-order interactions, which a tuned GBDT finds on its own.
   Hand-built crosses help only where they hand the model an interaction it would
   otherwise need several splits to express — here, `Contract` with the payment and
   internet type.

## What was not tried

Two feature ideas used by top-ranked competitors were not explored here — using
the original IBM Telco dataset (which the synthetic data was generated from) as a
source of aggregate features, and features built from the digits of the charge
columns, which can carry artefacts of the synthetic generator. See
[top_solutions_analysis.md](top_solutions_analysis.md).
