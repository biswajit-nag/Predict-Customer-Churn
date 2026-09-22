# Data

The competition files are not checked into the repository (they are Kaggle
competition data). Download them with the fetch script.

## Prerequisites

The `kaggle` CLI with an API token — see
[docs/reference/kaggle_setup.md](../docs/reference/kaggle_setup.md), or Kaggle's
[authentication instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
You must also have accepted the competition rules on the
[competition page](https://www.kaggle.com/competitions/playground-series-s6e3).

## Download

From the project root:

```bash
python data/fetch_data.py
```

This downloads `playground-series-s6e3.zip`, extracts `train.csv` and `test.csv`
into `data/`, and removes the zip.

## Derived files (not committed, regenerated on demand)

| Path | Produced by | Contents |
|---|---|---|
| `data/processed/train_df.parquet`, `test_df.parquet` | `src.data.prepare_data(encoding="onehot")` | 40 one-hot columns, for linear models |
| `data/processed/train_df_native.parquet`, `test_df_native.parquet` | `src.data.prepare_data(encoding="native")` | the 19 raw features, strings as `category` dtype |
| `data/processed/train_df_fe_v4_native.parquet`, … | `src.features.build_features("fe_v4_native")` | native base + engineered features |

Every file is written in raw `train.csv` row order — never reordered — which is
what keeps the shared cross-validation folds in
`experiments/cv_folds_seed42.csv.gz` valid for every model.
