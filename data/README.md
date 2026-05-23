# Data

Raw competition files are not checked into the repository. Run the fetch
script below to download them from Kaggle.

## Prerequisites

1. Install the `kaggle` package:
   ```
   pip install kaggle
   ```
   or, if using uv:
   ```
   uv add kaggle
   ```
2. Authenticate with your Kaggle credentials:
   ```
   kaggle auth login
   ```
   For further details, follow the [official instructions here](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).

## Download

From the project root:

```
python data/fetch_data.py
```

This will:
- Download `playground-series-s6e3.zip` from Kaggle
- Extract `train.csv` and `test.csv` into `data/raw/`
- Delete the zip file and the unused `sample_submission.csv`

## Processed files

After downloading the raw files, run `notebooks/02_baselines.ipynb` end-to-end
to produce the preprocessed parquets used by the remaining notebooks:

```
data/processed/train_df.parquet
data/processed/test_df.parquet
```
