#!/usr/bin/env python3
"""Download competition data for playground-series-s6e3.

Prerequisites
-------------
1. kaggle package installed:
       pip install kaggle
   or, if using uv:
       uv add kaggle

2. Kaggle API token at ~/.kaggle/kaggle.json
   Generate at: https://www.kaggle.com/settings/account -> API -> Create New Token

Usage
-----
    python data/fetch_data.py
"""
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "playground-series-s6e3"
OUT_DIR = Path(__file__).parent  # saves alongside this script in data/


def main() -> None:

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(OUT_DIR)],
        check=True,
    )

    zip_path = OUT_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
        extracted = zf.namelist()
    print(f"Extracted: {extracted}")

    zip_path.unlink()

    sample = OUT_DIR / "sample_submission.csv"
    if sample.exists():
        sample.unlink()
        print("Removed sample_submission.csv")

    print("\nData files:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
