"""Log (and submit) the six pre-registered headline blends as runs.csv entries.

The unattended equivalent of the logging section of 03_Blending.ipynb, so the
six headline blends — rank-mean (curated), the moderate-C curated logit stack, the
carried-forward best logit stack, hill-climb (curated), the best-softmax mean, and
the 50-trial Optuna LightGBM stack — get logged with full per-fold AUC mean/std
and, with --submit, uploaded to Kaggle so lb_public /
lb_private fill in.

Run:
    .venv\\Scripts\\python.exe scripts\\run_blend_logging.py            # log only
    .venv\\Scripts\\python.exe scripts\\run_blend_logging.py --submit   # log + submit
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blending import log_headline_blends  # noqa: E402


def main() -> None:
    submit = "--submit" in sys.argv[1:]
    print(f"Logging headline blends (submit={submit}) ...\n", flush=True)
    summary = log_headline_blends(submit=submit, wait=True)
    print("\n=== per-fold blend summary ===")
    print(summary.to_string(index=False, formatters={
        "oof_roc_auc": "{:.6f}".format, "fold_mean": "{:.6f}".format,
        "fold_std": "{:.6f}".format, "vs_best_single": "{:+.6f}".format,
        "gain_in_fold_stds": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
