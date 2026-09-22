# Archived experiment drivers

One-off scripts that produced specific ledger entries. They are kept because
`experiments/runs.csv` points at the runs they created, not because they are part
of the reusable workflow. All of them produced **negative results** — see
[docs/stacking_experiments.md](../../docs/stacking_experiments.md).

| Script | What it ran |
|---|---|
| `run_stacked_experiments.py` | GBDTs on `fe_v5_stack` (base features + six diverse models' OOF columns), with and without 5% confident pseudo-labels |
| `run_stacked_aggressive.py` | The same, with 100% pseudo-labelling and wider tuning |
| `run_perfold_aggressive.py` | GBDTs on `fe_v6_perfold` (five per-fold prediction columns per base model) |
| `run_perfold_pseudo_aggressive.py` | `fe_v6_perfold` plus pseudo-labels |
| `run_perfold_masking.py` | `PerFoldMaskingClassifier` repairs (test-time mask-bagging, train-time dropout) |
| `_blend_preview.py` | Early OOF-correlation / stack preview, superseded by `03_Blending.ipynb` |
| `_validate_subfold_artifacts.py` | One-off integrity check of the TabPFN/TabICL subfold notebooks' artifacts |

Run them from the project root, e.g. `python scripts/archive/run_perfold_masking.py`.
