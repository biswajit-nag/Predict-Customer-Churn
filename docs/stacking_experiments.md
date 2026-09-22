# Stacking, pseudo-labels & per-fold encoding — an experimental post-mortem

**Written:** 2026-06-27, updated with the final leaderboard results ·
**Competition:** Kaggle Playground S6E3 (ROC-AUC) ·
**Question:** can a level-2 GBDT, fed the diverse cohort's predictions and/or
pseudo-labelled test rows, beat the best single model?

**Answer: no.** Across every variant tried — collapsed vs per-fold OOF columns,
no / 5% / 100% pseudo-labels, conservative vs aggressive tuning — **not one beat
the `fe_v4_native` base** (private LB **0.91543** for the base model of the time).
The diverse signal is real but is best harvested by a **linear stack**
([03_Blending.ipynb](../03_Blending.ipynb): OOF 0.91719, private 0.91582); a GBDT
meta-learner is the wrong vehicle, for structural reasons mapped out below.
The value of the exercise is the *why*: several distinct failure mechanisms, each
illustrated by a clean run.

---

## 1. What we were trying to beat

The best single models, all on `fe_v4_native` (19 native categoricals +
`AverageMonthly` + 2 contract crosses), fully tuned (60-trial Optuna, full data):

| model | OOF ROC-AUC | private LB |
|---|---|---|
| **lgbm-catreg-fe-min3** | 0.916685 | **0.91543** |
| catboost-gpu-fe-min3 | 0.916485 | 0.91537 |
| xgb-coupled-fe-min3 | 0.916650 | 0.91529 |

OOF↔private-LB Spearman across the 40+ runs logged at the time was **0.953** (0.977
over all 68 scored runs by the end of the project) — CV is trustworthy, so OOF
deviations from LB below are *signal*, not noise.

The 6 diverse base models stacked in (best full-length run per non-GBDT
mechanism; GBDTs correlate at ρ>0.99 so adding one to another teaches nothing):

| column key | model | mechanism | OOF |
|---|---|---|---|
| tabpfn | TabPFN (subfold-bag3) | in-context (TFM) | 0.9142 |
| tabicl | TabICL (subfold-bag3) | in-context (TFM) | 0.9142 |
| realmlp | RealMLP | deep net | 0.9140 |
| tabm | TabM | deep net | 0.9137 |
| rf | RandomForest | bagged trees | 0.9147 |
| lr | LogReg + target-enc | linear | 0.9101 |

---

## 2. The two data versions

- **`fe_v5_stack`** — `fe_v4_native` + 6 *collapsed* OOF columns (one per model).
  Train rows get the model's leakage-free OOF; test rows get its bagged
  `test_proba_mean`. (`src/stacking.py::build_stacked_dataset`)
- **`fe_v6_perfold`** — `fe_v4_native` + 30 *per-fold* columns (6 models × 5
  folds). Train: `{model}_f{k}` holds the held-out OOF for fold-k rows only, **NaN
  for the other 4/5**. Test: each `{model}_f{k}` holds that fold-model's test
  prediction (**all 5 filled**). (`src/stacking.py::build_perfold_dataset`)

Pseudo-labels (when used): confident test rows appended to **each fold's training
set only** (OOF stays on real rows, full-length, leakage-free —
`run_cv_experiment(..., X_pseudo=, y_pseudo=)`). "5%" = top/bottom 5% of a
rank-mean blend; "100%" = all test rows, hard label `blend > 0.5`.

---

## 3. Results — every variant (gap = OOF − private LB)

| variant | data | n_pseudo | OOF | private LB | gap |
|---|---|---|---|---|---|
| **base** lgbm-catreg | fe_v4 | 0 | 0.91669 | **0.91543** | 0.00126 |
| stack + 5% pseudo (lgbm) | fe_v5 | 23 805 | 0.91569 | 0.91429 | 0.00140 |
| stack + 5% pseudo (xgb) | fe_v5 | 23 805 | 0.91575 | 0.91443 | 0.00132 |
| stack + 5% pseudo (cat) | fe_v5 | 23 805 | 0.91524 | 0.91376 | 0.00148 |
| stack + 100% pseudo (lgbm, aggr) | fe_v5 | 254 655 | 0.91480 | 0.91433 | 0.00047 |
| stack + 100% pseudo (xgb, aggr) | fe_v5 | 254 655 | 0.91500 | 0.91436 | 0.00064 |
| stack + 100% pseudo (cat, aggr) | fe_v5 | 254 655 | 0.91499 | 0.91388 | 0.00111 |
| **per-fold**, no pseudo (lgbm, aggr) | fe_v6 | 0 | 0.91578 | 0.91389 | 0.00189 |
| **per-fold**, no pseudo (xgb, aggr) | fe_v6 | 0 | **0.91599** | 0.91344 | **0.00255** |
| per-fold + 100% pseudo (lgbm, aggr) | fe_v6 | 254 655 | **0.79924** | 0.85668 | −0.05744 |
| per-fold + 100% pseudo (xgb, aggr) | fe_v6 | 254 655 | 0.84885 | 0.85909 | −0.01024 |

Read top-to-bottom it's a tour of failure modes: a steady ~0.001 deficit that
*widens* the harder you fit, then a total collapse. None clears 0.91543.

---

## 4. Why — five mechanisms

### 4.1 The OOF columns are redundant-with-noise
The 6 added models (0.910–0.9147) are individually *weaker* than the base GBDT
and rank-correlated with what it already learns from `fe_v4`. As features they
encode little new signal, but they do add variance. At matched (mediocre)
hyperparameters the stacked features are ~neutral (±0.0002); the deficit to the
base is the **tuning regime**, not the features (the conservative runs used a
regularization-biased subsample search to avoid overfitting the columns, which
also underfit the real categoricals; an aggressive search recovers OOF to ~0.916,
see §4.4).

### 4.2 Train/test *value* mismatch — and why a tree (not a linear stacker) breaks
A collapsed OOF column has **different distributions at train vs test**: train =
*single-fold* OOF (noisy), test = *5-fold bagged mean* (smoother, lower-variance).
- A **linear stacker** depends on each meta-feature only through one global,
  monotone coefficient. The bagged mean is ~a monotone/affine rescaling of the
  single-fold OOF, and `w·(a·x+b)` ranks identically to `w·x` — so AUC is
  preserved. The mismatch washes out. (This is *why* stacking is conventionally
  done with Ridge/logistic, not a GBDT.)
- A **tree** consumes the column as hard, distribution-specific **thresholds**
  (`x > 0.62`) and local interactions, calibrated to the *train* column's exact
  spread. A variance/scale shift at test time moves rows across those boundaries
  and the learned local rules misfire — the error doesn't average out.

### 4.3 Pseudo-labels add no information here; 100% pseudo poisons the base rate
Pseudo-labelling wins with scarce labels or train/test shift. Here there are 594k
labels and **no shift** (adversarial AUC ≈ 0.5; same synthetic generator). So the
confident rows just reinforce easy cases (confirmation bias). At **100%**, `blend
> 0.5` labels the **top 50% by rank** positive — a ~50% churn rate vs the true
22.5% — dragging the training base rate up and pulling OOF *accuracy* from 0.86
toward 0.69–0.77. Even so, on the collapsed columns this only cost ~0.001
(0.9149); the catastrophe needed the per-fold encoding (§4.5).

### 4.4 Per-fold encoding fixes the value mismatch but introduces a *missingness*-pattern mismatch
The per-fold trick was meant to cure §4.2: each `{model}_f{k}` is the *same fitted
model's out-of-sample output* on both its train rows (held-out fold) and all test
rows — same value distribution. It works for that. **But it creates a new
asymmetry:** train rows have **1/5** per-fold columns filled (4/5 NaN), test rows
have **5/5** filled. A split on `tabpfn_f2` is learned from the ~20% of train rows
where it's present (the rest routed to the NaN default); at test *every* row takes
the real-value branch — a routing regime the tree never trained on.

The tell is in the gaps: per-fold runs have the **widest OOF→LB gaps** (LGBM
0.00189, XGB **0.00255**), and **the harder you fit, the wider the gap** — the
aggressive XGB posts the **highest OOF of any stacked variant (0.91599)** yet
**among the lowest LB (0.91344)**. OOF and LB have decoupled: OOF rows share the
train (1/5-filled) pattern, so OOF can't see the test-time degradation.

### 4.5 Per-fold + 100% pseudo → degenerate shortcut + base-rate poison → collapse
Combining §4.3 and §4.4 produces a trap neither does alone (OOF **0.80/0.85**, LB
~0.857):
1. Each appended pseudo (test) row carries all 30 per-fold columns, and its label
   is `rank-mean(those columns) > 0.5` — so the label is ≈ **the mean of the
   row's own features**, a near-perfect **shortcut** the tree latches onto (it
   cleanly fits ~30% of the training data).
2. That shortcut needs the columns *filled* — but real train/OOF rows are **1/5
   filled** (4/5 NaN). The model, having spent capacity on the shortcut, **fails
   on real OOF rows → OOF craters to 0.80**.
3. Base-rate poison (§4.3) on top.

The missingness asymmetry didn't get fixed — it **inverted and amplified**: now
OOF (1/5-filled, shortcut fails) is *pessimistic* relative to test (5/5-filled,
shortcut partly works), and both real-OOF and true-test are wrecked.

**Corollary:** the clean form of the "show the model the all-5-filled pattern"
hypothesis is **untestable with pseudo-labels** — you can't add unlabeled test
rows to supervised GBDT training, and any label you attach is the confound that
builds the shortcut. The label-free way to harden a tree against the pattern would
be **training-time masking/dropout of the per-fold columns**, not pseudo-labels.

### 4.6 Postscript: the label-free repairs (2026-06-30)
Both repairs were implemented in `src/stacking.py::PerFoldMaskingClassifier` and run
on the aggressive per-fold GBDTs:

| repair | LightGBM OOF | XGBoost OOF |
|---|---|---|
| none (per-fold, aggressive) | 0.91578 | 0.91599 |
| mask-bagging (predict on 5 masked views — view *k* keeps only the fold-*k* columns — and average; applied to OOF and test rows alike) | 0.91578 | 0.91599 |
| train-time dropout (hide each filled per-fold value with probability 0.5 while fitting) | 0.91603 | 0.91633 |

Mask-bagging left OOF unchanged to six decimals: on train rows, which have at most
one fold's columns filled, the masked views add nothing. Train-time dropout lifts OOF by 0.0003 but still leaves both models
below the 0.91668 base, so neither was submitted. The per-fold route stays closed.

---

## 5. Takeaways

1. **The `fe_v4` single model (LB 0.91543) was never beaten** by any level-2 GBDT.
2. **Use a linear/logistic meta-learner for stacking, not a GBDT** — it's immune
   to the OOF value mismatch (§4.2) that the tree (and the per-fold "fix") fall to.
   The logistic stack in `03_Blending.ipynb` clears the base on both OOF (0.91719)
   and the private leaderboard (0.91582).
3. **Pseudo-labels don't help a large, no-shift dataset**; 100% pseudo actively
   poisons the base rate; per-fold + pseudo is degenerate (don't combine them).
4. **OOF can lie when the meta-features have a train/test distribution or
   missingness mismatch.** Always confirm level-2 work on the LB — the per-fold
   runs looked best on OOF and were among the worst on LB.
5. For this feature set and model pool the plateau is ~0.917: the level-1 models
   already capture what their inputs offer, so added level-2 complexity buys
   variance, not lift. It is not a ceiling for the dataset — the competition's
   top private scores were ~0.9185, reached with additional information such as
   the original IBM Telco data (see [top_solutions_analysis.md](top_solutions_analysis.md)).

---

## 6. Reproducibility

| run group | script | data_version |
|---|---|---|
| collapsed stack + 5% pseudo | `scripts/archive/run_stacked_experiments.py` | fe_v5_stack |
| collapsed stack + 100% pseudo (aggressive) | `scripts/archive/run_stacked_aggressive.py` | fe_v5_stack |
| per-fold, no pseudo (aggressive) | `scripts/archive/run_perfold_aggressive.py` | fe_v6_perfold |
| per-fold + 100% pseudo (aggressive) | `scripts/archive/run_perfold_pseudo_aggressive.py` | fe_v6_perfold |
| per-fold masking repairs (§4.6) | `scripts/archive/run_perfold_masking.py` | fe_v6_perfold |

Builders: `src/stacking.py::build_stacked_dataset` / `build_perfold_dataset`.
Every run is in `experiments/runs.csv` with OOF, `lb_public`/`lb_private`, and
`n_pseudo`; OOF row-alignment is asserted before any OOF vector is reused.
