# What the top solutions did differently

A retrospective, written after the competition closed, comparing this project with
the published write-ups of teams that finished near the top. The goal is to find out
which of this project's open hypotheses the write-ups settle and what a second
attempt should change.

## The gap

| | Private leaderboard ROC-AUC |
|---|---|
| 1st place | 0.91850 |
| 3rd place | 0.91845 |
| 21st place | 0.91834 |
| 49th place | 0.91825 |
| **This project, best blend** (submitted after the deadline, unranked) | **0.91582** |
| **This project, best single model** | **0.91558** |

The top of the leaderboard is extremely compressed: 49 teams scored within 0.00025 of
the winner. This project is about **0.0027** below them. That is about three times
the fold-to-fold noise of a single model, and larger than everything this project's
blending gained (+0.0004 OOF). The difference is real, and it is not a blending
difference.

## What the write-ups report

Sources were read on 2026-09-22 from the competition's write-up and discussion
pages. Numbers are quoted as the authors state them. Each team's out-of-fold (OOF)
scores are measured on its own CV setup, so they are not directly comparable with
this project's, but the offsets are similar: their OOF runs about 0.0015 above their
private score, this project's about 0.0013.

**17th place** ([write-up](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/17th-place-solution))
* **Used the original IBM Telco dataset** — the real data the synthetic set was
  generated from — as a source of group statistics: "groupby(Contract)['tenure'].mean()
  and its siblings (std, min, max, median, Q10–Q90, IQR) computed from the IBM
  dataset".
* **Periodic features on the charges**: "For MonthlyCharges: periods of 10, 25, 50,
  100 (billing round numbers)."
* Generated 800+ features, but noted that "throwing all 800+ features into a single
  GBDT caused overfitting"; the pool was used as a "feature buffet" for different
  models.
* "138 models across 15+ architecture families", combined by greedy hill climbing:
  final OOF 0.91974; best single model 0.91903.
* Pseudo-labelling: "marginal gain, not worth the complexity."

**21st place — "Final Blend Selection with Ridge and Nelder-Mead"**
([write-up](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/21st-place-solution-final-blend-selection-with-ri))
* "Around 50 to 70 single models in total"; the main twelve range from 0.913 to 0.919
  OOF. Several are named for **digit features** ("CAT DIGIT STRONGER", "XGB DIGIT").
* Submitted two blends as final candidates — a Ridge blend ("stable type") and a
  Nelder–Mead-weighted blend ("numerically strongest type") — rather than picking one
  by CV.
* Weak models earned places as "correction terms in the 'subtracting' direction", and
  a candidate was kept only if it improved the *blend*: "Do not adopt something just
  because the single-model CV improved slightly."

**3rd place — "An Ensemble of 100 OOFs"**
([write-up](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/3rd-place-solution-an-ensemble-of-100-oofs))
* About 100 OOF prediction sets: XGBoost variants, LightAutoML, field-aware
  factorisation machines, ResNet, RealMLP, DCN, TabTransformer, AutoGluon, CatBoost and
  others, including models from public notebooks.
* Hill climbing was tried; **a plain linear regression over the OOF predictions** was
  the best combiner on both public and private leaderboards.
* Folds were not consistent across all models, which the author notes widened the
  CV–leaderboard gap.

**Chris Deotte, "Bartz Starter — CV 0.916401"**
([discussion](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680976))
* Introduces BART (Bayesian additive regression trees) as a diverse ensemble member
  alongside XGBoost, logistic regression with target encoding, an MLP and a GNN.
* Points out that model families want different categorical encodings: "GBDT like TE
  categorical and LogReg likes OHE categorical and NN like Label encode to embedding
  categorical."

The 1st-place write-up could not be retrieved, so this comparison is based on the
teams above.

## What this settles

| Open question in this project | What the write-ups say |
|---|---|
| Would the **original IBM Telco data** help? (Never tried here.) | **Yes.** Used explicitly by the 17th-place team for group statistics. This is the largest single idea this project did not try. |
| Does the synthetic generator leave **artefacts in the charge values** that a model can exploit? (Never tried here.) | **Yes, probably.** Digit and "billing round number" features appear in both the 17th- and 21st-place solutions, including among the 21st-place team's strongest models. |
| Was **~0.917 the ceiling** of the data? | **No.** It was the ceiling of *this feature set*: other teams' best *single* models reached about 0.919 OOF on their own CV. The gap is in the inputs, not the models or the blending. |
| Are **pseudo-labels** worth it on this data? | **No** — consistent with this project's negative result ([stacking_experiments.md](stacking_experiments.md)). |
| Is a **linear combiner** over OOF predictions the right final layer? | **Yes.** The 3rd-place team's plain linear regression beat hill climbing; the 21st place used Ridge. This matches this project's result that the logistic stack generalised best. |
| How should the **final blend be chosen**? | Not by the OOF maximum alone. The 21st-place team submitted a stable and an aggressive candidate; this project pre-registered its candidates. Both address the same selection bias. |
| How many **base models**? | Far more than here: 50–140 against this project's 11-model curated pool, many of them the same algorithm on different feature sets or encodings. |

## What a second attempt would change

1. **Bring in the original IBM Telco data** — as a source of per-category statistics
   and as additional training rows, checking with the existing adversarial-validation
   code whether it shifts the distribution.
2. **Test generator artefacts directly** — the last digits and round-number structure
   of `MonthlyCharges` and `TotalCharges`, and whether `TotalCharges` is consistent with
   `MonthlyCharges × tenure`. The depth probe says a GBDT already finds the smooth
   structure, so the remaining signal is more likely in such discontinuities.
3. **Diversify by feature set and encoding, not just by algorithm.** This project's
   diversity came from different learners on nearly the same inputs; every pair
   ranked customers with Spearman ρ ≥ 0.96. The top teams trained the same strong
   learners on different feature views.
4. **Keep what worked here**: one frozen fold map for every model (the 3rd-place team
   notes what inconsistent folds cost them), OOF scores checked against the
   leaderboard, and a linear, pre-registered final combiner.
