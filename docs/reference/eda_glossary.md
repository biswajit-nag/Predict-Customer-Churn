# EDA glossary: plotting & explainability concepts

Reference notes for the terms used in the EDA workflow (see `01_EDA.ipynb`). Each
entry gives a **definition**, the **key features** (what it actually does / how to
read it), and **when to use it**. The final section gives the full game-theoretic
definition of the Shapley value and how SHAP specializes it to GBDTs and TabPFN.

---

## 1. LOWESS smoother

**Definition.** LOWESS / LOESS (LOcally WEighted Scatterplot Smoothing) is a
*non-parametric* regression curve. Instead of fitting one global line, it fits a
separate low-degree polynomial (usually degree 1 or 2) in a sliding window around
each x-value, weighting nearby points more heavily (a tricube kernel), and stitches
the local fits into one smooth curve.

**Key features.**
- No assumed functional form — the curve follows whatever shape the data has.
- One tuning knob, the **bandwidth / span** (`frac` in statsmodels): the fraction
  of points in each local window. Large span → smoother, more biased toward a
  straight line; small span → wigglier, higher variance (can chase noise).
- Computationally heavier than a global fit (one regression per evaluation point),
  and it does **not** give you an equation or coefficients — it's for *seeing*
  shape, not for extrapolation.

**When to use it.** Overlay a LOWESS curve on a feature-vs-target scatter to expose
**non-linearity** that binned means or a straight regression line would hide — e.g.
a U-shaped or saturating relationship between `tenure` and churn. It's an
EDA/diagnostic tool: it tells you whether a linear model will struggle and where a
tree-based model is likely finding structure. In seaborn: `sns.regplot(..., lowess=True)`
or `sns.lmplot(..., lowess=True)`.

---

## 2. Pearson and Spearman correlation heatmaps

A **correlation heatmap** is the feature-by-feature correlation matrix drawn as a
color grid (`sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)`).
The two flavors differ in *what kind* of relationship they measure.

**Pearson correlation** measures the strength of a **linear** relationship:

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y}, \quad r \in [-1, 1]$$

- $\pm 1$ = perfect straight-line relationship; $0$ = no *linear* relationship.
- Sensitive to outliers and assumes a roughly linear, ratio-scale relationship.
- Misses curved relationships: a perfect U-shape can have Pearson $r \approx 0$.

**Spearman correlation** is Pearson's $r$ computed on the **ranks** of the data, so it
measures any **monotonic** relationship (consistently increasing or decreasing),
linear or not:

- $\pm 1$ = perfectly monotonic; robust to outliers; invariant to any monotonic
  transform (log, sqrt) of either variable.
- Works for ordinal features (e.g. `Contract`: month-to-month < one year < two year).

**Key features / how to read.** Look for (a) features strongly correlated with the
**target** (candidate predictors), and (b) features strongly correlated with **each
other** (redundancy / multicollinearity — matters for linear models, less for GBDTs).
**Comparing the two heatmaps is the real value:** a pair with high Spearman but low
Pearson signals a *monotonic-but-curved* relationship — a hint to add a non-linear
transform for linear models, or just to let a tree handle it.

**When to use it.** Always, early in EDA. Pearson for linear-model feasibility and
collinearity; Spearman when features are skewed, ordinal, or non-linearly related.

---

## 3. Q–Q plot (quantile–quantile plot)

**Definition.** A scatter plot of the **quantiles of your data** against the
**quantiles of a reference distribution** (usually the standard Normal). If the two
distributions have the same shape, the points fall on a straight line.

**Key features / how to read.**
- **On the line** → data matches the reference distribution.
- **S-shape / ends curving off the line** → heavier or lighter tails than Normal.
- **Upward or downward bend** → skew (right-skew lifts the upper-right points above
  the line).
- The reference doesn't have to be Normal — a two-sample Q–Q plot compares two
  *empirical* distributions directly (e.g. a feature in train vs test).
- More informative than a single normality *test* (KS, Shapiro) because it shows
  *where* and *how* the distribution departs, not just a pass/fail p-value.

**When to use it.** Before fitting models that assume Gaussian-ish inputs or
residuals (linear/logistic regression, LDA): check the feature — or, better, the
model's **residuals** — for normality. Also handy as a train-vs-test drift check
alongside the KS test already in `01_EDA.ipynb`. `scipy.stats.probplot(x, plot=plt)`.

---

## 4. Violin and KDE plots

Both visualize a **distribution** (a smoothed alternative to the histogram),
typically split by class to compare churners vs non-churners.

**KDE (Kernel Density Estimate).** Places a small smooth "bump" (kernel, usually
Gaussian) on every data point and sums them into a continuous density curve — a
smoothed histogram. Knob: **bandwidth** (bump width); too small = spiky, too large
= washed out. `sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False)`.
Use `common_norm=False` so each class integrates to 1 and you compare *shapes*, not
class sizes (important with our ~23% churn imbalance).

**Violin plot.** A box plot with a **mirrored KDE** for its outline: the width at
each height is the density there. Shows median, IQR, *and* the full shape —
including **multimodality** (two humps) that a box plot completely hides.
`sns.violinplot(data=df, x='Churn', y='tenure')`; `split=True` puts two classes on
the two halves of one violin.

**Key features.** Both reveal shape (skew, bimodality, where the mass sits) far
better than summary statistics. KDE is best for **overlaying** several distributions
on shared axes; the violin is best for comparing a numeric feature **across discrete
categories** side by side and still seeing spread + outliers.

**When to use it.** When classes **overlap** and you want to see *how* a numeric
feature's distribution shifts between them (KDE overlay), or to compare a numeric
feature across the levels of a categorical one (violin). Caveat: KDE/violins can
invent density in empty regions or below natural bounds (e.g. negative `tenure`) —
sanity-check against the raw histogram.

---

## 5. SHAP and the Shapley value

SHAP (SHapley Additive exPlanations) attributes a model's prediction to its input
features using the **Shapley value** from cooperative game theory. To understand
why the SHAP summary plot in `01_EDA.ipynb` is trustworthy (and why it's exact for
LightGBM but only approximate for TabPFN), you need the underlying definition.

### 5.1 The Shapley value — game-theoretic definition

A **cooperative game** is a pair $(N, v)$:
- $N = \{1, 2, \ldots, n\}$ — a finite set of **players**.
- $v : 2^N \to \mathbb{R}$ — the **characteristic (value) function**, assigning each
  **coalition** $S \subseteq N$ a worth $v(S)$, with $v(\emptyset) = 0$. $v(N)$ is the total worth
  when everyone cooperates.

The question Shapley (1953) answered: *how should the total payout $v(N)$ be divided
among the players, fairly?* His answer, the **Shapley value** $\phi_i(v)$ of player $i$,
is the average of player $i$'s **marginal contribution** over every coalition it
could join:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n - |S| - 1)!}{n!} \cdot \bigl(v(S \cup \{i\}) - v(S)\bigr)$$

The term $v(S \cup \{i\}) - v(S)$ is $i$'s marginal contribution to coalition $S$; the
weight is the probability of seeing exactly that coalition form if players join in a
uniformly random order. Equivalently, averaging marginal contributions over all $n!$
orderings $\pi$:

$$\phi_i(v) = \frac{1}{n!} \sum_{\pi} \bigl[v(\text{Pre}_i(\pi) \cup \{i\}) - v(\text{Pre}_i(\pi))\bigr]$$

where $\text{Pre}_i(\pi)$ is the set of players that precede $i$ in ordering $\pi$.

**The four axioms** that uniquely pin down this formula (Shapley's theorem: the
Shapley value is the *only* allocation satisfying all four):

1. **Efficiency** — the parts sum to the whole: $\sum_i \phi_i(v) = v(N)$.
2. **Symmetry** — two players that contribute identically to every coalition get
   equal value: if $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$ excluding $i, j$, then
   $\phi_i = \phi_j$.
3. **Null (dummy) player** — a player that never adds anything gets zero: if
   $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity (linearity)** — for two games combined, value adds:
   $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$.

### 5.2 From Shapley value to SHAP (the ML mapping)

Lundberg & Lee (2017) cast **explaining one prediction** as a cooperative game:

| Game theory            | SHAP for a single instance $x$                                  |
|------------------------|-----------------------------------------------------------------|
| Players $N$            | The $n$ input **features**                                      |
| Coalition $S$          | A subset of features whose values are "known"                   |
| Payout $v(N)$          | The model output for $x$, relative to a baseline                |
| Value function $v(S)$  | $\mathbb{E}[f(X) \mid X_S = x_S] - \mathbb{E}[f(X)]$ — expected prediction when the features in $S$ are fixed to their values in $x$ and the rest are marginalized over a background distribution |

With this $v$: $v(\emptyset) = 0$ and $v(N) = f(x) - \mathbb{E}[f(X)]$. The **SHAP value** of feature
$i$ is its Shapley value $\phi_i$ in this game — its fair share of the gap between this
prediction and the average prediction.

The **efficiency axiom** is what makes SHAP exact and trustworthy: for any single
instance (also called "local accuracy" or additive feature attribution),

$$f(x) = \mathbb{E}[f(X)] + \sum_i \phi_i$$

i.e. the base value plus the SHAP values reconstruct the prediction *exactly*. This
is the property the force plot and beeswarm rely on.

### 5.3 GBDTs — TreeSHAP (exact and fast)

For tree ensembles (LightGBM, XGBoost, CatBoost, random forests), **TreeSHAP**
(Lundberg et al., 2020) computes the *exact* Shapley values in **polynomial time** —
roughly $O(T \cdot L \cdot D^2)$ for $T$ trees, $L$ leaves, depth $D$ — instead of the
exponential $2^n$. It exploits the tree structure: to evaluate $v(S)$ for a "missing"
feature, it pushes the instance down **both** branches at any split on that feature,
weighting each path by the fraction of training samples that went each way, and
aggregates the leaf values. This conditional-expectation trick is built into the
tree, so no sampling is needed.

This is exactly what `shap.TreeExplainer(lgbm)` does in `01_EDA.ipynb`: the SHAP values
are **exact**, computed in seconds, and the beeswarm is an honest additive
decomposition of the model's churn log-odds. On the **beeswarm summary plot**: each
dot is one customer's SHAP value for one feature; vertical order is mean $|\phi_i|$
(global importance); horizontal position is the signed impact on the prediction;
color is the feature value — so a band that goes blue-on-the-left to red-on-the-right
means "higher feature value → higher churn risk."

### 5.4 TabPFN — model-agnostic SHAP (approximate)

TabPFN is a **transformer (neural network)**, not a tree, so TreeSHAP does **not**
apply. You fall back to **model-agnostic** estimators that treat the model as a black
box and only call `predict_proba`:

- **KernelSHAP** (`shap.KernelExplainer`) — approximates the Shapley values by
  sampling coalitions, masking absent features with a background dataset, and solving
  a weighted linear regression whose coefficients are the SHAP values. Model-agnostic
  but **slow** (many model evaluations per explained row).
- **PermutationExplainer** (`shap.PermutationExplainer`) — directly averages marginal
  contributions over sampled feature **orderings** (the permutation form of the
  Shapley formula in §5.1), iterating forward and reverse for variance reduction.
- The **`tabpfn-extensions`** package ships an `interpretability` module that wraps
  SHAP for TabPFN so you don't wire this up by hand.

The **definition and the efficiency property are identical** to the GBDT case — only
the *algorithm for estimating* $v(S)$ changes (sampling instead of an exact tree
traversal). Two practical consequences: (1) it's far more expensive, and TabPFN
already caps the training set at a few thousand rows, so you compute SHAP on a small
sample; (2) because it's sampled, the values are approximate and will vary slightly
run to run unless you fix the seed and use enough samples.

---

### References
- L. Shapley, "A Value for n-Person Games," 1953.
- S. Lundberg & S.-I. Lee, "A Unified Approach to Interpreting Model Predictions,"
  NeurIPS 2017 (SHAP).
- S. Lundberg et al., "From local explanations to global understanding with
  explainable AI for trees," Nature Machine Intelligence 2020 (TreeSHAP).
