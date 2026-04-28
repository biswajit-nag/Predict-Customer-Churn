# Predicting Customer Churn

**Status: In progress**
- [x] EDA
- [x] Baselines (LGBM, XGB, CatBoost, Logistic Regression)
- [ ] Advanced feature engineering
- [ ] Model ensembling / stacking

Here, we use Kaggle's [synthetic Telecom customer data](https://www.kaggle.com/competitions/playground-series-s6e3/data?select=train.csv) (~600k rows, 21 columns) to identify customers who terminate their services. Customer data includes details about demographics, service subscriptions, account info, and charges. The target is imbalanced (~23% positive), so our primary evaluation metric is ROC-AUC. Our current best model achieves an ROC-AUC of 0.92. For more details, see `EDA.ipynb` for exploratory analysis and `Experiments.ipynb` for model training.

## Highlights

- Train-test distribution drift detection
- Git-based experiment logging with data versioning, artifact storage, and Python environment tracking
- Stateless/stateful feature engineering split along with testing to counter data leakage during training