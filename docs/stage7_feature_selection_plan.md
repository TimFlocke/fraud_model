# stage 7 — feature selection plan

## approach

Run in order. Each step narrows the field. Every cut has a documented reason.

---

## step 1 — variance threshold

Remove features with near-zero variance. These carry no signal.

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selector.fit(X)
low_variance_cols = X.columns[~selector.get_support()]
```

---

## step 2 — correlation matrix

Remove redundant features. Where two features are highly correlated
keep the one with higher importance (from step 5).

```python
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
```

---

## step 3 — mutual information

Rank features by non-linear statistical dependency with target.
Works on continuous and categorical features.

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
```

---

## step 4 — selectkbest + chi-squared

Statistical significance of each feature against target.
Best suited for non-negative integer features — aggregation
features (nunique, count) are ideal candidates here.

```python
from sklearn.feature_selection import SelectKBest, chi2

# requires non-negative values — apply to aggregation features only
selector = SelectKBest(chi2, k=50)
selector.fit(X_agg, y)
chi2_scores = pd.Series(selector.scores_, index=X_agg.columns).sort_values(ascending=False)
```

---

## step 5 — selectkbest + f_classif

ANOVA F-test. Works on continuous features where chi-squared
does not apply.

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=50)
selector.fit(X_continuous, y)
f_scores = pd.Series(selector.scores_, index=X_continuous.columns).sort_values(ascending=False)
```

---

## step 6 — catboost first pass importance

Train a lightweight CatBoost model on the full feature set.
Extract gain-based feature importances. Drop bottom N% by importance.

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=300, random_state=42, verbose=0)
model.fit(X_train, y_train, cat_features=cat_cols)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False)

# drop bottom 10% by importance
threshold = importance_df.importance.quantile(0.10)
drop_cols = importance_df[importance_df.importance < threshold].feature.tolist()
```

---

## step 7 — shap values

Most reliable importance measure. Captures true marginal
contribution of each feature accounting for interactions.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap_df = pd.DataFrame({
    'feature': X_train.columns,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)
```

---

## step 8 — featurewiz (sanity check)

Run featurewiz independently and compare its selected feature
set against the manual stack output. Agreement = validation.
Disagreement = investigate.

```python
from featurewiz import featurewiz

features, train = featurewiz(
    dataname=df_train,
    target='isFraud',
    corr_limit=0.70,
    verbose=2
)
```

---

## final selection logic

Take features that survive steps 1-2 (mandatory cuts),
then use steps 3-7 to rank. Keep features that appear
in the top tier of at least two ranking methods.

Present final selected features as a table:

| feature | mi_rank | chi2_rank | catboost_rank | shap_rank | selected |
|---|---|---|---|---|---|

---

## outputs

- outputs/plots/variance_threshold.png
- outputs/plots/correlation_heatmap.png
- outputs/plots/mutual_information.png
- outputs/plots/chi2_scores.png
- outputs/plots/catboost_importance.png
- outputs/plots/shap_summary.png
- outputs/results/selected_features.txt  ← final feature list
