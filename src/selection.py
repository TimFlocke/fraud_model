import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, mutual_info_classif


def remove_low_variance(X, threshold=0.01):
    """
    Drops columns whose variance falls below threshold.

    Parameters:
    - X: feature DataFrame
    - threshold: minimum variance to retain a column (default 0.01)

    Returns:
    - DataFrame with low-variance columns removed
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    dropped = [c for c, keep in zip(X.columns, selector.get_support()) if not keep]
    print(f'low variance columns dropped: {len(dropped)}')
    if dropped:
        print(dropped)
    return X[X.columns[selector.get_support()]]


def remove_correlated(X, threshold=0.95):
    """
    Drops columns that are highly correlated with an earlier column.

    Iterates through the upper triangle of the absolute correlation matrix.
    For each pair above threshold, the second column is marked for removal.

    Parameters:
    - X: feature DataFrame
    - threshold: absolute correlation above which a column is dropped (default 0.95)

    Returns:
    - DataFrame with correlated columns removed
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f'correlated columns dropped: {len(to_drop)}')
    if to_drop:
        print(to_drop)
    return X.drop(columns=to_drop)


def get_mutual_info_scores(X, y):
    """
    Computes mutual information between each feature and the target.

    Parameters:
    - X: feature DataFrame
    - y: binary target Series

    Returns:
    - Series of mutual information scores sorted descending, indexed by feature name
    """
    # median impute -- mutual_info_classif does not handle nulls
    X_filled = X.fillna(X.median())
    scores = mutual_info_classif(X_filled, y, random_state=42)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def get_chi2_scores(X_agg, y):
    """
    Computes chi-squared scores between non-negative features and the target.

    Only columns where the minimum value is >= 0 are included, since chi2
    requires non-negative inputs.

    Parameters:
    - X_agg: feature DataFrame
    - y: binary target Series

    Returns:
    - Series of chi2 scores sorted descending, indexed by feature name
    """
    non_neg_cols = [c for c in X_agg.columns if X_agg[c].min() >= 0]
    X_nn = X_agg[non_neg_cols]
    # fill nulls with 0 -- chi2 requires complete non-negative data
    X_filled = X_nn.fillna(0)
    selector = SelectKBest(chi2, k='all')
    selector.fit(X_filled, y)
    return pd.Series(selector.scores_, index=X_nn.columns).sort_values(ascending=False)


def get_catboost_importance(model, X):
    """
    Extracts feature importances from a fitted CatBoostClassifier.

    Parameters:
    - model: fitted CatBoostClassifier
    - X: feature DataFrame used during training (provides column names)

    Returns:
    - Series of importances sorted descending, indexed by feature name
    """
    importances = model.get_feature_importance()
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)


def get_shap_importance(model, X, sample_n=5000):
    """
    Computes mean absolute SHAP values using TreeExplainer.

    Samples sample_n rows for speed. Works with tree-based models
    (CatBoost, XGBoost, LightGBM).

    Parameters:
    - model: fitted tree-based model
    - X: feature DataFrame
    - sample_n: number of rows to sample (default 5000)

    Returns:
    - Series of mean absolute SHAP values sorted descending, indexed by feature name
    """
    sample = X.sample(n=min(sample_n, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    # catboost returns list for binary classification; take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False)


def combine_rankings(rank_dicts, top_n=100):
    """
    Combines multiple feature importance rankings into a single consensus table.

    Ranks features within each method (rank 1 = most important), then averages
    across methods. Features missing from a method (e.g. chi2 skips negative
    columns) receive the worst rank for that method.

    Parameters:
    - rank_dicts: dict of {method_name: importance_series}
    - top_n: number of top features to retain per method before ranking (default 100)

    Returns:
    - DataFrame with columns [feature, <method>_rank ..., mean_rank]
      sorted by mean_rank ascending
    """
    all_features = set()
    for s in rank_dicts.values():
        all_features.update(s.index.tolist())

    rows = {}
    for method, scores in rank_dicts.items():
        ranked = scores.rank(ascending=False, method='min')
        worst = len(all_features) + 1
        rows[method] = {f: ranked.get(f, worst) for f in all_features}

    df = pd.DataFrame(rows)
    df.index.name = 'feature'
    df = df.reset_index()
    rank_cols = list(rank_dicts.keys())
    df.columns = ['feature'] + [f'{m}_rank' for m in rank_cols]
    df['mean_rank'] = df[[f'{m}_rank' for m in rank_cols]].mean(axis=1)
    return df.sort_values('mean_rank').reset_index(drop=True)
