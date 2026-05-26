import config
from config import get_device
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.experiment import log_run


def train_baseline(X_train, y_train, X_val, y_val):
    """
    Trains a decision tree classifier with balanced class weights as a baseline model.
    Computes validation metrics and logs the run to the experiment tracker.

    Parameters:
    - X_train: training feature matrix
    - y_train: training binary target series
    - X_val: validation feature matrix
    - y_val: validation binary target series

    Returns:
    - fitted DecisionTreeClassifier
    """
    model = DecisionTreeClassifier(
        max_depth=10,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba > 0.5).astype(int)
    auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average='binary')

    log_run(
        model_name='baseline_decision_tree',
        params={
            'model_type': 'decision_tree',
            'max_depth': 10,
            'class_weight': 'balanced',
        },
        metrics={
            'val_auc': round(float(auc), 5),
            'val_gini': round(float(2 * auc - 1), 5),
            'val_pr_auc': round(float(pr_auc), 5),
            'val_precision': round(float(precision), 5),
            'val_recall': round(float(recall), 5),
            'val_f1': round(float(f1), 5),
        },
        results_path=config.RESULTS_PATH,
    )

    return model


def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    """
    Trains a CatBoostClassifier with early stopping evaluated on AUC.

    Parameters:
    - X_train: training feature matrix
    - y_train: training binary target series
    - X_val: validation feature matrix
    - y_val: validation binary target series
    - cat_features: list of column names to treat as categorical

    Returns:
    - fitted CatBoostClassifier at best iteration
    """
    from catboost import CatBoostClassifier

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=config.RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=50,
        task_type='GPU' if get_device() == 'gpu' else 'CPU',
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Trains an XGBClassifier with early stopping and class imbalance correction via
    scale_pos_weight set to the negative/positive class ratio.

    Parameters:
    - X_train: training feature matrix
    - y_train: training binary target series
    - X_val: validation feature matrix
    - y_val: validation binary target series

    Returns:
    - fitted XGBClassifier at best iteration
    """
    from xgboost import XGBClassifier

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='auc',
        random_state=config.RANDOM_STATE,
        early_stopping_rounds=50,
        scale_pos_weight=scale_pos_weight,
        device='cuda' if get_device() == 'gpu' else 'cpu',
        tree_method='hist',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    lightgbm native
    """
    import lightgbm as lgb

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 50,
        'scale_pos_weight': scale_pos_weight,
        'random_state': config.RANDOM_STATE,
        'verbosity': -1,
        'device': 'gpu' if get_device() == 'gpu' else 'cpu',
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks
    )

    return booster

def train_catboost_final(X_train, y_train, X_val, y_val, cat_features, params):
    """
    Trains a CatBoostClassifier using tuned hyperparameters from Optuna.

    Merges caller-supplied params with fixed operational settings
    (eval_metric, random_seed, verbose, early_stopping_rounds) so that
    the training run is reproducible and self-evaluating on the val set.

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - cat_features: list of column names to treat as categorical
    - params: dict of tuned hyperparameters (from best_params_catboost.json)

    Returns:
    - fitted CatBoostClassifier at best iteration
    """
    from catboost import CatBoostClassifier

    fixed = {
        'eval_metric': 'AUC',
        'random_seed': config.RANDOM_STATE,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'task_type': 'GPU' if get_device() == 'gpu' else 'CPU',
        'scale_pos_weight': float((y_train == 0).sum()) / float((y_train == 1).sum()),
    }
    model = CatBoostClassifier(**{**params, **fixed})
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    return model


def train_xgboost_final(X_train, y_train, X_val, y_val, params):
    """
    Trains an XGBClassifier using tuned hyperparameters from Optuna.

    Merges caller-supplied params with fixed operational settings
    (eval_metric, random_state, early_stopping_rounds, scale_pos_weight).
    scale_pos_weight is recomputed from y_train to handle class imbalance.

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - params: dict of tuned hyperparameters (from best_params_xgboost.json)

    Returns:
    - fitted XGBClassifier at best iteration
    """
    from xgboost import XGBClassifier

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    fixed = {
        'eval_metric': 'auc',
        'random_state': config.RANDOM_STATE,
        'early_stopping_rounds': 50,
        'scale_pos_weight': scale_pos_weight,
        'device': 'cuda' if get_device() == 'gpu' else 'cpu',
        'tree_method': 'hist',
    }
    model = XGBClassifier(**{**params, **fixed})
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm_final(X_train, y_train, X_val, y_val, params):
    """
    Trains a LightGBM booster using tuned hyperparameters from Optuna.

    Uses the native lgb.train API. Merges caller-supplied params with fixed
    operational settings (objective, metric, scale_pos_weight,
    feature_pre_filter, verbosity). num_boost_round is popped from params
    before passing to lgb.train since it is a separate argument.

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - params: dict of tuned hyperparameters (from best_params_lightgbm.json)

    Returns:
    - native LightGBM Booster at best iteration
    """
    import lightgbm as lgb

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    fixed = {
        'objective': 'binary',
        'metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'feature_pre_filter': False,
        'verbosity': -1,
        'device': 'gpu' if get_device() == 'gpu' else 'cpu',
    }
    merged = {**params, **fixed}
    num_boost_round = int(merged.pop('num_boost_round', 1000))

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    booster = lgb.train(
        merged,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return booster
