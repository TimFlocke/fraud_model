import config
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

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=config.RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=50,
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
        'verbosity': -1
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

# def train_lightgbm(X_train, y_train, X_val, y_val):
#     """
#     trains a lightgbm classifier with early stopping.
#     scale_pos_weight handles class imbalance explicitly.
#     """
#     from lightgbm import LGBMClassifier, early_stopping, log_evaluation

#     scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

#     model = LGBMClassifier(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=6,
#         num_leaves=31,
#         min_child_samples=20,
#         scale_pos_weight=scale_pos_weight,
#         random_state=config.RANDOM_STATE,
#         verbosity=-1
#     )

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric = 'auc',
#         callbacks=[early_stopping(50, verbose=False),
#                    log_evaluation(100)]
#     )

#     return model
