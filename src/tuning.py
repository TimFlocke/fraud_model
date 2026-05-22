import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from config import get_device


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=20):
    """
    Tunes CatBoostClassifier hyperparameters using Optuna TPE sampler.

    Maximises validation AUC. Each trial uses early stopping (50 rounds)
    and silent output.

    Search space:
    - iterations: 500-1500 (step 100)
    - learning_rate: 0.01-0.1 (log scale)
    - depth: 4-8
    - l2_leaf_reg: 1.0-10.0
    - bagging_temperature: 0.0-1.0

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - n_trials: number of Optuna trials (default 20)

    Returns:
    - (best_params dict, optuna Study object)
    """
    from catboost import CatBoostClassifier

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            'verbose': 0,
            'task_type': 'GPU' if get_device() == 'gpu' else 'CPU',
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=20):
    """
    Tunes XGBClassifier hyperparameters using Optuna TPE sampler.

    Maximises validation AUC. Each trial uses early stopping (50 rounds).
    scale_pos_weight is set to the negative/positive class ratio to handle
    class imbalance.

    Search space:
    - n_estimators: 500-1500 (step 100)
    - learning_rate: 0.01-0.1 (log scale)
    - max_depth: 4-8
    - subsample: 0.6-1.0
    - colsample_bytree: 0.6-1.0
    - min_child_weight: 1-10

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - n_trials: number of Optuna trials (default 20)

    Returns:
    - (best_params dict, optuna Study object)
    """
    from xgboost import XGBClassifier

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'scale_pos_weight': scale_pos_weight,
            'verbosity': 0,
            'device': 'cuda' if get_device() == 'gpu' else 'cpu',
            'tree_method': 'hist',
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=20):
    """
    Tunes LightGBM hyperparameters using Optuna TPE sampler and the native API.

    Maximises validation AUC. Each trial uses early stopping (50 rounds).
    scale_pos_weight is set to the negative/positive class ratio to handle
    class imbalance. Uses lgb.train (native booster API) inside the objective
    so that num_boost_round is treated as an integer parameter rather than
    a model attribute.

    Search space:
    - num_boost_round: 500-1500 (step 100)
    - learning_rate: 0.01-0.1 (log scale)
    - max_depth: 4-8
    - num_leaves: 20-60
    - min_child_samples: 20-100
    - subsample: 0.6-1.0
    - colsample_bytree: 0.6-1.0

    Parameters:
    - X_train: training feature DataFrame
    - y_train: training binary target Series
    - X_val: validation feature DataFrame
    - y_val: validation binary target Series
    - n_trials: number of Optuna trials (default 20)

    Returns:
    - (best_params dict, optuna Study object)
    """
    import lightgbm as lgb

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    def objective(trial):
        num_boost_round = trial.suggest_int('num_boost_round', 500, 1500, step=100)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'feature_pre_filter': False,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'verbosity': -1,
            'device': 'gpu' if get_device() == 'gpu' else 'cpu',
        }
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        proba = booster.predict(X_val)
        return roc_auc_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study
