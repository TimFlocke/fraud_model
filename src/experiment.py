import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def log_run(model_name, params, metrics, results_path):
    """
    Appends a single experiment run to experiment_log.json.

    Each run stores a timestamp, model name, hyperparameter dict, and metric dict.
    Creates results_path if it does not exist. Prints confirmation with model name and AUC.

    Parameters:
    - model_name: string identifier for the model
    - params: dict of hyperparameters
    - metrics: dict of evaluation metrics (must include val_auc)
    - results_path: Path or str; directory where experiment_log.json is saved
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    log_path = results_path / 'experiment_log.json'

    run = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'params': params,
        'metrics': metrics,
    }

    if log_path.exists():
        with open(log_path, 'r') as f:
            runs = json.load(f)
    else:
        runs = []

    runs.append(run)

    with open(log_path, 'w') as f:
        json.dump(runs, f, indent=2)

    print(f'logged: {model_name}  val_auc={metrics["val_auc"]:.5f}')


def load_experiment_log(results_path):
    """
    Loads experiment_log.json as a flattened DataFrame sorted by val_auc descending.

    Params and metrics nested dicts are flattened to columns named params.* and metrics.*.

    Parameters:
    - results_path: Path or str; directory containing experiment_log.json

    Returns:
    - DataFrame of all logged runs sorted by metrics.val_auc descending
    """
    log_path = Path(results_path) / 'experiment_log.json'
    with open(log_path, 'r') as f:
        runs = json.load(f)

    df = pd.json_normalize(runs)
    df = df.sort_values('metrics.val_auc', ascending=False).reset_index(drop=True)
    return df
