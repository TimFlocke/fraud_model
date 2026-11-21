from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / 'data'
TRANSACTION_PATH = ROOT / 'data' / 'train_transaction.csv'
IDENTITY_PATH = ROOT / 'data' / 'train_identity.csv'
OUTPUT_PATH = ROOT / 'outputs'
PLOTS_PATH = ROOT / 'outputs' / 'plots'
RESULTS_PATH = ROOT / 'outputs' / 'results'
RANDOM_STATE = 42
TARGET = 'isFraud'
NULL_THRESHOLD = 0.8
RULES_TOP_N_FEATURES = 10
STRATEGY_APPROVE_THRESHOLD = 500
STRATEGY_DECLINE_THRESHOLD = 900

# tabular foundation model context optimization (stage 14)
TFM = {
    'n_ctx': 10000,
    'fraud_rate': 0.2,
    'seeds': [0, 1, 2],
    'alert_rates': [0.01, 0.20],
    'tree_depth': 3,
    'min_samples_leaf': 5000,
    'min_leaf_fraud': 50,
    'kv_cache': 'repr',
    'force_rerun': False,
}

import subprocess


def get_device():
    """detects nvidia gpu availability, falls back to cpu"""
    try:
        subprocess.check_output('nvidia-smi', stderr=subprocess.DEVNULL)
        return 'gpu'
    except:
        return 'cpu'
