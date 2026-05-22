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

import subprocess


def get_device():
    """detects nvidia gpu availability, falls back to cpu"""
    try:
        subprocess.check_output('nvidia-smi', stderr=subprocess.DEVNULL)
        return 'gpu'
    except:
        return 'cpu'
