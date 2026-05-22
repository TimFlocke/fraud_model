import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def get_gains(df, target, score, top_n, save_path=None):
    """
    Computes and plots a gains chart showing the percentage of fraud captured
    at each score threshold.

    Parameters:
    - df: DataFrame containing target and score columns.
    - target: Column name of the binary fraud label.
    - score: Column name of the predicted probability score.
    - top_n: Fraction (0–1) of top-scored events to mark with a threshold line.
    - save_path: Optional file path to save the gains chart figure.

    Returns:
    - Single-row DataFrame with threshold, pct_flagged, n_fraud_cap, and fraud_pct_cap
      at the top_n cutoff.
    """

    d = df.copy()
    d = df.sort_values(by=score, ascending=False).reset_index(drop=True)
    d['frauds_cm'] = d[target].cumsum()

    frauds = d[target].astype(int).sum()

    d['pct_flagged'] = (d.index + 1) / len(df) * 100
    d['fraud_pct_cap'] = (d.frauds_cm / frauds) * 100

    top_n = top_n * 100

    out = d[d['pct_flagged'] >= top_n][[score, 'pct_flagged', 'frauds_cm', 'fraud_pct_cap']].rename({score: 'threshold', 'frauds_cm': 'n_fraud_cap'}, axis=1).iloc[0].to_frame(score).T.reset_index()
    thresh = out.threshold.values[0] * 100
    fraud_pct_cap = np.round(out.fraud_pct_cap.values[0], 2)

    # Plot #
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(d['pct_flagged'], d['fraud_pct_cap'], label=score)
    ax.plot([0, 100], [0, 100], linestyle='--', color='grey', label='Baseline')
    ax.axvline(x=top_n, linestyle='--', color='red')  # < - plot threshold
    ax.annotate(
        f"Top {top_n}% score threshold: {np.round(thresh, 2)} captures {fraud_pct_cap}% fraud",
        xy=(top_n, 120), color='red', ha='left', va='bottom',
        xycoords=('data', 'axes points'),
    )

    ax.set_title(f'{score} Gains Chart')
    ax.set_xlabel('% Events Flagged')
    ax.set_ylabel('% Frauds Captured')
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        out_path = Path('outputs/plots/gains_chart.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.show()
    plt.close()

    return out


def score_deciles(df, uid, score, target):
    """
    segments model scores into deciles and reports fraud capture
    statistics per decile. decile 1 = highest score (highest risk).

    parameters:
    - df: dataframe containing uid, score, and target columns
    - uid: column name for unique event identifier (used for count)
    - score: column name of predicted probability score
    - target: column name of binary fraud label

    returns:
    - dataframe with decile-level counts, score threshold,
      fraud counts, cumulative fraud count, and capture rate
    """
    d = df.copy()
    d = d.sort_values(score, ascending=False).reset_index(drop=True)

    # assign deciles by row position — avoids qcut duplicate bin issue
    d['decile'] = pd.qcut(d.index, 10, labels=False) + 1

    rpt = d.groupby('decile').agg(
        n_events=(uid, 'count'),
        score_thresh=(score, 'min'),
        n_frauds=(target, 'sum')
    )
    rpt.columns = ['n_events', f'{score}_thresh', 'n_frauds']
    rpt['fraud_cm'] = rpt['n_frauds'].cumsum()
    rpt['fraud_pct_cap'] = rpt['fraud_cm'] / d[target].sum()

    print(f'% frauds captured in top decile: {rpt.iloc[0]["n_frauds"] / d[target].sum():.4f}')

    return rpt


def fit_uniform_score(train_proba):
    """
    Fits a rank-based uniform score transformer on the training distribution.

    Sorts the training probabilities ascending and returns the sorted array.
    This array acts as the reference distribution for apply_uniform_score.
    Save the returned array for production reuse.

    Parameters:
    - train_proba: array-like of predicted probabilities from the training set

    Returns:
    - train_sorted: 1-D numpy array of sorted training probabilities (ascending)
    """
    return np.sort(np.array(train_proba))


def apply_uniform_score(proba, train_sorted, scale=1000):
    """
    Maps raw probabilities to integer scores on a 0-scale range using
    rank interpolation relative to the training distribution.

    Uses np.searchsorted to find where each probability falls in the sorted
    training reference, then converts to an integer score. A score of 800
    means the observation ranked higher than 80% of training transactions.

    Parameters:
    - proba: array-like of predicted probabilities to transform
    - train_sorted: sorted training reference array from fit_uniform_score
    - scale: upper bound of output score range (default 1000)

    Returns:
    - integer numpy array of scores in [0, scale]
    """
    ranks = np.searchsorted(train_sorted, np.array(proba), side='left')
    return (ranks / len(train_sorted) * scale).astype(int)


def compute_ks(y_true, y_proba):
    """
    Computes the KS (Kolmogorov-Smirnov) statistic between fraud and
    non-fraud score distributions.

    Sorts observations by predicted probability descending, then computes
    the cumulative fraud rate and cumulative non-fraud rate at each point.
    KS is the maximum absolute separation between the two curves.

    Parameters:
    - y_true: array-like of binary labels (1 = fraud, 0 = non-fraud)
    - y_proba: array-like of predicted probabilities

    Returns:
    - ks_stat: float, KS statistic (0-100 scale)
    - ks_threshold: float, predicted probability at the point of max separation
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    n_fraud = y_sorted.sum()
    n_nonfraud = len(y_sorted) - n_fraud

    cum_fraud = np.cumsum(y_sorted) / n_fraud
    cum_nonfraud = np.cumsum(1 - y_sorted) / n_nonfraud

    diffs = np.abs(cum_fraud - cum_nonfraud)
    ks_idx = np.argmax(diffs)
    ks_stat = diffs[ks_idx] * 100
    ks_threshold = p_sorted[ks_idx]

    return float(ks_stat), float(ks_threshold)


def plot_ks_curve(y_true, y_proba, model_name, output_path):
    """
    Plots the cumulative fraud and non-fraud distributions against
    the sorted score axis, marking the KS statistic.

    The KS point is shown as a vertical dashed line with the KS value
    annotated on the chart.

    Parameters:
    - y_true: array-like of binary labels
    - y_proba: array-like of predicted probabilities
    - model_name: string label used in the chart title
    - output_path: file path (str or Path) to save the figure
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    n_fraud = y_sorted.sum()
    n_nonfraud = len(y_sorted) - n_fraud

    cum_fraud = np.cumsum(y_sorted) / n_fraud
    cum_nonfraud = np.cumsum(1 - y_sorted) / n_nonfraud

    diffs = np.abs(cum_fraud - cum_nonfraud)
    ks_idx = np.argmax(diffs)
    ks_stat = diffs[ks_idx] * 100
    ks_threshold = p_sorted[ks_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_sorted[::-1], cum_fraud[::-1],    color='red',      label='cumulative fraud rate')
    ax.plot(p_sorted[::-1], cum_nonfraud[::-1], color='steelblue', label='cumulative non-fraud rate')
    ax.axvline(x=ks_threshold, linestyle='--', color='grey')
    ax.annotate(
        f'KS = {ks_stat:.1f}%',
        xy=(ks_threshold, (cum_fraud[ks_idx] + cum_nonfraud[ks_idx]) / 2),
        xytext=(10, 0), textcoords='offset points',
        fontsize=11, color='grey',
    )
    ax.set_xlabel('predicted probability (ascending)')
    ax.set_ylabel('cumulative rate')
    ax.set_title(f'KS Curve -- {model_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def compute_psi(expected, actual, bins=10):
    """
    Computes the Population Stability Index (PSI) between two score distributions.

    Bins both distributions into equal-width bins between 0 and 1. PSI measures
    how much the actual distribution has shifted relative to the expected
    (reference) distribution. Zero bins are protected with a small epsilon to
    avoid log(0).

    PSI interpretation:
    - PSI < 0.1  : stable
    - PSI 0.1-0.2: monitor
    - PSI > 0.2  : investigate / retrain

    Parameters:
    - expected: array-like, reference score distribution (e.g. train or val)
    - actual: array-like, current score distribution (e.g. val or test)
    - bins: number of equal-width bins between 0 and 1 (default 10)

    Returns:
    - psi_value: float
    """
    eps = 1e-10
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts,   _ = np.histogram(actual,   bins=breakpoints)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct   = actual_counts   / len(actual)   + eps

    psi_value = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi_value


def plot_psi_distributions(expected, actual, expected_label, actual_label,
                           model_name, output_path):
    """
    Plots overlapping score histograms for the expected and actual distributions,
    annotating the PSI value with a traffic-light status indicator.

    Traffic light thresholds:
    - PSI < 0.1  : stable (green)
    - PSI 0.1-0.2: monitor (amber)
    - PSI > 0.2  : investigate (red)

    Parameters:
    - expected: array-like, reference score distribution
    - actual: array-like, current score distribution
    - expected_label: string label for the expected distribution
    - actual_label: string label for the actual distribution
    - model_name: string label used in the chart title
    - output_path: file path (str or Path) to save the figure
    """
    psi = compute_psi(expected, actual)

    if psi < 0.1:
        status, color = 'stable', 'green'
    elif psi < 0.2:
        status, color = 'monitor', 'orange'
    else:
        status, color = 'investigate', 'red'

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(expected, bins=40, alpha=0.5, label=expected_label, color='steelblue', density=True)
    ax.hist(actual,   bins=40, alpha=0.5, label=actual_label,   color='darkorange', density=True)
    ax.set_xlabel('predicted probability')
    ax.set_ylabel('density')
    ax.set_title(f'PSI Distribution -- {model_name}  |  PSI={psi:.4f}  [{status}]',
                 color=color)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
