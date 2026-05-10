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
    Segments model scores into deciles and reports fraud capture statistics per decile.

    Parameters:
    - df: DataFrame containing uid, score, and target columns.
    - uid: Column name for the unique event identifier (used for count).
    - score: Column name of the predicted probability score.
    - target: Column name of the binary fraud label.

    Returns:
    - DataFrame with decile-level counts, score threshold, fraud counts,
      cumulative fraud count, and cumulative fraud capture rate.
    """

    d = df.copy()
    d = d.sort_values(score, ascending=False).reset_index(drop=True)
    d['decile'] = pd.qcut(d[score], 10, labels=False,
                        #   duplicates='drop'
                          )

    rpt = d.groupby('decile').agg({uid: 'count', score: 'min', target: 'sum'})
    rpt['fraud_cm'] = rpt[target].cumsum()
    rpt['fraud_pct_cap'] = rpt['fraud_cm'] / d[target].sum()
    rpt.columns = ['n_events', f'{score}_thresh', 'n_frauds', 'fraud_cm', 'fraud_pct_cap']

    print("% Frauds captured in upper decile:", rpt.loc[9]['n_frauds']/d[target].sum())

    return rpt
