import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DECISION_COLORS = {'DECLINE': 'red', 'STEP-UP': 'darkorange', 'APPROVE': 'green'}


def assign_score_based_decisions(df, score_col,
                                 approve_threshold=None,
                                 decline_threshold=None):
    """
    Assigns DECLINE / STEP-UP / APPROVE decisions using fixed score thresholds.

    score <= approve_threshold  -> APPROVE  (low risk, no friction)
    score >  decline_threshold  -> DECLINE  (high risk, block)
    everything else             -> STEP-UP  (elevated risk, challenge)

    Parameters:
    - df: DataFrame containing the score column
    - score_col: name of the integer score column (0-1000 range)
    - approve_threshold: upper bound for APPROVE (default config.STRATEGY_APPROVE_THRESHOLD)
    - decline_threshold: lower bound for DECLINE (default config.STRATEGY_DECLINE_THRESHOLD)

    Returns:
    - df with decision column added
    """
    if approve_threshold is None:
        approve_threshold = config.STRATEGY_APPROVE_THRESHOLD
    if decline_threshold is None:
        decline_threshold = config.STRATEGY_DECLINE_THRESHOLD

    df = df.copy()

    def _decide(score):
        if score <= approve_threshold:
            return 'APPROVE'
        elif score > decline_threshold:
            return 'DECLINE'
        return 'STEP-UP'

    df['decision'] = df[score_col].apply(_decide)
    return df


def get_shap_top_feature(shap_values, feature_names, lookup=None):
    """
    Returns the feature with the highest absolute SHAP value per transaction.

    Parameters:
    - shap_values: 2-D numpy array (n_samples, n_features)
    - feature_names: list of feature name strings aligned with shap_values columns
    - lookup: optional dict {feature_name: plain_english_description}

    Returns:
    - pandas Series of length n_samples; values are feature names or
      plain english descriptions if lookup is provided
    """
    top_idx = np.abs(shap_values).argmax(axis=1)
    top_features = [feature_names[i] for i in top_idx]

    if lookup is not None:
        top_features = [lookup.get(f, f) for f in top_features]

    return pd.Series(top_features)


def build_feature_lookup(feature_names):
    """
    Generates plain english descriptions from feature naming conventions.

    Rules applied in order:
    - known fields mapped directly (TransactionAmt, ensemble_score, has_identity)
    - aggregation features containing '__': entity__target__agg ->
      '{agg_desc} {target} for {entity}'
    - log-transformed features ending with '_log': strip suffix, describe base
    - encoded categoricals ending with '_encoded': strip suffix, add 'category'
    - fallback: replace underscores with spaces

    Parameters:
    - feature_names: list of feature name strings

    Returns:
    - dict of {feature_name: description}
    """
    known = {
        'TransactionAmt': 'transaction amount',
        'ensemble_score':  'overall fraud risk score',
        'has_identity':    'device identity present',
    }
    agg_map = {
        'nunique': 'number of unique',
        'mean':    'average',
        'std':     'variability of',
        'count':   'total count of',
    }

    lookup = {}
    for name in feature_names:
        if name in known:
            lookup[name] = known[name]
        elif '__' in name:
            parts = name.split('__')
            if len(parts) >= 3:
                entity, target, agg = parts[0], parts[1], parts[2]
                agg_desc = agg_map.get(agg, agg)
                lookup[name] = f'{agg_desc} {target} for {entity}'
            else:
                lookup[name] = name.replace('_', ' ')
        elif name.endswith('_log'):
            lookup[name] = name[:-4].replace('_', ' ')
        elif name.endswith('_encoded'):
            lookup[name] = f'{name[:-8]} category'
        else:
            lookup[name] = name.replace('_', ' ')

    return lookup


def plot_shap_by_decision(shap_values, X_sample, decisions, output_path):
    """
    Three-panel horizontal bar chart of mean absolute SHAP values per decision tier.

    Shows the top 10 features driving each of DECLINE, STEP-UP, and APPROVE.
    Bars are coloured by decision tier. Useful for understanding which features
    most strongly influence each tier independently.

    Parameters:
    - shap_values: 2-D numpy array (n_samples, n_features) aligned with X_sample
    - X_sample: feature DataFrame whose columns provide feature names
    - decisions: array-like of decision labels aligned with shap_values rows
    - output_path: file path to save the figure
    """
    feature_names = X_sample.columns.tolist()
    decisions = np.array(decisions)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, tier in zip(axes, ['DECLINE', 'STEP-UP', 'APPROVE']):
        mask = decisions == tier
        color = _DECISION_COLORS.get(tier, 'grey')

        if mask.sum() == 0:
            ax.set_title(f'{tier} (no data in sample)')
            ax.set_xlabel('mean |SHAP|')
            continue

        mean_abs = pd.Series(
            np.abs(shap_values[mask]).mean(axis=0),
            index=feature_names,
        ).sort_values(ascending=True).tail(10)

        ax.barh(mean_abs.index, mean_abs.values, color=color)
        ax.set_title(f'{tier} -- top features by mean |SHAP|')
        ax.set_xlabel('mean |SHAP|')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def evaluate_decisions(df, target, decision_col):
    """
    Computes per-decision-tier fraud capture statistics.

    Parameters:
    - df: DataFrame with target and decision columns
    - target: name of binary target column
    - decision_col: name of decision column (DECLINE / STEP-UP / APPROVE)

    Returns:
    - DataFrame with columns: decision, n_transactions, n_fraud, precision,
      pct_of_total_transactions, pct_of_total_fraud; sorted by precision descending
    """
    total_n = len(df)
    total_fraud = int(df[target].sum())

    rows = []
    for decision, group in df.groupby(decision_col):
        n_transactions = len(group)
        n_fraud = int(group[target].sum())
        precision = n_fraud / n_transactions if n_transactions > 0 else 0.0
        rows.append({
            'decision': decision,
            'n_transactions': n_transactions,
            'n_fraud': n_fraud,
            'precision': round(float(precision), 4),
            'pct_of_total_transactions': round(float(n_transactions / total_n), 4),
            'pct_of_total_fraud': round(float(n_fraud / total_fraud), 4),
        })

    return (pd.DataFrame(rows)
              .sort_values('precision', ascending=False)
              .reset_index(drop=True))


def plot_decision_summary(eval_df, output_path):
    """
    Three-panel chart of precision, transaction volume, and fraud capture
    percentage by decision tier.

    Parameters:
    - eval_df: DataFrame from evaluate_decisions
    - output_path: file path to save the figure
    """
    df = eval_df.sort_values('precision', ascending=False)
    colors = [_DECISION_COLORS.get(d, 'grey') for d in df['decision']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(df['decision'], df['precision'], color=colors)
    axes[0].set_title('precision by decision tier')
    axes[0].set_ylabel('precision')

    axes[1].bar(df['decision'], df['n_transactions'], color=colors)
    axes[1].set_title('transaction volume by decision tier')
    axes[1].set_ylabel('n transactions')

    axes[2].bar(df['decision'], df['pct_of_total_fraud'], color=colors)
    axes[2].set_title('pct fraud captured by decision tier')
    axes[2].set_ylabel('pct of total fraud')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_score_distribution_by_decision(df, score_col, decision_col,
                                        target, output_path):
    """
    Overlapping density histograms of the ensemble score split by decision
    tier, with a step-line overlay for the actual fraud distribution.

    Parameters:
    - df: DataFrame with score, decision, and target columns
    - score_col: name of the score column (e.g. 'ensemble_score')
    - decision_col: name of the decision column
    - target: name of binary target column
    - output_path: file path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for decision in ['APPROVE', 'STEP-UP', 'DECLINE']:
        mask = df[decision_col] == decision
        if mask.sum() == 0:
            continue
        ax.hist(df.loc[mask, score_col], bins=50, alpha=0.5,
                label=decision, color=_DECISION_COLORS[decision], density=True)

    fraud_mask = df[target] == 1
    ax.hist(df.loc[fraud_mask, score_col], bins=50, alpha=0.9,
            label='actual fraud', color='black', density=True,
            histtype='step', linewidth=2)

    ax.set_xlabel(score_col)
    ax.set_ylabel('density')
    ax.set_title(f'{score_col} distribution by decision tier')
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
