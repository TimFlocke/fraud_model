import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from builtins import round
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def get_mult_roc(df, target, model_list, dataset_cols):
    """
    Plots ROC curves for a comparison of multiple models

    Parameters:
    df: The df containing the target and predicted probabilities.
    target: binary fraud target
    model_list: list of column names as strings containing probabalistic scores

    Returns:
    Matplotlib plot with multiple curves
    """

    plt.figure(figsize=(6, 6))

    for m, d in zip(model_list, dataset_cols):
        filtered = df[df[d] == 'valid'].reset_index(drop=True)
        fpr, tpr, _ = metrics.roc_curve(filtered[target], filtered[m])
        auc = metrics.roc_auc_score(filtered[target], filtered[m])
        plt.plot(fpr, tpr, label=f"{m}, auc={round(auc, 2)}")

    plt.legend(loc=0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.grid(False)
    plt.show()


def display_mod_performance(y_test, preds, save_path=None):
    """
    Displays a confusion matrix heatmap and classification report for model predictions.

    Parameters:
    - y_test: True binary labels.
    - preds: Predicted binary labels from the classifier.
    - save_path: Optional file path to save the confusion matrix figure before display.
    """

    # CONFUSION MATRIX VISUAL
    cm = confusion_matrix( y_test, preds )
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    ax.set_title('Fraud Model \n\n');
    ax.set_xlabel('\nFraud Predictions')
    ax.set_ylabel('Confirmed Fraud');

    ## LABELS
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # CLASSIFICATION REPORT
    print(classification_report( y_test, preds, digits=4))


def get_pr_curve(truth, predicted_prob):
    """
    Plots Precision-Recall Curve for Truth Labels vs Predicted probs, to help understand trade-off between precision and recall at different thresholds.

    Parameters:
    - truth: True binary labels (0 or 1).
    - predicted_prob: Predicted probabilities from classifier.

    Return:
    - Matplotlib plot of PR curve
    """

    precisions, recalls, thresholds = precision_recall_curve(truth, predicted_prob)
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def evaluate_performance(df, score_field, y, threshold=0.8):
    """
    Computes a comprehensive set of classification metrics at a given decision threshold.

    Parameters:
    - df: DataFrame containing the score column.
    - score_field: Column name of the predicted probability scores.
    - y: True binary labels.
    - threshold: Decision threshold for converting scores to binary predictions (default 0.8).

    Returns:
    - dict of metric name to rounded value.
    """

    score = df[score_field].values
    y_pred = (score > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    accuracy = accuracy_score(y, y_pred)
    pr_auc = average_precision_score(y, score)
    auc = roc_auc_score(y, score)
    f2 = fbeta_score(y, y_pred, beta=2)
    g_mean = (precision * recall) ** 0.5
    mcc = matthews_corrcoef(y, y_pred)
    gini = 2 * auc - 1

    return {
        'Score': score_field,
        'Precision': round(precision, 5),
        'Recall': round(recall, 5),
        'F1 Score': round(f1, 5),
        'Accuracy': round(accuracy, 5),
        'PR AUC': round(pr_auc, 5),
        'AUC': round(auc, 5),
        'F2 Score': round(f2, 5),
        'G-Mean': round(g_mean, 5),
        'MCC': round(mcc, 5),
        'Gini': round(gini, 5)
    }


def plot_roc_comparison(model_scores, y_true, output_path):
    """
    Plots ROC curves for multiple models on a single chart.

    Parameters:
    - model_scores: dict of {model_name: proba_array}
    - y_true: true binary labels (array-like)
    - output_path: file path to save the figure

    Returns:
    - None
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_scores)))
    fig, ax = plt.subplots(figsize=(7, 6))

    for (name, proba), color in zip(model_scores.items(), colors):
        fpr, tpr, _ = metrics.roc_curve(y_true, proba)
        auc = metrics.roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f'{name}  auc={auc:.3f}', color=color)

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random baseline')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_gains_comparison(model_scores, y_true, output_path):
    """
    Plots gains curves for multiple models on a single chart.

    Gains curve shows the percentage of fraud captured as a function of
    the percentage of transactions flagged, when scored by each model.
    A vertical red dashed line marks the top 10% threshold.

    Parameters:
    - model_scores: dict of {model_name: proba_array}
    - y_true: true binary labels (array-like)
    - output_path: file path to save the figure

    Returns:
    - None
    """
    y_arr = np.array(y_true)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_scores)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for (name, proba), color in zip(model_scores.items(), colors):
        order = np.argsort(-np.array(proba))
        sorted_labels = y_arr[order]
        cum_fraud_pct = np.cumsum(sorted_labels) / sorted_labels.sum() * 100
        pct_flagged = np.arange(1, len(y_arr) + 1) / len(y_arr) * 100
        ax.plot(pct_flagged, cum_fraud_pct, label=name, color=color)

    ax.plot([0, 100], [0, 100], linestyle='--', color='grey', label='random baseline')
    ax.axvline(x=10, linestyle='--', color='red', label='top 10%')
    ax.set_xlabel('% events flagged')
    ax.set_ylabel('% frauds captured')
    ax.set_title('Gains Curve Comparison')
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_pr_comparison(model_scores, y_true, output_path):
    """
    Plots precision-recall curves for multiple models on a single chart.

    Parameters:
    - model_scores: dict of {model_name: proba_array}
    - y_true: true binary labels (array-like)
    - output_path: file path to save the figure

    Returns:
    - None
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_scores)))
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, proba), color in zip(model_scores.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_auc = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f'{name}  pr_auc={pr_auc:.3f}', color=color)

    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_auc_progression(exp_log, output_path):
    """
    Plots val AUC across experiment runs in chronological order.

    Runs without a val AUC (e.g. stage 14 tfm runs, which use val as the
    context pool) and identical reruns (same model and val AUC) are dropped.

    Parameters:
    - exp_log: experiment log dataframe from load_experiment_log
    - output_path: file path to save the figure

    Returns:
    - the filtered log that was plotted
    """
    log = exp_log[exp_log['metrics.val_auc'].notna()]
    log = log.sort_values('timestamp').drop_duplicates(['model', 'metrics.val_auc']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(range(len(log)), log['metrics.val_auc'], 'o-', color='steelblue', markersize=6)
    for i, row in log.iterrows():
        ax.annotate(row['model'], xy=(i, row['metrics.val_auc']), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=7, rotation=40)
    ax.set_xticks(range(len(log)))
    ax.set_xticklabels(range(1, len(log) + 1))
    ax.set_xlabel('experiment run (chronological, identical reruns removed)')
    ax.set_ylabel('val AUC (tuning split)')
    ax.set_ylim(0.79, 0.95)
    ax.set_title('AUC Progression Across All Experiments')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return log
