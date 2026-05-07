import matplotlib.pyplot as plt
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


def display_mod_performance(y_test, preds):
    """
    Displays a confusion matrix heatmap and classification report for model predictions.

    Parameters:
    - y_test: True binary labels.
    - preds: Predicted binary labels from the classifier.
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
