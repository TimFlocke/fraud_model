import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score, fbeta_score, recall_score, matthews_corrcoef, classification_report, roc_auc_score, confusion_matrix

# Databricks notebook source
# DBTITLE 1,full value counts
def full_value_counts(df, col):
    """
    combines value counts with normalized counts
    """
    a = df[col].value_counts().to_frame('cnt')
    b = df[col].value_counts(normalize=True).to_frame('pct')

    return a.merge(b, left_index=True, right_index=True)

# COMMAND ----------

# DBTITLE 1,roc
import matplotlib.pyplot as plt
from sklearn import metrics
from builtins import round

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
        df = df[df[d] == 'valid'].reset_index(drop=True)
        fpr, tpr, _ = metrics.roc_curve(df[target], df[m])
        auc = metrics.roc_auc_score(df[target], df[m])
        plt.plot(fpr, tpr, label=f"{m}, auc={round(auc, 2)}")

    plt.legend(loc=0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.grid(False)
    plt.show()

# COMMAND ----------

def display_mod_performance(y_test, preds):

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

# COMMAND ----------

def get_pr_curve(truth, predicted_prob):
    """
    Plots Precision-Recall Curve for Truth Labels vs Predicted probs, to help understand trade-off between precision and recall at different thresholds.

    Parameters:
    - truth: True binary labels (0 or 1).
    - predicted_prob: Predicted probabilities from classifier.

    Return:
    - Matplotlib plot of PR curve
    """
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(truth, predicted_prob)
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

# COMMAND ----------

# DBTITLE 1,metrics table
def evaluate_performance(df, score_field, y, threshold=0.8):
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

# COMMAND ----------

# DBTITLE 1,gains
def get_gains(df, target, score, top_n):
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
    fig = px.line(d, x='pct_flagged', y='fraud_pct_cap', 
                  title=f'{score} Gains Chart',
                  height=400,
                  width=900,
                  template='plotly_white',
                  labels={'pct_flagged': '% Events Flagged',
                          'fraud_pct_cap': '% Frauds Captured'})
    
    fig.add_shape(
        type='line', 
        x0=0, y0=0, x1=100, y1=100, 
        line=dict(dash='dash'))
    
    fig.add_shape(
        type='line', 
        x0=top_n, y0=-20, x1=top_n, y1=120, 
        line=dict(dash='dash', color='red')) # < - plot threshold
    
    fig.add_annotation(x=top_n, y=120, text=f"Top {top_n}% score threshold: {np.round(thresh, 2)} captures {fraud_pct_cap}% fraud", font=dict(color='red'), xanchor='left', yanchor='bottom', showarrow=False)
    
    fig.show()  

    return out

# COMMAND ----------

# DBTITLE 1,score deciles
def score_deciles(df, uid, score, target):
    
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