import matplotlib.pyplot as plt
import numpy as np
import shap


def get_shap_values(models, weights, X, sample_n=2000):
    """
    computes weighted ensemble SHAP values across multiple models.

    parameters:
    - models: dict of {model_name: model_object}
    - weights: dict of {model_name: float} -- must sum to 1
    - X: feature dataframe
    - sample_n: number of rows to sample for speed

    returns:
    - shap_values: weighted average shap values array
    - X_sample: sampled feature dataframe
    - explainers: dict of {model_name: explainer}
    """
    X_sample = X.sample(n=min(sample_n, len(X)), random_state=42)

    shap_values_weighted = None
    explainers = {}

    for name, model in models.items():
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)

        # catboost binary returns a list [neg_class, pos_class]
        if isinstance(sv, list):
            sv = sv[1]

        weight = weights.get(name, 0)

        if shap_values_weighted is None:
            shap_values_weighted = weight * sv
        else:
            shap_values_weighted += weight * sv

        explainers[name] = explainer

    return shap_values_weighted, X_sample, explainers


def plot_shap_summary(shap_values, X_sample, output_path):
    """
    Plots a SHAP beeswarm summary for the top 20 features.

    Each dot represents one observation; colour encodes feature value
    (red = high, blue = low) and x-position encodes SHAP value magnitude.

    Parameters:
    - shap_values: 2-D numpy array from get_shap_values
    - X_sample: sampled feature DataFrame aligned with shap_values
    - output_path: file path (str or Path) to save the figure
    """
    shap.summary_plot(shap_values, X_sample, max_display=20,
                      show=False, plot_type='dot')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_shap_bar(shap_values, X_sample, output_path):
    """
    Plots a bar chart of mean absolute SHAP values for the top 20 features.

    Provides a single-number importance ranking comparable to model-native
    feature importance but grounded in the model's prediction logic.

    Parameters:
    - shap_values: 2-D numpy array from get_shap_values
    - X_sample: sampled feature DataFrame aligned with shap_values
    - output_path: file path (str or Path) to save the figure
    """
    shap.summary_plot(shap_values, X_sample, max_display=20,
                      show=False, plot_type='bar')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_shap_waterfall(explainer, X_sample, idx, output_path):
    """
    Plots a SHAP waterfall chart for a single observation.

    Shows how each feature pushes the prediction from the base value
    (expected model output) toward the final prediction for that row.

    Parameters:
    - explainer: fitted shap.TreeExplainer from get_shap_values
    - X_sample: sampled feature DataFrame aligned with explainer
    - idx: integer row index within X_sample to explain
    - output_path: file path (str or Path) to save the figure
    """
    shap_explanation = explainer(X_sample.iloc[[idx]])
    # explainer() returns Explanation object; waterfall needs single-row
    if len(shap_explanation.shape) == 3:
        # catboost multi-output: take positive class
        shap_explanation = shap_explanation[:, :, 1]
    shap.waterfall_plot(shap_explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_shap_dependence(shap_values, X_sample, feature, output_path):
    """
    Plots a SHAP dependence plot for a single feature.

    Shows the relationship between feature value and its SHAP contribution,
    with colour encoding of an automatically selected interaction feature.

    Parameters:
    - shap_values: 2-D numpy array from get_shap_values
    - X_sample: sampled feature DataFrame aligned with shap_values
    - feature: column name (str) of the feature to plot
    - output_path: file path (str or Path) to save the figure
    """
    shap.dependence_plot(feature, shap_values, X_sample,
                         show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
