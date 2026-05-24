# Fraud Modeling Pipeline

End-to-end fraud detection pipeline built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset. Covers the full model development lifecycle — from raw data to an operationalized decision strategy — following production ML standards.

This is intended to be a re-usable pipeline for fraud modeling. There are 13 stages, with functions extracted to src files, and demonstrated using modularized notebooks. Extensive due dilligence spent on feature engineering, selection, and transformations - including creation of graph-related features. 

---

## results

| model | features | val auc | val gini |
|---|---|---|---|
| baseline — decision tree | full | 0.806 | 0.612 |
| catboost | full, default | 0.906 | 0.812 |
| xgboost | full, default | 0.915 | 0.830 |
| lightgbm | full, default | 0.914 | 0.829 |
| catboost final | selected, tuned | 0.918 | 0.836 |
| xgboost final | selected, tuned | 0.923 | 0.846 |
| lightgbm final | selected, tuned | 0.916 | 0.832 |
| **ensemble weighted** | selected, tuned | **0.925** | **0.849** |

**KS statistic:** 65+ (exceptional) | **Score PSI train vs test:** 0.006 (stable)

![AUC progression](assets/auc_progression.png)
![Gains comparison](assets/gains_comparison_final.png)
![Precision-Recall Curve comparison](assets/pr_curves_comparison_final.png)
![Best model: gains chart](assets/best_model_test_gains.png)


---

## pipeline

| stage | notebook | description |
|---|---|---|
| 01 | 01_data_creation | load, merge, clean, null analysis |
| 02 | 02_eda | time series, categoricals, trigger rate analysis |
| 03 | 03_feature_engineering | graph features, velocity features, transforms |
| 04 | 04_train_test_split | chronological 70/15/15 split |
| 05 | 05_baseline | decision tree baseline |
| 06 | 06_model_training | catboost, xgboost, lightgbm — full feature set |
| 07 | 07_feature_selection | variance, correlation, MI, chi2, catboost, shap |
| 08 | 08_tuning | optuna bayesian optimization, 50 trials each |
| 09 | 09_final_training | tuned models + weighted ensemble |
| 10 | 10_evaluation | full model comparison on test set |
| 11 | 11_mrm | ks statistic, psi, uniform score transformation |
| 12 | 12_explainability | weighted ensemble shap |
| 13 | 13_fraud_rules | score-based decision strategy with shap explanations |

---

## key features

### graph-based network features
transactions are represented as a graph where nodes are entities
(cards, email domains, addresses, devices) and edges are shared
transactions. four graph metrics computed per entity:

- pagerank — structural influence in the network
- weighted pagerank — influence scaled by transaction amount
- degree centrality — number of unique connections
- connected component size — fraud ring membership

![fraud dense subgraph](assets/fraud_dense_subgraph.png)

### velocity and linkage features
21 aggregation features capturing historical entity behavior:

```
card1__addr1__nunique         — unique addresses per card
card1__R_emaildomain__nunique — unique recipient domains per card
card1__TransactionID__count   — transaction velocity per card
P_emaildomain__card1__nunique — unique cards per purchaser email
```

### feature selection
multi-method selection stack reduces 457 features to 115:

1. variance threshold
2. correlation filter (>0.95 threshold)
3. mutual information
4. selectkbest + chi-squared
5. catboost importance
6. shap values

features surviving at least two ranking methods are retained.

### model risk management
- **ks statistic:** 65+ on test set — exceptional separation
- **score psi:** 0.006 train vs test — stable population
- **uniform score transformation:** raw probabilities mapped to
  0-1000 scale fitted on training distribution for operational use

![ensemble ks curve](assets/ks_curve_ensemble_weighted.png)

### decision strategy
score-based tiers with shap-driven explanations:

| tier | threshold | precision | pct transactions | pct fraud caught |
|---|---|---|---|---|
| decline | score > 900 | 24.4% | 9.7% | 68.3% |
| step-up | 500 < score ≤ 900 | 2.4% | 38.5% | 27.0% |
| approve | score ≤ 500 | 0.3% | 51.8% | 4.8% |

thresholds configurable in `config.py`. shap identifies the
primary fraud driver per transaction for the analyst queue.

![decision summary](assets/decision_summary.png)

---

## top fraud signals (shap)

| feature | description |
|---|---|
| C13 | number of linked accounts on card |
| C1 | number of addresses associated with card |
| D15, D1 | timedelta features — account age and recency |
| TransactionAmt | transaction amount |
| card1__R_emaildomain__nunique | unique recipient domains per card |
| card1__TransactionAmt__mean | average transaction amount per card |

![top shap values](assets/shap_summary_ensemble.png)

---

## eda highlights

- **76.2%** of transactions have no identity data — structural,
  not random. fraud rate with identity: **7.96%** vs without: **2.10%**
- recipient gmail address appears **4x** more often in fraud (trigger rate ix=375)
- mobile transactions **3x** more likely to be fraud (ix=315)
- discover card **2x** baseline fraud rate (ix=230)
- m-columns (verification matches) consistently protective — all negative trigger rates

![relative trigger rate](assets/relative_trigger_rate.png)
---

## repo structure

```
fraud_model/
├── README.md
├── STATUS.md              ← full project status and results
├── SETUP.md               ← environment setup instructions
├── config.py              ← paths, constants, thresholds
├── requirements.txt
├── mod_funcs.py           ← original utility functions
├── assets/                ← key plots
├── data/                  ← gitignored — see data setup below
├── docs/
│   ├── project_decisions.md
│   ├── stage_specs.md
│   └── stage7_feature_selection_plan.md
├── src/
│   ├── utils.py           ← full_value_counts, get_relative_trigger
│   ├── features.py        ← data loading, cleaning, engineering
│   ├── graph.py           ← entity graph construction, graph features
│   ├── model.py           ← train functions for all models
│   ├── selection.py       ← feature selection methods
│   ├── tuning.py          ← optuna objective functions
│   ├── evaluation.py      ← metrics, plots, comparison functions
│   ├── mrm.py             ← ks, psi, score transformation
│   ├── explainability.py  ← ensemble shap functions
│   ├── experiment.py      ← experiment logger
│   └── rules.py           ← decision strategy functions
├── notebooks/             ← 13 notebooks, run in order
└── outputs/               ← gitignored, regenerated by running notebooks
    ├── models/
    ├── plots/
    └── results/
```

---

## setup

### 1. clone the repo

```bash
git clone https://github.com/TimFlocke/fraud_model.git
cd fraud_model
```

### 2. create virtual environment

```bash
python -m venv fraud_env
fraud_env\Scripts\activate      # windows
source fraud_env/bin/activate   # mac/linux
```

### 3. install dependencies

```bash
pip install -r requirements.txt
```

### 4. download data

- go to https://www.kaggle.com/c/ieee-fraud-detection/data
- accept competition rules
- download `train_transaction.csv` and `train_identity.csv`
- place both files in `data/`

### 5. run notebooks in order

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

each notebook loads outputs from the previous stage.
models load from disk if already trained — no unnecessary retraining.

---

## gpu support

gpu detected automatically via `get_device()` in `config.py`.
catboost, xgboost, and lightgbm all use gpu if nvidia gpu is available,
falling back to cpu on any machine. no configuration required.

---

## experiment tracking

all training runs logged to `outputs/results/experiment_log.json`.
view full progression from baseline to final ensemble:

```python
from src.experiment import load_experiment_log
log = load_experiment_log('outputs/results')
print(log[['model', 'val_auc', 'val_gini']].sort_values('val_auc', ascending=False))
```

---

## configurable thresholds

all key parameters in `config.py`:

```python
NULL_THRESHOLD = 0.80              # null drop threshold
RANDOM_STATE = 42
TARGET = 'isFraud'
RULES_TOP_N_FEATURES = 10          # shap features for decision strategy
STRATEGY_APPROVE_THRESHOLD = 400   # score below this = approve
STRATEGY_DECLINE_THRESHOLD = 700   # score above this = decline
```

---

## future experiments

- **rule-based model routing** — segment transactions using hard
  business rules (has_identity, device type, card network) and route
  each segment to the model with highest segment-level auc
- **tabular transformer** — TabNet or FT-Transformer comparison
  against gradient boosting
- **reusable pipeline agent** — generalize to any labeled transaction
  dataset via cli: `python run_pipeline.py --data dataset.csv --target fraud_label`
- **PaySim validation** — run pipeline on secondary dataset to
  confirm generalization

---

## dependencies

key libraries: `catboost` `xgboost` `lightgbm` `shap` `optuna`
`networkx` `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn`

see `requirements.txt` for full list with pinned versions.
