# fraud_model (project status)

End-to-end fraud detection pipeline using the IEEE-CIS dataset.
Demonstrates feature engineering, graph-based network features,
multi-model comparison, model risk management, explainability,
and an operationalized decision strategy.

---

## pipeline stages

| stage | description | status |
|---|---|---|
| 01 | data creation | complete |
| 02 | exploratory data analysis | complete |
| 03 | feature engineering | complete |
| 04 | train / val / test split | complete |
| 05 | baseline model | complete |
| 06 | model training: catboost, xgboost, lightgbm | complete |
| 07 | feature selection | complete |
| 08 | hyperparameter tuning: optuna 50 trials | complete |
| 09 | final training + weighted ensemble | complete |
| 10 | evaluation: full model comparison | complete |
| 11 | model risk management: ks, psi | complete |
| 12 | explainability: ensemble shap | complete |
| 13 | decision strategy: score based with shap explanations | complete |
| 14 | tabular foundation model (tabicl v2), context optimization | complete (experimental) |

stage 14 is experimental and not part of the production scoring path, which remains stages 01-13.

---

## data

| file | description | rows | columns |
|---|---|---|---|
| train_transaction.csv | raw transaction data | 590,540 | 394 |
| train_identity.csv | raw identity data | 144,233 | 41 |
| data_clean.parquet | merged, cleaned | 590,540 | 340 |
| data_features.parquet | full enriched feature set | 590,540 | 457 |
| features.parquet | engineered features only | 590,540 | 117 |
| X_train/val/test_selected.parquet | selected feature splits | varies | 115 |

---

## splits

| split | rows | fraud rate |
|---|---|---|
| train | 413,378 | 3.52% |
| val | 88,581 | 3.43% |
| test | 88,581 | 3.48% |

split method: chronological by TransactionDT, no random shuffling.
time-based split prevents data leakage from time-ordered features.

---

## features

| feature group | count | description |
|---|---|---|
| original cleaned | 318 | post null-threshold drop |
| aggregation / velocity | 21 | groupby entity linkage features |
| log transforms | 22 | log1p of TransactionAmt and agg features |
| categorical encoded | 31 | label encoded categorical columns |
| graph: pagerank | 7 | unweighted pagerank per entity |
| graph: pagerank weighted | 7 | pagerank weighted by TransactionAmt |
| graph: degree centrality | 7 | degree centrality per entity |
| graph: component size | 7 | connected component size per entity |
| missingness | 1 | has_identity flag |
| **total engineered** | **457** | |
| **selected for modeling** | **115** | survived multi-method feature selection |

entity columns used for graph:
card1, card2, P_emaildomain, R_emaildomain, addr1, addr2, DeviceInfo

note: graph features (28 columns) eliminated at variance threshold step.
aggregation features capture equivalent linkage signal more directly.

---

## key findings from eda

- 76.2% of transactions have no identity data, structural,
  not random. likely reflects different transaction channels.
- fraud rate with identity data: 7.96%
- fraud rate without identity data: 2.10%
- decision: retain all rows, has_identity flag added as feature

relative trigger rate highlights:
- r_email_gmail ix=375, recipient gmail 4x more common in fraud
- is_mobile ix=315, mobile transactions 3x more likely to be fraud
- is_discover ix=230, discover card 2x+ baseline fraud rate
- M columns all negative, verification matches are protective

top shap features (ensemble):
- C13 (number of linked accounts on card)
- C1 (number of addresses associated with card)
- D15, D1 (timedelta features)
- TransactionAmt (transaction amount)
- card1__R_emaildomain__nunique (aggregation feature)
- card1__TransactionAmt__mean (aggregation feature)

---

## model results

| model | features | params | test auc | test gini | val auc (tuning split) | val gini (tuning split) |
|---|---|---|---|---|---|---|
| baseline decision tree (depth=10) | full | default | 0.791 | 0.581 | 0.806 | 0.612 |
| catboost | full | default | 0.907 | 0.814 | 0.923 | 0.847 |
| xgboost | full | default | 0.893 | 0.785 | 0.915 | 0.830 |
| lightgbm | full | default | 0.902 | 0.805 | 0.914 | 0.829 |
| catboost final | selected | tuned 50 trials | 0.905 | 0.810 | 0.918 | 0.836 |
| xgboost final | selected | tuned 50 trials | 0.899 | 0.797 | 0.923 | 0.846 |
| lightgbm final | selected | tuned 50 trials | 0.900 | 0.800 | 0.916 | 0.832 |
| **ensemble weighted** | selected | tuned | **0.905** | **0.811** | 0.925 | 0.849 |

val was used for early stopping, tuning and blend weights, so it reads higher than test. On test, the weighted ensemble (0.905) is level with CatBoost final (0.905) and just below the retrained full-feature CatBoost (0.907); its value is stability across models.

tuning: optuna bayesian optimization, 50 trials per model
best single model: catboost (full features) and xgboost final, val auc 0.923

---

## mrm results

| metric | value | assessment |
|---|---|---|
| ks statistic (ensemble, test) | 65+ | exceptional |
| score psi train vs test | 0.006 | stable |
| feature psi > 0.2 | 9 features | monitor |

score psi of 0.006 confirms model output distribution is stable
despite minor feature-level drift in timedelta and sparse identity fields.

---

## decision strategy

score transformed to 0-1000 uniform scale fitted on training distribution.
thresholds configurable in config.py.

| tier | threshold | precision | pct transactions | pct fraud captured |
|---|---|---|---|---|
| decline | score > 900 | 0.244 | 9.7% | 68.3% |
| step-up | 500 < score <= 900 | 0.024 | 38.5% | 27.0% |
| approve | score <= 500 | 0.003 | 51.8% | 4.8% |

shap explanations identify primary fraud driver per transaction
for analyst queue and audit trail.

---

## environment

- python 3.13
- virtual environment: fraud_env
- gpu: nvidia geforce rtx 3050 (cuda 13.0)
  gpu detection auto-configures training, falls back to cpu
- key libraries: catboost, xgboost, lightgbm, shap, optuna,
  networkx, scikit-learn, pandas, numpy, matplotlib, seaborn

see requirements.txt for full list with versions.

---

## repo structure

```
fraud_model/
├── README.md
├── STATUS.md
├── SETUP.md
├── config.py
├── requirements.txt
├── mod_funcs.py
├── assets/               <- key plots for README
├── data/                 <- gitignored, download instructions in README
├── docs/
│   ├── project_decisions.md
│   ├── stage_specs.md
│   └── stage7_feature_selection_plan.md
├── src/
│   ├── utils.py
│   ├── features.py
│   ├── graph.py
│   ├── model.py
│   ├── selection.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── mrm.py
│   ├── explainability.py
│   ├── experiment.py
│   └── rules.py
├── notebooks/
│   ├── 01_data_creation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_train_test_split.ipynb
│   ├── 05_baseline.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_feature_selection.ipynb
│   ├── 08_tuning.ipynb
│   ├── 09_final_training.ipynb
│   ├── 10_evaluation.ipynb
│   ├── 11_mrm.ipynb
│   ├── 12_explainability.ipynb
│   └── 13_fraud_rules.ipynb
└── outputs/              <- gitignored, regenerate by running notebooks
    ├── models/
    ├── plots/
    └── results/
```

---

## future experiments

- rule-based model routing using hard business segments
  (has_identity, device type, card network). has_identity routing
  tested in stage 14 (D_seg) ties a single balanced tfm context
  at ~18% lower scoring cost
- tabular transformer comparison (TabNet, FT-Transformer)
- reusable pipeline agent: python run_pipeline.py --data dataset.csv
- test on secondary dataset (PaySim) to validate generalization

---

## status: complete

all 13 stages implemented and committed.
final clean run and readme remaining before submission.
