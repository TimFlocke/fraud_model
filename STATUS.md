# fraud_model — project status

End-to-end fraud detection pipeline using the IEEE-CIS dataset.
Demonstrates feature engineering, graph-based network features,
multi-model comparison, model risk management, and fraud rules.

---

## pipeline stages

| stage | description | status |
|---|---|---|
| 01 | data creation | complete |
| 02 | exploratory data analysis | complete |
| 03 | feature engineering | complete |
| 04 | train / val / test split | complete |
| 05 | baseline model | in progress |
| 06 | model training — catboost, xgboost, lightgbm | not started |
| 07 | feature selection | not started |
| 08 | hyperparameter tuning | not started |
| 09 | final training | not started |
| 10 | evaluation | not started |
| 11 | model risk management — ks, psi | not started |
| 12 | explainability — shap | not started |
| 13 | fraud rules | not started |

---

## data

| file | description | rows | columns |
|---|---|---|---|
| train_transaction.csv | raw transaction data | 590,540 | 394 |
| train_identity.csv | raw identity data | 144,233 | 41 |
| data_clean.parquet | merged, cleaned | 590,540 | 340 |
| data_features.parquet | full enriched feature set | 590,540 | 457 |
| features.parquet | engineered features only | 590,540 | 117 |

---

## splits

| split | rows | fraud rate |
|---|---|---|
| train | 413,378 | 3.52% |
| val | 88,581 | 3.43% |
| test | 88,581 | 3.48% |

split method: chronological by TransactionDT — no random shuffling.
time-based split prevents data leakage from time-ordered features.

---

## features

| feature group | count | description |
|---|---|---|
| original cleaned | 318 | post null-threshold drop |
| aggregation / velocity | 21 | groupby entity linkage features |
| log transforms | 22 | log1p of TransactionAmt and agg features |
| categorical encoded | 31 | label encoded categorical columns |
| graph — pagerank | 7 | unweighted pagerank per entity |
| graph — pagerank weighted | 7 | pagerank weighted by TransactionAmt |
| graph — degree centrality | 7 | degree centrality per entity |
| graph — component size | 7 | connected component size per entity |
| missingness | 1 | has_identity flag |
| **total** | **457** | |

entity columns used for graph:
card1, card2, P_emaildomain, R_emaildomain, addr1, addr2, DeviceInfo

---

## key findings from eda

- 76.2% of transactions have no identity data — structural,
  not random. likely reflects different transaction channels.
- fraud rate with identity data: 7.96%
- fraud rate without identity data: 2.10%
- decision: retain all rows, has_identity flag added as feature

relative trigger rate highlights:
- r_email_gmail ix=375 — recipient gmail 4x more common in fraud
- is_mobile ix=315 — mobile transactions 3x more likely to be fraud
- is_discover ix=230 — discover card 2x+ baseline fraud rate
- M columns all negative — verification matches are protective

---

## models

| model | val auc | val ks | status |
|---|---|---|---|
| baseline — decision tree (depth=5) | tbd | tbd | in progress |
| catboost | tbd | tbd | not started |
| xgboost | tbd | tbd | not started |
| lightgbm | tbd | tbd | not started |
| catboost (tuned, selected features) | tbd | tbd | not started |
| xgboost (tuned, selected features) | tbd | tbd | not started |
| lightgbm (tuned, selected features) | tbd | tbd | not started |

---

## environment

- python 3.13
- virtual environment: fraud_env
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

## next steps

1. complete stage 05 — baseline decision tree
2. stage 06 — train catboost, xgboost, lightgbm
3. stage 07 — feature selection
4. stage 08 — hyperparameter tuning with optuna
5. stage 09 — final training on selected features and best params
6. stage 10 — head to head model evaluation
7. stage 11 — mrm: ks statistic and psi
8. stage 12 — shap explainability
9. stage 13 — fraud rules
10. write readme
11. clean and final commit
