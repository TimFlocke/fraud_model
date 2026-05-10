# stage specs — claude code prompts

All Claude Code prompts used to build this project, in order.
Reuse these to rebuild any stage or adapt for a new dataset.

---

## reusable style rules

Add to every prompt:

```
STYLE RULES:
- comments lowercase, minimal, no punctuation
- no obvious comments restating what the code does
- no AI-style section headers
- no excessive blank lines
- markdown cells concise and direct
- write like a practitioner
- plain ASCII only in strings and comments
```

---

## reusable architecture rules

Add to every prompt:

```
ARCHITECTURE RULES:
- all logic lives in src/ files
- no function definitions in notebooks
- use config.py values throughout
- every plot saved to outputs/plots/ before display
- use plt.close() after every plot
- use matplotlib only, no plotly
- add markdown cells between sections explaining what is happening
- kernel is Python (fraud_env)
```

---

## migrate existing functions into src/

```
I have a file called mod_funcs.py in my project root that contains
several functions I want to migrate into the correct src/ files.

mapping:
full_value_counts()        -> src/utils.py
get_mult_roc()             -> src/evaluation.py
display_mod_performance()  -> src/evaluation.py
get_pr_curve()             -> src/evaluation.py
evaluate_performance()     -> src/evaluation.py
get_gains()                -> src/mrm.py
score_deciles()            -> src/mrm.py

rules:
- read mod_funcs.py first before doing anything
- copy each function into the correct src/ file
- add appropriate imports at the top of each src/ file
- do not delete mod_funcs.py yet
- do not modify the functions themselves
- add a docstring to any function missing one
- fix this bug in get_mult_roc() -- df is being reassigned
  inside the loop, filtering an already filtered dataframe.
  fix it so each iteration filters from the original dataframe.
```

---

## stage 01 — data creation

```
I am building a fraud detection model using the IEEE-CIS dataset.

my project structure is:
fraud_model/
├── config.py
├── mod_funcs.py
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
├── src/
│   ├── __init__.py
│   ├── utils.py          <- contains full_value_counts()
│   ├── features.py       <- empty, will hold cleaning functions
│   ├── evaluation.py     <- contains model evaluation functions
│   └── mrm.py            <- contains get_gains(), score_deciles()
├── notebooks/
│   └── 01_data_creation.ipynb
└── outputs/
    ├── plots/
    └── results/

TASK 1 -- populate config.py with:
- DATA_PATH = 'data/'
- TRANSACTION_PATH = 'data/train_transaction.csv'
- IDENTITY_PATH = 'data/train_identity.csv'
- OUTPUT_PATH = 'outputs/'
- PLOTS_PATH = 'outputs/plots/'
- RESULTS_PATH = 'outputs/results/'
- RANDOM_STATE = 42
- TARGET = 'isFraud'
- NULL_THRESHOLD = 0.8

use pathlib Path pattern so paths work regardless of where
notebook is run from:

from pathlib import Path
ROOT = Path(__file__).parent
TRANSACTION_PATH = ROOT / 'data' / 'train_transaction.csv'
etc.

TASK 2 -- populate src/features.py with these functions:
- load_data(transaction_path, identity_path)
- merge_data(df_transaction, df_identity)
- drop_null_cols(df, threshold)
- convert_datetime(df)
all functions must have docstrings.

TASK 3 -- populate notebooks/01_data_creation.ipynb

section 1 - imports and setup
- %load_ext autoreload / %autoreload 2
- import sys, os, pandas, numpy, matplotlib, seaborn
- add project root to sys.path
- import config
- import functions from src/features.py and src/utils.py

section 2 - load data
- call load_data()
- print shape of each dataframe
- print first 5 rows of each

section 3 - merge
- call merge_data()
- display shape before and after
- size the missingness:
  * count rows with/without identity data (DeviceType not null)
  * print percentage missing identity
  * print fraud count in no-identity rows
  * print fraud rate with vs without identity data
  * plot side by side bar chart comparing count and fraud rate
  * save to outputs/plots/identity_missingness.png
  * markdown cell documenting decision to retain all rows

section 4 - null analysis
- calculate null percentage per column
- plot top 30 columns by null percentage (horizontal bar)
- save to outputs/plots/null_analysis.png
- call drop_null_cols()

section 5 - datetime conversion
- call convert_datetime()
- print sample of TransactionDT, hour_of_day, day_of_week

section 6 - target analysis
- call full_value_counts() from src/utils.py on isFraud
- plot isFraud distribution as bar chart
- save to outputs/plots/target_distribution.png
- print fraud rate as percentage
- print explicit confirmation that zero rows were dropped

section 7 - save
- save to outputs/results/data_clean.parquet
- print final shape and confirmation
```

---

## stage 02 — eda

```
populate notebooks/02_eda.ipynb

input: outputs/results/data_clean.parquet
do not reload raw CSVs.

section 1 - fraud time series
- aggregate by calendar date: total transactions, fraud count, fraud rate
- plot daily transaction volume (bar)
- plot daily fraud count (line)
- plot daily fraud rate (line) -- annotate days where rate > 2x rolling 7-day mean
- save to outputs/plots/fraud_timeseries_daily.png
- aggregate by hour timestamp (date + hour, not just hour of day)
- plot hourly fraud count and rate
- save to outputs/plots/fraud_timeseries_hourly.png

section 2 - fraud over time (cyclical)
- fraud volume by hour_of_day (bar)
- fraud volume by day_of_week (bar)
- fraud rate by hour_of_day (line)
- fraud rate by day_of_week (line)

section 3 - transaction amount
- histogram of TransactionAmt
- histogram of log TransactionAmt
- side by side box plots fraud vs non-fraud
- print median and mean for fraud vs non-fraud

section 4 - categorical analysis
for each field use full_value_counts() AND plot fraud penetration:
- ProductCD
- card4, card6
- top 15 P_emaildomain by volume
- top 15 R_emaildomain by volume
- top 15 addr1 by volume

section 5 - device analysis
- full_value_counts() on DeviceType
- top 15 DeviceInfo with fraud penetration

section 6 - match fields M1-M9
- fraud rate by value for each M column
- single figure with subplots
- save to outputs/plots/match_fields_fraud_rate.png

section 7 - correlation
- correlation of all numeric columns with isFraud
- top 20 positive and negative features as horizontal bar chart
- save to outputs/plots/correlation_with_target.png

section 8 - relative trigger rate
binary features:
for col in ['M1','M2','M3','M4','M5','M6','M7','M8','M9']:
    df[col] = (df[col] == 'T').astype(int)

df['is_mobile'] = (df['DeviceType'] == 'mobile').astype(int)
df['is_credit'] = (df['card6'] == 'credit').astype(int)
df['is_visa'] = (df['card4'] == 'visa').astype(int)
df['is_mastercard'] = (df['card4'] == 'mastercard').astype(int)
df['is_amex'] = (df['card4'] == 'amex').astype(int)
df['is_discover'] = (df['card4'] == 'discover').astype(int)
df['p_email_gmail'] = (df['P_emaildomain'] == 'gmail.com').astype(int)
df['r_email_gmail'] = (df['R_emaildomain'] == 'gmail.com').astype(int)
df['is_product_W'] = (df['ProductCD'] == 'W').astype(int)
df['has_identity'] = df['DeviceType'].notna().astype(int)

binary_vars = [
    'M1','M2','M3','M4','M5','M6','M7','M8','M9',
    'is_mobile','is_credit',
    'is_visa','is_mastercard','is_amex','is_discover',
    'p_email_gmail','r_email_gmail',
    'is_product_W','has_identity'
]

- import get_relative_trigger from src/utils.py
- upper_thresh=120, lower_thresh=70
- plot ix as horizontal bar chart, red > 120, blue < 70
- vertical dashed line at 100
- save to outputs/plots/relative_trigger_rate.png
```

---

## stage 03 — feature engineering

```
TASK 1 -- populate src/graph.py

build_entity_graph(df):
- undirected graph using networkx
- entity_cols = ['card1','card2','P_emaildomain','R_emaildomain',
  'addr1','addr2','DeviceInfo']
- edges connect entities appearing in same transaction
- use itertools.combinations for all pairs
- edge weight = TransactionAmt (cumulative sum if edge exists)
- use groupby.sum() before touching graph for performance

get_graph_features(G, df, entity_cols):
- four metrics per node:
  pagerank (unweighted)
  pagerank_weighted (weight='weight')
  degree_centrality
  connected_component_size
- maps all four to df per entity = 28 new columns
- entity_cols = ['card1','card2','P_emaildomain','R_emaildomain',
  'addr1','addr2','DeviceInfo']

get_fraud_dense_subgraph(G, df, entity_cols, top_n=1):
- find most fraud-dense connected components
- score components by mean fraud rate of member nodes
- return top_n components as subgraph

plot_fraud_subgraph(subgraph, node_fraud_rates, output_path,
                   min_degree_label=10):
- node color = fraud rate (RdYlBu_r colormap)
- node size = degree
- edge thickness = transaction amount weight
- only label nodes with degree > min_degree_label
- colorbar for fraud rate
- save before display, plt.close() after

TASK 2 -- add to src/features.py

build_aggregation_features(df):
- sort by TransactionDT ascending
- map-based pattern for 21 combinations:
entity_target_agg = [
    ('card1','addr1','nunique'),
    ('card1','P_emaildomain','nunique'),
    ('card1','R_emaildomain','nunique'),
    ('card1','TransactionAmt','mean'),
    ('card1','TransactionAmt','std'),
    ('card1','TransactionID','count'),
    ('P_emaildomain','card1','nunique'),
    ('P_emaildomain','addr1','nunique'),
    ('P_emaildomain','R_emaildomain','nunique'),
    ('P_emaildomain','TransactionAmt','mean'),
    ('R_emaildomain','card1','nunique'),
    ('R_emaildomain','addr1','nunique'),
    ('R_emaildomain','P_emaildomain','nunique'),
    ('R_emaildomain','TransactionAmt','mean'),
    ('addr1','card1','nunique'),
    ('addr1','P_emaildomain','nunique'),
    ('addr1','R_emaildomain','nunique'),
    ('addr1','TransactionAmt','mean'),
    ('DeviceInfo','card1','nunique'),
    ('DeviceInfo','P_emaildomain','nunique'),
    ('DeviceInfo','addr1','nunique'),
]
- column naming: entity__target__agg
- comment: global stats on time-sorted data, production
  would use real-time feature store

add_missingness_features(df):
- df['has_identity'] = df['DeviceType'].notna().astype(int)

add_transformations(df):
- df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
- log transform all columns where '__' in col name

encode_categoricals(df, cat_cols):
- LabelEncoder per column
- fillna('missing').astype(str) before encoding
- write to col_encoded, preserve originals

cat_cols = [
    'ProductCD','card4','card6',
    'P_emaildomain','R_emaildomain',
    'M1','M2','M3','M4','M5','M6','M7','M8','M9',
    'DeviceType','DeviceInfo',
    'id_12','id_15','id_16','id_23','id_27','id_28',
    'id_29','id_30','id_31','id_33','id_34','id_35',
    'id_36','id_37','id_38'
]

split_data(df, target):
- sort by TransactionDT ascending
- train 70%, val 15%, test 15% chronologically
- drop TransactionID and TransactionDT from features
- print shape and fraud rate per split
- return X_train, X_val, X_test, y_train, y_val, y_test

TASK 3 -- populate notebooks/03_feature_engineering.ipynb

section 1 - imports
section 2 - load data_clean.parquet
section 3 - missingness features
section 4 - aggregation features
section 5 - log transformations + side by side histogram
section 6 - categorical encoding
section 7 - graph features + pagerank distribution plot
section 8 - fraud dense subgraph visualization (ego graph)
section 9 - save:
  - data_clean.parquet untouched
  - features.parquet = TransactionID + all new columns
  - data_features.parquet = full enriched frame
```

---

## stage 04 — train/val/test split

```
populate notebooks/04_train_test_split.ipynb
add split_data() to src/features.py (see stage 03 spec)

section 1 - imports
section 2 - load data_features.parquet
section 3 - call split_data(), plot fraud rates across splits
  save to outputs/plots/split_fraud_rates.png
  markdown: time-based split prevents data leakage
section 4 - save all six parquet files to outputs/results/
```

---

## stage 05 — baseline model

```
populate notebooks/05_baseline.ipynb
add train_baseline() to src/model.py

src/model.py -- train_baseline(X_train, y_train):
- DecisionTreeClassifier
- max_depth=5, random_state=config.RANDOM_STATE,
  class_weight='balanced'
- returns fitted model

existing functions to use:
- evaluate_performance() from src/evaluation.py
- display_mod_performance() from src/evaluation.py
- get_pr_curve() from src/evaluation.py
- score_deciles() from src/mrm.py
- get_gains() from src/mrm.py

section 1 - imports
section 2 - load X_train, X_val, y_train, y_val
section 3 - train baseline
section 4 - evaluate on val set:
  - evaluate_performance() as dataframe
  - display_mod_performance() confusion matrix
  - get_pr_curve()
section 5 - gains and deciles:
  - results df with TransactionID, isFraud, baseline_score
  - score_deciles()
  - get_gains(top_n=0.1)
section 6 - feature importance top 20 horizontal bar chart
section 7 - save model to outputs/models/baseline.pkl
```

---

## stage 06 — model training (coming soon)

train catboost, xgboost, lightgbm on full feature set.
same evaluation suite as baseline for apples to apples comparison.

---

## stage 07 — feature selection (coming soon)

see docs/stage7_feature_selection_plan.md for full spec.

---

## stage 08 — tuning (coming soon)

optuna bayesian optimization for all three models.

---

## stage 09 — final training (coming soon)

retrain all three models on selected features and best params.

---

## stage 10 — evaluation (coming soon)

head to head comparison table across all models.

---

## stage 11 — mrm (coming soon)

ks statistic on decile scores, psi train vs val vs test.

---

## stage 12 — explainability (coming soon)

shap summary plot, waterfall plot, bar plot for best model.

---

## stage 13 — fraud rules (coming soon)

model-derived rules from shap + analytical rules from eda.
