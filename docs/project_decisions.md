# project decisions

A record of every key design decision made during development,
the alternatives considered, and the rationale for each choice.

---

## dataset

**decision:** IEEE-CIS Fraud Detection (Kaggle)

**alternatives considered:**
- PaySim — rejected, synthetic and too simple
- Credit Card Fraud ULB — rejected, only 30 PCA features, no graph opportunity
- Elliptic Bitcoin — rejected, niche domain, less recognizable

**rationale:** IEEE-CIS offers two joinable tables, rich entity fields
for graph construction, class imbalance, and anonymized features that
reflect real production constraints. well known in the fraud/ml community.

---

## development environment

**decision:** VS Code primary, Databricks for graph stage only

**alternatives considered:**
- pure Databricks — rejected, slower iteration, notebook export issues
- pure VS Code — accepted with one exception

**rationale:** Claude Code extension lives in VS Code, enabling faster
iteration. Git workflow is native. Databricks awareness demonstrated via
graph construction notebook and README documentation.

---

## code architecture

**decision:** src/ modules + thin notebooks

**alternatives considered:**
- pure notebooks — rejected, signals analyst not engineer
- pure scripts — rejected, loses narrative flow

**rationale:** notebooks import from src/, call functions, display results.
all logic lives in reusable, importable, testable src/ files. this pattern
signals production ml awareness to a hiring manager.

---

## data — identity missingness

**decision:** retain all rows with missing identity data

**alternatives considered:**
- drop rows with no identity data — rejected

**rationale:** 76.2% of transactions have no identity data. this is
structural, not random — likely reflects different transaction channels.
dropping these rows would lose 9,452 fraud cases. missing identity is
itself a signal. gradient boosting models handle NaN natively.
has_identity binary flag added to capture the missingness pattern.

**finding:** fraud rate with identity 7.96%, without identity 2.10%.
transactions with device data are nearly 4x more likely to be fraud,
consistent with card-not-present and web-based fraud patterns.

---

## null threshold

**decision:** drop columns with >80% null rate

**rationale:** columns above this threshold carry insufficient signal
to justify the noise they introduce. primarily affects sparse V-columns
from Vesta. reduced column count from 434 to 340 after merge.

---

## train/val/test split

**decision:** chronological split — 70/15/15 by TransactionDT

**alternatives considered:**
- random split — rejected
- stratified random split — rejected

**rationale:** fraud detection is a temporal problem. random splitting
leaks future data into training, corrupting time-based velocity features.
chronological split reflects production reality — models always predict
future fraud from past data. fraud rates are stable across splits
(train 3.52%, val 3.43%, test 3.48%) confirming the split is clean.

---

## feature engineering — missingness

**decision:** has_identity flag only, leave all other nulls as NaN

**rationale:** catboost, xgboost, lightgbm handle NaN natively.
imputation may remove signal embedded in missingness patterns.
has_identity validated as strong fraud signal in EDA (ix=240).

---

## feature engineering — aggregation features

**decision:** global groupby aggregations with time-sort, map-based pattern

**alternatives considered:**
- expanding window with time constraint — rejected, too slow on 590k rows
- no time constraint at all — rejected, methodologically weak

**rationale:** df sorted by TransactionDT before aggregation.
map-based pattern (groupby then map) is faster than transform.
production caveat documented in code comments — in production these
would be computed via a real-time feature store using only historical data.

**combinations:** 21 entity-target-aggregation combinations across
card1, card2, P_emaildomain, R_emaildomain, addr1, addr2, DeviceInfo.

---

## feature engineering — graph

**decision:** networkx graph with pagerank, weighted pagerank,
degree centrality, connected component size per entity node

**entity columns:**
card1, card2, P_emaildomain, R_emaildomain, addr1, addr2, DeviceInfo

**alternatives considered:**
- GraphFrames on Databricks — deferred, not needed at this data size
- GNN embeddings — out of scope for demo
- more entity columns (id_20, id_30, id_31) — deferred

**rationale:** three interpretable graph metrics that demonstrably add
signal. weighted pagerank uses TransactionAmt as edge weights — nodes
connected by high-value transactions rank higher. produces 28 graph
feature columns (7 entities x 4 metrics).

**production note:** in production entity columns would include full
email addresses, phone numbers, IP addresses, and device fingerprint
hashes. the methodology is identical — only entity granularity differs.

---

## feature engineering — transformations

**decision:** log1p transform on TransactionAmt and all aggregation features

**rationale:** TransactionAmt is heavily right-skewed. log transform
compresses the tail and can improve split quality. saved alongside
originals — feature selection decides which version is more useful.

---

## feature engineering — encoding

**decision:** label encoding for categoricals, new columns only

**rationale:** catboost handles categoricals natively via cat_features.
xgboost and lightgbm need numeric encoding. label encoding is sufficient
for tree-based models — ordinal relationships do not matter.
original columns preserved, encoded versions in col_encoded columns.
nulls filled with string 'missing' before encoding.

---

## plotting library

**decision:** matplotlib only, no plotly

**rationale:** plotly interactive plots go blank on notebook export.
matplotlib produces static plots that export and render reliably on
GitHub. every plot saved to outputs/plots/ before display.
plt.close() called after every plot.

---

## model selection

**decision:** decision tree baseline + catboost, xgboost, lightgbm comparison

**alternatives considered:**
- logistic regression baseline — rejected, assumes linearity,
  needs scaling, will underperform badly and inflate the apparent
  improvement from gradient boosting
- random forest — not included, gradient boosting dominates on
  tabular fraud data

**rationale:** decision tree is the same model family as the final models.
max_depth=5 keeps it interpretable and plottable. the improvement from
catboost/xgboost/lightgbm reflects tuning and ensemble gains, not a
fundamental model family change. three gradient boosting models gives
a head-to-head comparison with different strengths.

**catboost:** handles categoricals natively, likely strongest out of box
**lightgbm:** fastest to train, leaf-wise growth, watch for overfitting
**xgboost:** most established ecosystem, SHAP originated here

---

## hyperparameter tuning

**decision:** optuna (bayesian optimization)

**alternatives considered:**
- grid search — rejected, too slow for three models
- random search — rejected, less efficient than bayesian

**rationale:** optuna is the modern standard. more efficient than grid
or random search. n_trials capped to keep runtime manageable.

---

## feature selection stack

**decision:** multi-method approach in order:
1. variance threshold
2. correlation matrix (drop >0.95 correlated pairs)
3. mutual information
4. selectkbest + chi-squared (aggregation features)
5. selectkbest + f_classif (continuous features)
6. catboost first pass importance
7. shap values
8. featurewiz (sanity check)

**rationale:** no single method is sufficient. features surviving
multiple methods have stronger evidence for inclusion. the narrative
is defensible — every cut has a documented reason.

---

## mrm approach

**decision:** ks statistic on decile scores + psi (train vs val vs test)

**rationale:** ks and psi are standard model risk management metrics
in regulated financial institutions. ks > 40 considered strong for
fraud models. psi < 0.1 stable, 0.1-0.2 minor shift, >0.2 investigate.
required for any model deployed at a bank or fintech.

---

## fraud rules approach

**decision:** model-derived rules from shap + analytical rules from eda

**rationale:** shap identifies top drivers, rules operationalize them.
this reflects how fraud teams work in practice — model tells you what
matters, rules make it actionable for operations teams.

---

## gitignore decisions

**decision:** gitignore outputs/, data/, fraud_env/

**rationale:** parquet files exceed github 100mb limit. raw data should
never be committed. virtual environment is local only. key plots
copied to assets/ folder and committed selectively for README.

---

## graph visualization

**decision:** ego graph centered on highest-degree high-fraud node,
filtered to neighbors with fraud rate > 0.1, min_degree_label=10

**rationale:** full graph has 15,970 nodes — unreadable. ego graph
with fraud rate filter produces a readable 200-400 node subgraph.
node color = fraud rate, size = degree, edge thickness = transaction amount.
this visual is the single best illustration of why graph features
add value that tabular models miss.
