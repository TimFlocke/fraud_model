# fraud_pipeline — agent transition notes

Notes for building the reusable pipeline agent on top of fraud_model.
Start a new Claude project for this work. Reference fraud_model repo
as the foundation.

---

## what exists in fraud_model that transfers directly

all src/ modules are already written to be dataset-agnostic:

| file | status | notes |
|---|---|---|
| src/utils.py | ready | full_value_counts, get_relative_trigger |
| src/features.py | ready | load, merge, clean, engineer, split functions |
| src/graph.py | ready | entity graph, graph features, subgraph viz |
| src/model.py | ready | train functions for all 4 models, gpu detection |
| src/selection.py | ready | 6-method feature selection stack |
| src/tuning.py | ready | optuna objectives for all 3 models |
| src/evaluation.py | ready | metrics, comparison plots |
| src/mrm.py | ready | ks, psi, uniform score transform |
| src/explainability.py | ready | weighted ensemble shap |
| src/experiment.py | ready | json-based experiment logger |
| src/rules.py | ready | score-based decision strategy |
| config.py | needs update | make dynamic, accept runtime params |

---

## what needs to be built

### 1. dynamic config

current config.py uses hardcoded paths and constants.
needs to accept dataset-specific params at runtime:

```python
def load_config(data_path, target_col, date_col, id_col,
                output_dir='outputs/'):
    return {
        'DATA_PATH': Path(data_path).parent,
        'TARGET': target_col,
        'DATE_COL': date_col,
        'ID_COL': id_col,
        'OUTPUT_PATH': Path(output_dir),
        'RESULTS_PATH': Path(output_dir) / 'results',
        'PLOTS_PATH': Path(output_dir) / 'plots',
        'RANDOM_STATE': 42,
        'NULL_THRESHOLD': 0.80,
        'RULES_TOP_N_FEATURES': 10,
        'STRATEGY_APPROVE_THRESHOLD': 400,
        'STRATEGY_DECLINE_THRESHOLD': 700,
    }
```

### 2. pipeline runner

single script that calls each src/ module in sequence:

```python
# run_pipeline.py
def run_pipeline(config):
    # stage 1 - data
    df = load_and_clean(config)
    
    # stage 2 - features
    df = engineer_features(df, config)
    
    # stage 3 - split
    splits = split_data(df, config)
    
    # stage 4 - baseline
    baseline = train_baseline(*splits, config)
    
    # stage 5 - models
    models = train_all_models(*splits, config)
    
    # stage 6 - feature selection
    splits_selected, selected_features = select_features(
        models, *splits, config)
    
    # stage 7 - tuning
    best_params = tune_all_models(*splits_selected, config)
    
    # stage 8 - final training
    final_models, ensemble = train_final(
        best_params, *splits_selected, config)
    
    # stage 9 - evaluation
    evaluate_all(final_models, ensemble, *splits_selected, config)
    
    # stage 10 - mrm
    run_mrm(ensemble, *splits_selected, config)
    
    # stage 11 - explainability
    run_shap(final_models, splits_selected[2], config)
    
    # stage 12 - decision strategy
    run_decision_strategy(ensemble, splits_selected[2], config)
    
    return {
        'best_model': max(final_models, key=lambda m: m.val_auc),
        'ensemble_auc': ensemble.val_auc,
        'selected_features': selected_features,
        'experiment_log': load_experiment_log(config)
    }
```

### 3. cli entry point

```python
# cli.py
import argparse

parser = argparse.ArgumentParser(
    description='fraud modeling pipeline')
parser.add_argument('--data',       required=True,
    help='path to dataset csv')
parser.add_argument('--target',     required=True,
    help='target column name')
parser.add_argument('--date_col',   required=True,
    help='datetime or timedelta column for chronological split')
parser.add_argument('--id_col',     required=True,
    help='unique transaction identifier column')
parser.add_argument('--output_dir', default='outputs/',
    help='output directory for results and plots')
parser.add_argument('--n_trials',   default=20, type=int,
    help='optuna trials per model (default 20, use 50 for overnight)')

args = parser.parse_args()
config = load_config(args.data, args.target,
                     args.date_col, args.id_col,
                     args.output_dir)
run_pipeline(config)
```

usage:
```bash
python cli.py \
  --data data/train.csv \
  --target isFraud \
  --date_col TransactionDT \
  --id_col TransactionID \
  --n_trials 50
```

### 4. schema detector (optional)

auto-infer config from dataset without user specifying columns:

```python
def detect_schema(df):
    # find target: binary column, low cardinality, named fraud/label/target
    # find date: datetime dtype or large integer
    # find id: high cardinality, near-unique
    # find categoricals: object dtype or low cardinality int
    return inferred_config
```

with schema detector:
```bash
python cli.py --data data/train.csv
```

---

## key design decisions to preserve

**time-based split** — always chronological, never random.
requires date_col to be passed in.

**gpu detection** — get_device() in config.py already handles this.
no changes needed.

**experiment logging** — src/experiment.py already agentic.
results saved to json, loadable as dataframe.

**uniform score transform** — train_sorted.npy fitted on train,
applied to val and test. save path needs to be dynamic.

**feature lookup** — build_feature_lookup() in src/rules.py
already dynamic. parses naming conventions, no hardcoding.

**graph entity cols** — currently hardcoded in src/graph.py:
['card1', 'card2', 'P_emaildomain', 'R_emaildomain',
 'addr1', 'addr2', 'DeviceInfo']
needs to be passed as a config param for new datasets.

---

## datasets to test generalization

**PaySim** (synthetic mobile money fraud)
- available on kaggle: kaggle datasets download -d ealaxi/paysim1
- target: isFraud
- date_col: step (time step integer)
- id_col: nameOrig
- no identity table — single file
- tests: does pipeline handle single-file datasets?

**Credit Card Fraud (ULB)**
- kaggle: kaggle datasets download -d mlg-ulb/creditcardfraud
- target: Class
- date_col: Time
- id_col: generate sequential
- only PCA features — no categoricals, no graph opportunity
- tests: does pipeline handle fully numeric datasets?

---

## suggested new repo structure

```
fraud_pipeline/
├── README.md
├── cli.py                 ← entry point
├── run_pipeline.py        ← pipeline orchestrator
├── config.py              ← dynamic config
├── requirements.txt
├── src/                   ← copied from fraud_model/src/
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
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_mrm.py
└── examples/
    ├── ieee_cis/          ← example run on ieee-cis
    └── paysim/            ← example run on paysim
```

---

## things to watch for on new datasets

**no identity table** — merge_data() assumes two tables.
needs a single-table code path.

**different null patterns** — NULL_THRESHOLD may need tuning.

**different class imbalance** — scale_pos_weight computed
dynamically from y_train so this is already handled.

**no timedelta column** — convert_datetime() assumes TransactionDT.
needs to handle generic datetime or integer columns.

**different entity columns** — graph construction entity_cols
currently hardcoded. needs to be inferred or passed as config.

**categorical encoding** — encode_categoricals() uses a hardcoded
cat_cols list. needs to detect object dtype columns dynamically.

---

```
