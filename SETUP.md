# fraud_model — project setup manual

A step by step guide to reproducing the project environment from scratch.

---

## prerequisites

- Python 3.9+ installed
- Git installed
- VS Code installed
- VS Code extensions: **Jupyter**, **Python**, **Claude Code**
- Kaggle account (for data download)

---

## step 1 — create github repo

- go to github.com
- click `+` → new repository
- name: `fraud_model`
- visibility: public
- check: add a README file
- check: add .gitignore → select `Python` from dropdown
- click create repository

---

## step 2 — clone into vs code

- in your new github repo click the green **code** button
- copy the https url
- in vs code: `ctrl+shift+p` → type `git: clone` → paste the url
- choose where to save locally
- when prompted: click **open** (not add to workspace)
- when asked to save workspace configuration: click **don't save**

---

## step 3 — create virtual environment

open the vs code terminal (`ctrl + backtick`) and run:

```bash
python -m venv fraud_env
```

a `fraud_env/` folder will appear in your project directory.

---

## step 4 — activate virtual environment (windows)

```bash
fraud_env\Scripts\activate
```

you should see `(fraud_env)` at the start of your terminal prompt:

```
(fraud_env) PS C:\Users\you\fraud_model>
```

**if you get a permission error:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then activate again.

---

## step 5 — update .gitignore

open `.gitignore` and add these lines at the bottom:

```
# virtual environment
fraud_env/

# data - never commit raw data
data/

# model artifacts
outputs/models/
```

save the file.

---

## step 6 — install packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel catboost xgboost lightgbm shap optuna networkx pyarrow
```

this covers all project stages. takes a few minutes.

---

## step 7 — register jupyter kernel

```bash
python -m ipykernel install --user --name=fraud_env --display-name "Python (fraud_env)"
```

this makes `fraud_env` selectable as a kernel in notebooks.

to verify it worked, open any `.ipynb` file, click **select kernel** in the top right, and confirm `Python (fraud_env)` appears.

---

## step 8 — download ieee-cis data

- go to: `https://www.kaggle.com/c/ieee-fraud-detection/data`
- log in to kaggle
- click **join competition** to accept the rules if not already done
- download these two files:
  - `train_transaction.csv`
  - `train_identity.csv`
- move both files into your `data/` folder

---

## step 9 — scaffold repo structure

run each mkdir individually (powershell requirement):

```bash
mkdir notebooks
mkdir outputs
mkdir outputs\models
mkdir outputs\plots
mkdir outputs\results
mkdir tests
mkdir src
```

create all project files:

```bash
New-Item data\.gitkeep -ItemType File
New-Item outputs\models\.gitkeep -ItemType File
New-Item outputs\plots\.gitkeep -ItemType File
New-Item outputs\results\.gitkeep -ItemType File
New-Item src\__init__.py -ItemType File
New-Item src\utils.py -ItemType File
New-Item src\features.py -ItemType File
New-Item src\graph.py -ItemType File
New-Item src\model.py -ItemType File
New-Item src\selection.py -ItemType File
New-Item src\tuning.py -ItemType File
New-Item src\evaluation.py -ItemType File
New-Item src\mrm.py -ItemType File
New-Item src\explainability.py -ItemType File
New-Item src\rules.py -ItemType File
New-Item config.py -ItemType File
New-Item requirements.txt -ItemType File
New-Item tests\test_features.py -ItemType File
New-Item notebooks\01_data_creation.ipynb -ItemType File
New-Item notebooks\02_eda.ipynb -ItemType File
New-Item notebooks\03_feature_engineering.ipynb -ItemType File
New-Item notebooks\04_train_test_split.ipynb -ItemType File
New-Item notebooks\05_baseline.ipynb -ItemType File
New-Item notebooks\06_model_training.ipynb -ItemType File
New-Item notebooks\07_feature_selection.ipynb -ItemType File
New-Item notebooks\08_tuning.ipynb -ItemType File
New-Item notebooks\09_final_training.ipynb -ItemType File
New-Item notebooks\10_evaluation.ipynb -ItemType File
New-Item notebooks\11_mrm.ipynb -ItemType File
New-Item notebooks\12_explainability.ipynb -ItemType File
New-Item notebooks\13_fraud_rules.ipynb -ItemType File
```

---

## step 10 — silence line ending warnings (windows)

```bash
git config --global core.autocrlf true
```

run once. prevents LF/CRLF warnings on every commit.

---

## step 11 — first git commit

```bash
git add .
git commit -m "initial project scaffold"
git push origin main
```

verify on github — full folder structure should be visible.

---

## step 12 — generate requirements.txt

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "add requirements"
git push origin main
```

---

## step 13 — migrate existing functions into src/

if you have existing utility functions, map them to the correct src/ file:

| function | destination |
|---|---|
| `full_value_counts()` | `src/utils.py` |
| `get_mult_roc()` | `src/evaluation.py` |
| `display_mod_performance()` | `src/evaluation.py` |
| `get_pr_curve()` | `src/evaluation.py` |
| `evaluate_performance()` | `src/evaluation.py` |
| `get_gains()` | `src/mrm.py` |
| `score_deciles()` | `src/mrm.py` |

use claude code to migrate them with this prompt pattern:

```
read [source_file.py] and copy [function_name()] into [src/target.py].
add appropriate imports at the top. add a docstring if missing.
do not modify the function logic.
```

commit when done:

```bash
git add .
git commit -m "migrate existing functions into src modules"
git push origin main
```

---

## project structure (final)

```
fraud_model/
├── .gitignore
├── README.md
├── SETUP.md
├── config.py
├── requirements.txt
├── data/
│   ├── .gitkeep
│   ├── train_transaction.csv     ← gitignored
│   └── train_identity.csv        ← gitignored
├── src/
│   ├── __init__.py
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
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
└── tests/
    └── test_features.py
```

---

## git workflow (ongoing)

after completing each stage:

```bash
git add .
git commit -m "complete stage X — brief description"
git push origin main
```

commit after every stage. never commit data files or fraud_env/.

---

## useful commands

| task | command |
|---|---|
| activate environment | `fraud_env\Scripts\activate` |
| deactivate environment | `deactivate` |
| check installed packages | `pip list` |
| update requirements.txt | `pip freeze > requirements.txt` |
| check git status | `git status` |
| view commit history | `git log --oneline` |
| check remote url | `git remote -v` |
