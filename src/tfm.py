import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def log_msg(msg):
    """prints msg with a timestamp and flushes so progress shows while a cell runs"""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def memory_report():
    """
    Returns a one-line string with free system ram and free / total gpu memory.
    """
    import psutil
    import torch

    ram = psutil.virtual_memory()
    out = f"ram free {ram.available / 1024**3:.1f} of {ram.total / 1024**3:.1f} gb"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out += f"  gpu free {free / 1024**3:.2f} of {total / 1024**3:.2f} gb"
    return out


def load_cached_run(scores_path, runs_path, run, seed):
    """
    Returns saved scores and timing for one run and seed, or None if either file
    lacks that run and seed.

    Parameters:
    - scores_path: parquet with columns run, seed, row, score
    - runs_path: parquet with columns run, seed, fit_s, predict_s
    - run: run name, e.g. 'C'
    - seed: int seed

    Returns:
    - (score array in row order, dict with fit_s and predict_s) or None
    """
    scores_path, runs_path = Path(scores_path), Path(runs_path)
    if not scores_path.exists() or not runs_path.exists():
        return None
    runs = pd.read_parquet(runs_path)
    meta = runs[(runs['run'] == run) & (runs['seed'] == seed)]
    if meta.empty:
        return None
    scores = pd.read_parquet(scores_path, filters=[('run', '==', run), ('seed', '==', seed)])
    if scores.empty:
        return None
    score = scores.sort_values('row')['score'].to_numpy()
    return score, {'fit_s': float(meta['fit_s'].iloc[0]), 'predict_s': float(meta['predict_s'].iloc[0])}


def save_run(scores_path, runs_path, run, seed, score, timing):
    """
    Upserts one run and seed into the scores and timing parquets, replacing any
    earlier rows for the same run and seed. Called after every seed so a crash
    keeps finished seeds.

    Parameters:
    - scores_path: parquet with columns run, seed, row, score
    - runs_path: parquet with columns run, seed, fit_s, predict_s
    - run: run name
    - seed: int seed
    - score: array of scores in row order
    - timing: dict with fit_s and predict_s
    """
    new_scores = pd.DataFrame({'run': run, 'seed': seed, 'row': np.arange(len(score)), 'score': score})
    new_meta = pd.DataFrame([{'run': run, 'seed': seed, **timing}])
    for path, new in [(Path(scores_path), new_scores), (Path(runs_path), new_meta)]:
        if path.exists():
            old = pd.read_parquet(path)
            old = old[~((old['run'] == run) & (old['seed'] == seed))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_parquet(path, index=False)


def build_balanced_context(pool, feats, label, n_ctx=10000, fraud_rate=0.2, seed=0, fill_to_n_ctx=False):
    """
    Samples a context of at most n_ctx rows from pool.

    With fraud_rate set: keeps all fraud up to n_ctx * fraud_rate, then adds random
    negatives to reach fraud_rate, capped at n_ctx rows in total. With fill_to_n_ctx,
    negatives are added until the context reaches n_ctx rows (or pool runs out), so the
    context fraud rate can fall below fraud_rate. With fraud_rate=None: a plain random
    sample of n_ctx rows.

    keep_rate is the sampling rate of negatives relative to fraud
    ((negatives kept / negatives in pool) / (fraud kept / fraud in pool)), which is
    the odds factor correct_odds undoes. When all fraud is kept it equals
    negatives kept / negatives in pool.

    Parameters:
    - pool: DataFrame with feats and label
    - feats: list of columns to keep
    - label: name of the binary target column
    - n_ctx: maximum context rows
    - fraud_rate: target fraud share in the context, or None for a random sample
    - seed: random seed
    - fill_to_n_ctx: fill with negatives up to n_ctx rows instead of stopping at fraud_rate

    Returns:
    - (ctx DataFrame with feats + label, keep_rate float)
    """
    rng = np.random.default_rng(seed)
    cols = list(feats) + [label]
    if fraud_rate is None:
        idx = rng.choice(len(pool), min(n_ctx, len(pool)), replace=False)
        return pool.iloc[np.sort(idx)][cols], 1.0

    y = pool[label].values
    fraud_pos, neg_pos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_fraud = min(len(fraud_pos), int(n_ctx * fraud_rate))
    n_neg = min(len(neg_pos), n_ctx - n_fraud)
    if not fill_to_n_ctx:
        n_neg = min(n_neg, int(round(n_fraud * (1 - fraud_rate) / fraud_rate)))
    keep = np.concatenate([
        rng.choice(fraud_pos, n_fraud, replace=False),
        rng.choice(neg_pos, n_neg, replace=False),
    ])
    keep_rate = (n_neg / len(neg_pos)) / (n_fraud / len(fraud_pos))
    return pool.iloc[np.sort(keep)][cols], keep_rate


def correct_odds(p, keep_rate):
    """
    Maps probabilities from a context with negatives sampled at keep_rate back to
    the pool base rate: odds_true = odds_ctx * keep_rate.

    Parameters:
    - p: array of probabilities from the tfm
    - keep_rate: negative sampling rate relative to fraud (from build_balanced_context)

    Returns:
    - array of corrected probabilities
    """
    p = np.asarray(p, dtype=np.float64)
    return p * keep_rate / (p * keep_rate + (1 - p))


def fit_predict_tfm(ctx, target, feats, label, flag_cols=(), seed=0, kv_cache='repr', device='cuda',
                    chunk_rows=20000, verbose=False):
    """
    Fits TabICL on ctx and returns fraud probabilities for target.

    Both frames go through prep_tfm (feats plus is-missing flags for flag_cols).
    kv_cache='repr' caches the context representation at fit time so predict does not
    reprocess it (about 20% faster predict on the full test set at the same gpu memory).
    target is predicted in chunks of chunk_rows: TabICL buffers its intermediate outputs
    for all rows in pinned cpu memory (~6.7 GB for 88k rows), which can fail to allocate.

    Parameters:
    - ctx: DataFrame with feats and label
    - target: DataFrame with feats
    - feats: list of model input columns
    - label: name of the binary target column in ctx
    - flag_cols: columns that get an _isna flag
    - seed: TabICL random_state (ensemble shuffles)
    - kv_cache: TabICL kv_cache setting (False, 'kv' or 'repr')
    - device: torch device
    - chunk_rows: target rows per predict call
    - verbose: print a timestamped line after fit and after each predict chunk

    Returns:
    - (array of raw probabilities, dict with fit_s and predict_s)
    """
    import torch
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, random_state=seed, kv_cache=kv_cache)
    t0 = time.perf_counter()
    clf.fit(prep_tfm(ctx, feats, flag_cols), ctx[label].values)
    t1 = time.perf_counter()
    if verbose:
        log_msg(f"  fit done {t1 - t0:.1f}s  ctx {len(ctx):,} rows")
    X_target = prep_tfm(target, feats, flag_cols)
    starts = range(0, len(X_target), chunk_rows)
    parts = []
    for i, start in enumerate(starts, 1):
        parts.append(clf.predict_proba(X_target.iloc[start:start + chunk_rows])[:, 1])
        if verbose:
            log_msg(f"  predict chunk {i}/{len(starts)}  rows {start + len(parts[-1]):,}/{len(X_target):,}  "
                    f"{time.perf_counter() - t1:.1f}s")
    proba = np.concatenate(parts)
    t2 = time.perf_counter()
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return proba, {'fit_s': t1 - t0, 'predict_s': t2 - t1}


def build_context_bank(pool, route_feats, feats, label, max_depth=3, min_samples_leaf=5000,
                       n_ctx=10000, fraud_rate=0.2, min_leaf_fraud=50, seed=0, segment_col=None,
                       fill_to_n_ctx=False):
    """
    Builds a routed context bank: one class-balanced context per routing key.

    Keys are leaf ids of a DecisionTreeClassifier fit on pool[route_feats] (sklearn 1.8
    routes nans natively with a learned direction per split), or the values of
    segment_col when it is set (no tree). Each key's context is built from that key's
    rows only with build_balanced_context, so fraud is never mixed across keys and each
    keep_rate is the key's own negative keep rate / fraud keep rate. Keys with fewer than
    min_leaf_fraud frauds use the global context from build_balanced_context on the full pool.

    Parameters:
    - pool: DataFrame with route_feats (or segment_col), feats and label
    - route_feats: features the tree splits on (ignored when segment_col is set)
    - feats: model input columns kept in each context
    - label: binary target column
    - max_depth, min_samples_leaf: tree settings
    - n_ctx, fraud_rate: per-context settings, as in build_balanced_context
    - min_leaf_fraud: minimum frauds for a key to get its own context
    - seed: tree random_state and context sampling seed
    - segment_col: optional column whose values are the routing keys
    - fill_to_n_ctx: fill each key's context with its own negatives up to n_ctx rows
      (key fraud rate is whatever results; the global fallback context is unchanged)

    Returns:
    - dict with tree, segment_col, route_feats, nan_feats, contexts, keep_rates,
      fallback, pool_stats, global_ctx, global_keep_rate
    """
    from sklearn.tree import DecisionTreeClassifier

    tree = None
    if segment_col is None:
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        tree.fit(pool[route_feats], pool[label])
        keys = tree.apply(pool[route_feats])
    else:
        keys = pool[segment_col].to_numpy()

    global_ctx, global_keep = build_balanced_context(pool, feats, label, n_ctx=n_ctx, fraud_rate=fraud_rate, seed=seed)
    contexts, keep_rates, fallback, pool_stats = {}, {}, {}, {}
    for key in np.unique(keys):
        sub = pool[keys == key]
        n_fraud = int(sub[label].sum())
        pool_stats[key] = {'pool_rows': len(sub), 'pool_fraud': n_fraud}
        if n_fraud < min_leaf_fraud:
            contexts[key], keep_rates[key], fallback[key] = global_ctx, global_keep, True
        else:
            ctx, keep = build_balanced_context(sub, feats, label, n_ctx=n_ctx, fraud_rate=fraud_rate, seed=seed,
                                               fill_to_n_ctx=fill_to_n_ctx)
            contexts[key], keep_rates[key], fallback[key] = ctx, keep, False

    nan_feats = [] if tree is None else [f for f in route_feats if pool[f].isna().any()]
    return {
        'tree': tree, 'segment_col': segment_col, 'route_feats': list(route_feats), 'nan_feats': nan_feats,
        'contexts': contexts, 'keep_rates': keep_rates, 'fallback': fallback, 'pool_stats': pool_stats,
        'global_ctx': global_ctx, 'global_keep_rate': global_keep, 'fill_to_n_ctx': fill_to_n_ctx,
    }


def route_bank(bank, df):
    """returns the routing key for each row of df (tree leaf id or segment value)"""
    if bank['tree'] is not None:
        return bank['tree'].apply(df[bank['route_feats']])
    return df[bank['segment_col']].to_numpy()


def score_with_bank(bank, target, feats, label, flag_cols=(), seed=0, kv_cache='repr', chunk_rows=20000,
                    verbose=False):
    """
    Scores target with a context bank: each row is routed to its key, scored by TabICL
    against that key's context, and odds corrected with that key's keep_rate so scores
    from different keys share the pool base-rate scale. Keys not seen in the pool use
    the global context.

    Parameters:
    - bank: dict from build_context_bank
    - target: DataFrame with feats and the routing columns
    - feats, label, flag_cols, seed, kv_cache, chunk_rows: passed to fit_predict_tfm
    - verbose: timestamped line per key (and per predict chunk)

    Returns:
    - (score array in target row order, dict with fit_s, predict_s and per_key timing)
    """
    import gc

    import torch

    keys = route_bank(bank, target)
    out = np.full(len(target), np.nan)
    timing = {'fit_s': 0.0, 'predict_s': 0.0, 'per_key': {}}
    uniq = np.unique(keys)
    for i, key in enumerate(uniq, 1):
        idx = np.flatnonzero(keys == key)
        ctx = bank['contexts'].get(key, bank['global_ctx'])
        keep = bank['keep_rates'].get(key, bank['global_keep_rate'])
        if verbose:
            log_msg(f" key {key} ({i}/{len(uniq)})  target rows {len(idx):,}  ctx rows {len(ctx):,}  "
                    f"fallback {bank['fallback'].get(key, True)}")
        raw, t = fit_predict_tfm(ctx, target.iloc[idx], feats, label, flag_cols, seed=seed,
                                 kv_cache=kv_cache, chunk_rows=chunk_rows, verbose=verbose)
        out[idx] = correct_odds(raw, keep)
        timing['fit_s'] += t['fit_s']
        timing['predict_s'] += t['predict_s']
        timing['per_key'][key] = t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert len(out) == len(target) and not np.isnan(out).any(), 'bank scoring dropped rows'
    return out, timing


def _leaf_rules(tree, feature_names, nan_feats):
    """returns {leaf id: rule text} by walking each leaf's path from the root"""
    t = tree.tree_
    parent = {}
    for node in range(t.node_count):
        for child, side in [(t.children_left[node], 'left'), (t.children_right[node], 'right')]:
            if child != -1:
                parent[child] = (node, side)
    rules = {}
    for leaf in np.flatnonzero(t.children_left == -1):
        conds, node = [], leaf
        while node in parent:
            p, side = parent[node]
            name = feature_names[t.feature[p]]
            cond = f"{name} {'<=' if side == 'left' else '>'} {t.threshold[p]:.4g}"
            nan_left = bool(t.missing_go_to_left[p])
            if name in nan_feats and nan_left == (side == 'left'):
                cond = f'({cond} or nan)'
            conds.append(cond)
            node = p
        rules[leaf] = ' and '.join(reversed(conds))
    return rules


def describe_bank(bank, route_feats=None):
    """
    One row per routing key: rule text, pool rows and fraud rate, context rows and
    fraud rate, keep_rate and whether the key uses the global fallback context.

    Parameters:
    - bank: dict from build_context_bank
    - route_feats: feature names for the rule text (defaults to the bank's route_feats)

    Returns:
    - DataFrame indexed by key
    """
    route_feats = route_feats or bank['route_feats']
    if bank['tree'] is not None:
        rules = _leaf_rules(bank['tree'], route_feats, bank['nan_feats'])
    else:
        rules = {k: f"{bank['segment_col']} == {k}" for k in bank['contexts']}
    rows = []
    for key, stats in bank['pool_stats'].items():
        ctx = bank['contexts'][key]
        label_col = ctx.columns[-1]
        rows.append({
            'key': key, 'rule': rules[key],
            'pool_rows': stats['pool_rows'], 'pool_fraud_rate': stats['pool_fraud'] / stats['pool_rows'],
            'ctx_rows': len(ctx), 'ctx_fraud_rate': ctx[label_col].mean(),
            'keep_rate': bank['keep_rates'][key], 'fallback': 'yes' if bank['fallback'][key] else 'no',
        })
    return pd.DataFrame(rows).set_index('key')


def fit_predict_stacked(pool, target, feats, label, flag_cols=(), fit_frac=0.8, seed=0):
    """
    Control for the tfm runs: a CatBoost model trained on the same pool and inputs.

    pool must be in time order. The first fit_frac rows are used to fit and the rest
    only for early stopping. scale_pos_weight is set from the fit rows. No tuning.

    Parameters:
    - pool: time-ordered DataFrame with feats and label
    - target: DataFrame with feats
    - feats: list of model input columns
    - label: name of the binary target column in pool
    - flag_cols: columns that get an _isna flag (same inputs as the tfm)
    - fit_frac: share of pool rows used to fit
    - seed: catboost random_seed

    Returns:
    - (array of probabilities, dict with fit_s, predict_s, best_iteration, fit_rows, es_rows)
    """
    from catboost import CatBoostClassifier

    from config import get_device

    cut = int(len(pool) * fit_frac)
    X = prep_tfm(pool, feats, flag_cols)
    y = pool[label].values
    X_fit, y_fit, X_es, y_es = X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        early_stopping_rounds=100,
        scale_pos_weight=float((y_fit == 0).sum()) / float((y_fit == 1).sum()),
        random_seed=seed,
        task_type='GPU' if get_device() == 'gpu' else 'CPU',
        verbose=0,
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    t1 = time.perf_counter()
    proba = model.predict_proba(prep_tfm(target, feats, flag_cols))[:, 1]
    t2 = time.perf_counter()
    return proba, {
        'fit_s': t1 - t0, 'predict_s': t2 - t1, 'best_iteration': int(model.get_best_iteration()),
        'fit_rows': cut, 'es_rows': len(pool) - cut,
    }


def recall_at_alert(y, score, rate):
    """
    Share of all fraud captured in the top rate share of rows by score.

    Parameters:
    - y: binary labels
    - score: scores, higher means more likely fraud
    - rate: alert rate, e.g. 0.01 for the top 1%

    Returns:
    - recall as a float
    """
    y = np.asarray(y)
    k = int(np.ceil(rate * len(y)))
    top = np.argsort(-np.asarray(score), kind='stable')[:k]
    return float(y[top].sum() / y.sum())


def nan_flag_cols(pool, feats, min_rate=0.01):
    """
    Returns the features in feats whose missing rate in pool exceeds min_rate.

    TabICL mean-imputes numeric NaNs, which drops the missingness signal.
    These columns get an explicit is-missing indicator in prep_tfm.

    Parameters:
    - pool: DataFrame the rates are measured on (the context pool)
    - feats: list of candidate feature names
    - min_rate: minimum missing rate to get a flag

    Returns:
    - list of column names
    """
    rates = pool[feats].isna().mean()
    return [c for c in feats if rates[c] > min_rate]


def prep_tfm(df, feats, flag_cols):
    """
    Builds the TabICL input matrix: feats as float32 plus one 0/1 is-missing
    flag per column in flag_cols. NaNs are left in place for TabICL to impute.

    Label-encoded categoricals stay numeric (they are int columns, which TabICL
    treats as continuous, the same way the tree models saw them).

    Parameters:
    - df: DataFrame containing feats
    - feats: list of feature names, including prior_score if used
    - flag_cols: list of columns that get a <col>_isna flag

    Returns:
    - float32 DataFrame with the same index as df
    """
    out = df[feats].astype(np.float32)
    flags = pd.DataFrame(
        {f'{c}_isna': df[c].isna().astype(np.float32) for c in flag_cols},
        index=df.index,
    )
    return pd.concat([out, flags], axis=1)
