import numpy as np
import pandas as pd

from src.tfm import build_balanced_context, correct_odds, nan_flag_cols, prep_tfm, recall_at_alert


def _pool(n=10000, fraud=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, fraud, replace=False)] = 1
    return pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n), 'y': y})


def test_balanced_context_keeps_all_fraud_when_under_cap():
    pool = _pool()
    ctx, k = build_balanced_context(pool, ['a', 'b'], 'y', n_ctx=5000, fraud_rate=0.2, seed=1)
    assert ctx['y'].sum() == 300
    assert len(ctx) == 1500
    assert np.isclose(k, 1200 / 9700)


def test_balanced_context_caps_fraud_and_rows():
    pool = _pool(fraud=3000)
    ctx, k = build_balanced_context(pool, ['a', 'b'], 'y', n_ctx=5000, fraud_rate=0.2, seed=1)
    assert len(ctx) == 5000 and ctx['y'].sum() == 1000
    assert np.isclose(k, (4000 / 7000) / (1000 / 3000))


def test_random_context():
    pool = _pool()
    ctx, k = build_balanced_context(pool, ['a'], 'y', n_ctx=2000, fraud_rate=None, seed=1)
    assert len(ctx) == 2000 and k == 1.0 and list(ctx.columns) == ['a', 'y']


def test_correct_odds_restores_base_rate():
    p_ctx = 0.2
    keep = 1200 / 9700
    p = correct_odds([p_ctx], keep)[0]
    assert np.isclose(p, 300 / 10000)
    assert np.allclose(correct_odds([0.3, 0.7], 1.0), [0.3, 0.7])


def test_recall_at_alert():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    s = np.arange(10)[::-1].astype(float)
    assert recall_at_alert(y, s, 0.1) == 1 / 3
    assert recall_at_alert(y, s, 0.4) == 2 / 3


def test_prep_tfm_flags():
    df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan], 'b': [1, 2, 3, 4]})
    flags = nan_flag_cols(df, ['a', 'b'])
    out = prep_tfm(df, ['a', 'b'], flags)
    assert flags == ['a']
    assert list(out.columns) == ['a', 'b', 'a_isna']
    assert out['a_isna'].tolist() == [0, 1, 0, 1]


def test_save_and_load_run(tmp_path=None):
    import tempfile
    from pathlib import Path
    from src.tfm import load_cached_run, save_run
    d = Path(tmp_path or tempfile.mkdtemp())
    sp, rp = d / 'scores.parquet', d / 'runs.parquet'
    assert load_cached_run(sp, rp, 'C', 0) is None
    save_run(sp, rp, 'C', 0, np.array([0.1, 0.2, 0.3]), {'fit_s': 1.0, 'predict_s': 2.0})
    save_run(sp, rp, 'C', 1, np.array([0.4, 0.5, 0.6]), {'fit_s': 1.5, 'predict_s': 2.5})
    save_run(sp, rp, 'C', 0, np.array([0.7, 0.8, 0.9]), {'fit_s': 3.0, 'predict_s': 4.0})
    score, t = load_cached_run(sp, rp, 'C', 0)
    assert np.allclose(score, [0.7, 0.8, 0.9]) and t == {'fit_s': 3.0, 'predict_s': 4.0}
    assert len(pd.read_parquet(sp)) == 6 and len(pd.read_parquet(rp)) == 2
    assert load_cached_run(sp, rp, 'C0', 0) is None


def _bank_pool(seed=0):
    rng = np.random.default_rng(seed)
    n = 20000
    a = rng.normal(size=n)
    a[rng.random(n) < 0.1] = np.nan
    seg = (rng.random(n) < 0.3).astype(int)
    p = np.where(np.nan_to_num(a, nan=2.0) > 1, 0.15, 0.01)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({'a': a, 'b': rng.normal(size=n), 'seg': seg, 'y': y})


def test_context_bank_tree_keys_and_isolation():
    from src.tfm import build_context_bank, describe_bank, route_bank
    pool = _bank_pool()
    bank = build_context_bank(pool, ['a', 'b'], ['a', 'b'], 'y', max_depth=2, min_samples_leaf=2000,
                              n_ctx=3000, fraud_rate=0.2, min_leaf_fraud=50, seed=0)
    keys = route_bank(bank, pool)
    assert set(bank['contexts']) == set(np.unique(keys))
    for k, ctx in bank['contexts'].items():
        if bank['fallback'][k]:
            assert ctx is bank['global_ctx']
            continue
        # every context row comes from its own leaf
        assert set(ctx.index) <= set(pool.index[keys == k])
        assert len(ctx) <= 3000
    desc = describe_bank(bank)
    assert len(desc) == len(bank['contexts'])
    assert desc['pool_rows'].sum() == len(pool)


def test_context_bank_segment_and_fallback():
    from src.tfm import build_context_bank, describe_bank
    pool = _bank_pool()
    bank = build_context_bank(pool, [], ['a', 'b'], 'y', n_ctx=3000, fraud_rate=0.2,
                              min_leaf_fraud=10 ** 6, seed=0, segment_col='seg')
    assert bank['tree'] is None and set(bank['contexts']) == {0, 1}
    assert all(bank['fallback'].values())
    assert describe_bank(bank).loc[1, 'rule'] == 'seg == 1'


def test_score_with_bank_keeps_row_order(monkeypatch=None):
    import src.tfm as tfm
    pool = _bank_pool()
    bank = tfm.build_context_bank(pool, ['a', 'b'], ['a', 'b'], 'y', max_depth=2, min_samples_leaf=2000,
                                  n_ctx=3000, fraud_rate=0.2, seed=0)
    target = _bank_pool(seed=1).sample(frac=1, random_state=0)
    original = tfm.fit_predict_tfm
    # stub the tfm with a score equal to column b so the output order can be checked
    tfm.fit_predict_tfm = lambda ctx, tgt, *a, **k: (1 / (1 + np.exp(-tgt['b'].to_numpy())), {'fit_s': 0.0, 'predict_s': 0.0})
    try:
        score, _ = tfm.score_with_bank(bank, target, ['a', 'b'], 'y')
    finally:
        tfm.fit_predict_tfm = original
    keys = tfm.route_bank(bank, target)
    keep = np.array([bank['keep_rates'][k] for k in keys])
    assert np.allclose(score, tfm.correct_odds(1 / (1 + np.exp(-target['b'].to_numpy())), keep))


def test_fill_to_n_ctx():
    pool = _pool(fraud=300)
    ctx, k = build_balanced_context(pool, ['a'], 'y', n_ctx=5000, fraud_rate=0.2, seed=1, fill_to_n_ctx=True)
    assert len(ctx) == 5000 and ctx['y'].sum() == 300
    assert np.isclose(k, 4700 / 9700)
    small = _pool(n=1000, fraud=30)
    ctx, k = build_balanced_context(small, ['a'], 'y', n_ctx=5000, fraud_rate=0.2, seed=1, fill_to_n_ctx=True)
    assert len(ctx) == 1000 and k == 1.0


def test_context_bank_fill_stays_in_leaf():
    from src.tfm import build_context_bank, route_bank
    pool = _bank_pool()
    bank = build_context_bank(pool, ['a', 'b'], ['a', 'b'], 'y', max_depth=2, min_samples_leaf=2000,
                              n_ctx=3000, fraud_rate=0.2, min_leaf_fraud=10, seed=0, fill_to_n_ctx=True)
    keys = route_bank(bank, pool)
    for k, ctx in bank['contexts'].items():
        leaf = pool.index[keys == k]
        assert set(ctx.index) <= set(leaf)
        assert len(ctx) == min(3000, len(leaf))
