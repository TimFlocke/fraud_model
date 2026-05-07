import pandas as pd
import numpy as np


def full_value_counts(df, col):
    """
    combines value counts with normalized counts
    """
    a = df[col].value_counts().to_frame('cnt')
    b = df[col].value_counts(normalize=True).to_frame('pct')

    return a.merge(b, left_index=True, right_index=True)


def get_relative_trigger(df, target, vars, upper_thresh, lower_thresh):
    """
    Create a table showing relative trigger rate for fraud
    vs non fraud
    """
    
    a = df.query(f"{target} == 1")[vars].mean().to_frame(target)
    b = df.query(f"{target} == 0")[vars].mean().to_frame("not_" +target)

    c = a.merge(b, left_index=True, right_index=True)
    c['ix'] = np.round((c[target] / c["not_"+target])*100)

    c['n_fraud'] = df.query(f"{target} == 1")[vars].sum().astype(int)
    c['n_non_fraud'] = df.query(f"{target} == 0")[vars].sum().astype(int)

    out = c[(c.ix < lower_thresh) | (c.ix > upper_thresh)]
    return out.sort_values('ix', ascending = False)
