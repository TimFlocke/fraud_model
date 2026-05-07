import pandas as pd


def full_value_counts(df, col):
    """
    combines value counts with normalized counts
    """
    a = df[col].value_counts().to_frame('cnt')
    b = df[col].value_counts(normalize=True).to_frame('pct')

    return a.merge(b, left_index=True, right_index=True)
