import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_aggregation_features(df):
    """
    Adds entity-level aggregation features using global statistics on time-sorted data.

    Features are computed after sorting by TransactionDT ascending. Column naming
    follows the pattern entity__target__agg.

    Parameters:
    - df: DataFrame with entity and transaction columns

    Returns:
    - df with aggregation feature columns added
    """
    # global statistics on time-sorted data; production would use only pre-transaction history
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    entity_target_agg = [
        ('card1', 'addr1', 'nunique'),
        ('card1', 'P_emaildomain', 'nunique'),
        ('card1', 'R_emaildomain', 'nunique'),
        ('card1', 'TransactionAmt', 'mean'),
        ('card1', 'TransactionAmt', 'std'),
        ('card1', 'TransactionID', 'count'),
        ('P_emaildomain', 'card1', 'nunique'),
        ('P_emaildomain', 'addr1', 'nunique'),
        ('P_emaildomain', 'R_emaildomain', 'nunique'),
        ('P_emaildomain', 'TransactionAmt', 'mean'),
        ('R_emaildomain', 'card1', 'nunique'),
        ('R_emaildomain', 'addr1', 'nunique'),
        ('R_emaildomain', 'P_emaildomain', 'nunique'),
        ('R_emaildomain', 'TransactionAmt', 'mean'),
        ('addr1', 'card1', 'nunique'),
        ('addr1', 'P_emaildomain', 'nunique'),
        ('addr1', 'R_emaildomain', 'nunique'),
        ('addr1', 'TransactionAmt', 'mean'),
        ('DeviceInfo', 'card1', 'nunique'),
        ('DeviceInfo', 'P_emaildomain', 'nunique'),
        ('DeviceInfo', 'addr1', 'nunique'),
    ]

    for entity, target, agg in entity_target_agg:
        col_name = f'{entity}__{target}__{agg}'
        if agg == 'nunique':
            mapping = df.groupby(entity)[target].nunique()
        elif agg == 'mean':
            mapping = df.groupby(entity)[target].mean()
        elif agg == 'std':
            mapping = df.groupby(entity)[target].std()
        elif agg == 'count':
            mapping = df.groupby(entity)[target].count()
        df[col_name] = df[entity].map(mapping)

    return df


def add_missingness_features(df):
    """
    Adds a binary flag indicating whether identity data is present for a transaction.

    Parameters:
    - df: DataFrame with DeviceType column from the merged identity table

    Returns:
    - df with has_identity column added
    """
    df = df.copy()
    df['has_identity'] = df['DeviceType'].notna().astype(int)
    return df


def add_transformations(df):
    """
    Adds log-transformed versions of TransactionAmt and all aggregation feature columns.

    Uses np.log1p to handle zero values safely. New columns are saved alongside originals
    using the naming convention {col}_log.

    Parameters:
    - df: DataFrame with TransactionAmt and aggregation columns (names containing '__')

    Returns:
    - df with log-transformed columns added
    """
    df = df.copy()
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    for col in [c for c in df.columns if '__' in c]:
        df[f'{col}_log'] = np.log1p(df[col])
    return df


def encode_categoricals(df, cat_cols):
    """
    Label-encodes categorical columns, storing results in new {col}_encoded columns.

    Nulls are filled with the string 'missing' before encoding so the encoder does not
    fail on NaN values. Original columns are not modified.

    Parameters:
    - df: DataFrame containing the columns to encode
    - cat_cols: list of column names to encode

    Returns:
    - df with {col}_encoded columns added for each column in cat_cols
    """
    df = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        filled = df[col].fillna('missing').astype(str)
        df[f'{col}_encoded'] = le.fit_transform(filled)
    return df


def load_data(transaction_path, identity_path):
    """
    Loads the raw IEEE-CIS transaction and identity CSVs.

    Parameters:
    - transaction_path: path to train_transaction.csv
    - identity_path: path to train_identity.csv

    Returns:
    - df_transaction, df_identity: two DataFrames
    """
    df_transaction = pd.read_csv(transaction_path)
    df_identity = pd.read_csv(identity_path)
    return df_transaction, df_identity


def merge_data(df_transaction, df_identity):
    """
    Left joins identity onto transaction data on TransactionID.

    Prints shape before and after the join and the count of rows
    with no identity match.

    Parameters:
    - df_transaction: transaction DataFrame
    - df_identity: identity DataFrame

    Returns:
    - merged DataFrame
    """
    print(f'transaction shape: {df_transaction.shape}')
    print(f'identity shape:    {df_identity.shape}')

    df = df_transaction.merge(df_identity, on='TransactionID', how='left')

    print(f'merged shape:      {df.shape}')
    n_missing = df['DeviceType'].isna().sum()
    print(f'rows without identity data: {n_missing:,}')

    return df


def drop_null_cols(df, threshold):
    """
    Drops columns where the fraction of null values exceeds threshold.

    Parameters:
    - df: input DataFrame
    - threshold: float (0-1); columns with null fraction above this are dropped

    Returns:
    - DataFrame with high-null columns removed
    """
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f'columns dropped:   {len(cols_to_drop)}')
    print(f'columns remaining: {df.shape[1]}')
    return df


def convert_datetime(df):
    """
    Converts TransactionDT from seconds elapsed to datetime and extracts time features.

    TransactionDT in the IEEE-CIS dataset is seconds elapsed from a reference date
    (~2017-11-30). Adds hour_of_day and day_of_week columns.

    Parameters:
    - df: DataFrame containing a TransactionDT column

    Returns:
    - DataFrame with TransactionDT as datetime, plus hour_of_day and day_of_week
    """
    START_DATE = pd.Timestamp('2017-11-30')
    df = df.copy()
    df['TransactionDT'] = START_DATE + pd.to_timedelta(df['TransactionDT'], unit='s')
    df['hour_of_day'] = df['TransactionDT'].dt.hour
    df['day_of_week'] = df['TransactionDT'].dt.dayofweek
    return df


def split_data(df, target):
    """
    Splits df into train, validation, and test sets using a chronological cutoff.

    Rows are sorted by TransactionDT ascending before splitting to avoid data leakage.
    Split proportions: 70% train, 15% val, 15% test.
    TransactionID and TransactionDT are excluded from feature matrices.

    Parameters:
    - df: DataFrame containing all features, the target column, TransactionID, TransactionDT
    - target: name of the target column

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    exclude = {target, 'TransactionID', 'TransactionDT'}
    feature_cols = [c for c in df.columns if c not in exclude]

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_train, y_train = train[feature_cols], train[target]
    X_val,   y_val   = val[feature_cols],   val[target]
    X_test,  y_test  = test[feature_cols],  test[target]

    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        print(f'{name:<6} shape: {X.shape}  fraud rate: {y.mean():.4f}')

    return X_train, X_val, X_test, y_train, y_val, y_test
