import pandas as pd


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
