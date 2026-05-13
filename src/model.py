import config
from sklearn.tree import DecisionTreeClassifier


def train_baseline(X_train, y_train):
    """
    Trains a decision tree classifier with balanced class weights as a baseline model.

    Parameters:
    - X_train: feature matrix
    - y_train: binary target series

    Returns:
    - fitted DecisionTreeClassifier
    """
    model = DecisionTreeClassifier(
        max_depth=10,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)
    return model
