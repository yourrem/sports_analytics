"""Train and predict wrappers for the sports outcome model."""

import pandas as pd


def train(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None):
    pass


def predict(model, X: pd.DataFrame) -> pd.Series:
    pass


def save_model(model, path: str) -> None:
    pass


def load_model(path: str):
    pass
