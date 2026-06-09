"""Metrics and SHAP plot helpers for model evaluation."""

import pandas as pd
import matplotlib.pyplot as plt


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    pass


def plot_shap_summary(model, X: pd.DataFrame) -> plt.Figure:
    pass


def plot_feature_importance(model, feature_names: list[str]) -> plt.Figure:
    pass
