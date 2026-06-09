"""Feature construction for sports prediction models."""

import pandas as pd


def add_rolling_averages(df: pd.DataFrame, stat_cols: list[str], window: int) -> pd.DataFrame:
    pass


def add_rest_days(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    pass


def add_home_away_flag(df: pd.DataFrame, matchup_col: str) -> pd.DataFrame:
    pass
