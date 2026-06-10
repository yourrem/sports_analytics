"""Feature construction for sports prediction models."""

import pandas as pd


def add_rolling_averages(
    df: pd.DataFrame,
    stat_cols: list[str],
    window: int,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """Add rolling mean of each stat column over the previous `window` games.

    Column names are suffixed with _roll{window} (e.g. PTS_roll5).
    shift(1) ensures the current game is never included (no data leakage).
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    for col in stat_cols:
        df[f"{col}_roll{window}"] = (
            df[col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
        )

    df = df.drop(columns=["_date_parsed"])
    return df


def add_rest_days(df: pd.DataFrame, date_col: str = "GAME_DATE") -> pd.DataFrame:
    """Add rest_days column: calendar days since the team's previous game.

    First game of the season gets rest_days = 0.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    df["rest_days"] = df["_date_parsed"].diff().dt.days.fillna(0).astype(int)

    df = df.drop(columns=["_date_parsed"])
    return df


def add_home_away_flag(df: pd.DataFrame, matchup_col: str = "MATCHUP") -> pd.DataFrame:
    """Add is_home column: 1 if home game ('vs.'), 0 if away ('@')."""
    df = df.copy()
    df["is_home"] = df[matchup_col].str.contains(r"vs\.", regex=True).astype(int)
    return df


def add_win_streak(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    wl_col: str = "WL",
) -> pd.DataFrame:
    """Add win_streak column: consecutive-game streak going INTO each game.

    Positive = win streak, negative = loss streak, 0 = no prior games.
    Sorted by date ascending; original row order is restored afterward.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    streak = 0
    streaks = []
    for w in (df[wl_col] == "W"):
        streaks.append(streak)
        streak = (max(streak, 0) + 1) if w else (min(streak, 0) - 1)

    df["win_streak"] = streaks
    df = df.drop(columns=["_date_parsed"])
    return df


def add_last_10_wins(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    wl_col: str = "WL",
) -> pd.DataFrame:
    """Add last_10_w column: wins in the 10 games before each game (0–10 integer).

    Uses shift(1) so the current game result is never included.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    win = (df[wl_col] == "W").astype(int)
    df["last_10_w"] = (
        win.shift(1).rolling(window=10, min_periods=1).sum().fillna(0).astype(int)
    )

    df = df.drop(columns=["_date_parsed"])
    return df
