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


def add_season_record(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    w_col: str = "W",
    l_col: str = "L",
) -> pd.DataFrame:
    """Add season_wins and season_losses: cumulative record going INTO each game.

    The raw W/L columns reflect the record after the game, so shift(1) is
    applied to avoid leakage. First game of the season gets 0/0.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    df["season_wins"]   = df[w_col].shift(1).fillna(0).astype(int)
    df["season_losses"] = df[l_col].shift(1).fillna(0).astype(int)

    df = df.drop(columns=["_date_parsed"])
    return df


def add_opponent_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    matchup_col: str = "MATCHUP",
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """Add opponent situational features and per-stat rolling differentials.

    Must be called on the combined DataFrame (all teams) after per-team feature
    functions have already been applied.  Joins each row against its opponent's
    pre-computed roll5 stats on the same game date, then computes:
      - opp_rest_days, opp_win_streak, opp_last_10_w
      - {stat}_diff_roll5 = team_roll5 − opp_roll5  (for each stat in stat_cols)

    Team and opponent abbreviations are both parsed from MATCHUP
    (e.g. "ATL vs. HOU" → team=ATL, opp=HOU; "ATL @ IND" → team=ATL, opp=IND).
    Unmatched rows are filled with 0.
    """
    df = df.copy()

    # Team abbreviation is always the first token in MATCHUP
    df["_team_abbr"] = df[matchup_col].str.split(r"\s+").str[0]
    df["_opp_abbr"]  = df[matchup_col].str.extract(r"(?:vs\.|@)\s+(\w+)")

    roll5_cols = [f"{c}_roll5" for c in stat_cols]
    situational_cols = ["rest_days", "win_streak", "last_10_w"]
    lookup_cols = [date_col, "_team_abbr"] + situational_cols + roll5_cols

    lookup = df[lookup_cols].rename(
        columns={
            "_team_abbr": "_opp_abbr",
            "rest_days": "opp_rest_days",
            "win_streak": "opp_win_streak",
            "last_10_w": "opp_last_10_w",
            **{c: f"_opp_{c}" for c in roll5_cols},
        }
    )

    df = df.merge(lookup, on=[date_col, "_opp_abbr"], how="left")

    opp_roll5_cols = [f"_opp_{c}_roll5" for c in stat_cols]
    for stat, opp_col in zip(stat_cols, opp_roll5_cols):
        df[f"{stat}_diff_roll5"] = df[f"{stat}_roll5"].sub(df[opp_col]).fillna(0)

    df = df.drop(columns=["_team_abbr", "_opp_abbr"] + opp_roll5_cols)

    opp_situational = ["opp_rest_days", "opp_win_streak", "opp_last_10_w"]
    df[opp_situational] = df[opp_situational].fillna(0)

    return df
