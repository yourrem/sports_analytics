"""Fetch NBA game and player data via nba_api."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog, teamgamelog
from nba_api.stats.static import teams

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_SLEEP = 0.6


def fetch_all_teams() -> pd.DataFrame:
    """Return all 30 NBA teams and save to data/raw/teams.csv."""
    path = _RAW_DIR / "teams.csv"
    if path.exists():
        return pd.read_csv(path)
    df = pd.DataFrame(teams.get_teams())
    df.to_csv(path, index=False)
    return df


def fetch_team_game_logs(team_id: int, season: str) -> pd.DataFrame:
    """Fetch per-game box scores for one team/season (e.g. season='2023-24')."""
    path = _RAW_DIR / f"team_game_logs_{team_id}_{season}.csv"
    if path.exists():
        return pd.read_csv(path)
    logs = teamgamelog.TeamGameLog(team_id=team_id, season=season)
    time.sleep(_SLEEP)
    df = logs.get_data_frames()[0]
    df.to_csv(path, index=False)
    return df


def fetch_team_roster(team_id: int, season: str) -> pd.DataFrame:
    """Fetch active roster for one team/season."""
    path = _RAW_DIR / f"roster_{team_id}_{season}.csv"
    if path.exists():
        return pd.read_csv(path)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    time.sleep(_SLEEP)
    df = roster.get_data_frames()[0]
    df.to_csv(path, index=False)
    return df


def fetch_player_stats(player_id: int, season: str) -> pd.DataFrame:
    """Fetch per-game stats for one player/season."""
    path = _RAW_DIR / f"player_game_logs_{player_id}_{season}.csv"
    if path.exists():
        return pd.read_csv(path)
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    time.sleep(_SLEEP)
    df = logs.get_data_frames()[0]
    df.to_csv(path, index=False)
    return df
