# Sports Analytics — NBA Game Outcome Prediction

## Overview

This project aims to build a machine learning model that can predict the outcome of a given NBA matchup. All team data is pulled from public NBA APIs. We used pulled data to then feature engineer columns such as rolling team stats, rest-day features, and home/away splits across 5 seasons (2020-21 through 2024-25). Model will then be used to predict outcomes of the latter half of the 2025-2026 NBA seaason.

**Data:** [nba_api](https://github.com/swar/nba_api) — official NBA stats, no API key required

## Setup

```bash
pip install -r requirements.txt
```

## Notebooks

Run in order:

| Notebook | Description |
|---|---|
| `01_data_collection.ipynb` | Fetch team game logs, rosters, and player stats from nba_api and save to `data/raw/` |
| `02_eda.ipynb` | Distributions, correlations, class balance |
| `03_feature_engineering.ipynb` | Build feature matrix, save to `data/processed/features.csv` |
| `04_modeling.ipynb` | Train XGBoost/LightGBM, cross-validation, hyperparameter tuning |
| `05_shap_analysis.ipynb` | SHAP summary, dependence plots, feature importance |

> `data/processed/features.csv` is committed so you can skip notebook 01 and start from notebook 03 without re-fetching the API data.

## Features Engineered

| Feature | Description |
|---|---|
| `is_home` | 1 = home game, 0 = away |
| `rest_days` | Calendar days since previous game (0 for season opener) |
| `win_streak` | Consecutive W/L streak going into the game (+win, −loss) |
| `last_10_w` | Wins in the previous 10 games (0–10) |
| `{stat}_roll5` | 5-game rolling average of the stat before the game |
| `{stat}_roll10` | 10-game rolling average of the stat before the game |

Rolling stats cover: `PTS`, `FG_PCT`, `FG3_PCT`, `FT_PCT`, `REB`, `AST`, `TOV`, `STL`, `BLK` — 22 feature columns total. All rolling features use `shift(1)` to prevent data leakage.

## Project Structure

```
data/raw/          — raw API responses (gitignored; regenerate with notebook 01)
data/processed/    — feature-engineered data ready for modeling
notebooks/         — numbered notebooks; run in order
src/               — importable modules used by notebooks
outputs/figures/   — saved plots (EDA, SHAP summaries, feature importance)
outputs/models/    — serialized model artifacts (.pkl)
```

## Source Modules

| Module | Responsibility |
|---|---|
| `src/scraper.py` | Fetch game logs, rosters, and player stats via nba_api |
| `src/features.py` | Rolling averages, rest days, home/away flag, win streak, last-10 wins |
| `src/model.py` | Train/predict wrappers, save/load helpers |
| `src/evaluation.py` | Classification metrics, SHAP summary plot, feature importance plot |
