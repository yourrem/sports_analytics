# Sports Analytics — NBA Game Outcome Prediction

Predicting NBA game outcomes and player performance using rolling stats, rest-day features, and home/away splits. Includes SHAP analysis for model explainability.

**Data:** [nba_api](https://github.com/swar/nba_api) — official NBA stats API wrapper

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
data/raw/          — raw API responses saved as CSVs
data/processed/    — cleaned and feature-engineered data
notebooks/         — step-by-step analysis notebooks
src/               — reusable Python modules
outputs/           — saved figures and model artifacts
```
