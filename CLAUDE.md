# Sports Analytics — Project Guide

## Goal
Predict NBA game outcomes using rolling team stats, rest-day features, and home/away splits. Portfolio project emphasizing model explainability via SHAP analysis.

**Data source:** `nba_api` Python package (official NBA stats, no API key required)  
**Target variable:** win/loss per game (binary classification)

## Project Structure

```
data/raw/          — raw CSVs fetched from nba_api, one file per endpoint/season
data/processed/    — feature-engineered data ready for modeling
notebooks/         — numbered notebooks; run in order
src/               — importable modules used by notebooks
outputs/figures/   — saved plots (EDA, SHAP summaries, feature importance)
outputs/models/    — serialized model artifacts (.pkl)
```

## Source Modules

| File | Responsibility |
|------|---------------|
| `src/scraper.py` | Fetch game logs and player stats via nba_api |
| `src/features.py` | Rolling averages, rest days, home/away flag |
| `src/model.py` | Train/predict wrappers, save/load helpers |
| `src/evaluation.py` | Classification metrics, SHAP summary plot, feature importance plot |

## Notebook Order

1. `01_data_collection.ipynb` — pull and save raw data from nba_api
2. `02_eda.ipynb` — distributions, correlations, class balance check
3. `03_feature_engineering.ipynb` — build feature matrix, save to processed/
4. `04_modeling.ipynb` — train XGBoost/LightGBM, cross-validation, hyperparameter tuning
5. `05_shap_analysis.ipynb` — SHAP summary, dependence plots, feature importance

## Conventions

- Raw API responses are saved to `data/raw/` immediately after fetching — avoids redundant API calls.
- nba_api has rate limiting; add `time.sleep(0.6)` between requests in `scraper.py`.
- All feature construction logic lives in `src/features.py`, not in notebooks.
- Save every trained model to `outputs/models/` with `joblib.dump`.
- SHAP plots go to `outputs/figures/shap_*.png`.

## Environment

```bash
pip install -r requirements.txt
```

Main dependencies: pandas, numpy, nba_api, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn
