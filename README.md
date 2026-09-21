# Daily Minimum Temperature Forecasting - Melbourne, Australia

Capstone 2 · Applied DS, ML & AI · E & ICT Academy, IIT Guwahati

Predicts the next day's minimum temperature for Melbourne using ten years of daily
historical readings (1981–1990). Seven models - spanning linear regression, tree
ensembles, gradient boosting, classical time series, and deep learning - are trained
and compared under an identical, leakage-free chronological evaluation protocol.

**Live app:** [https://daily-minimum-temperature-forecasting-by-vineeth.streamlit.app/]

**Full write-up:** [VINEETH_Capstone_2_MDD .pdf](./VINEETH_Capstone_2_MDD%20.pdf)

---

## Overview

| | |
|---|---|
| **Dataset** | [Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne) (Kaggle), 3,650 daily readings, 1981-1990 |
| **Task** | One-step-ahead regression - predict tomorrow's minimum temperature (°C) |
| **Split** | Chronological 80/20 (never random - see MDD Section 4.6 for why) |
| **Best result** | LSTM, RMSE 2.145 °C, MAE 1.692 °C, R² 0.694 |
| **Recommended for deployment** | Linear Regression - statistically indistinguishable from the LSTM, far simpler to run and maintain (MDD Section 8.3) |

## Results

All models evaluated on the identical 658-day test window (1989-03-14 to 1990-12-31):

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| **LSTM** | **2.145** | 1.692 | 0.694 |
| GRU | 2.148 | 1.696 | 0.693 |
| Blend (Linear Regression + XGBoost) | 2.149 | 1.691 | 0.693 |
| Linear Regression | 2.155 | 1.699 | 0.691 |
| Ridge Regression | 2.155 | 1.699 | 0.691 |
| SARIMAX + Fourier (walk-forward) | 2.163 | 1.714 | 0.688 |
| Random Forest | 2.163 | 1.698 | 0.688 |
| XGBoost (tuned) | 2.180 | 1.715 | 0.683 |
| Naive (persistence) | 2.513 | 1.974 | 0.579 |
| Seasonal naive (1 yr) | 3.706 | 2.926 | 0.085 |

Every trained model converges to a tight ~2.1-2.2 °C RMSE band regardless of methodology -
evidence of an irreducible noise floor given single-station inputs, not a modeling
limitation. Full discussion in the MDD, Section 7.2.

## Repository structure

```
├── app.py                          # Streamlit forecasting app
├── lr_model.pkl                    # Linear Regression (recommended deployment model)
├── xgb_model.json                  # Tuned XGBoost (version-stable format, not pickled)
├── model_meta.pkl                  # Feature list, residual std, valid date range
├── temperature_history.csv         # Cleaned series the app computes live features from
├── .streamlit/config.toml          # Dark theme
├── assets/bg_original.svg          # Original background artwork
├── requirements.txt                # App dependencies (read by Streamlit Cloud)
│
├── code.ipynb                      # Full analysis notebook (EDA → 7 models → comparison)
├── Temperature.csv                 # Raw Kaggle download
├── requirements-notebook.txt       # Notebook dependencies (pandas, xgboost, statsmodels, tensorflow, ...)
│
└── VINEETH_Capstone_2_MDD .pdf/.docx   # Model Development Document (full write-up)
```

## Running the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook code.ipynb
```

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Pick a date, choose a model, click **Predict**.

## Methodology summary

1. **Data cleaning** - the raw Kaggle export has three data-quality issues (a trailing
   non-data row, three Bureau-of-Meteorology quality-flagged readings, two missing
   calendar days), all resolved before modeling (MDD Section 4.1).
2. **Feature engineering** - 11 features: cyclical calendar (Fourier) terms, six lag
   features (1–365 days), three rolling-window statistics (MDD Section 4.4).
3. **Baselines** - naive persistence and seasonal-naive, established before any ML model
   (MDD Section 5).
4. **Models** - Linear Regression, Ridge, Random Forest, XGBoost (tuned via
   RandomizedSearchCV + TimeSeriesSplit), SARIMAX with Fourier terms (dynamic harmonic
   regression, evaluated via proper walk-forward forecasting), LSTM, GRU.
5. **Evaluation** - RMSE, MAE, R² on an identical chronological test window across every
   model family, so results are directly comparable.

Full methodology, justification for every decision, and diagnostics are in the MDD.

## Tech stack

Python · pandas · scikit-learn · XGBoost · statsmodels · TensorFlow/Keras · Streamlit · Plotly

## Author

**Vineeth M**
Applied DS, ML & AI - E & ICT Academy, IIT Guwahati
[GitHub](https://github.com/Vineeth-Muraleedharan)
