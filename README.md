# Volatility Surface Intelligence Engine

A historical-data machine-learning research project that learns implied-volatility structure from real SPY option-market observations.

## Research question

**Can a model trained on earlier SPY volatility surfaces generalize to later market conditions?**

## Dataset

This project uses the free **Historical Options Dataset: SPY, QQQ, IWM (2008–2025)** preservation mirror by `anahatsingh-ui`, originally created by Philipp D. Dubach. The SPY portion contains about 24.7 million contract-day observations and is distributed as yearly Parquet files. The dataset includes bid/ask, volume, open interest, option type, expiration, strike, vendor IV, and vendor Greeks. Historical underlying prices are included separately. The repository is MIT licensed. See the source repository:

https://github.com/anahatsingh-ui/options-dataset-hist

The code does **not** bundle the large market-data files in this repository. Download only the years needed for the experiment.

## Pipeline

```text
SPY historical option chains
        ↓
Historical SPY underlying close
        ↓
Date alignment
        ↓
Quote-quality filtering
        ↓
Bid / Ask → Mid
        ↓
BSM + Brent implied-volatility extraction
        ↓
Chronological train / validation / test split
        ↓
Gradient Boosting or MLP
        ↓
Out-of-sample IV error
        ↓
BSM downstream pricing error
```

## Important design choice

The source dataset already contains `implied_volatility` and Greeks. Those fields are retained as **reference data**, not used as the ML target.

The model target is `calculated_iv`, which is independently recovered from the observed bid/ask midpoint using the implied-volatility solver from Project 1.

This keeps the progression between the projects meaningful:

- **Project 1:** implement and validate option pricing and IV extraction.
- **Project 2:** use historical market observations to learn the structure of implied volatility.

The vendor/source IV is used only for comparison.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Download free SPY data

The downloader retrieves yearly Parquet files from the public dataset mirror. Start with a few years rather than downloading the full 2008–2025 history.

```powershell
python -m src.download_spy --years 2021 2022 2023 2024 2025
```

The files are saved under:

```text
data/raw/spy/
    options_2021.parquet
    options_2022.parquet
    options_2023.parquet
    options_2024.parquet
    options_2025.parquet
    underlying_prices.parquet
```

For a smaller first test:

```powershell
python -m src.download_spy --years 2024
```

## Run

```powershell
streamlit run app.py
```

The sidebar lets you select which downloaded years to load, the date range, quote-quality thresholds, risk-free rate, dividend yield, and ML model.

## ML features

The model currently uses:

- log-moneyness
- time to expiration
- call/put indicator
- volume
- open interest
- relative bid/ask spread
- risk-free rate
- dividend yield

The target is `calculated_iv`.

`implied_volatility` from the source dataset is deliberately excluded from the feature set to avoid target leakage.

## Validation

The split is chronological rather than random. Earlier observations are used for training, an intermediate period is used for validation, and the latest period is held out for the final test.

Reported metrics include:

- MAE
- RMSE
- residuals
- downstream BSM price difference
- comparison against the source dataset's reference IV

## Data and reproducibility notes

The source data is end-of-day and represents a 4:00 PM ET snapshot. It is market data for research and education. Do not commit the downloaded Parquet files to Git; they are intentionally ignored by `.gitignore`.

The risk-free rate is currently supplied as a research input rather than sourced as a date-specific historical curve. Dividend yield is also supplied as an input. These are limitations to document when presenting results.

## Tests

```powershell
pytest
```

Do not make empirical claims about model performance until the project has been run on the downloaded historical dataset.
