# SPY historical data

The application expects the free historical SPY Parquet files here:

```text
options_YYYY.parquet
underlying_prices.parquet
```

Download them with:

```powershell
python -m src.download_spy --years 2021 2022 2023 2024 2025
```

Do not commit the Parquet files to Git. They are ignored by `.gitignore`.

Dataset source: https://github.com/anahatsingh-ui/options-dataset-hist
