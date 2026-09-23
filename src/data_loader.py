from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st


SPY_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "anahatsingh-ui/options-dataset-hist/main/spy"
)


def _download_file(url, destination):
    """Download a missing dataset file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    request = Request(
        url,
        headers={
            "User-Agent": "VolatilitySurfaceIntelligenceEngine/1.0"
        },
    )

    try:
        with urlopen(request, timeout=180) as response:
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

    except Exception:
        if destination.exists():
            destination.unlink()

        raise


def _ensure_spy_data(data_dir, years):
    """
    Download missing SPY Parquet files.

    Existing files are reused.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(set(int(year) for year in years))

    missing_years = [
        year
        for year in years
        if not (data_dir / f"options_{year}.parquet").exists()
    ]

    underlying_path = data_dir / "underlying_prices.parquet"
    underlying_missing = not underlying_path.exists()

    total = len(missing_years) + int(underlying_missing)

    if total == 0:
        return

    progress = st.progress(0)
    status = st.empty()

    completed = 0

    try:
        for year in missing_years:
            status.info(
                f"Downloading SPY options data for {year}..."
            )

            _download_file(
                f"{SPY_DATA_BASE_URL}/options_{year}.parquet",
                data_dir / f"options_{year}.parquet",
            )

            completed += 1
            progress.progress(completed / total)

        if underlying_missing:
            status.info(
                "Downloading SPY underlying price data..."
            )

            _download_file(
                f"{SPY_DATA_BASE_URL}/underlying_prices.parquet",
                underlying_path,
            )

            completed += 1
            progress.progress(completed / total)

    except Exception as exc:
        progress.empty()
        status.empty()

        raise RuntimeError(
            "Unable to download the historical SPY dataset. "
            f"Download error: {exc}"
        ) from exc

    progress.empty()
    status.empty()


def _read_parquet(path, columns=None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return pd.read_parquet(
        path,
        columns=columns,
        engine="pyarrow",
    )


@st.cache_data(show_spinner=False)
def load_spy_options(
    data_dir="data/raw/spy",
    start_date=None,
    end_date=None,
    years=None,
):
    """
    Load historical SPY option data.

    Missing yearly Parquet files are downloaded automatically.
    """

    if years is None:
        years = [2024]

    years = sorted(
        set(int(year) for year in years)
    )

    data_dir = Path(data_dir)

    # THIS IS THE IMPORTANT CHANGE.
    # The old version only searched for files.
    # This version downloads missing files.
    _ensure_spy_data(
        data_dir,
        years,
    )

    columns = [
        "contract_id",
        "symbol",
        "expiration",
        "strike",
        "type",
        "last",
        "mark",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "date",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "in_the_money",
    ]

    frames = []

    for year in years:
        path = (
            data_dir
            / f"options_{year}.parquet"
        )

        df = _read_parquet(
            path,
            columns=columns,
        )

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No SPY option data could be loaded."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["expiration"] = pd.to_datetime(
        df["expiration"],
        errors="coerce",
    )

    if start_date is not None:
        df = df[
            df["date"]
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        df = df[
            df["date"]
            <= pd.Timestamp(end_date)
        ]

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_spy_underlying(
    path="data/raw/spy/underlying_prices.parquet"
):
    """
    Load historical SPY underlying prices.

    The file is downloaded automatically if missing.
    """

    path = Path(path)

    if not path.exists():
        _ensure_spy_data(
            path.parent,
            years=[],
        )

    df = _read_parquet(
        path,
        columns=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividend_amount",
            "split_coefficient",
        ],
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return (
        df[
            [
                "date",
                "close",
            ]
        ]
        .dropna(
            subset=[
                "date",
                "close",
            ]
        )
        .drop_duplicates("date")
    )


def prepare_dataset(options, underlying):
    """Merge options with SPY prices and create research features."""

    df = options.copy()
    underlying = underlying.copy()

    df["strike"] = pd.to_numeric(
        df["strike"],
        errors="coerce",
    )

    numeric_columns = [
        "bid",
        "ask",
        "mark",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    raw_type = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["option_type"] = raw_type.map(
        {
            "c": "call",
            "call": "call",
            "p": "put",
            "put": "put",
        }
    )

    df = df.merge(
        underlying.rename(
            columns={
                "close": "spot"
            }
        ),
        on="date",
        how="left",
        validate="many_to_one",
    )

    df["T"] = (
        df["expiration"]
        - df["date"]
    ).dt.total_seconds() / (
        365.25 * 24 * 60 * 60
    )

    valid_quotes = (
        df["bid"].notna()
        & df["ask"].notna()
        & (df["bid"] > 0)
        & (df["ask"] >= df["bid"])
    )

    df["mid"] = np.where(
        valid_quotes,
        (df["bid"] + df["ask"]) / 2.0,
        np.nan,
    )

    df["spread"] = np.where(
        valid_quotes,
        df["ask"] - df["bid"],
        np.nan,
    )

    df["relative_spread"] = np.where(
        valid_quotes
        & (df["mid"] > 0),
        df["spread"] / df["mid"],
        np.nan,
    )

    df["log_moneyness"] = np.where(
        (df["spot"] > 0)
        & (df["strike"] > 0),
        np.log(
            df["strike"]
            / df["spot"]
        ),
        np.nan,
    )

    df["option_type_is_put"] = (
        df["option_type"] == "put"
    ).astype(float)

    return df


def clean_quotes(
    df,
    max_spread_pct=0.20,
    min_open_interest=10,
    min_volume=0,
):
    """Remove unusable or illiquid option observations."""

    df = df.copy()

    keep = (
        df["date"].notna()
        & df["expiration"].notna()
        & df["strike"].gt(0)
        & df["spot"].gt(0)
        & df["T"].gt(0)
        & df["bid"].gt(0)
        & df["ask"].ge(df["bid"])
        & df["mid"].gt(0)
        & df["relative_spread"].notna()
        & df["relative_spread"].le(
            max_spread_pct
        )
        & df["option_type"].isin(
            [
                "call",
                "put",
            ]
        )
        & (
            df["open_interest"]
            .fillna(0)
            .ge(min_open_interest)
            |
            df["volume"]
            .fillna(0)
            .ge(min_volume)
        )
    )

    return df.loc[keep].copy()