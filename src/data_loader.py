from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st


OPTION_COLUMNS = [
    "contract_id",
    "symbol",
    "expiration",
    "strike",
    "type",
    "last",
    "mark",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
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


# Public historical SPY dataset.
# Files are downloaded only when they are missing locally.
SPY_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "anahatsingh-ui/options-dataset-hist/main/spy"
)


def _download_file(url, destination):
    """
    Download a file from the public SPY dataset.

    Existing files are not downloaded again.
    """
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
        with urlopen(request, timeout=120) as response:
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

    except Exception:
        # Remove incomplete file if the download fails.
        if destination.exists():
            destination.unlink()

        raise


def _ensure_spy_data(data_dir, years):
    """
    Make sure all requested SPY Parquet files exist.

    Missing files are downloaded automatically.
    Existing files are left untouched.
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

    total_downloads = len(missing_years) + int(underlying_missing)

    if total_downloads == 0:
        return

    progress = st.progress(0)
    status = st.empty()

    completed = 0

    try:
        for year in missing_years:
            filename = f"options_{year}.parquet"

            status.info(
                f"Downloading historical SPY options data for {year}..."
            )

            _download_file(
                f"{SPY_DATA_BASE_URL}/{filename}",
                data_dir / filename,
            )

            completed += 1
            progress.progress(completed / total_downloads)

        if underlying_missing:
            status.info(
                "Downloading historical SPY underlying prices..."
            )

            _download_file(
                f"{SPY_DATA_BASE_URL}/underlying_prices.parquet",
                underlying_path,
            )

            completed += 1
            progress.progress(completed / total_downloads)

        status.success("Historical SPY data is ready.")

    except Exception as exc:
        progress.empty()
        status.empty()

        raise RuntimeError(
            "The app could not download the required historical SPY "
            f"data. Please try again later.\n\nDetails: {exc}"
        ) from exc

    progress.empty()
    status.empty()


def _read_parquet(path: Path, columns=None):
    """
    Read a Parquet file with PyArrow.
    """
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
    Load yearly SPY option Parquet files.

    Missing files are automatically downloaded.
    """
    if years is None:
        years = [2024]

    years = sorted(set(int(year) for year in years))

    data_dir = Path(data_dir)

    # Automatically download anything Streamlit Cloud is missing.
    _ensure_spy_data(data_dir, years)

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
        path = data_dir / f"options_{year}.parquet"

        df = _read_parquet(
            path,
            columns=columns,
        )

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No historical SPY option data was found."
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
            df["date"] >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        df = df[
            df["date"] <= pd.Timestamp(end_date)
        ]

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_spy_underlying(
    path="data/raw/spy/underlying_prices.parquet"
):
    """
    Load historical SPY underlying prices.

    The underlying file is automatically downloaded
    if it does not exist.
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
    """
    Merge the option chain with the same-day SPY close
    and create quantitative research features.
    """
    df = options.copy()
    underlying = underlying.copy()

    df["strike"] = pd.to_numeric(
        df["strike"],
        errors="coerce",
    )

    for col in [
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
    ]:
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
        df["expiration"] - df["date"]
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
        valid_quotes & (df["mid"] > 0),
        df["spread"] / df["mid"],
        np.nan,
    )

    df["log_moneyness"] = np.where(
        (df["spot"] > 0)
        & (df["strike"] > 0),
        np.log(
            df["strike"] / df["spot"]
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
    """
    Remove crossed, zero-bid, wide-spread,
    stale/illiquid, and otherwise unusable observations.
    """
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
            ["call", "put"]
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