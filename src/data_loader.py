from pathlib import Path
import numpy as np
import pandas as pd

OPTION_COLUMNS = [
    "contract_id", "symbol", "expiration", "strike", "type", "last", "mark",
    "bid", "bid_size", "ask", "ask_size", "volume", "open_interest", "date",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho", "in_the_money"
]


def _read_parquet(path: Path, columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_spy_options(
    data_dir="data/raw/spy",
    start_date=None,
    end_date=None,
    years=None,
):
    """Load yearly SPY option Parquet files from the free historical dataset."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("options_*.parquet"))

    if years is not None:
        wanted = {int(y) for y in years}
        files = [p for p in files if p.stem.split("_")[-1].isdigit() and int(p.stem.split("_")[-1]) in wanted]

    if not files:
        raise FileNotFoundError(
            f"No SPY Parquet files found in {data_dir}. "
            "Run: python -m src.download_spy --years 2021 2022 2023 2024 2025"
        )

    frames = []
    columns = [
        "contract_id", "symbol", "expiration", "strike", "type", "last", "mark",
        "bid", "ask", "volume", "open_interest", "date", "implied_volatility",
        "delta", "gamma", "theta", "vega", "rho", "in_the_money"
    ]

    for path in files:
        df = _read_parquet(path, columns=columns)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


def load_spy_underlying(path="data/raw/spy/underlying_prices.parquet"):
    """Load historical SPY underlying prices."""
    df = _read_parquet(
        Path(path),
        columns=["symbol", "date", "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient"],
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", "close"]].dropna(subset=["date", "close"]).drop_duplicates("date")


def prepare_dataset(options, underlying):
    """Merge the option chain with the same-day SPY close and create research features."""
    df = options.copy()
    underlying = underlying.copy()

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    for col in ["bid", "ask", "mark", "volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "rho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_type = df["type"].astype(str).str.strip().str.lower()
    df["option_type"] = raw_type.map({"c": "call", "call": "call", "p": "put", "put": "put"})

    df = df.merge(
        underlying.rename(columns={"close": "spot"}),
        on="date",
        how="left",
        validate="many_to_one",
    )

    df["T"] = (df["expiration"] - df["date"]).dt.total_seconds() / (365.25 * 24 * 60 * 60)

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
    df["spread"] = np.where(valid_quotes, df["ask"] - df["bid"], np.nan)
    df["relative_spread"] = np.where(
        valid_quotes & (df["mid"] > 0),
        df["spread"] / df["mid"],
        np.nan,
    )
    df["log_moneyness"] = np.where(
        (df["spot"] > 0) & (df["strike"] > 0),
        np.log(df["strike"] / df["spot"]),
        np.nan,
    )
    df["option_type_is_put"] = (df["option_type"] == "put").astype(float)

    return df


def clean_quotes(df, max_spread_pct=0.20, min_open_interest=10, min_volume=0):
    """Remove crossed, zero-bid, stale/illiquid and otherwise unusable observations."""
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
        & df["relative_spread"].le(max_spread_pct)
        & df["option_type"].isin(["call", "put"])
        & (
            df["open_interest"].fillna(0).ge(min_open_interest)
            | df["volume"].fillna(0).ge(min_volume)
        )
    )
    return df.loc[keep].copy()
