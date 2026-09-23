import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "log_moneyness",
    "T",
    "option_type_is_put",
    "volume",
    "open_interest",
    "relative_spread",
    "r",
    "q",
]


def chronological_split(df, train_fraction=0.60, validation_fraction=0.20):
    df = df.sort_values("date").copy()
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) < 3:
        raise ValueError("Need at least three unique dates for chronological splitting.")

    train_end = max(1, int(len(dates) * train_fraction))
    val_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    val_end = min(val_end, len(dates) - 1)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(val_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )


def _training_frame(df):
    return df.dropna(subset=FEATURES + ["calculated_iv"]).copy()


def train_gradient_boosting(df):
    x = _training_frame(df)
    if x.empty:
        raise ValueError("No valid training observations remain after feature filtering.")
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(x[FEATURES], x["calculated_iv"])
    return model


def train_mlp(df):
    x = _training_frame(df)
    if x.empty:
        raise ValueError("No valid training observations remain after feature filtering.")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            early_stopping=True,
            random_state=42,
        )),
    ])
    model.fit(x[FEATURES], x["calculated_iv"])
    return model


def predict(model, df):
    df = df.copy()
    valid = df[FEATURES].notna().all(axis=1)
    df["predicted_iv"] = np.nan
    if valid.any():
        df.loc[valid, "predicted_iv"] = model.predict(df.loc[valid, FEATURES])
    return df


def evaluate(df):
    x = df.dropna(subset=["calculated_iv", "predicted_iv"])
    if x.empty:
        return {"MAE": np.nan, "RMSE": np.nan}
    return {
        "MAE": mean_absolute_error(x["calculated_iv"], x["predicted_iv"]),
        "RMSE": mean_squared_error(x["calculated_iv"], x["predicted_iv"]) ** 0.5,
    }
