import numpy as np
import plotly.express as px
import streamlit as st

from src.black_scholes import call_price, put_price
from src.data_loader import load_spy_options, load_spy_underlying, prepare_dataset, clean_quotes
from src.iv import add_iv
from src.models import FEATURES, chronological_split, train_gradient_boosting, train_mlp, predict, evaluate

st.set_page_config(page_title="Volatility Surface Intelligence Engine", layout="wide")
st.title("Volatility Surface Intelligence Engine")
st.caption("Historical SPY option-market data → implied-volatility extraction → out-of-sample ML")
st.caption("About this demo " \
"The underlying research pipeline processes a large historical options dataset and calculates implied volatility using numerical root-finding. This computation is intentionally performed rather than relying on precomputed IV values. The hosted version may take several minutes to process large date ranges. For faster execution and full experimentation, run the project locally.")

st.sidebar.header("Dataset")
start_date = st.sidebar.date_input("Start Date", value=None)
end_date = st.sidebar.date_input("End Date", value=None)
year_text = st.sidebar.text_input(
    "Parquet Years",
    "2024",
    help="Enter years separated by commas, for example: 2023,2024",
)

st.sidebar.header("Research Settings")
model_name = st.sidebar.selectbox("ML Model", ["Gradient Boosting", "MLP Neural Network"])
train_frac = st.sidebar.slider("Training Fraction", .50, .80, .60, .05)
val_frac = st.sidebar.slider("Validation Fraction", .10, .30, .20, .05)
max_spread = st.sidebar.slider("Maximum Relative Spread", .05, .50, .20, .05)
min_oi = st.sidebar.number_input("Minimum Open Interest", 0, 100000, 10)
rate = st.sidebar.number_input("Risk-Free Rate", value=.045, step=.005, format="%.4f")
q = st.sidebar.number_input("Dividend Yield", value=.0, step=.005, format="%.4f")

try:
    years = [
        int(x.strip())
        for x in year_text.split(",")
        if x.strip()
    ]

    if not years:
        st.error(
            "Please enter at least one Parquet year."
        )
        st.stop()

    raw = load_spy_options(
        data_dir="data/raw/spy_backup",
        start_date=start_date,
        end_date=end_date,
        years=years,
    )

    underlying = load_spy_underlying(
        "data/raw/spy/underlying_prices.parquet"
    )

    raw = prepare_dataset(
        raw,
        underlying,
    )

    clean = clean_quotes(
        raw,
        max_spread,
        min_oi,
    )

except Exception as exc:
    st.error(
        "Unable to load the historical SPY dataset."
    )

    st.warning(
        "The app automatically downloads missing "
        "SPY Parquet files. Please refresh the app "
        "if the download was interrupted."
    )

    with st.expander("Technical details"):
        st.code(str(exc))

    st.stop()

with st.spinner("Extracting implied volatility from market mids..."):
    clean = add_iv(clean, rate, q)

clean = clean.replace([np.inf, -np.inf], np.nan)
clean = clean.dropna(subset=FEATURES + ["calculated_iv"]).copy()

if clean.empty:
    st.error("No observations have a valid calculated IV after cleaning.")
    st.stop()

train, val, test = chronological_split(clean, train_frac, val_frac)
if train.empty or val.empty or test.empty:
    st.error("Chronological split produced an empty partition. Use more dates or adjust the split fractions.")
    st.stop()

model = train_gradient_boosting(train) if model_name == "Gradient Boosting" else train_mlp(train)
val_pred = predict(model, val)
test_pred = predict(model, test)
val_metrics = evaluate(val_pred)
test_metrics = evaluate(test_pred)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean Observations", f"{len(clean):,}")
c2.metric("Train", f"{len(train):,}")
c3.metric("Test", f"{len(test_pred):,}")
c4.metric("Test RMSE", f"{test_metrics['RMSE']:.5f}")

st.caption(
    f"Validation MAE: {val_metrics['MAE']:.5f} | "
    f"Validation RMSE: {val_metrics['RMSE']:.5f} | "
    f"Test MAE: {test_metrics['MAE']:.5f}"
)

st.subheader("Out-of-Sample IV Surface")
surface = test_pred.sample(min(len(test_pred), 2500), random_state=42)
fig = px.scatter_3d(
    surface,
    x="log_moneyness",
    y="T",
    z="calculated_iv",
    color="option_type",
    labels={
        "log_moneyness": "Log-Moneyness",
        "T": "Time to Expiry",
        "calculated_iv": "Calculated IV",
    },
)
fig.update_traces(marker=dict(size=1.5))
st.plotly_chart(fig, use_container_width=True)

view = test_pred.dropna(subset=["calculated_iv", "predicted_iv"]).copy()
st.subheader("Observed vs Predicted IV")
if not view.empty:
    fig = px.scatter(
        view,
        x="calculated_iv",
        y="predicted_iv",
        trendline="ols",
        trendline_color_override="green",
        labels={"calculated_iv": "Observed IV", "predicted_iv": "Predicted IV"},
    )
    lo = min(view.calculated_iv.min(), view.predicted_iv.min())
    hi = max(view.calculated_iv.max(), view.predicted_iv.max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="red"))
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)

    view["residual"] = view.predicted_iv - view.calculated_iv
    fig.update_traces(marker=dict(size=1.5))
    fig.update_traces(
    line=dict(color="green", width=3), 
    selector=dict(mode="lines"))
    st.plotly_chart(
        px.scatter(view, x="calculated_iv", y="residual", title="IV Prediction Residuals"),
        use_container_width=True,
    )

st.subheader("Downstream BSM Pricing Error")
if not view.empty:
    p = view.sample(min(len(view), 1000), random_state=42).copy()

    def price(row, sigma):
        fn = call_price if row["option_type"] == "call" else put_price
        return fn(row["spot"], row["strike"], row["T"], row["r"], row["q"], sigma)

    p["observed_iv_price"] = p.apply(lambda x: price(x, x["calculated_iv"]), axis=1)
    p["ml_iv_price"] = p.apply(lambda x: price(x, x["predicted_iv"]), axis=1)
    p["price_error"] = p["ml_iv_price"] - p["observed_iv_price"]
    st.metric("Mean Absolute Price Difference", f"${p['price_error'].abs().mean():.4f}")
    st.plotly_chart(
        px.scatter(p, x="observed_iv_price", y="ml_iv_price", title="BSM Price: Observed IV vs ML IV"),
        use_container_width=True,
    )

st.subheader("Reference IV Comparison")
if "implied_volatility" in test_pred.columns:
    ref = test_pred[["calculated_iv", "implied_volatility"]].dropna().copy()
    if not ref.empty:
        ref["vendor_iv_error"] = ref.implied_volatility - ref.calculated_iv
        st.metric("Reference IV Mean Absolute Difference", f"{ref.vendor_iv_error.abs().mean():.5f}")

with st.expander("Filtered Historical Data"):
    columns = [
        "date", "expiration", "strike", "option_type", "spot", "bid", "ask", "mid",
        "relative_spread", "volume", "open_interest", "implied_volatility", "calculated_iv",
    ]
    st.dataframe(test_pred[[c for c in columns if c in test_pred.columns]].head(1000), use_container_width=True)

with st.expander("Research Notes"):
    st.markdown(
        "The dataset is the free historical SPY end-of-day options dataset. "
        "The model target is implied volatility independently extracted from bid/ask market mids using BSM and Brent root finding. "
        "The dataset's supplied IV is retained only as a reference field. The main evaluation is chronological out-of-sample testing. "
        "Quote-quality filters are applied before IV extraction, and the same-day historical SPY close is used as the underlying spot price."
    )
