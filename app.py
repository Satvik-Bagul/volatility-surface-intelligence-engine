import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Volatility Surface Intelligence Engine",
    layout="wide",
)

st.title("Volatility Surface Intelligence Engine")

DATA_DIR = Path("data/raw/spy")

st.write("### Diagnostic")

if not DATA_DIR.exists():
    st.error(f"Data directory does not exist: {DATA_DIR}")
    st.stop()

st.write("Files currently inside `data/raw/spy`:")

for item in DATA_DIR.iterdir():
    st.write({
        "name": item.name,
        "size_mb": round(item.stat().st_size / (1024**2), 2),
        "is_file": item.is_file(),
    })

files = sorted(DATA_DIR.glob("*.parquet"))

st.write(f"Found {len(files)} Parquet file(s).")

if not files:
    st.error("No Parquet files found.")
    st.stop()

for file in files:
    st.write(f"- `{file}` — {file.stat().st_size / (1024**2):.2f} MB")

# Only inspect the first file
file = files[0]

st.write(f"### Testing: `{file.name}`")

try:
    df = pd.read_parquet(file)

    st.success("Parquet file loaded successfully.")

    st.write("Rows:", len(df))
    st.write("Columns:", len(df.columns))
    st.write("Memory usage:", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    st.write("Column names:")
    st.write(df.columns.tolist())

    st.write("First 5 rows:")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"Parquet loading failed: {type(e).__name__}: {e}")