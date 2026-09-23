import streamlit as st

st.title("Volatility Surface Intelligence Engine")

st.write("A: app started")

from src.data_loader import load_spy_options, load_spy_underlying, prepare_dataset, clean_quotes
st.write("B: imports completed")