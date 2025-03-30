import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Configurar la app
st.title("Cálculo de VaR y Expected Shortfall")

# Input del usuario
ticker = st.text_input("Ingrese el ticker del activo financiero:", "AAPL")
data = yf.download(ticker, start="2010-01-01")["Adj Close"]

# Mostrar datos
st.write("Últimos datos descargados:")
st.write(data.tail())

# Cálculo de rendimientos
returns = data.pct_change().dropna()

# Cálculo del VaR y Expected Shortfall
alpha = 0.05
VaR = np.percentile(returns, alpha * 100)
ES = returns[returns <= VaR].mean()

# Mostrar resultados
st.write(f"**Value-at-Risk (VaR) al {alpha*100}%:** {VaR:.5f}")
st.write(f"**Expected Shortfall (ES) al {alpha*100}%:** {ES:.5f}")



