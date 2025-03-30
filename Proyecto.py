import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 🔹 Título y subtítulo
st.title(" Calculo de Value-At-Risk y de Expected Shortfall")
st.subheader("Proyecto 1")
st.subheader("Métodos Cuantitativos en Finanzas 2025-2")
# 🔹 Información del equipo
st.markdown("""
### 👥 Integrantes del equipo:
- Alix Sue Rangel Mondragón - No. de cuenta: 320219515
""")

# Cargar datos del petróleo (Ticker: CL=F para Crude Oil WTI)
 ticker = "CL=F"
 data = yf.download(ticker, start="2010-01-01", end="2025-01-01")
