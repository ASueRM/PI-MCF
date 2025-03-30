import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 🔹 Título y subtítulo
st.set_page_config(page_title="Calculo de Value-At-Risk y de Expected Shortfall", layout="wide")
st.subheader("Proyecto 1")
st.subheader("Métodos Cuantitativos en Finanzas 2025-2")
# 🔹 Información del equipo
st.markdown("""
### 👥 Integrantes del equipo:
- Alix Sue Rangel Mondragón - No. de cuenta: 320219515
""")

# Cargar datos del petróleo (Ticker: CL=F para Crude Oil WTI)
activo = "CL=F"
data = yf.download(activo, start="2010-01-01", end="2025-01-01")

st.subheader("Activo financiero: Petróleo crudo WTI (CL=F)")
st.write("Nota: Para este proyecto estamos considerando datos desde 01/01/2010 hasta 01/01/2025")
