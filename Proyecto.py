import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis

# Título y subtítulo
st.set_page_config(page_title="Calculo de Value-At-Risk y de Expected Shortfall", layout="wide")
st.title("Calculo de Value-At-Risk y de Expected Shortfall")
st.write("Proyecto 1")
st.write("Métodos Cuantitativos en Finanzas 2025-2")
# Información del equipo
st.markdown("""
# Integrantes del equipo:
- Alix Sue Rangel Mondragón - No. de cuenta: 320219515
""")

# Cargar datos del petróleo (Ticker: CL=F para Crude Oil WTI)
activo = "CL=F"
data = yf.download(activo, start="2010-01-01", end="2025-01-01")
st.subheader("Activo financiero: Petróleo crudo WTI (CL=F)")
st.write("Nota: Para este proyecto estamos considerando datos desde 01/01/2010 hasta 01/01/2025")

# Rendimientos diarios y métricas
if 'Adj Close' in data.columns:
    data["RD"] = data["Adj Close"].pct_change()
data["RD"] = data["Close"].pct_change()
mean_return = np.mean(data["RD"])
skewness = skew(data["RD"].dropna())
excess_kurtosis = kurtosis(data["RD"].dropna())



# Mostrar métricas de rendimiento
st.subheader("Estadísticas de Rendimientos Diarios")
st.write(f"Media: {mean_return:.5f}")
st.write(f"Sesgo: {skewness:.5f}")
st.write(f"Exceso de Curtosis: {excess_kurtosis:.5f}")
