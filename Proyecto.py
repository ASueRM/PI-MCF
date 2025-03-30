import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis, norm, t
import matplotlib.pyplot as plt

# Título y subtítulo
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
    data["RD"] = data["Adj Close"].pct_change().dropna()
data["RD"] = data["Close"].pct_change()
mean_return = np.mean(data["RD"])
skewness = skew(data["RD"].dropna())
excess_kurtosis = kurtosis(data["RD"].dropna())

# Mostrar métricas de rendimiento
st.subheader("Estadísticas de Rendimientos Diarios")
st.write(f"Media: {mean_return:.5f}")
st.write(f"Sesgo: {skewness:.5f}")
st.write(f"Exceso de Curtosis: {excess_kurtosis:.5f}")

# VaR y ES
nconf = [0.95, 0.975, 0.99]
Var = {}
ES= {}
# Método Paramétrico (Normal)
mean_r = np.mean(data['RD'])
std_r = np.std(data['RD'])
for alpha in nconf:
    z_score = norm.ppf(1 - alpha)
    VaR[f'Normal {alpha}'] = -(mean_r + z_score * std_r)
    ES[f'Normal {alpha}'] = -(mean_r + std_r * norm.pdf(z_score) / (1 - alpha))
# Aproximación paramétrica (t-Student)
dof, loc, scale = t.fit(data['RD'].dropna())
for alpha in nconf:
    t_score = t.ppf(1 - alpha, dof)
    VaR[f't-Student {alpha}'] = -(loc + t_score * scale)
    ES[f't-Student {alpha}'] = -(loc + scale * t.pdf(t_score, dof) / (1 - alpha))
# Aproximación historica
for alpha in nconf:
    VaR[f'Histórico {alpha}'] = -np.percentile(data['RD'].dropna(), 100 * (1 - alpha))
    ES[f'Histórico {alpha}'] = -data['RD'][data['RD'] <= -VaR[f'Histórico {alpha}']].mean()
# Aptoximación Monte Carlo (Simulación de 10,000 escenarios)
np.random.seed(42)
simulations = np.random.normal(mean_r, std_r, (10000,))
for alpha in nconf:
    VaR[f'Monte Carlo {alpha}'] = -np.percentile(simulations, 100 * (1 - alpha))
    ES[f'Monte Carlo {alpha}'] = -simulations[simulations <= -VaR[f'Monte Carlo {alpha}']].mean()
#Rresultados
VaRESdf = pd.DataFrame({'Método': list(var_results.keys()), 'VaR': list(var_results.values()), 'ES': list(es_results.values())})
st.subheader("Resultados de VaR y Expected Shortfall")
st.dataframe(var_es_df.style.format({'VaR': '{:.6f}', 'ES': '{:.6f}'}).set_properties(**{'text-align': 'center'}))

