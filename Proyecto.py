import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis, norm, t
import matplotlib.pyplot as plt
import streamlit as st


st.markdown("<h1 style='color:#1E3A8A;'>Cálculo de Value-At-Risk y de Expected Shortfall</h1>", unsafe_allow_html=True)


st.markdown("<h3 style='color:#4B5563;'>Proyecto 1</h3>", unsafe_allow_html=True)


st.markdown("<h4 style='color:#6B7280;'>Métodos Cuantitativos en Finanzas 2025-2</h4>", unsafe_allow_html=True)

# Información del equipo 
st.markdown("""
    <h3 style='color:#111827;'>Integrantes del equipo:</h3>
    <ul>
        <li><b style='color:#2D3748;'>Alix Sue Rangel Mondragón</b> - No. de cuenta: 320219515</li>
        <li><b style='color:#2D3748;'>Edgar Giovanny Caravantes Román</b> - No. de cuenta: 421015887</li>
    </ul>
""", unsafe_allow_html=True)


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
VaR = {}
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
VaRES = pd.DataFrame({'Método de aproximación': list(VaR.keys()), 'VaR': list(VaR.values()), 'ES': list(ES.values())})
VaRES = VaRES.replace([np.inf, -np.inf], np.nan).dropna()
st.subheader("Resultados de VaR y Expected Shortfall")
def color_negative_red(val):
    color = 'blue' if val > 0 else 'gray'
    return f'color: {color}'
styled_VaRES = VaRES.style.applymap(color_negative_red, subset=['VaR', 'ES'])
st.write(styled_VaRES)

#Rolling window
window = 252  
a95 = 0.95
a99 = 0.99
var_95_historico = []
var_99_historico = []
es_95_historico = []
es_99_historico = []
retornos_pred = [] 
#Cálculos y resultados
for i in range(window, len(data)):
    rolling_retornos = data["RD"].iloc[i - window:i]
    # Predicción del retorno (usamos el retorno promedio de los últimos 252 días como predicción)
    prediccion = np.mean(rolling_retornos)
    retornos_pred.append(prediccion)
    # Cálculo del VaR y ES
    var_95_historico.append(np.percentile(rolling_retornos, 100 * (1 - a95)))
    var_99_historico.append(np.percentile(rolling_retornos, 100 * (1 - a99)))
    es_95_historico.append(rolling_retornos[rolling_retornos <= var_95_historico[-1]].mean())
    es_99_historico.append(rolling_retornos[rolling_retornos <= var_99_historico[-1]].mean())
resultados = pd.DataFrame({
    'Retorno Predicho': retornos_pred,
    'VaR 95% Histórico': var_95_historico,
    'VaR 99% Histórico': var_99_historico,
    'ES 95% Histórico': es_95_historico,
    'ES 99% Histórico': es_99_historico
}, index=data.index[window:])
#Gráfica
plt.figure(figsize=(12, 6))
plt.plot(resultados.index, resultados['Retorno Predicho'], label='Retorno Predicho', color='green', linestyle='-', linewidth=2)
plt.plot(resultados.index, resultados['VaR 95% Histórico'], label='VaR 95% Histórico', color='#E56717', linestyle='-', linewidth=2)
plt.plot(resultados.index, resultados['VaR 99% Histórico'], label='VaR 99% Histórico', color='#FFA800', linestyle='-', linewidth=2)
plt.plot(resultados.index, resultados['ES 95% Histórico'], label='ES 95% Histórico', color='#191970', linestyle='-', linewidth=2)
plt.plot(resultados.index, resultados['ES 99% Histórico'], label='ES 99% Histórico', color='#014D4E', linestyle='-', linewidth=2)
plt.title('Rolling windows (Ganancias y Pérdidas, VaR y ES)')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend(loc='upper left')
plt.grid(True)
st.pyplot(plt)


