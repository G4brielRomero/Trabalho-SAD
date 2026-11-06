import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configuracao import COLUNAS_ENTRADA, DATA_CORTE, REFERENCIA_ANO, REFERENCIA_MEDIA

base = pd.read_csv("dados/base_modelo.csv", parse_dates=["data"])
teste = base.loc[base["data"] >= DATA_CORTE]
X_teste = teste[COLUNAS_ENTRADA]
y_teste = teste["temperatura"]
print(f"Teste: {len(teste)} meses medidos ({teste['data'].min():%m/%Y} a {teste['data'].max():%m/%Y})")

rede = joblib.load("modelos/rede_neural.joblib")
arvore = joblib.load("modelos/arvore_decisao.joblib")
previsoes = pd.DataFrame({
    "data": teste["data"],
    "Real": y_teste,
    "Rede Neural (MLP)": rede.predict(X_teste),
    "Árvore de Decisão": arvore.predict(X_teste),
    REFERENCIA_ANO: X_teste["t-12"],  # Já usa a média do treino se faltou a medição.
    REFERENCIA_MEDIA: teste["media_treino"],
})

linhas = []
for nome in previsoes.columns[2:]:
    previsto = previsoes[nome]
    mse = mean_squared_error(y_teste, previsto)
    linhas.append({
        "Modelo": nome,
        "MAE": mean_absolute_error(y_teste, previsto),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R²": r2_score(y_teste, previsto),
    })

metricas = pd.DataFrame(linhas).round(3)
print(metricas.to_string(index=False))
os.makedirs("resultados", exist_ok=True)
metricas.to_csv("resultados/metricas.csv", index=False)
previsoes.to_csv("resultados/previsoes_teste.csv", index=False, float_format="%.6f")
