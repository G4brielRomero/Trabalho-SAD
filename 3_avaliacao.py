"""
ETAPA 3 - AVALIAÇÃO DOS MODELOS
================================

O que este script faz:
  1. Carrega os modelos salvos na etapa 2.
  2. Faz previsões nos dados de TESTE (que os modelos nunca viram).
  3. Calcula as métricas de regressão: MAE, MSE, RMSE e R².
  4. Salva a tabela de métricas e as previsões na pasta resultados/.

Como rodar:
    python 3_avaliacao.py
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PORCENTAGEM_TREINO = 0.8  # tem que ser igual ao do script 2

# ---------------------------------------------------------------
# 1. Ler a base e separar o TESTE (mesma divisão do script 2)
# ---------------------------------------------------------------
df = pd.read_csv("dados/base_modelo.csv", parse_dates=["data"])
X = df.drop(columns=["data", "temperatura"])
y = df["temperatura"]

corte = int(len(df) * PORCENTAGEM_TREINO)
X_teste, y_teste = X.iloc[corte:], y.iloc[corte:]

# ---------------------------------------------------------------
# 2. Carregar os modelos salvos
# ---------------------------------------------------------------
modelos = {
    "Rede Neural (MLP)": joblib.load("modelos/rede_neural.joblib"),
    "Árvore de Decisão": joblib.load("modelos/arvore_decisao.joblib"),
}

# ---------------------------------------------------------------
# 3. Prever e calcular as métricas
#    MAE  = erro médio absoluto (em °C) -> "em média erro X graus"
#    MSE  = erro quadrático médio (pune mais os erros grandes)
#    RMSE = raiz do MSE (volta para °C)
#    R²   = quanto da variação da temperatura o modelo explica (1 = perfeito)
# ---------------------------------------------------------------
previsoes = pd.DataFrame({"data": df["data"].iloc[corte:], "Real": y_teste})
linhas = []

for nome, modelo in modelos.items():
    y_previsto = modelo.predict(X_teste)
    previsoes[nome] = y_previsto.round(2)

    mse = mean_squared_error(y_teste, y_previsto)
    linhas.append({
        "Modelo": nome,
        "MAE": mean_absolute_error(y_teste, y_previsto),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R²": r2_score(y_teste, y_previsto),
    })

# Referência simples (sem IA): "a temperatura vai ser igual à do mesmo mês
# do ano passado". Se o modelo não for melhor que isso, ele não está ajudando.
y_referencia = X_teste["t-12"]
mse = mean_squared_error(y_teste, y_referencia)
linhas.append({
    "Modelo": "Referência (repetir o ano anterior)",
    "MAE": mean_absolute_error(y_teste, y_referencia),
    "MSE": mse,
    "RMSE": mse ** 0.5,
    "R²": r2_score(y_teste, y_referencia),
})

# ---------------------------------------------------------------
# 4. Mostrar e salvar a tabela de métricas
# ---------------------------------------------------------------
metricas = pd.DataFrame(linhas).round(3)

print("\nTABELA DE MÉTRICAS (dados de teste)")
print("=" * 70)
print(metricas.to_string(index=False))
print("=" * 70)

os.makedirs("resultados", exist_ok=True)  # cria a pasta se ela não existir
metricas.to_csv("resultados/metricas.csv", index=False)
previsoes.to_csv("resultados/previsoes_teste.csv", index=False)
print("\nArquivos salvos: resultados/metricas.csv e resultados/previsoes_teste.csv")
