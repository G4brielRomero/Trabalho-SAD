"""
ETAPA 2 - TREINAMENTO DOS MODELOS
==================================

O que este script faz:
  1. Lê a base criada na etapa 1.
  2. Separa os dados em TREINO (80% mais antigos) e TESTE (20% mais recentes).
  3. Treina dois modelos:
       - Rede Neural (MLP)  -> modelo principal
       - Árvore de Decisão  -> para comparação
  4. Salva os modelos treinados na pasta modelos/ (arquivos .joblib).

Como rodar:
    python 2_treinamento.py
"""

import os

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

PORCENTAGEM_TREINO = 0.8

# ---------------------------------------------------------------
# 1. Ler a base e separar entradas (X) e alvo (y)
# ---------------------------------------------------------------
df = pd.read_csv("dados/base_modelo.csv", parse_dates=["data"])

X = df.drop(columns=["data", "temperatura"])  # entradas: t-1 ... t-12
y = df["temperatura"]                          # alvo: temperatura do mês

# ---------------------------------------------------------------
# 2. Separar treino e teste
#    Em série temporal NÃO embaralhamos os dados: o modelo treina com
#    o passado e é testado no "futuro" (os anos mais recentes).
# ---------------------------------------------------------------
corte = int(len(df) * PORCENTAGEM_TREINO)

X_treino, y_treino = X.iloc[:corte], y.iloc[:corte]
X_teste, y_teste = X.iloc[corte:], y.iloc[corte:]

print(f"Treino: {len(X_treino)} meses ({df['data'].iloc[0]:%m/%Y} a {df['data'].iloc[corte - 1]:%m/%Y})")
print(f"Teste:  {len(X_teste)} meses ({df['data'].iloc[corte]:%m/%Y} a {df['data'].iloc[-1]:%m/%Y})")

# ---------------------------------------------------------------
# 3a. Rede Neural (MLP = Multi-Layer Perceptron)
#     - StandardScaler: normaliza as entradas (média 0, desvio 1).
#       Redes neurais aprendem muito melhor com dados normalizados.
#     - make_pipeline junta normalização + rede em um único objeto.
#       Assim o scaler aprende a média/desvio SÓ com os dados de treino,
#       e na hora de prever a normalização é aplicada sozinha.
#     - hidden_layer_sizes=(16,): 1 camada oculta com 16 neurônios.
#       (12 entradas -> 16 neurônios -> 1 saída). Rede pequena, pois temos
#       poucos dados (cerca de 700 meses).
#     - alpha=1.0: regularização. Impede que a rede "decore" o treino
#       (overfitting) e aí erre nos dados novos.
#     - learning_rate_init=0.01: tamanho do passo de aprendizado.
#     - max_iter=2000: número máximo de épocas de treino.
#     - random_state=42: fixa a aleatoriedade para o resultado ser reproduzível.
# ---------------------------------------------------------------
rede_neural = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=1.0,
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=42,
    ),
)
rede_neural.fit(X_treino, y_treino)
print("\nRede Neural treinada!")

# ---------------------------------------------------------------
# 3b. Árvore de Decisão (para comparar com a rede)
#     max_depth=5 limita o tamanho da árvore para ela não "decorar" o treino.
#     Árvores não precisam de normalização.
# ---------------------------------------------------------------
arvore = DecisionTreeRegressor(max_depth=5, random_state=42)
arvore.fit(X_treino, y_treino)
print("Árvore de Decisão treinada!")

# ---------------------------------------------------------------
# 4. Salvar os modelos em arquivo binário (.joblib) para reusar depois
# ---------------------------------------------------------------
os.makedirs("modelos", exist_ok=True)  # cria a pasta se ela não existir
joblib.dump(rede_neural, "modelos/rede_neural.joblib")
joblib.dump(arvore, "modelos/arvore_decisao.joblib")
print("\nModelos salvos em: modelos/rede_neural.joblib e modelos/arvore_decisao.joblib")
