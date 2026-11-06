import os

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from configuracao import COLUNAS_ENTRADA, DATA_CORTE

base = pd.read_csv("dados/base_modelo.csv", parse_dates=["data"])
treino = base.loc[base["data"] < DATA_CORTE]
X_treino = treino[COLUNAS_ENTRADA]
y_treino = treino["temperatura"]

print(f"Treino: {len(treino)} meses medidos ({treino['data'].min():%m/%Y} a {treino['data'].max():%m/%Y})")

# A padronização aprende a média e o desvio de cada entrada durante o fit.
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

arvore = DecisionTreeRegressor(max_depth=5, random_state=42)
arvore.fit(X_treino, y_treino)

os.makedirs("modelos", exist_ok=True)
joblib.dump(rede_neural, "modelos/rede_neural.joblib")
joblib.dump(arvore, "modelos/arvore_decisao.joblib")
print("Modelos salvos na pasta modelos/")
