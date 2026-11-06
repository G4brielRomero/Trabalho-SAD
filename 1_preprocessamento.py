import pandas as pd

from configuracao import COLUNAS_ENTRADA, DATA_CORTE

# ------------------------- CONFIGURAÇÃO -------------------------
ARQUIVO_KAGGLE = "dados/station_goiania.csv"  # arquivo baixado do Kaggle
NOME_CIDADE = "Goiânia"
# ----------------------------------------------------------------

# O dataset usa os nomes dos meses em inglês
MESES = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
         "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


# ---------------------------------------------------------------
# 1. Ler o arquivo original
# ---------------------------------------------------------------
df = pd.read_csv(ARQUIVO_KAGGLE)
df.columns = df.columns.str.strip()  # remove espaços extras nos nomes das colunas
print(f"Arquivo original: {len(df)} anos ({df['YEAR'].min()} a {df['YEAR'].max()})")

# ---------------------------------------------------------------
# 2. Transformar "uma linha por ano" em "uma linha por mês"
#    As colunas D-J-F, M-A-M, J-J-A, S-O-N e metANN são médias já
#    calculadas, então não usamos: ficamos só com o ano e os 12 meses.
# ---------------------------------------------------------------
df = df[["YEAR"] + list(MESES.keys())]
df = df.melt(id_vars="YEAR", var_name="mes", value_name="temperatura")

df["mes"] = df["mes"].map(MESES)  # "JAN" -> 1, "FEB" -> 2, ...
df["data"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["mes"].astype(str) + "-01")
df = df.sort_values("data").reset_index(drop=True)
df = df[["data", "mes", "temperatura"]]

# ---------------------------------------------------------------
# 3. Valores ausentes: no dataset, 999.90 significa "sem medição"
# ---------------------------------------------------------------
df.loc[df["temperatura"] >= 999, "temperatura"] = None
print(f"Meses sem medição: {df['temperatura'].isna().sum()} de {len(df)}")

# A série vai da primeira à última medição disponível.
primeiro = df["temperatura"].first_valid_index()
ultimo = df["temperatura"].last_valid_index()
df = df.loc[primeiro:ultimo].reset_index(drop=True).copy()

# A média usa só medições do treino. A coluna temperatura guarda os alvos observados.
medias_treino = df.loc[df["data"] < DATA_CORTE].groupby("mes")["temperatura"].mean()
df["media_treino"] = df["mes"].map(medias_treino)
if df["media_treino"].isna().any():
    raise ValueError("Faltam medições de treino para calcular a média de algum mês do ano.")

df["preenchido"] = df["temperatura"].isna()
df["temperatura_entrada"] = df["temperatura"].fillna(df["media_treino"].round(2))
df["cidade"] = NOME_CIDADE
df[["data", "temperatura", "temperatura_entrada", "preenchido", "media_treino", "cidade"]].to_csv(
    "dados/serie_tratada.csv", index=False
)
print(f"Meses preenchidos nas entradas: {df['preenchido'].sum()}")

# Criar defasagens com o calendário completo antes de retirar alvos ausentes.
base = df[["data", "media_treino", "temperatura"]].copy()
for meses_atras, coluna in enumerate(COLUNAS_ENTRADA, start=1):
    base[coluna] = df["temperatura_entrada"].shift(meses_atras)

base = base.dropna(subset=[*COLUNAS_ENTRADA, "temperatura"])
base[["data", *COLUNAS_ENTRADA, "media_treino", "temperatura"]].to_csv(
    "dados/base_modelo.csv", index=False
)
print(f"Base do modelo: {len(base)} meses com alvo medido e janela completa")
