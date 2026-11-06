"""
ETAPA 1 - PRÉ-PROCESSAMENTO DOS DADOS
======================================

O que este script faz:
  1. Lê o arquivo baixado do Kaggle (uma linha por ANO, uma coluna por MÊS).
  2. Transforma em série temporal (uma linha por MÊS).
  3. Troca o código 999.90 (que no dataset significa "sem medição") por vazio (NaN).
  4. Preenche os meses vazios com a média histórica daquele mês.
  5. Cria as colunas de entrada do modelo: a temperatura dos 12 meses anteriores.
  6. Salva o resultado na pasta dados/.

Como rodar:
    python 1_preprocessamento.py
"""

import pandas as pd

# ------------------------- CONFIGURAÇÃO -------------------------
ARQUIVO_KAGGLE = "dados/station_goiania.csv"  # arquivo baixado do Kaggle
NOME_CIDADE = "Goiânia"
MESES_ANTERIORES = 12  # quantos meses do passado o modelo usa para prever o próximo
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

# Os meses vazios no começo e no fim da série (ex.: meses de 2019 que
# ainda não tinham acontecido quando o dataset foi criado) são removidos.
primeiro = df["temperatura"].first_valid_index()
ultimo = df["temperatura"].last_valid_index()
df = df.loc[primeiro:ultimo].reset_index(drop=True)

# ---------------------------------------------------------------
# 4. Preencher os meses vazios que sobraram (no meio da série)
#    Usamos a média histórica do MESMO mês. Exemplo: se falta julho
#    de 1980, colocamos a média de todos os julhos. Assim respeitamos
#    a sazonalidade (julho é sempre mais frio que outubro).
# ---------------------------------------------------------------
df["preenchido"] = df["temperatura"].isna()  # guarda quais meses foram preenchidos
media_de_cada_mes = df.groupby("mes")["temperatura"].transform("mean")
df["temperatura"] = df["temperatura"].fillna(media_de_cada_mes).round(2)
print(f"Meses preenchidos com a média do mês: {df['preenchido'].sum()}")

df["cidade"] = NOME_CIDADE
df[["data", "temperatura", "preenchido", "cidade"]].to_csv("dados/serie_tratada.csv", index=False)

# ---------------------------------------------------------------
# 5. Criar as entradas do modelo (janela de 12 meses)
#    Para prever a temperatura de um mês, o modelo recebe as
#    temperaturas dos 12 meses anteriores:
#       t-1  = temperatura do mês anterior
#       t-2  = temperatura de 2 meses atrás
#       ...
#       t-12 = temperatura do mesmo mês no ano passado
# ---------------------------------------------------------------
base = df[["data", "temperatura"]].copy()
for i in range(1, MESES_ANTERIORES + 1):
    base[f"t-{i}"] = base["temperatura"].shift(i)

# As 12 primeiras linhas não têm 12 meses anteriores, então saem
base = base.dropna().reset_index(drop=True)

# Coloca o alvo (temperatura) como última coluna, só para ficar mais legível
base = base[["data"] + [f"t-{i}" for i in range(1, MESES_ANTERIORES + 1)] + ["temperatura"]]
base.to_csv("dados/base_modelo.csv", index=False)

print(f"\nBase do modelo: {len(base)} linhas")
print(base.head())
print("\nArquivos salvos: dados/serie_tratada.csv e dados/base_modelo.csv")
