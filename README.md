# 🌡️ Previsão de Temperatura Média Mensal com Rede Neural

Trabalho bimestral de **Sistemas de Apoio à Decisão**, Bacharelado em Sistemas de Informação, UFMS (Câmpus de Três Lagoas).

Dashboard interativo que prevê a temperatura média mensal de uma cidade brasileira usando uma **Rede Neural (MLP)** e mostra os resultados para apoiar decisões.

---

## 1. Problema

A temperatura afeta decisões em várias áreas:

- **Agronegócio:** planejamento de plantio, irrigação e proteção de rebanhos contra o calor.
- **Energia:** meses mais quentes aumentam o consumo (ar-condicionado).
- **Saúde pública:** ondas de calor pedem campanhas de hidratação e atenção a idosos.

**Pergunta:** *com base nas temperaturas dos últimos 12 meses, qual será a temperatura média dos próximos meses? E ela vai ficar acima ou abaixo do normal para a época?*

## 2. Dataset

- **Fonte:** Kaggle, [Temperature Time-Series for some Brazilian cities](https://www.kaggle.com/datasets/volpatto/temperature-timeseries-for-some-brazilian-cities). Os dados originais são da NOAA GHCN v4, processados pelo NASA GISTEMP v4.
- **Arquivo usado:** `station_goiania.csv` (Goiânia, região Centro-Oeste).
- **Formato original:** uma linha por ano e uma coluna por mês (`JAN` ... `DEC`), com temperaturas médias mensais em °C.
- **Valores ausentes:** meses sem medição aparecem com o código `999.90`.

| Variável | Tipo | Descrição |
|---|---|---|
| `t-1`, `t-2`, ..., `t-12` | **Entrada** | Temperatura média de 1, 2, ..., 12 meses atrás |
| `temperatura` | **Alvo** | Temperatura média do mês que queremos prever |

## 3. Estrutura do projeto

```
previsao-temperatura/
├── dados/
│   ├── station_goiania.csv      # arquivo original do Kaggle
│   ├── serie_tratada.csv        # gerado pelo script 1
│   └── base_modelo.csv          # gerado pelo script 1
├── modelos/
│   ├── rede_neural.joblib       # gerado pelo script 2
│   └── arvore_decisao.joblib    # gerado pelo script 2
├── resultados/
│   ├── metricas.csv             # gerado pelo script 3
│   └── previsoes_teste.csv      # gerado pelo script 3
├── 1_preprocessamento.py        # limpeza e preparação dos dados
├── 2_treinamento.py             # treino dos modelos
├── 3_avaliacao.py               # métricas (MAE, MSE, RMSE, R²)
├── app.py                       # dashboard (Streamlit)
└── requirements.txt
```

## 4. Como executar

**Requisito:** Python 3.10 ou mais novo.

```bash
# 1. Instalar as bibliotecas
pip install -r requirements.txt

# 2. Baixar o dataset do Kaggle e colocar o arquivo station_goiania.csv na pasta dados/

# 3. Rodar os scripts, nesta ordem
python 1_preprocessamento.py
python 2_treinamento.py
python 3_avaliacao.py

# 4. Abrir o dashboard (abre sozinho no navegador)
streamlit run app.py
```

> Para usar outra cidade, troque `ARQUIVO_KAGGLE` e `NOME_CIDADE` no início do `1_preprocessamento.py` e rode os 3 scripts de novo.

## 5. O que cada etapa faz

### 5.1 Pré-processamento (`1_preprocessamento.py`)
1. Transforma a tabela de "uma linha por ano" em "uma linha por mês" (série temporal).
2. Troca o código `999.90` por valor vazio.
3. Remove os meses vazios do começo e do fim da série.
4. Preenche os meses vazios do meio com a **média histórica daquele mês**. Por exemplo, um julho faltante recebe a média de todos os julhos. Isso respeita a sazonalidade.
5. Cria as entradas `t-1` ... `t-12` (janela dos 12 meses anteriores).

A **normalização** (`StandardScaler`) é feita no script 2, dentro do pipeline da rede neural. Assim, a média e o desvio são calculados só com os dados de treino.

### 5.2 Modelos (`2_treinamento.py`)
- **Divisão treino/teste:** 80% dos meses mais antigos para treino, 20% mais recentes para teste. Em série temporal os dados **não são embaralhados**: o modelo aprende com o passado e é testado no "futuro".
- **Rede Neural (MLP)**, o modelo principal: 12 entradas → 1 camada oculta com 16 neurônios (ReLU) → 1 saída. Usa regularização `alpha=1.0` para evitar overfitting.
- **Árvore de Decisão**, para comparação: `max_depth=5`.

### 5.3 Avaliação (`3_avaliacao.py`)
Calcula **MAE, MSE, RMSE e R²** nos dados de teste. Também compara com uma **referência sem IA**: "repetir a temperatura do mesmo mês do ano anterior". Um modelo útil precisa ser melhor que essa referência.

### 5.4 Persistência
Os modelos treinados são salvos com `joblib` em `modelos/*.joblib`. O dashboard carrega esses arquivos, sem precisar treinar de novo.

### 5.5 Dashboard (`app.py`)
- **Previsão:** gráfico e tabela dos próximos 1 a 24 meses, com comparação com a média histórica e alertas de "acima/abaixo do normal".
- **Desempenho do modelo:** tabela de métricas e gráfico real × previsto.
- **Histórico e insights:** série completa, média por mês do ano (sazonalidade) e média por década (tendência de aquecimento).
- **Simulador:** o usuário edita as temperaturas dos últimos 12 meses e vê a previsão mudar ao vivo.

## 6. Resultados

Saída do `3_avaliacao.py` (dados de teste):

| Modelo | MAE (°C) | MSE | RMSE (°C) | R² |
|---|---|---|---|---|
| Rede Neural (MLP) | – | – | – | – |
| Árvore de Decisão | – | – | – | – |
| Referência (repetir o ano anterior) | – | – | – | – |

> Preencha com os valores do arquivo `resultados/metricas.csv` depois de rodar o script 3.

## 7. Limitações e melhorias futuras

- O dataset termina em 2019, então a "previsão futura" começa logo depois do último mês com dados.
- O modelo usa só a temperatura passada. Outras variáveis (chuva, umidade, El Niño) poderiam melhorar a previsão.
- Na previsão de vários meses, cada previsão vira entrada da próxima, então o erro vai se acumulando.
- Melhorias possíveis: testar outras configurações da rede, usar validação cruzada para séries temporais (`TimeSeriesSplit`) e testar redes recorrentes (LSTM).

## 8. Tecnologias

Python · pandas · scikit-learn · joblib · Streamlit · Plotly
