# Previsão da temperatura média mensal em Goiânia

Trabalho bimestral de Sistemas de Apoio à Decisão da UFMS, Câmpus de Três Lagoas.

**Pergunta do painel** — A previsão do próximo mês sugere revisar o plano mensal de ventilação? A demonstração usa uma série histórica de Goiânia. Uma temperatura prevista acima da média do mês serve como sinal para investigar as necessidades da operação. Decisões diárias exigem dados e previsões atualizados.

## Dados e método

A [série original](dados/station_goiania.csv) vem do [conjunto de séries de temperatura de cidades brasileiras no Kaggle](https://www.kaggle.com/datasets/volpatto/temperature-timeseries-for-some-brazilian-cities). A primeira medição é de janeiro de 1961 e a última, de outubro de 2019.

- **Entradas e alvo.** Cada linha usa 12 temperaturas mensais anteriores para prever a temperatura média do mês-alvo, em °C. `t-1` é o mês anterior e `t-12`, o mesmo mês do ano anterior.
- **Separação cronológica.** O treino usa 453 alvos medidos de janeiro de 1962 a março de 2008. O teste usa 136 alvos medidos de abril de 2008 a outubro de 2019. Em cada passo do teste, os meses anteriores já estão disponíveis como entradas. Os modelos permanecem fixos durante o teste.
- **Lacunas.** A série tem 105 meses sem medição entre o primeiro e o último registro. O valor `999.90` indica medição ausente. Para formar as entradas, cada lacuna recebe a média daquele mês do ano entre janeiro de 1961 e março de 2008, arredondada para duas casas. Meses sem medição ficam fora dos alvos de treino e teste.
- **Modelos.** A rede MLP tem 16 neurônios em uma camada oculta, ativação ReLU e entradas padronizadas com estatísticas do treino. A árvore de regressão tem profundidade máxima 5. Ambos são exportados em arquivos `.joblib`.

## Desempenho de um mês à frente

| Método                     | MAE (°C) | MSE (°C²) | RMSE (°C) |    R² |
| -------------------------- | -------: | --------: | --------: | ----: |
| Rede neural (MLP)          |    0,661 |     0,740 |     0,860 | 0,650 |
| Árvore de decisão          |    0,798 |     1,215 |     1,102 | 0,426 |
| Mesmo mês do ano anterior¹ |    0,764 |     0,927 |     0,963 | 0,562 |
| Média do mês no treino     |    1,016 |     1,470 |     1,212 | 0,305 |

¹ Quando falta a medição do ano anterior, a regra usa a média mensal do treino arredondada para duas casas, assim como nas entradas. A referência que sempre usa a média mensal mantém a precisão original do cálculo.

A rede reduziu o MAE em **0,103 °C (cerca de 13,5%)** frente à regra do ano anterior neste teste. Teve menor erro absoluto em **80 dos 136 meses**. Setembro teve o maior MAE da rede por mês do ano, **1,16 °C em 12 medições**. O ganho agregado e o recorte por mês descrevem o desempenho na série histórica.

Em [abril de 2008](resultados/previsoes_teste.csv), a medição foi de 25,50 °C, a rede previu 25,17 °C e o erro absoluto foi de 0,33 °C. A regra do ano anterior usou 26,35 °C e errou 0,85 °C. Em [abril de 2018](dados/serie_tratada.csv), a medição ficou vazia. A média de abril no treino, 25,11 °C, aparece como `t-2` na linha de junho de 2018. Abril de 2018 fica fora do cálculo das métricas.

## Como ler o painel

- **Previsão.** A rede estima 26,84 °C para novembro de 2019, ou 1,28 °C acima da média dos novembros no treino (25,56 °C). O gráfico distingue medição, média mensal do treino e previsão.
- **Desempenho.** A tabela compara os métodos, o gráfico mostra o MAE por mês do ano e o seletor permite recalcular ao vivo a previsão de um mês medido no teste.
- **Histórico.** O filtro de anos mostra sazonalidade e cobertura das medições no recorte. Na década de 1990, por exemplo, apenas 57 de 120 meses têm medição.
- **Simulador.** As 12 entradas podem ser editadas. De 2 a 24 meses, cada previsão alimenta a seguinte. A avaliação reportada cobre um mês à frente. Os resultados do simulador ilustram a resposta do modelo a um cenário e a previsão é pontual.

## Como reproduzir

Execute a partir da pasta que contém este README. O fluxo completo foi conferido com Python 3.14 e as versões de `requirements.txt`. Só a criação e a ativação do ambiente virtual mudam conforme o sistema.

**Linux e macOS**

```sh
python3.14 -m venv .venv
source .venv/bin/activate
```

**Windows (Prompt de Comando)**

```bat
py -3.14 -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear o script de ativação, execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` uma vez e tente de novo. Com o ambiente ativo, o prompt começa com `(.venv)`. Os comandos a seguir são iguais em todos os sistemas:

```sh
python -m pip install -r requirements.txt
python 1_preprocessamento.py
python 2_treinamento.py
python 3_avaliacao.py
python -m streamlit run app.py
```

O painel abre em `http://localhost:8501`. Para encerrar, use `Ctrl+C` no terminal.

Os scripts geram a série e a matriz de entradas em `dados/`, os dois modelos em `modelos/` e as previsões e métricas em `resultados/`.
