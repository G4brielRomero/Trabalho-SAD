# Previsão da temperatura média mensal em Goiânia

Trabalho de Sistemas de Apoio à Decisão da UFMS, Câmpus de Três Lagoas.

**A rede neural ajuda a prever a temperatura média do mês seguinte?** Comparamos uma rede neural, uma árvore de decisão e duas regras simples. A principal referência repete a temperatura do mesmo mês do ano anterior, aproveitando a sazonalidade.

## Dados e método

A [série original](dados/station_goiania.csv) vai de janeiro de 1961 a outubro de 2019, considerando a primeira e a última medição disponível. O arquivo vem do [conjunto de séries de temperatura de cidades brasileiras](https://www.kaggle.com/datasets/volpatto/temperature-timeseries-for-some-brazilian-cities).

- Cada previsão usa 12 entradas. `t-1` é a temperatura do mês anterior e `t-12`, a do mesmo mês do ano anterior.
- O treino usa 453 meses com alvo medido, de janeiro de 1962 a março de 2008. O teste usa 136 meses com alvo medido, de abril de 2008 a outubro de 2019. A ordem do calendário é mantida.
- A série tem 105 lacunas internas. Quando uma lacuna aparece nas entradas, usamos a média das medições daquele mês do ano entre janeiro de 1961 e março de 2008, arredondada para duas casas. Mantemos o campo de medição vazio e excluímos esse mês como alvo.
- A rede tem uma camada oculta de 16 neurônios. As entradas são padronizadas com dados do treino. A árvore tem profundidade máxima 5.

## Resultado para um mês à frente

| Método | MAE (°C) | RMSE (°C) | R² |
| --- | ---: | ---: | ---: |
| Rede neural (MLP) | 0,661 | 0,860 | 0,650 |
| Árvore de decisão | 0,798 | 1,102 | 0,426 |
| Mesmo mês do ano anterior¹ | 0,764 | 0,963 | 0,562 |
| Média do mês no treino | 1,016 | 1,212 | 0,305 |

¹ Quando falta a medição do ano anterior, a regra usa a média mensal do treino arredondada para duas casas, assim como nas entradas. A referência que sempre usa a média mensal mantém a precisão original do cálculo.

A rede errou, em média, **0,103 °C a menos** que a regra do ano anterior nesse recorte. O MAE é a média dos erros absolutos. Em [abril de 2008](resultados/previsoes_teste.csv), a medição foi 25,50 °C e a rede previu 25,17 °C. O erro foi de aproximadamente 0,33 °C. A regra do ano anterior usou 26,35 °C e errou 0,85 °C.

Em [abril de 2018](dados/serie_tratada.csv), faltou a medição. A série preserva esse mês no calendário e usa 25,11 °C quando precisa dele como entrada. Por exemplo, esse valor entra como `t-2` em junho de 2018. Abril de 2018 fica fora do cálculo das métricas.

## Painel

A demonstração histórica começa em novembro de 2019. O painel mostra o desvio em relação à média do mesmo mês no treino. O resultado principal é a previsão de um mês. De 2 a 24 meses, cada previsão alimenta a seguinte. O MAE da tabela foi calculado para um mês à frente e o erro da simulação encadeada fica em aberto.

## Como executar

Na pasta do projeto, use Python 3.12 em um ambiente virtual.

```sh
python -m pip install -r requirements.txt
python 1_preprocessamento.py
python 2_treinamento.py
python 3_avaliacao.py
streamlit run app.py
```

Os scripts geram os dados tratados em `dados/`, os modelos em `modelos/` e as previsões e métricas em `resultados/`. O [roteiro da apresentação](roteiro_apresentacao.md) segue esses mesmos exemplos.
