# Roteiro da apresentação

1. **Comece pela pergunta.** Se eu já conheço a temperatura média dos últimos 12 meses em Goiânia, quanto uma rede neural melhora a previsão do mês seguinte em relação a repetir o valor do mesmo mês do ano passado?
2. **Mostre abril de 2008 na base.** Para prever esse mês, `t-1` é março de 2008 (25,27 °C) e `t-12` é abril de 2007 (26,35 °C). Os 12 valores entram na rede, que devolve uma temperatura. A medição foi 25,50 °C. Mostre a conta da rede, `|25,50 − 25,17| ≈ 0,33 °C`, e a da referência, `|25,50 − 26,35| = 0,85 °C`. O MAE faz a média dessas diferenças absolutas.
3. **Mostre abril de 2018 na série tratada.** O campo da medição está vazio. Usamos 25,11 °C, a média de abril no treino arredondada para duas casas, como entrada quando necessário. Em junho de 2018 esse valor aparece como `t-2`. Para calcular o erro de uma previsão de abril de 2018, precisaríamos da medição de abril. Por isso o mês fica fora do teste.
4. **Mostre a tabela de desempenho.** Comparamos os quatro métodos nos mesmos 136 meses medidos. A rede teve MAE de 0,661 °C, contra 0,764 °C da regra do ano anterior, 0,798 °C da árvore e 1,016 °C da média mensal do treino.
5. **Feche no painel.** A previsão exibida começa em novembro de 2019, após o último registro da série. Compare a previsão de novembro com a média dos novembros do treino. Depois mude o horizonte para mostrar a simulação. Nela, uma previsão passa a ser entrada da próxima. Os resultados da tabela sustentam a comparação de um mês à frente.

Antes de apresentar, refaça a conta de abril de 2008 e localize abril e junho de 2018 nos CSVs.
