# Roteiro falado para apresentar o painel de temperatura em dupla

**Duração-alvo de 20 minutos.** Tudo entre **[colchetes]** é uma ação na tela. Os trechos em bloco são falas que vocês podem usar como estão no primeiro ensaio. Os horários são marcas no cronômetro, contado desde o início da apresentação.

| Quem fala | Tempo | Responsabilidade |
| --- | --- | --- |
| Pessoa A · Moisés | 00:00–10:00 | Abre o painel, apresenta a decisão, explica os dados, o treino e os modelos. |
| Pessoa B · Gabriel | 10:00–20:00 | Recalcula abril de 2008, troca de modelo, mostra o simulador e fecha com os resultados. |

A troca acontece **uma vez**, com a aba **Desempenho** aberta. A Pessoa A entrega a fala e o controle do computador à Pessoa B. Se preferirem inverter quem faz cada parte, troquem os nomes e mantenham a ordem das telas.

## Prepare o computador antes do ensaio

1. [Abra o painel seguindo a seção **Como reproduzir** do README. Deixe a aba **Previsão** selecionada e marque **Rede Neural (MLP)** na barra lateral.] [README](README.md#como-reproduzir)
2. [Abra os três CSVs no editor de código e deixe cada arquivo na linha indicada. No original, mantenha o cabeçalho visível ou saiba que as primeiras colunas são `YEAR, JAN, FEB, MAR, APR, MAY, JUN`.] [Original, linha 59](dados/station_goiania.csv#L59) · [Série tratada, linha 689](dados/serie_tratada.csv#L689) · [Base do modelo, linha 574](dados/base_modelo.csv#L574)
3. [No editor, abra também `configuracao.py` e `2_treinamento.py`. Deixe a pasta `modelos/` expandida na lateral.] [Corte do teste](configuracao.py#L1) · [Treinamento](2_treinamento.py#L19)
4. [Faça a checagem rápida. Em **Desempenho**, expanda **Reproduzir uma previsão do teste**. O seletor **Mês com medição** deve estar em **04/2008**. Em **Simulador**, o horizonte deve estar em **1** e a última entrada, **10/2019**, deve estar em **28.30**. Volte para **Previsão**.]

**Para se localizar antes de decorar a fala:** os arquivos mostram de onde vêm as entradas. A aba **Previsão** apresenta um mês depois do fim da série. A aba **Desempenho** volta a meses em que conhecemos a medição para conferir o erro. O **Simulador** muda uma entrada para mostrar como o modelo reage.

## Pessoa A · problema, dados e treino

### A1 · 00:00–02:30 · Apresente a decisão

**[Abra a aba Previsão. Confirme Rede Neural (MLP) na lateral. Aponte a pergunta abaixo do título.]**

> Bom dia. Nosso trabalho usa uma série de temperatura de Goiânia para responder a uma pergunta de planejamento. A previsão do próximo mês dá motivo para revisar um plano mensal de ventilação?
>
> Pensem em um gestor que quer se preparar para um mês mais quente do que costuma ser naquela época do ano. O painel reúne uma previsão para o próximo mês e uma referência para entender se o valor está alto ou baixo para aquele mês.

**[Aponte os três cartões da esquerda para a direita. Dê uma pausa curta entre os valores.]**

> A primeira previsão é para novembro de 2019. A rede estima 26,84 graus. A média dos novembros no período de treino foi 25,56 graus. Subtraindo a média da previsão, chegamos a 1,28 grau acima da referência.
>
> A decisão que esse número sugere é revisar o planejamento mensal e consultar previsões atualizadas antes de ajustar a operação em dias específicos. Estamos demonstrando essa decisão em um caso histórico.

**[Aponte a linha azul, a linha verde pontilhada e o losango laranja no gráfico.]**

> A linha azul é o que foi medido. A verde pontilhada é a média sazonal do treino. O losango laranja é a previsão. A última medição da série é de outubro de 2019, então o losango vem logo depois, em novembro.
>
> Agora vou mostrar de onde vieram os valores que a rede recebe.

### A2 · 02:30–04:30 · Mostre a origem dos dados

**[Aponte Fonte dos dados no Kaggle na lateral. Troque para `station_goiania.csv`, na linha do ano 2018. Aponte o ano e os primeiros meses.]**

> A fonte é um conjunto de séries de temperatura de cidades brasileiras no Kaggle. Escolhemos Goiânia. No arquivo original, cada linha corresponde a um ano. Aqui, na linha de 2018, as colunas de janeiro a dezembro guardam temperaturas médias mensais.
>
> O primeiro script transforma esse formato em uma série com uma linha por mês e coloca tudo em ordem cronológica. A primeira medição é de janeiro de 1961 e a última é de outubro de 2019.

**[Aponte os valores 999.90 em APR e MAY. Troque para `serie_tratada.csv`, na linha 2018-04-01. Aponte a coluna de medição vazia e a coluna de entrada com 25.11.]**

> Em abril e maio de 2018 aparece 999,90. É o código do arquivo para ausência de medição. A coluna de temperatura medida fica vazia. Há 105 lacunas desse tipo entre a primeira e a última medição da série.
>
> Quando o modelo precisa de abril de 2018 como entrada, usamos 25,11 graus, que é a média dos meses de abril no treino. O arquivo tratado guarda a medição e a entrada em colunas separadas. Assim dá para ver de onde veio cada valor.

### A3 · 04:30–06:30 · Explique o que entra e o que o modelo prevê

**[Troque para `base_modelo.csv`, na linha 2018-06-01. Aponte a data, as duas primeiras entradas e a última coluna, `temperatura`.]**

> Esta linha representa junho de 2018. O modelo recebe doze temperaturas dos meses anteriores e devolve uma estimativa da temperatura média de junho. Chamamos o mês anterior de t menos um, o mês anterior a ele de t menos dois, e seguimos até t menos doze, que é o mesmo mês do ano anterior.
>
> Nesta linha, t menos um é maio de 2018, com 23,38 graus preenchidos pela média de maio no treino. T menos dois é abril de 2018, com os 25,11 graus que acabamos de ver.
>
> O alvo de junho é 23,77 graus, que foi medido. Essa é a temperatura que usaríamos para verificar o erro em junho. Se quiséssemos verificar o erro de uma previsão para abril de 2018, precisaríamos da medição de abril. Como ela está vazia, abril fica fora da avaliação.
>
> Resumindo a linha que vocês estão vendo, as doze primeiras temperaturas da janela entram no modelo. A temperatura do mês-alvo fica separada e serve para aprender no treino ou para conferir o erro no teste.

### A4 · 06:30–10:00 · Apresente o corte, os modelos e passe a fala

**[Abra `configuracao.py` e aponte a data 2008-04-01.]**

> O corte foi feito pelo calendário. O treino vai até março de 2008 e tem 453 meses com alvo medido. O teste começa em abril de 2008 e termina em outubro de 2019, com 136 meses com alvo medido.
>
> Para prever cada mês do teste, usamos o que estaria disponível até o mês anterior. Por exemplo, na previsão de abril, março é o mês mais recente. Quando avançamos para maio, abril já pode entrar como medição na janela. O modelo continua sendo o mesmo. Ele foi ajustado no treino.

**[Abra `2_treinamento.py`. Mostre `StandardScaler()` e `hidden_layer_sizes=(16,)`. Depois aponte `max_depth=5`.]**

> Treinamos dois modelos. O primeiro é uma rede neural MLP, com uma camada intermediária de 16 neurônios. Antes de entrar na rede, cada coluna é padronizada com a média e o desvio-padrão aprendidos no treino. A rede aprende combinações das doze entradas para estimar uma temperatura.
>
> O segundo modelo é uma árvore de regressão. Ela faz perguntas sobre as entradas, por exemplo se a temperatura de um mês anterior está acima ou abaixo de certo valor. Conforme as respostas, a árvore chega a uma estimativa. A profundidade máxima de cinco limita quantos níveis de decisões ela pode ter.

**[Aponte `rede_neural.joblib` e `arvore_decisao.joblib` na pasta modelos. Volte ao painel, abra Desempenho e pare na tabela.]**

> Os dois modelos foram salvos em arquivos joblib. A rede foi salva junto com a etapa de padronização. Assim o painel consegue carregar o modelo escolhido e usá-lo para calcular uma previsão na hora.
>
> Na tabela, também temos duas referências simples. A primeira repete a medição do mesmo mês do ano anterior. Quando essa medição falta, usa a média do mês no treino. A segunda referência usa sempre essa média. Comparamos os quatro métodos nos mesmos 136 meses.
>
> Agora minha dupla vai abrir o primeiro mês do teste, fazer o cálculo na tela e mostrar como a gente chega ao resultado da tabela.

**[Entregue o controle à Pessoa B. Mantenha Desempenho aberto e Rede Neural (MLP) selecionada.]**

## Gabriel · demonstração, resultados e conclusão

### B1 · 10:00–12:30 · Recalcule abril de 2008 ao vivo

**[Em Desempenho, role até Reproduzir uma previsão do teste. Clique para expandir. Em Mês com medição, confirme 04/2008.]**

> Vou usar abril de 2008, o primeiro mês do teste. Aqui o painel carrega as doze entradas daquele mês e executa o modelo selecionado.

**[Aponte `t-1` na tabela. Depois aponte `t-12`. Se a última coluna estiver fora da tela, role a tabela para a direita.]**

> A entrada mais recente é março de 2008, com 25,27 graus. A mais antiga é abril de 2007, com 26,35 graus. A rede usa essas duas e as outras dez temperaturas.

**[Aponte Medição de 04/2008, Previsão calculada agora e Erro absoluto, nessa ordem.]**

> A medição de abril foi 25,50. A rede previu 25,17. A distância entre os dois valores é de cerca de 0,33 grau. Esse é o erro absoluto da rede neste mês.
>
> A regra do ano anterior usaria 26,35 graus, que é justamente o valor de t menos doze. A distância até os 25,50 medidos é de 0,85 grau. Neste mês, a rede ficou mais perto da medição do que essa regra.

### B2 · 12:30–14:00 · Troque de modelo e explique a comparação

**[Na barra lateral, marque Árvore de Decisão. Continue em Desempenho, com 04/2008 selecionado. Se necessário, abra novamente Reproduzir uma previsão do teste. Aponte 25,30 °C e 0,20 °C.]**

> Agora mantive o mesmo mês e as mesmas entradas. Troquei só o modelo. A árvore previu 25,30 graus e ficou a 0,20 grau da medição.
>
> Neste abril, a árvore ficou mais perto que a rede. Para comparar o desempenho geral, precisamos fazer essa conta mês a mês. O MAE é a média dos erros absolutos dos 136 meses. Então um modelo pode ganhar em um exemplo e ter erro maior em média.
>
> Se a gente elevar cada erro ao quadrado antes de tirar a média, obtém o MSE. Um erro de dois graus, por exemplo, contribui com quatro para essa média. Tirar a raiz do MSE devolve a unidade em graus Celsius. Esse é o RMSE que aparece na tabela.

### B3 · 14:00–16:30 · Edite uma entrada no simulador

**[Na lateral, marque novamente Rede Neural (MLP). Volte ao topo e clique em Simulador. Confirme Horizonte do cenário (meses) em 1. Aponte 26,84 °C no cartão.]**

> Agora vou mostrar a resposta da rede a um cenário. Com as temperaturas originais, ela volta a prever 26,84 graus para novembro de 2019.

**[Na tabela, encontre a última linha, 10/2019. Dê dois cliques na célula de temperatura dessa linha. Substitua `28.30` por `30.30` e pressione Enter. Aguarde o cartão mostrar 27,83 °C.]**

> Somei dois graus somente à entrada de outubro. As outras onze entradas continuam iguais. Nesse cenário, a previsão de novembro subiu para 27,83 graus. O cartão mostra a diferença de 0,99 grau em relação à previsão com os valores originais.
>
> Esse número descreve a resposta da rede ao cenário que acabei de montar. Se eu escolhesse mais de um mês, a previsão de novembro entraria no cálculo de dezembro, e assim por diante. As métricas do nosso teste foram calculadas para um mês à frente.

**[Restaure `28.30`, pressione Enter e confirme que o cartão voltou a 26,84 °C. Volte para Desempenho.]**

### B4 · 16:30–18:30 · Diga o que os resultados permitem concluir

**[No topo de Desempenho, aponte a coluna MAE (°C). Mostre primeiro Rede Neural (MLP) e depois Referência (ano anterior).]**

> Aqui está a comparação dos 136 meses. A rede teve MAE de 0,661 grau. A regra do ano anterior teve 0,764. A diferença foi de 0,103 grau de erro absoluto médio, uma redução de cerca de 13,5% em relação à regra.
>
> A rede teve erro menor que essa regra em 80 dos 136 meses. A árvore teve MAE de 0,798 grau, apesar de ter saído melhor no caso de abril que acabamos de mostrar. As outras colunas ajudam a completar a avaliação. O RMSE dá mais peso aos erros grandes, e o R² compara o modelo com uma previsão constante baseada na média dos alvos do teste.

**[Role até Em quais meses o erro foi maior? Passe o mouse sobre Set e deixe a turma ver o valor e a contagem. Depois passe sobre Fev.]**

> A qualidade também muda conforme o mês. Setembro teve o maior MAE da rede, 1,16 grau, calculado com doze medições de setembro. Aqui em fevereiro, o MAE ficou perto de 0,37 grau, com onze medições. O gráfico agrupa os mesmos meses do calendário em anos diferentes. Esse recorte ajuda a ver onde vale olhar o erro com mais atenção. Os resultados descrevem o teste histórico de Goiânia, com horizonte de um mês.

### B5 · 18:30–20:00 · Volte à decisão e encerre

**[Volte ao topo e clique em Previsão. Confirme Rede Neural (MLP). Aponte +1,28 °C no cartão Desvio previsto.]**

> Voltando à pergunta do início, a previsão de novembro ficou 1,28 grau acima da média do mês no treino. No exemplo de ventilação, eu usaria esse desvio para revisar o plano mensal e olhar informações atualizadas antes de agir.
>
> O que sustenta essa leitura é a comparação com uma regra simples. No teste de um mês à frente, a rede reduziu o erro médio em relação a repetir a temperatura do mesmo mês do ano anterior. O painel também mostra onde essa vantagem variou e permite refazer um caso para conferir o cálculo.
>
> Obrigado. Podemos abrir qualquer etapa do painel ou do treinamento para responder às perguntas.

**[Deixe o painel aberto em Previsão para a arguição.]**

## Como ensaiar sem depender da leitura

1. **Primeira passada, sem cronômetro.** Façam todos os cliques lendo as falas em voz alta. Ao terminar cada trecho, digam em uma frase o que acabaram de mostrar.
2. **Segunda passada, com cronômetro.** A Pessoa A deve chegar à tabela de desempenho perto de **10:00**. A Pessoa B deve sair do simulador perto de **16:30**. Se o ensaio ficar abaixo de 15 minutos, façam pausas para mostrar as colunas dos CSVs, os valores de abril e a troca de modelo. Se passar do tempo, encurtem a leitura do código e a descrição das métricas, mantendo as duas ações ao vivo.
3. **Terceira passada, só com as indicações entre colchetes.** Cada pessoa pratica também uma pergunta da parte da outra. A arguição é individual.

## Respostas prontas para a arguição

| Se perguntarem | Você pode responder |
| --- | --- |
| Qual é o alvo e o que entra no modelo? | O alvo é a temperatura média do mês a prever. Entram as temperaturas dos 12 meses anteriores. Em abril de 2008, março de 2008 é t menos um e abril de 2007 é t menos doze. |
| Por que 12 meses? | A janela cobre um ciclo anual. Assim o modelo vê tanto o mês anterior quanto o mesmo mês do ano passado. |
| Por que preencher abril com média de abril? | Isso preserva a sazonalidade. A média vem do treino e entra como substituição quando abril faz parte da janela. Para medir o erro de uma previsão de abril, ainda precisaríamos da observação real. |
| Por que separar por data? | A avaliação deve simular o uso de meses anteriores para prever o próximo. Uma divisão aleatória pode colocar no treino informações de meses posteriores aos que queremos avaliar. |
| O teste usa valores de outros meses do teste? | Usa meses que já passaram como entradas. Em maio de 2008, abril já pode estar na janela com a temperatura medida. Os modelos continuam fixos e as médias de preenchimento foram calculadas antes do corte. |
| Por que 16 neurônios e profundidade 5? | São configurações de tamanho limitado escolhidas para comparar a rede e a árvore neste conjunto. Para refinar a escolha, eu compararia alternativas com validação cronológica dentro do treino. |
| Como explicar MAE, MSE e RMSE? | MAE é a média das distâncias entre previsões e medições, em graus Celsius. MSE calcula a média dessas diferenças ao quadrado, em graus Celsius ao quadrado. O RMSE é a raiz do MSE, volta a graus Celsius e dá mais peso aos erros grandes do que o MAE. |
| Como explicar R² de 0,650? | A soma dos erros quadráticos da rede ficou cerca de 65% abaixo da obtida ao usar uma constante igual à média dos alvos do teste. |
| Qual foi o erro de novembro de 2019? | O arquivo marca novembro de 2019 como mês sem medição. O painel mostra a previsão e o desvio em relação à média de novembro no treino. O erro exigiria a temperatura medida naquele novembro. |
| Posso usar o MAE como margem de erro? | O MAE resume o tamanho médio dos erros no teste. Para dizer que a próxima previsão tem uma faixa com certa probabilidade, precisaríamos calcular e validar um intervalo de previsão. |
| Aumentar outubro em dois graus causa o aumento de novembro? | O simulador mostra que, mantendo as outras entradas fixas, essa edição fez o modelo subir a estimativa em 0,99 grau. Isso descreve o comportamento da rede no cenário. |
| Como reproduzir? | O primeiro script trata os dados. O segundo ajusta e salva os modelos. O terceiro avalia o teste e gera os CSVs de resultados. O painel usa esses arquivos para exibir as métricas e executar o modelo. Os comandos estão no README. |

Antes da aula, confirmem a versão a ser apresentada no repositório público e enviem o link na submissão.
