"""
DASHBOARD - Previsão de Temperatura Média Mensal
=================================================

Antes de abrir o dashboard, rode os scripts 1, 2 e 3 (nessa ordem).

Como rodar:
    streamlit run app.py
"""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Previsão de Temperatura", page_icon="🌡️", layout="wide")

# Cores dos gráficos
AZUL = "#2a78d6"     # dados reais / histórico
LARANJA = "#eb6834"  # previsões do modelo

# Diferença (em °C) em relação à média do mês para considerar "acima/abaixo do normal"
LIMITE_ALERTA = 0.5

ARQUIVOS_MODELOS = {
    "Rede Neural (MLP)": "modelos/rede_neural.joblib",
    "Árvore de Decisão": "modelos/arvore_decisao.joblib",
}

NOMES_MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
               "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


# =================================================================
# CARREGAR OS DADOS (gerados pelos scripts 1 e 3)
# =================================================================
serie = pd.read_csv("dados/serie_tratada.csv", parse_dates=["data"])
metricas = pd.read_csv("resultados/metricas.csv")
previsoes_teste = pd.read_csv("resultados/previsoes_teste.csv", parse_dates=["data"])

cidade = serie["cidade"].iloc[0]
serie["mes"] = serie["data"].dt.month
serie["ano"] = serie["data"].dt.year

# Temperatura média de cada mês do ano (Jan, Fev, ...) em todo o histórico
media_de_cada_mes = serie.groupby("mes")["temperatura"].mean()


# =================================================================
# FUNÇÃO PARA DESENHAR GRÁFICOS DE LINHA (com Plotly)
# =================================================================
def grafico_de_linhas(tabela, colunas, cores, formato_data="%m/%Y"):
    fig = px.line(tabela, x="data", y=colunas, color_discrete_sequence=cores,
                  labels={"data": "Mês", "value": "Temperatura (°C)", "variable": ""})
    fig.update_xaxes(tickformat=formato_data, hoverformat="%m/%Y")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
    return fig


# =================================================================
# FUNÇÃO DE PREVISÃO
# O modelo prevê 1 mês por vez. Para prever vários meses, usamos a
# própria previsão como entrada do mês seguinte (previsão recursiva).
# =================================================================
def prever_proximos_meses(modelo, ultimas_temperaturas, quantidade_meses):
    temperaturas = list(ultimas_temperaturas)
    previsoes = []
    for _ in range(quantidade_meses):
        # t-1 = último valor da lista, t-2 = penúltimo, ..., t-12
        entrada = pd.DataFrame([{f"t-{i}": temperaturas[-i] for i in range(1, 13)}])
        valor_previsto = modelo.predict(entrada)[0]
        previsoes.append(valor_previsto)
        temperaturas.append(valor_previsto)
    return previsoes


# =================================================================
# BARRA LATERAL (filtros)
# =================================================================
st.sidebar.title("⚙️ Configurações")

nome_modelo = st.sidebar.radio("Modelo", list(ARQUIVOS_MODELOS.keys()))
meses_para_prever = st.sidebar.slider("Quantos meses prever", min_value=1, max_value=24, value=12)

ano_min, ano_max = int(serie["ano"].min()), int(serie["ano"].max())
anos = st.sidebar.slider("Período do histórico", ano_min, ano_max, (ano_min, ano_max))

st.sidebar.markdown("---")
st.sidebar.caption("Dados: Kaggle — *Temperature Time-Series for some Brazilian cities* "
                   "(NASA GISTEMP / NOAA GHCN).")

modelo = joblib.load(ARQUIVOS_MODELOS[nome_modelo])


# =================================================================
# PREVISÃO DOS PRÓXIMOS MESES
# =================================================================
ultimas_12 = serie["temperatura"].iloc[-12:]
valores_previstos = prever_proximos_meses(modelo, ultimas_12, meses_para_prever)

ultima_data = serie["data"].iloc[-1]
datas_futuras = pd.date_range(ultima_data + pd.DateOffset(months=1), periods=meses_para_prever, freq="MS")

previsao = pd.DataFrame({"data": datas_futuras, "previsao": valores_previstos})
previsao["media_historica"] = previsao["data"].dt.month.map(media_de_cada_mes)
previsao["diferenca"] = previsao["previsao"] - previsao["media_historica"]


def situacao(diferenca):
    if diferenca > LIMITE_ALERTA:
        return "🔺 Acima do normal"
    if diferenca < -LIMITE_ALERTA:
        return "🔻 Abaixo do normal"
    return "✅ Normal"


previsao["situacao"] = previsao["diferenca"].apply(situacao)


# =================================================================
# CABEÇALHO E INDICADORES
# =================================================================
st.title(f"🌡️ Previsão de Temperatura Média Mensal — {cidade}")
st.caption(f"Histórico de {serie['data'].iloc[0]:%m/%Y} a {ultima_data:%m/%Y} · Modelo selecionado: **{nome_modelo}**")

mae = metricas.loc[metricas["Modelo"] == nome_modelo, "MAE"].iloc[0]
primeira = previsao.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Média histórica", f"{serie['temperatura'].mean():.1f} °C")
col2.metric(f"Último registro ({ultima_data:%m/%Y})", f"{serie['temperatura'].iloc[-1]:.1f} °C")
col3.metric(f"Previsão para {primeira['data']:%m/%Y}", f"{primeira['previsao']:.1f} °C",
            delta=f"{primeira['diferenca']:+.1f} °C vs. média do mês", delta_color="inverse")
col4.metric("Erro médio do modelo (MAE)", f"± {mae:.2f} °C")


# =================================================================
# ABAS
# =================================================================
aba_previsao, aba_modelo, aba_historico, aba_simulador = st.tabs(
    ["📈 Previsão", "🎯 Desempenho do modelo", "📊 Histórico e insights", "🧪 Simulador"]
)

# -----------------------------------------------------------------
# ABA 1 - PREVISÃO
# -----------------------------------------------------------------
with aba_previsao:
    st.subheader(f"Previsão para os próximos {meses_para_prever} meses")

    # Últimos 3 anos do histórico + previsão no mesmo gráfico
    historico = serie[["data", "temperatura"]].tail(36).rename(columns={"temperatura": "Histórico"})
    futuro = previsao[["data", "previsao"]].rename(columns={"previsao": "Previsão"})
    # repete o último ponto real na linha de previsão, só para as linhas se ligarem
    ligacao = pd.DataFrame({"data": [ultima_data], "Previsão": [serie["temperatura"].iloc[-1]]})
    grafico = pd.concat([historico, ligacao, futuro]).groupby("data").first().reset_index()

    st.plotly_chart(grafico_de_linhas(grafico, ["Histórico", "Previsão"], [AZUL, LARANJA]))

    tabela = pd.DataFrame({
        "Mês": previsao["data"].dt.strftime("%m/%Y"),
        "Previsão (°C)": previsao["previsao"].round(1),
        "Média histórica do mês (°C)": previsao["media_historica"].round(1),
        "Diferença (°C)": previsao["diferenca"].round(1),
        "Situação": previsao["situacao"],
    })
    st.dataframe(tabela, hide_index=True, width="stretch")

    # Apoio à decisão: destacar os meses fora do normal
    quentes = tabela.loc[previsao["diferenca"] > LIMITE_ALERTA, "Mês"].tolist()
    frios = tabela.loc[previsao["diferenca"] < -LIMITE_ALERTA, "Mês"].tolist()

    st.markdown("#### 💡 Apoio à decisão")
    if quentes:
        st.warning(
            f"**Meses previstos acima do normal:** {', '.join(quentes)}.  \n"
            "Sugestões: planejar maior demanda de energia (ar-condicionado), reforçar a "
            "irrigação e o manejo de lavouras e rebanhos contra estresse térmico, e preparar "
            "campanhas de saúde pública (hidratação)."
        )
    if frios:
        st.info(
            f"**Meses previstos abaixo do normal:** {', '.join(frios)}.  \n"
            "Sugestões: atenção a culturas sensíveis ao frio e planejamento de estoques "
            "de produtos de inverno."
        )
    if not quentes and not frios:
        st.success("Todos os meses previstos estão dentro do normal para a época (±"
                   f"{LIMITE_ALERTA} °C da média histórica).")

# -----------------------------------------------------------------
# ABA 2 - DESEMPENHO DO MODELO
# -----------------------------------------------------------------
with aba_modelo:
    st.subheader("Métricas nos dados de teste")
    st.caption("O teste usa os 20% de meses mais recentes, que o modelo **não viu** durante o treino.")
    st.dataframe(metricas, hide_index=True, width="stretch")

    st.markdown(
        "- **MAE**: em média, quantos °C a previsão erra.\n"
        "- **RMSE**: parecido com o MAE, mas pune mais os erros grandes.\n"
        "- **R²**: quanto da variação da temperatura o modelo explica (1 = perfeito).\n"
        "- **Referência**: estratégia sem IA, que repete a temperatura do mesmo mês do ano anterior. "
        "Um bom modelo precisa ser melhor que ela."
    )

    st.subheader(f"Real × Previsto — {nome_modelo}")
    st.plotly_chart(grafico_de_linhas(previsoes_teste, ["Real", nome_modelo], [AZUL, LARANJA], "%Y"))

# -----------------------------------------------------------------
# ABA 3 - HISTÓRICO E INSIGHTS
# -----------------------------------------------------------------
with aba_historico:
    filtrado = serie[(serie["ano"] >= anos[0]) & (serie["ano"] <= anos[1])]

    st.subheader(f"Série histórica ({anos[0]} a {anos[1]})")
    st.plotly_chart(grafico_de_linhas(filtrado, ["temperatura"], [AZUL], "%Y").update_layout(showlegend=False))
    st.caption(f"{int(serie['preenchido'].sum())} meses sem medição foram preenchidos "
               "com a média histórica do mês (pré-processamento).")

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Média por mês do ano")
        por_mes = filtrado.groupby("mes")["temperatura"].mean()
        # "01 Jan", "02 Fev"... o número na frente mantém a ordem correta no gráfico
        por_mes.index = [f"{m:02d} {NOMES_MESES[m - 1]}" for m in por_mes.index]
        st.bar_chart(por_mes, color=AZUL, x_label="Mês", y_label="Temperatura (°C)")
        st.write(f"Mês mais quente: **{por_mes.idxmax()[3:]}** ({por_mes.max():.1f} °C) · "
                 f"Mês mais frio: **{por_mes.idxmin()[3:]}** ({por_mes.min():.1f} °C)")

    with col_dir:
        st.subheader("Média por década")
        filtrado = filtrado.assign(decada=(filtrado["ano"] // 10 * 10).astype(str) + "s")
        por_decada = filtrado.groupby("decada")["temperatura"].mean()
        st.bar_chart(por_decada, color=AZUL, x_label="Década", y_label="Temperatura (°C)")
        if len(por_decada) > 1:
            variacao = por_decada.iloc[-1] - por_decada.iloc[0]
            st.write(f"Variação entre a primeira ({por_decada.index[0]}) e a última "
                     f"({por_decada.index[-1]}) década: **{variacao:+.2f} °C**")

    with st.expander("Ver tabela de dados tratados"):
        st.dataframe(filtrado[["data", "temperatura", "preenchido"]], hide_index=True, width="stretch")

# -----------------------------------------------------------------
# ABA 4 - SIMULADOR
# -----------------------------------------------------------------
with aba_simulador:
    st.subheader("Simulador: teste o modelo com seus próprios valores")
    st.write("Edite as temperaturas dos últimos 12 meses e veja como a previsão do modelo para o "
             "mês seguinte muda. Exemplo: aumente os 3 últimos meses em 2 °C (uma onda de calor).")

    entrada = pd.DataFrame({
        "Mês": serie["data"].iloc[-12:].dt.strftime("%m/%Y").values,
        "Temperatura (°C)": serie["temperatura"].iloc[-12:].values,
    })
    editado = st.data_editor(entrada, hide_index=True, disabled=["Mês"], width="stretch", height=460)

    previsto = prever_proximos_meses(modelo, editado["Temperatura (°C)"], 1)[0]
    proximo_mes = ultima_data + pd.DateOffset(months=1)
    st.metric(f"Previsão do modelo ({nome_modelo}) para o mês seguinte", f"{previsto:.2f} °C",
              delta=f"{previsto - media_de_cada_mes[proximo_mes.month]:+.2f} °C vs. média de "
                    f"{NOMES_MESES[proximo_mes.month - 1]}", delta_color="inverse")
