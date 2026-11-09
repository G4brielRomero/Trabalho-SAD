"""Painel de previsão mensal. Gere os artefatos com os scripts 1, 2 e 3."""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from configuracao import COLUNAS_ENTRADA, DATA_CORTE, REFERENCIA_ANO, REFERENCIA_MEDIA

st.set_page_config(page_title="Temperatura mensal em Goiânia", layout="wide")

AZUL = "#24528A"
LARANJA = "#B64A13"
VERDE = "#0F766E"
ARQUIVOS_MODELOS = {
    "Rede Neural (MLP)": "modelos/rede_neural.joblib",
    "Árvore de Decisão": "modelos/arvore_decisao.joblib",
}
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
FONTE = "https://www.kaggle.com/datasets/volpatto/temperature-timeseries-for-some-brazilian-cities"


def decimal(valor, casas=2, sinal=False):
    formato = f"+.{casas}f" if sinal else f".{casas}f"
    return format(valor, formato).replace(".", ",")


def grafico_de_linhas(tabela, colunas, cores, formato_data="%m/%Y"):
    fig = px.line(
        tabela, x="data", y=colunas, color_discrete_sequence=cores,
        labels={"data": "Mês", "value": "Temperatura (°C)", "variable": ""},
    )
    fig.update_xaxes(tickformat=formato_data, hoverformat="%m/%Y")
    fig.update_traces(line_width=2.5, hovertemplate="%{y:.2f} °C", connectgaps=False)
    fig.update_layout(
        height=335, margin=dict(l=12, r=18, t=45, b=18), hovermode="x unified",
        legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0),
    )
    return fig


def prever_proximos_meses(modelo, ultimas_temperaturas, quantidade_meses):
    temperaturas = list(ultimas_temperaturas)
    previsoes = []
    for _ in range(quantidade_meses):
        # t-1 recebe o mês mais recente e t-12, o mais antigo.
        entrada = pd.DataFrame([temperaturas[-len(COLUNAS_ENTRADA):][::-1]], columns=COLUNAS_ENTRADA)
        valor_previsto = modelo.predict(entrada)[0]
        previsoes.append(valor_previsto)
        temperaturas.append(valor_previsto)
    return previsoes


serie = pd.read_csv("dados/serie_tratada.csv", parse_dates=["data"])
metricas = pd.read_csv("resultados/metricas.csv").set_index("Modelo")
teste = pd.read_csv("resultados/previsoes_teste.csv", parse_dates=["data"]).set_index("data")
serie["mes"] = serie["data"].dt.month
serie["ano"] = serie["data"].dt.year

media_por_mes = serie.groupby("mes")["media_treino"].first()
fim_referencia = pd.Timestamp(DATA_CORTE) - pd.DateOffset(months=1)
periodo_referencia = f"{serie['data'].min():%m/%Y} a {fim_referencia:%m/%Y}"
ultima_data = serie["data"].iloc[-1]
proximo_mes = ultima_data + pd.DateOffset(months=1)
ultimas_12 = serie["temperatura_entrada"].tail(len(COLUNAS_ENTRADA))

st.sidebar.header("Modelo da demonstração")
nome_modelo = st.sidebar.radio("Modelo", list(ARQUIVOS_MODELOS), label_visibility="collapsed")
st.sidebar.caption("A seleção atualiza a previsão, a análise de erros e o simulador.")
st.sidebar.divider()
st.sidebar.markdown(f"[Fonte dos dados no Kaggle]({FONTE})")
st.sidebar.caption("UFMS · Sistemas de Apoio à Decisão")

modelo = joblib.load(ARQUIVOS_MODELOS[nome_modelo])
previsto = prever_proximos_meses(modelo, ultimas_12, 1)[0]
media_referencia = media_por_mes[proximo_mes.month]
desvio = previsto - media_referencia

st.caption("SISTEMAS DE APOIO À DECISÃO · ESTUDO HISTÓRICO")
st.title(f"Temperatura mensal em {serie['cidade'].iloc[0]}")
st.markdown("**Vale revisar o plano de ventilação para o próximo mês?**")
st.caption(
    f"Exercício de planejamento mensal. Última medição em {ultima_data:%m/%Y} "
    f"e primeira previsão em {proximo_mes:%m/%Y}. Modelo selecionado **{nome_modelo}**."
)

aba_previsao, aba_modelo, aba_historico, aba_simulador = st.tabs(
    ["Previsão", "Desempenho", "Histórico", "Simulador"]
)

with aba_previsao:
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Previsão para {proximo_mes:%m/%Y}", f"{decimal(previsto)} °C", border=True)
    col2.metric("Média do mês no treino", f"{decimal(media_referencia)} °C", border=True)
    col3.metric("Desvio previsto", f"{decimal(desvio, sinal=True)} °C", border=True)

    if desvio >= 0:
        leitura = "Use o desvio acima da média como sinal para revisar o plano mensal de ventilação."
    else:
        leitura = "Use o desvio abaixo da média para reavaliar a necessidade de reforço da ventilação."
    st.info(f"{leitura} Para decisões do dia a dia, consulte previsões atualizadas.")

    st.subheader("Medições recentes e próximo mês")
    historico = serie[["data", "temperatura", "media_treino"]].tail(24).rename(
        columns={"temperatura": "Medição", "media_treino": "Média do mês no treino"}
    )
    futuro = pd.DataFrame({
        "data": [proximo_mes], "Média do mês no treino": [media_referencia]
    })
    fig = grafico_de_linhas(
        pd.concat([historico, futuro], ignore_index=True),
        ["Medição", "Média do mês no treino"], [AZUL, VERDE],
    )
    fig.update_traces(selector={"name": "Média do mês no treino"}, line_dash="dot")
    fig.add_scatter(
        x=[proximo_mes], y=[previsto], name="Previsão", mode="markers",
        marker=dict(color=LARANJA, size=13, symbol="diamond"),
        hovertemplate="%{y:.2f} °C",
    )
    fig.update_xaxes(range=[historico["data"].min(), proximo_mes + pd.Timedelta(days=20)])
    st.plotly_chart(fig)
    st.caption(f"Referência sazonal calculada com medições de {periodo_referencia}.")

with aba_modelo:
    previsoes_teste = teste.reset_index()
    st.subheader("Métricas nos dados de teste")
    inicio_teste = previsoes_teste["data"].min()
    fim_teste = previsoes_teste["data"].max()
    st.caption(
        f"Previsões de um mês à frente para {len(previsoes_teste)} meses com medição "
        f"de {inicio_teste:%m/%Y} a {fim_teste:%m/%Y}."
    )
    st.dataframe(metricas.reset_index(), hide_index=True, width="stretch")

    st.markdown(
        "- **MAE**: em média, quantos °C a previsão erra.\n"
        "- **RMSE**: parecido com o MAE, mas pune mais os erros grandes.\n"
        "- **R²**: quanto da variação da temperatura o modelo explica (1 = perfeito).\n"
        "- **Referências**: repetir o mesmo mês do ano anterior ou usar a média do mês no treino. "
        "Quando falta a medição do ano anterior, a primeira regra também usa essa média."
    )

    st.subheader(f"Real × Previsto — {nome_modelo}")
    st.plotly_chart(grafico_de_linhas(previsoes_teste, ["Real", nome_modelo, REFERENCIA_ANO, REFERENCIA_MEDIA], [AZUL, LARANJA, "#38825d", "#7558a5"], "%Y"))

# -----------------------------------------------------------------
# ABA 3 - HISTÓRICO E INSIGHTS
# -----------------------------------------------------------------
with aba_historico:
    ano_min, ano_max = int(serie["ano"].min()), int(serie["ano"].max())
    anos = st.slider("Período exibido no histórico", ano_min, ano_max, (ano_min, ano_max))
    filtrado = serie[(serie["ano"] >= anos[0]) & (serie["ano"] <= anos[1])]

    st.subheader(f"Série histórica ({anos[0]} a {anos[1]})")
    st.plotly_chart(grafico_de_linhas(filtrado, ["temperatura"], [AZUL], "%Y").update_layout(showlegend=False))
    st.caption(f"{int(serie['preenchido'].sum())} meses sem medição foram preenchidos "
               "nas entradas usando a média daquele mês no treino.")

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Média por mês do ano")
        por_mes = filtrado.groupby("mes")["temperatura"].mean().dropna()
        # "01 Jan", "02 Fev"... o número na frente mantém a ordem correta no gráfico
        por_mes.index = [f"{m:02d} {MESES[m - 1]}" for m in por_mes.index]
        if por_mes.empty:
            st.info("O período selecionado não tem temperaturas medidas.")
        else:
            st.bar_chart(por_mes, color=AZUL, x_label="Mês", y_label="Temperatura (°C)")
            st.write(f"Mês mais quente: **{por_mes.idxmax()[3:]}** ({por_mes.max():.1f} °C) · "
                     f"Mês mais frio: **{por_mes.idxmin()[3:]}** ({por_mes.min():.1f} °C)")

    with col_dir:
        st.subheader("Média por década")
        filtrado = filtrado.assign(decada=(filtrado["ano"] // 10 * 10).astype(str) + "s")
        por_decada = filtrado.groupby("decada")["temperatura"].mean().dropna()
        if not por_decada.empty:
            st.bar_chart(por_decada, color=AZUL, x_label="Década", y_label="Temperatura (°C)")
        if len(por_decada) > 1:
            variacao = por_decada.iloc[-1] - por_decada.iloc[0]
            st.write(f"Variação entre a primeira ({por_decada.index[0]}) e a última "
                     f"({por_decada.index[-1]}) década: **{variacao:+.2f} °C**")

    with st.expander("Ver tabela de dados tratados"):
        st.dataframe(filtrado[["data", "temperatura", "temperatura_entrada", "preenchido"]], hide_index=True, width="stretch")

with aba_simulador:
    st.subheader("Cenário com entradas editáveis")
    st.write(
        f"Edite as temperaturas para observar a resposta do modelo. "
        f"Experimente somar 2 °C ao valor de {ultima_data:%m/%Y}."
    )
    meses_para_prever = st.slider("Horizonte do cenário (meses)", 1, 24, 1)
    st.caption(
        "A partir do segundo mês, cada previsão alimenta a seguinte. "
        "As métricas do teste cobrem apenas o primeiro passo."
    )
    resultado_cenario = st.empty()
    entrada = pd.DataFrame({
        "Mês": serie["data"].tail(len(COLUNAS_ENTRADA)).dt.strftime("%m/%Y").values,
        "Temperatura (°C)": ultimas_12.values,
    })
    editado = st.data_editor(
        entrada, hide_index=True, disabled=["Mês"], width="stretch", height=458,
        key="cenario_temperaturas",
    )
    temperaturas = editado["Temperatura (°C)"]
    if temperaturas.isna().any():
        resultado_cenario.info("Preencha as 12 temperaturas para calcular o cenário.")
    else:
        valores = prever_proximos_meses(modelo, temperaturas, meses_para_prever)
        datas = pd.date_range(proximo_mes, periods=meses_para_prever, freq="MS")
        medias = media_por_mes.loc[datas.month].to_numpy()
        resultado_cenario.metric(
            f"Previsão do cenário para {proximo_mes:%m/%Y}", f"{decimal(valores[0])} °C",
            delta=f"{decimal(valores[0] - previsto, sinal=True)} °C vs. previsão com as entradas originais",
            delta_color="off", border=True,
        )
        if meses_para_prever > 1:
            tabela = pd.DataFrame({
                "Mês": datas.strftime("%m/%Y"),
                "Cenário (°C)": valores,
                "Média do mês no treino (°C)": medias,
                "Desvio (°C)": valores - medias,
            })
            st.dataframe(tabela.round(2), hide_index=True, width="stretch")
