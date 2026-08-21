from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.forecast import (
    calcular_resumo_forecast,
    calcular_variacao_semanal,
    carregar_modelo,
    comparar_com_ultimo_ano,
    converter_forecast_csv,
    gerar_forecast,
)


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="CO₂ Forecast Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CAMINHOS DO PROJETO
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

CAMINHO_DADOS = (
    ROOT
    / "data"
    / "co2_clean.csv"
)


# ==================================================
# HTML
# ==================================================
def renderizar_html(
    conteudo: str,
) -> None:
    st.html(
        dedent(conteudo).strip()
    )


# ==================================================
# ESTILO VISUAL
# ==================================================
renderizar_html(
    """
    <style>

        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F4F7FC;
            color: #0F172A;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #052E2B 0%,
                #064E3B 100%
            );
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }

        /* Hero */
        .hero {
            padding: 2.3rem 2.5rem;
            border-radius: 24px;

            background: linear-gradient(
                135deg,
                #065F46 0%,
                #059669 55%,
                #06B6D4 100%
            );

            margin-bottom: 1.8rem;

            box-shadow:
                0 18px 45px
                rgba(5, 150, 105, 0.20);
        }

        .hero h1 {
            margin: 0;
            color: #FFFFFF !important;
            font-size: 2.65rem;
            line-height: 1.15;
        }

        .hero p {
            max-width: 950px;
            margin-top: 1rem;
            margin-bottom: 0;

            color: #D1FAE5 !important;

            font-size: 1.05rem;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;

            padding: 0.38rem 0.85rem;
            margin-bottom: 1rem;

            border-radius: 999px;

            background-color:
                rgba(255, 255, 255, 0.18);

            color: #FFFFFF !important;

            font-size: 0.82rem;
            font-weight: 700;
        }

        /* Títulos */
        .section-title {
            margin-top: 1.6rem;
            margin-bottom: 1rem;

            color: #0F172A !important;

            font-size: 1.45rem;
            font-weight: 800;
        }

        /* Cards */
        .info-card {
            min-height: 185px;
            padding: 1.5rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;

            box-shadow:
                0 8px 25px
                rgba(15, 23, 42, 0.06);
        }

        .info-card h3 {
            margin-top: 0;
            margin-bottom: 0.8rem;

            color: #047857 !important;

            font-size: 1.35rem;
        }

        .info-card p {
            margin-bottom: 0;

            color: #475569 !important;

            line-height: 1.65;
        }

        /* Cards de análise */
        .analysis-card {
            min-height: 260px;
            padding: 1.5rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;

            box-shadow:
                0 8px 24px
                rgba(15, 23, 42, 0.06);
        }

        .analysis-card h3 {
            margin-top: 0;
            color: #0F172A !important;
        }

        .analysis-card p,
        .analysis-card li {
            color: #475569 !important;
            line-height: 1.65;
        }

        .card-green {
            border-top: 6px solid #059669;
        }

        .card-blue {
            border-top: 6px solid #2563EB;
        }

        .card-orange {
            border-top: 6px solid #F97316;
        }

        /* Fluxo */
        .flow-card {
            min-height: 420px;
            padding: 1.6rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;

            box-shadow:
                0 8px 24px
                rgba(15, 23, 42, 0.06);
        }

        .flow-card h3 {
            margin-top: 0;

            color: #047857 !important;
        }

        .flow-step {
            margin: 0.45rem 0;
            padding: 0.75rem 1rem;

            border-radius: 10px;

            background-color: #ECFDF5;

            color: #047857 !important;

            font-weight: 700;
            text-align: center;
        }

        .flow-arrow {
            color: #64748B !important;
            font-size: 1.1rem;
            font-weight: bold;
            text-align: center;
        }

        /* Nota */
        .model-note {
            padding: 1.1rem 1.3rem;

            border-left: 5px solid #059669;
            border-radius: 14px;

            background-color: #ECFDF5;

            color: #065F46 !important;

            line-height: 1.65;
        }

        /* Métricas */
        div[data-testid="stMetric"] {
            min-height: 112px;

            padding: 1rem 1.1rem;

            border: 1px solid #DCE4F0;
            border-radius: 16px;

            background-color: #FFFFFF !important;

            box-shadow:
                0 6px 18px
                rgba(15, 23, 42, 0.05);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: #475569 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: #0F172A !important;
            opacity: 1 !important;
        }

        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] * {
            opacity: 1 !important;
        }

        /* Tabelas */
        [data-testid="stDataFrame"] {
            overflow: hidden;

            border: 1px solid #DCE4F0;
            border-radius: 14px;
        }

        /* Botão */
        .stButton > button[kind="primary"] {
            border: none;
            border-radius: 10px;

            background: linear-gradient(
                90deg,
                #059669,
                #06B6D4
            );

            color: #FFFFFF;
            font-weight: 700;
        }

        /* Rodapé */
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;

            border-top: 1px solid #DCE4F0;

            color: #64748B !important;

            font-size: 0.9rem;
            text-align: center;
        }

    </style>
    """
)


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================
@st.cache_data
def carregar_dados() -> pd.DataFrame:
    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {CAMINHO_DADOS}"
        )

    dados = pd.read_csv(
        CAMINHO_DADOS,
        index_col=0,
        parse_dates=True,
    )

    dados.index = pd.DatetimeIndex(
        dados.index
    )

    return dados.sort_index()


# ==================================================
# CACHE DO MODELO
# ==================================================
@st.cache_resource
def carregar_modelo_cache():
    return carregar_modelo()


# ==================================================
# GRÁFICO HISTÓRICO
# ==================================================
def criar_grafico_historico(
    dados: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        dados.index,
        dados["co2"],
        color="#2563EB",
        linewidth=1.2,
    )

    ax.set_title(
        "Evolução histórica da concentração de CO₂",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_ylabel("CO₂")
    ax.set_xlabel("")

    ax.grid(alpha=0.2)

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO MÉDIA ANUAL
# ==================================================
def criar_grafico_media_anual(
    dados: pd.DataFrame,
):
    media_anual = (
        dados["co2"]
        .resample("YE")
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        media_anual.index.year,
        media_anual.values,
        marker="o",
        color="#059669",
        linewidth=2,
    )

    ax.set_title(
        "Média anual da concentração de CO₂",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("CO₂ médio")

    ax.grid(alpha=0.2)

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO SAZONALIDADE
# ==================================================
def criar_grafico_sazonalidade(
    dados: pd.DataFrame,
):
    dados_temp = dados.copy()

    dados_temp["mes"] = (
        dados_temp.index.month
    )

    media_mensal = (
        dados_temp
        .groupby("mes")["co2"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        media_mensal.index,
        media_mensal.values,
        marker="o",
        color="#F97316",
        linewidth=2,
    )

    ax.set_title(
        "Padrão sazonal médio",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Mês")
    ax.set_ylabel("CO₂ médio")

    ax.set_xticks(
        range(1, 13)
    )

    ax.grid(alpha=0.2)

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO FORECAST
# ==================================================
def criar_grafico_forecast(
    dados: pd.DataFrame,
    forecast: pd.DataFrame,
    semanas_historico: int = 104,
):
    historico = dados.tail(
        semanas_historico
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        historico.index,
        historico["co2"],
        color="#2563EB",
        linewidth=2,
        label="Histórico",
    )

    ax.plot(
        forecast.index,
        forecast["co2_previsto"],
        color="#059669",
        linewidth=2.3,
        linestyle="--",
        label="Previsão",
    )

    ax.fill_between(
        forecast.index,
        forecast["limite_inferior"],
        forecast["limite_superior"],
        color="#10B981",
        alpha=0.18,
        label="Intervalo de confiança 95%",
    )

    ax.axvline(
        dados.index.max(),
        color="#64748B",
        linestyle="--",
        alpha=0.7,
    )

    ax.set_title(
        "Forecast da concentração atmosférica de CO₂",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("CO₂")

    ax.legend(
        frameon=False
    )

    ax.grid(alpha=0.2)

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO VARIAÇÃO SEMANAL
# ==================================================
def criar_grafico_variacao(
    forecast: pd.DataFrame,
):
    resultado = (
        calcular_variacao_semanal(
            forecast
        )
    )

    cores = [
        (
            "#059669"
            if valor >= 0
            else "#DC2626"
        )
        for valor
        in resultado[
            "variacao_semanal"
        ].fillna(0)
    ]

    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.bar(
        resultado.index,
        resultado["variacao_semanal"],
        color=cores,
        width=5,
    )

    ax.axhline(
        0,
        color="#64748B",
        linewidth=1,
    )

    ax.set_title(
        "Variação semanal prevista",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Δ CO₂")

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# CARREGAMENTO PRINCIPAL
# ==================================================
try:
    dados = carregar_dados()
    modelo = carregar_modelo_cache()

except Exception as erro:
    st.error(
        f"Erro ao carregar a aplicação: {erro}"
    )

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown(
        "## 🌍 CO₂ Forecast"
    )

    st.caption(
        "Forecasting da concentração "
        "atmosférica de CO₂"
    )

    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão geral",
            "Análise histórica",
            "Forecast de CO₂",
            "Sobre o projeto",
        ],
    )

    st.divider()

    st.markdown(
        "### Modelo selecionado"
    )

    st.markdown(
        """
**SARIMAX**

`(1,1,1)(1,1,1,52)`

**Frequência**

Semanal

**Sazonalidade**

52 semanas
        """
    )

    st.divider()

    st.caption(
        "Projeto desenvolvido por "
        "Renan Assis Trevelim."
    )


# ==================================================
# VISÃO GERAL
# ==================================================
if pagina == "Visão geral":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Time Series Forecasting
            </span>

            <h1>
                CO₂ Forecast Intelligence
            </h1>

            <p>
                Aplicação de Ciência de Dados para
                análise da concentração atmosférica
                de CO₂ e geração de previsões futuras
                utilizando séries temporais.
            </p>

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            O que esta aplicação entrega
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        renderizar_html(
            """
            <div class="info-card">
                <h3>📈 Análise histórica</h3>

                <p>
                    Visualização da evolução da série,
                    tendência de longo prazo e padrão
                    sazonal da concentração de CO₂.
                </p>
            </div>
            """
        )

    with col2:
        renderizar_html(
            """
            <div class="info-card">
                <h3>🔮 Forecast</h3>

                <p>
                    Geração de previsões futuras
                    utilizando o modelo SARIMAX
                    selecionado na validação temporal.
                </p>
            </div>
            """
        )

    with col3:
        renderizar_html(
            """
            <div class="info-card">
                <h3>📊 Inteligência temporal</h3>

                <p>
                    Identificação de máximos,
                    mínimos, variações semanais
                    e comparação com o histórico.
                </p>
            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Resultado do modelo final
        </div>
        """
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "MAE",
        "0.335",
    )

    col2.metric(
        "RMSE",
        "0.421",
    )

    col3.metric(
        "MAPE",
        "0.090%",
    )

    renderizar_html(
        """
        <div class="section-title">
            Fluxo da solução
        </div>
        """
    )

    fluxo1, fluxo2 = (
        st.columns(2)
    )

    with fluxo1:
        renderizar_html(
            """
            <div class="flow-card">

                <h3>
                    🧠 Desenvolvimento
                </h3>

                <div class="flow-step">
                    Dados históricos
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Validação e tratamento
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Análise temporal
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Modelagem
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Walk-forward validation
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    SARIMAX selecionado
                </div>

            </div>
            """
        )

    with fluxo2:
        renderizar_html(
            """
            <div class="flow-card">

                <h3>
                    🌍 Aplicação
                </h3>

                <div class="flow-step">
                    Modelo serializado
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Forecast
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Intervalo de confiança
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Indicadores
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Monitoramento
                </div>

            </div>
            """
        )


# ==================================================
# ANÁLISE HISTÓRICA
# ==================================================
elif pagina == "Análise histórica":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Exploratory Time Series Analysis
            </span>

            <h1>
                Análise histórica
            </h1>

            <p>
                Explore a evolução temporal da
                concentração atmosférica de CO₂,
                sua tendência de longo prazo e
                seu comportamento sazonal.
            </p>

        </div>
        """
    )

    total = len(dados)

    inicio = dados.index.min()
    fim = dados.index.max()

    media = dados["co2"].mean()

    minimo = dados["co2"].min()
    maximo = dados["co2"].max()

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Observações",
        f"{total:,}".replace(",", "."),
    )

    col2.metric(
        "Início",
        inicio.strftime("%d/%m/%Y"),
    )

    col3.metric(
        "Fim",
        fim.strftime("%d/%m/%Y"),
    )

    col4.metric(
        "Média",
        f"{media:.2f}",
    )

    col5.metric(
        "Amplitude",
        f"{maximo - minimo:.2f}",
    )

    st.markdown("<br>")

    st.pyplot(
        criar_grafico_historico(
            dados
        ),
        use_container_width=True,
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        st.pyplot(
            criar_grafico_media_anual(
                dados
            ),
            use_container_width=True,
        )

    with col2:
        st.pyplot(
            criar_grafico_sazonalidade(
                dados
            ),
            use_container_width=True,
        )

    renderizar_html(
        """
        <div class="model-note">

            A série apresenta tendência crescente
            de longo prazo e um padrão sazonal anual
            bem definido. Essa estrutura temporal
            contribuiu para o desempenho do SARIMAX.

        </div>
        """
    )


# ==================================================
# FORECAST
# ==================================================
elif pagina == "Forecast de CO₂":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Predictive Analytics
            </span>

            <h1>
                Forecast de CO₂
            </h1>

            <p>
                Utilize o modelo SARIMAX treinado
                com o histórico completo para
                gerar projeções futuras da
                concentração atmosférica de CO₂.
            </p>

        </div>
        """
    )

    horizonte = st.slider(
        "Horizonte da previsão",
        min_value=4,
        max_value=104,
        value=52,
        step=4,
        help=(
            "Quantidade de semanas "
            "que serão previstas."
        ),
    )

    forecast = gerar_forecast(
        modelo=modelo,
        horizonte=horizonte,
    )

    resumo = (
        calcular_resumo_forecast(
            forecast
        )
    )

    comparacao = (
        comparar_com_ultimo_ano(
            historico=dados["co2"],
            forecast=forecast,
        )
    )

    renderizar_html(
        """
        <div class="section-title">
            Resumo da previsão
        </div>
        """
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Horizonte",
        f"{horizonte} semanas",
    )

    col2.metric(
        "Média prevista",
        f"{resumo['media']:.2f}",
    )

    col3.metric(
        "Máximo previsto",
        f"{resumo['maximo']:.2f}",
    )

    col4.metric(
        "Mínimo previsto",
        f"{resumo['minimo']:.2f}",
    )

    col5.metric(
        "Variação vs. último ano",
        (
            f"{comparacao['variacao_percentual']:+.2f}%"
        ),
    )

    st.pyplot(
        criar_grafico_forecast(
            dados=dados,
            forecast=forecast,
        ),
        use_container_width=True,
    )

    renderizar_html(
        """
        <div class="section-title">
            Principais pontos do forecast
        </div>
        """
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        renderizar_html(
            f"""
            <div class="analysis-card card-green">

                <h3>
                    📈 Maior concentração prevista
                </h3>

                <p>
                    <strong>Valor:</strong>
                    {resumo["maximo"]:.2f}
                </p>

                <p>
                    <strong>Data:</strong>
                    {
                        resumo["data_maximo"]
                        .strftime("%d/%m/%Y")
                    }
                </p>

                <p>
                    Maior concentração estimada
                    dentro do horizonte selecionado.
                </p>

            </div>
            """
        )

    with col2:
        renderizar_html(
            f"""
            <div class="analysis-card card-blue">

                <h3>
                    📉 Menor concentração prevista
                </h3>

                <p>
                    <strong>Valor:</strong>
                    {resumo["minimo"]:.2f}
                </p>

                <p>
                    <strong>Data:</strong>
                    {
                        resumo["data_minimo"]
                        .strftime("%d/%m/%Y")
                    }
                </p>

                <p>
                    Menor concentração estimada
                    dentro do horizonte selecionado.
                </p>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Evolução semanal prevista
        </div>
        """
    )

    st.pyplot(
        criar_grafico_variacao(
            forecast
        ),
        use_container_width=True,
    )

    renderizar_html(
        """
        <div class="section-title">
            Comparação com o histórico recente
        </div>
        """
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Média — últimas 52 semanas",
        (
            f"{comparacao['media_ultimo_ano']:.2f}"
        ),
    )

    col2.metric(
        "Média prevista",
        (
            f"{comparacao['media_prevista']:.2f}"
        ),
    )

    col3.metric(
        "Variação absoluta",
        (
            f"{comparacao['variacao_absoluta']:+.2f}"
        ),
    )

    col4.metric(
        "Variação percentual",
        (
            f"{comparacao['variacao_percentual']:+.3f}%"
        ),
    )

    renderizar_html(
        """
        <div class="section-title">
            Dados previstos
        </div>
        """
    )

    tabela = forecast.copy()

    tabela.index = (
        tabela.index.strftime(
            "%d/%m/%Y"
        )
    )

    tabela.index.name = "Data"

    st.dataframe(
        tabela,
        use_container_width=True,
        column_config={
            "co2_previsto":
                st.column_config.NumberColumn(
                    "CO₂ previsto",
                    format="%.3f",
                ),

            "limite_inferior":
                st.column_config.NumberColumn(
                    "Limite inferior",
                    format="%.3f",
                ),

            "limite_superior":
                st.column_config.NumberColumn(
                    "Limite superior",
                    format="%.3f",
                ),
        },
    )

    st.download_button(
        "⬇️ Baixar forecast",
        data=converter_forecast_csv(
            forecast
        ),
        file_name="forecast_co2.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# SOBRE O PROJETO
# ==================================================
else:

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Projeto de portfólio
            </span>

            <h1>
                Sobre o projeto
            </h1>

            <p>
                Projeto de séries temporais envolvendo
                preparação de dados, análise exploratória,
                modelos estatísticos, Machine Learning,
                validação temporal e aplicação do
                modelo final.
            </p>

        </div>
        """
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:
        renderizar_html(
            """
            <div class="analysis-card card-green">

                <h3>
                    📊 Modelos estatísticos
                </h3>

                <ul>
                    <li>Holt-Winters</li>
                    <li>SARIMAX</li>
                </ul>

                <p>
                    O SARIMAX apresentou o melhor
                    desempenho durante a validação
                    temporal.
                </p>

            </div>
            """
        )

    with col2:
        renderizar_html(
            """
            <div class="analysis-card card-blue">

                <h3>
                    🤖 Machine Learning
                </h3>

                <ul>
                    <li>Random Forest</li>
                    <li>XGBoost</li>
                    <li>Lags</li>
                    <li>Médias móveis</li>
                    <li>Features sazonais</li>
                </ul>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Comparação final
        </div>
        """
    )

    comparacao_modelos = (
        pd.DataFrame(
            {
                "Modelo": [
                    "SARIMAX",
                    "Holt-Winters",
                    "Random Forest Tuned",
                    "XGBoost Tuned",
                ],
                "MAE": [
                    0.504,
                    0.764,
                    0.772,
                    0.827,
                ],
                "RMSE": [
                    0.622,
                    0.983,
                    0.974,
                    1.052,
                ],
                "MAPE": [
                    0.138,
                    0.209,
                    0.210,
                    0.225,
                ],
            }
        )
    )

    st.dataframe(
        comparacao_modelos,
        use_container_width=True,
        hide_index=True,
    )

    renderizar_html(
        """
        <div class="model-note">

            Mesmo após o tuning dos modelos de
            Machine Learning, o SARIMAX apresentou
            os menores erros médios.

            O resultado reforça que a complexidade
            do algoritmo não determina, por si só,
            a qualidade da previsão.

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Tecnologias utilizadas
        </div>
        """
    )

    st.code(
        """Python
Pandas
NumPy
Matplotlib
Statsmodels
Scikit-learn
XGBoost
Streamlit
Joblib""",
        language="text",
    )

    renderizar_html(
        """
        <div class="section-title">
            Autor
        </div>
        """
    )

    renderizar_html(
        """
        <div class="analysis-card card-green">

            <h3>
                Renan Assis Trevelim
            </h3>

            <p>
                Projeto desenvolvido como
                aplicação prática de Ciência
                de Dados, Machine Learning
                e Time Series Forecasting.
            </p>

        </div>
        """
    )


# ==================================================
# RODAPÉ
# ==================================================
renderizar_html(
    """
    <div class="footer">

        CO₂ Forecast Intelligence
        • Time Series Forecasting
        • Renan Assis Trevelim

    </div>
    """
)