from pathlib import Path

import joblib
import pandas as pd


# ==================================================
# CAMINHOS DO PROJETO
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

CAMINHO_MODELO = (
    ROOT
    / "models"
    / "sarimax_final.pkl"
)


# ==================================================
# CARREGAMENTO DO MODELO
# ==================================================
def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(CAMINHO_MODELO)


# ==================================================
# FORECAST
# ==================================================
def gerar_forecast(
    modelo,
    horizonte: int = 52,
) -> pd.DataFrame:
    resultado = modelo.get_forecast(
        steps=horizonte
    )

    previsao = resultado.predicted_mean

    intervalo = resultado.conf_int(
        alpha=0.05
    )

    forecast_df = pd.DataFrame(
        {
            "co2_previsto": previsao,
            "limite_inferior": intervalo.iloc[:, 0],
            "limite_superior": intervalo.iloc[:, 1],
        }
    )

    forecast_df.index.name = "data"

    return forecast_df


# ==================================================
# RESUMO DO FORECAST
# ==================================================
def calcular_resumo_forecast(
    forecast: pd.DataFrame,
) -> dict:
    data_maximo = (
        forecast["co2_previsto"]
        .idxmax()
    )

    data_minimo = (
        forecast["co2_previsto"]
        .idxmin()
    )

    return {
        "media": (
            forecast["co2_previsto"]
            .mean()
        ),
        "maximo": forecast.loc[
            data_maximo,
            "co2_previsto",
        ],
        "data_maximo": data_maximo,
        "minimo": forecast.loc[
            data_minimo,
            "co2_previsto",
        ],
        "data_minimo": data_minimo,
    }


# ==================================================
# VARIAÇÃO SEMANAL
# ==================================================
def calcular_variacao_semanal(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    resultado = forecast.copy()

    resultado["variacao_semanal"] = (
        resultado["co2_previsto"]
        .diff()
    )

    resultado["variacao_percentual"] = (
        resultado["co2_previsto"]
        .pct_change()
        * 100
    )

    return resultado


# ==================================================
# COMPARAÇÃO COM O ÚLTIMO ANO
# ==================================================
def comparar_com_ultimo_ano(
    historico: pd.Series,
    forecast: pd.DataFrame,
) -> dict:
    media_ultimo_ano = (
        historico
        .tail(52)
        .mean()
    )

    media_prevista = (
        forecast["co2_previsto"]
        .mean()
    )

    variacao_absoluta = (
        media_prevista
        - media_ultimo_ano
    )

    variacao_percentual = (
        variacao_absoluta
        / media_ultimo_ano
        * 100
    )

    return {
        "media_ultimo_ano": media_ultimo_ano,
        "media_prevista": media_prevista,
        "variacao_absoluta": variacao_absoluta,
        "variacao_percentual": variacao_percentual,
    }


# ==================================================
# CONVERSÃO PARA CSV
# ==================================================
def converter_forecast_csv(
    forecast: pd.DataFrame,
) -> bytes:
    resultado = (
        forecast
        .reset_index()
        .to_csv(
            index=False,
            encoding="utf-8-sig",
        )
    )

    return resultado.encode(
        "utf-8-sig"
    )