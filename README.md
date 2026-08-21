# CO₂ Forecasting — Statistical Models vs Machine Learning

Projeto de séries temporais para análise e previsão da concentração atmosférica de CO₂ utilizando modelos estatísticos clássicos e algoritmos de Machine Learning.

O objetivo principal é construir um pipeline completo de forecasting, desde a validação e tratamento dos dados até a comparação entre diferentes modelos, avaliação em dados futuros, geração de previsões e disponibilização dos resultados em uma aplicação Streamlit.

---

## Objetivo do Projeto

Este projeto busca responder à seguinte pergunta:

> Modelos estatísticos clássicos conseguem superar modelos de Machine Learning na previsão de uma série temporal univariada com tendência e sazonalidade bem definidas?

Para isso, foram avaliadas diferentes abordagens de forecasting utilizando a série histórica de concentração atmosférica de CO₂ de Mauna Loa.

Além da etapa de modelagem, o projeto também explora como transformar previsões em informações úteis para monitoramento e tomada de decisão.

---

## Dataset

O dataset utilizado é a série histórica de concentração atmosférica de CO₂ disponível na biblioteca `statsmodels`.

A série possui frequência semanal e cobre o período de:

- **Início:** 1958-03-29
- **Fim:** 2001-12-29
- **Observações:** 2.284
- **Frequência:** semanal (`W-SAT`)

A variável analisada é `co2`, representando a concentração atmosférica de dióxido de carbono.

---

## Estrutura do Projeto

```text
co2-forecasting-statistical-vs-ml/
│
├── data/
│   └── co2_clean.csv
│
├── models/
│   └── sarimax_final.pkl
│
├── notebooks/
│   ├── 01_data_validation_and_cleaning.ipynb
│   ├── 02_time_series_eda.ipynb
│   └── 03_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── forecast.py
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Validação e Tratamento dos Dados

Notebook:

`01_data_validation_and_cleaning.ipynb`

Nesta etapa, o objetivo foi garantir que a série estivesse consistente e adequada para análise e modelagem.

Foram realizadas:

- inspeção inicial dos dados;
- validação do índice temporal;
- verificação da frequência da série;
- identificação de valores ausentes;
- análise das datas com valores ausentes;
- verificação de duplicidades;
- tratamento dos valores ausentes;
- validação após o tratamento;
- exportação da série limpa.

Foram identificados **59 valores ausentes** na variável `co2`.

O tratamento foi realizado utilizando interpolação temporal:

```python
df_clean["co2"] = df_clean["co2"].interpolate(
    method="time"
)
```

A interpolação utiliza as observações conhecidas ao redor do ponto ausente e considera a distância temporal entre elas.

Após o tratamento, a série permaneceu com:

- **2.284 observações**
- **0 valores ausentes**
- **0 datas duplicadas**
- índice temporal ordenado corretamente

A série tratada foi exportada para utilização nas etapas seguintes.

---

# 2. Análise Exploratória da Série Temporal

Notebook:

`02_time_series_eda.ipynb`

A análise exploratória foi desenvolvida com foco em compreender a estrutura temporal da série.

Foram analisados:

- estatísticas descritivas;
- distribuição da concentração de CO₂;
- evolução temporal;
- médias móveis;
- tendência de longo prazo;
- sazonalidade;
- média por mês;
- média por ano;
- decomposição da série;
- comportamento dos resíduos;
- estacionariedade;
- ACF;
- PACF;
- diferenciação regular;
- diferenciação sazonal.

## Tendência

A série apresenta uma tendência crescente de longo prazo.

A concentração de CO₂ aumenta de aproximadamente 315 no início da série para valores superiores a 370 ao final do período analisado.

Médias móveis foram utilizadas para suavizar as oscilações e facilitar a identificação da tendência.

## Sazonalidade

A análise revelou um padrão sazonal anual bem definido.

Como os dados possuem frequência semanal, foi considerado um período sazonal de:

`52 semanas`

A decomposição foi realizada utilizando um modelo aditivo:

```python
seasonal_decompose(
    dados["co2"],
    model="additive",
    period=52
)
```

A decomposição evidenciou três componentes principais:

```text
Série Observada
      ↓
Tendência
      +
Sazonalidade
      +
Resíduo
```

## Estacionariedade

O teste Augmented Dickey-Fuller (ADF) aplicado à série original apresentou:

`p-value = 0.9612`

indicando que a série original não era estacionária.

Foram avaliadas diferentes transformações:

```text
Diferenciação Regular
        ↓
Diferenciação Sazonal
        ↓
Diferenciação Regular + Sazonal
```

A transformação final utilizada foi:

```python
serie_diff = (
    dados["co2"]
    .diff(1)
    .diff(52)
    .dropna()
)
```

Após a transformação, o teste ADF apresentou:

```text
ADF Statistic = -17.1904
p-value ≈ 0.0000
```

indicando evidências de estacionariedade.

## ACF e PACF

As funções de autocorrelação e autocorrelação parcial foram utilizadas para analisar a dependência temporal da série.

A análise contribuiu para a definição dos parâmetros utilizados posteriormente no modelo SARIMAX.

A estrutura considerada foi:

```text
d = 1
D = 1
s = 52
```

levando à investigação de modelos da forma:

```text
SARIMA(p,1,q)(P,1,Q,52)
```

---

# 3. Modelagem e Forecasting

Notebook:

`03_models.ipynb`

A etapa de modelagem foi dividida entre modelos estatísticos e modelos de Machine Learning.

Os modelos foram avaliados utilizando validação temporal.

## Estratégia de Validação

As últimas **52 semanas** foram reservadas como conjunto de teste final.

```text
Dados de Modelagem
1958 ----------------------- 2000

Teste Final
2001 ----------------------- 2001
```

O conjunto de teste final permaneceu isolado durante o processo de seleção dos modelos.

Nos dados de modelagem foi utilizado:

```python
TimeSeriesSplit(
    n_splits=5,
    test_size=52
)
```

Essa estratégia permite simular diferentes cenários históricos de previsão mantendo a ordem cronológica da série.

## Métricas

Foram utilizadas três métricas:

### MAE — Mean Absolute Error

Representa o erro absoluto médio das previsões.

### RMSE — Root Mean Squared Error

Penaliza com maior intensidade erros elevados.

### MAPE — Mean Absolute Percentage Error

Representa o erro médio percentual das previsões.

Como todas representam métricas de erro, valores menores indicam melhor desempenho.

O **MAE** foi utilizado como principal critério de seleção.

---

# 4. Modelos Estatísticos

Foram avaliados:

- Holt-Winters
- SARIMAX

## Holt-Winters

O modelo foi configurado com:

```python
ExponentialSmoothing(
    treino,
    trend="add",
    seasonal="add",
    seasonal_periods=52
)
```

Essa configuração permite representar tendência e sazonalidade anual.

## SARIMAX

O modelo utilizado foi:

`SARIMAX(1,1,1)(1,1,1,52)`

Configuração:

```python
SARIMAX(
    treino,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52)
)
```

A parametrização foi baseada nos resultados obtidos durante a análise exploratória da série.

---

# 5. Modelos de Machine Learning

Foram avaliados:

- Random Forest
- XGBoost

Como esses modelos não trabalham diretamente com a estrutura temporal, foi necessário transformar a série em um problema supervisionado.

## Feature Engineering

Foram criadas features baseadas no histórico da série.

### Lags

```text
lag_1
lag_2
lag_4
lag_13
lag_26
lag_52
```

Essas variáveis representam informações de curto, médio e longo prazo.

### Médias Móveis

Foram utilizadas:

```text
rolling_mean_4
rolling_mean_13
rolling_mean_52
```

Para evitar data leakage, as médias móveis foram construídas utilizando apenas valores anteriores:

```python
dados["co2"].shift(1).rolling(52).mean()
```

### Sazonalidade Cíclica

A semana do ano foi representada utilizando seno e cosseno:

```text
week_sin
week_cos
```

permitindo representar a natureza cíclica do calendário.

---

# 6. Otimização de Hiperparâmetros

Random Forest e XGBoost foram submetidos a tuning utilizando:

```python
RandomizedSearchCV
```

com validação temporal baseada em:

```python
TimeSeriesSplit
```

A métrica utilizada para otimização foi:

```text
neg_mean_absolute_error
```

permitindo selecionar os hiperparâmetros com menor MAE médio.

---

# 7. Comparação dos Modelos

Após o tuning, os modelos apresentaram os seguintes resultados médios durante a validação temporal:

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMAX | **0.504** | **0.622** | **0.138** |
| Holt-Winters | 0.764 | 0.983 | 0.209 |
| Random Forest Tuned | 0.772 | 0.974 | 0.210 |
| XGBoost Tuned | 0.827 | 1.052 | 0.225 |

O SARIMAX apresentou o melhor desempenho nas três métricas avaliadas.

Esse resultado mostra que, para esta série univariada com forte tendência e sazonalidade, o modelo estatístico conseguiu representar a estrutura temporal de forma mais eficiente que os modelos de Machine Learning avaliados.


---

# 8. Modelo Final

O modelo selecionado foi:

`SARIMAX(1,1,1)(1,1,1,52)`

Após a seleção, o modelo foi treinado utilizando todo o conjunto destinado à modelagem e avaliado sobre as últimas 52 semanas da série.

Resultados no teste final:

```text
MAE  = 0.335
RMSE = 0.421
MAPE = 0.090%
```

Visualmente, as previsões acompanharam de forma consistente:

- o crescimento no início do ano;
- o pico sazonal;
- a redução no segundo semestre;
- o ponto mínimo do ciclo;
- a recuperação no final do período.

Os resultados reforçaram a capacidade de generalização do modelo.

---

# 9. Forecast Futuro

Após a validação final, o SARIMAX foi treinado novamente utilizando todo o histórico disponível.

Foi então realizado um forecast para as próximas:

`52 semanas`

Também foram calculados intervalos de confiança para representar a incerteza associada às previsões.

O forecast manteve o padrão sazonal observado historicamente, com crescimento no início do ano, pico sazonal, redução ao longo do segundo semestre e recuperação no final do ciclo.

---

# 10. Geração de Valor

A previsão foi transformada em indicadores capazes de apoiar um cenário real de monitoramento ambiental.

Foram analisados:

- média prevista;
- máximo previsto;
- mínimo previsto;
- intervalo de confiança;
- variação semanal;
- comparação com o último ano observado.

## Resumo do Forecast

```text
Média prevista  = 372.42
Máximo previsto = 375.18
Mínimo previsto = 369.05
```

Maior concentração prevista:

```text
375.18
```

em:

```text
18/05/2002
```

Menor concentração prevista:

```text
369.05
```

em:

```text
21/09/2002
```

## Evolução Semanal Prevista

Também foi analisada a variação entre semanas consecutivas.

As barras positivas representam aumento da concentração prevista de CO₂, enquanto valores negativos representam redução em relação à semana anterior.

Os principais movimentos identificados foram:

```text
Maior aumento semanal = +0.628
Data                   = 16/03/2002

Maior redução semanal = -0.677
Data                   = 27/07/2002
```

Visualmente, observa-se um período predominantemente crescente no início do ano, seguido por uma sequência de reduções durante o meio do ciclo anual e posterior retomada do crescimento.

Esse comportamento é consistente com o padrão sazonal identificado anteriormente na análise exploratória.

## Comparação com o Último Ano

A média observada nas últimas 52 semanas foi:

```text
370.865
```

A média prevista para as próximas 52 semanas foi:

```text
372.420
```

Variação absoluta:

```text
+1.554
```

Variação percentual:

```text
+0.419%
```

O modelo projeta, portanto, continuidade da tendência crescente da concentração atmosférica de CO₂.

---

# 11. Exportação do Modelo Final

Após a validação e geração das previsões futuras, o modelo SARIMAX treinado com todo o histórico foi serializado para reutilização fora do notebook.

A exportação foi realizada com `joblib`:

```python
import joblib

joblib.dump(
    modelo_producao,
    "sarimax_final.pkl"
)
```

O arquivo foi armazenado em:

```text
models/sarimax_final.pkl
```

Essa etapa permite reutilizar o modelo diretamente na aplicação Streamlit sem necessidade de novo treinamento a cada execução.

---

# 12. Aplicação Streamlit

O projeto também inclui uma aplicação interativa construída com Streamlit.

A aplicação utiliza o modelo SARIMAX exportado e a base histórica tratada para disponibilizar os principais resultados de forma visual e interativa.

A camada de aplicação está organizada dentro da pasta `src/`:

```text
src/
├── __init__.py
├── forecast.py
└── app.py
```

## `forecast.py`

Responsável pela lógica de forecasting e funções reutilizáveis, incluindo:

- carregamento do modelo serializado;
- geração de forecasts;
- cálculo de intervalos de confiança;
- resumo das previsões;
- cálculo da variação semanal;
- comparação com o último ano observado;
- exportação do forecast em CSV.

## `app.py`

Responsável pela interface Streamlit.

A aplicação possui quatro áreas principais:

- **Visão geral**
- **Análise histórica**
- **Forecast de CO₂**
- **Sobre o projeto**

Na página de forecast, o usuário pode selecionar o horizonte desejado e visualizar:

- média prevista;
- maior concentração prevista;
- menor concentração prevista;
- intervalo de confiança;
- variação em relação ao último ano;
- gráfico histórico e forecast;
- variação semanal;
- tabela com previsões;
- download dos resultados em CSV.

O horizonte oficial utilizado no projeto é de **52 semanas**, enquanto a aplicação permite exploração de diferentes horizontes de forma interativa.

---

# Possível Aplicação

Em um cenário real, esse tipo de solução poderia ser integrado a:

- dashboards ambientais;
- sistemas de monitoramento;
- relatórios de sustentabilidade;
- plataformas de indicadores climáticos;
- sistemas de alerta;
- pipelines automatizados de previsão.

Uma arquitetura futura poderia seguir:

```text
Novas medições
      ↓
Pipeline de dados
      ↓
Tratamento e validação
      ↓
Modelo SARIMAX
      ↓
Forecast
      ↓
Intervalo de confiança
      ↓
Dashboard / Monitoramento
      ↓
Tomada de decisão
```

É importante destacar que o modelo possui finalidade preditiva.

Portanto, ele estima o comportamento futuro da série com base nos padrões históricos, mas não identifica relações causais sobre os fatores responsáveis pelas variações observadas na concentração de CO₂.

---

# Principais Tecnologias

O projeto utiliza:

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# Como Executar o Projeto

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
```

Acesse o diretório:

```bash
cd co2-forecasting-statistical-vs-ml
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

---

## Executando os Notebooks

Abra os notebooks na seguinte ordem:

```text
01_data_validation_and_cleaning.ipynb
02_time_series_eda.ipynb
03_models.ipynb
```

---

## Executando a Aplicação Streamlit

A partir da raiz do projeto, execute:

```bash
python -m streamlit run src/app.py
```

A aplicação utilizará automaticamente:

```text
data/co2_clean.csv
models/sarimax_final.pkl
```

---

# Fluxo Completo do Projeto

```text
Dados brutos
      ↓
Validação dos Dados
      ↓
Tratamento
      ↓
Análise Exploratória
      ↓
Tendência e Sazonalidade
      ↓
Estacionariedade
      ↓
ACF / PACF
      ↓
Feature Engineering
      ↓
Modelos Estatísticos
      +
Machine Learning
      ↓
Walk-Forward Validation
      ↓
Tuning
      ↓
Comparação dos Modelos
      ↓
SARIMAX selecionado
      ↓
Teste Final
      ↓
Forecast Futuro
      ↓
Geração de Indicadores
      ↓
Exportação do Modelo
      ↓
Aplicação Streamlit
```

---

# Conclusão

O projeto percorreu as principais etapas de um problema completo de forecasting, desde a preparação dos dados até a disponibilização do modelo em uma aplicação interativa.

A análise exploratória revelou uma série com forte tendência crescente e sazonalidade anual bem definida.

Após os testes de estacionariedade e análise de autocorrelação, foram avaliados modelos estatísticos e algoritmos de Machine Learning.

Mesmo após a otimização dos modelos de Machine Learning, o **SARIMAX(1,1,1)(1,1,1,52)** apresentou o melhor desempenho médio durante a validação temporal.

No conjunto de teste final, o modelo apresentou:

```text
MAE  = 0.335
RMSE = 0.421
MAPE = 0.090%
```

Além dos bons resultados quantitativos, a previsão acompanhou visualmente de forma consistente o comportamento real da série, capturando o ciclo sazonal e a dinâmica anual da concentração de CO₂.

O modelo final foi então treinado utilizando todo o histórico disponível e empregado para gerar previsões futuras de 52 semanas, intervalos de confiança e indicadores de evolução temporal.

O projeto também avançou além dos notebooks, incluindo a serialização do modelo com `joblib` e a construção de uma aplicação Streamlit para visualização e exploração interativa das previsões.

O principal aprendizado do projeto é que a escolha de um modelo deve ser orientada pelas características da série e pelos resultados de validação, e não apenas pela complexidade do algoritmo.

Para esta série temporal univariada, com tendência e sazonalidade fortemente estruturadas, a abordagem estatística apresentou desempenho superior aos modelos de Machine Learning avaliados.

---

# Próximas Evoluções

Como próximos passos, o projeto pode ser expandido com:

- inclusão de variáveis exógenas;
- comparação com SARIMAX multivariado;
- LightGBM e CatBoost;
- Prophet;
- modelos de Deep Learning para séries temporais;
- otimização mais ampla dos parâmetros SARIMA;
- backtesting com diferentes horizontes;
- MLflow para rastreamento de experimentos;
- testes automatizados;
- criação de API com FastAPI;
- automação do pipeline;
- atualização automática do modelo;
- monitoramento de drift;
- deploy da aplicação Streamlit;
- deploy em ambiente cloud;
- integração com uma fonte de dados atualizada.

---

# Autor

**Renan Assis Trevelim**

Projeto desenvolvido como estudo e aplicação prática de:

- Data Science
- Time Series Forecasting
- Machine Learning
- Statistical Modeling
- Deploy de Modelos
- Streamlit