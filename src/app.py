# app.py
# ==============================
# IA para Predição de Risco de Diabetes
# Protótipo acadêmico com foco em triagem, prevenção,
# explicabilidade, métricas e reflexão ética.
# ==============================

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# CONFIGURAÇÕES DA PÁGINA


st.set_page_config(
    page_title="IA para Predição de Diabetes", page_icon="🩺", layout="wide"
)


# CAMINHOS E CONSTANTES


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATASET_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

# fallback caso o dataset esteja dentro da pasta src
if not DATASET_PATH.exists():
    DATASET_PATH = BASE_DIR / "diabetes.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COLUNAS_ZERO_INVALIDO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

NOMES_VARIAVEIS = {
    "Pregnancies": "Gestações",
    "Glucose": "Glicose",
    "BloodPressure": "Pressão arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Histórico familiar",
    "Age": "Idade",
    "Outcome": "Resultado",
}


# CARREGAMENTO E PREPARAÇÃO


@st.cache_data
def carregar_dados():
    return pd.read_csv(DATASET_PATH)


def preparar_dados(df):
    X = df.drop("Outcome", axis=1).copy()
    y = df["Outcome"].copy()

    X[COLUNAS_ZERO_INVALIDO] = X[COLUNAS_ZERO_INVALIDO].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X, y, X_train, X_test, y_train, y_test


# FUNÇÕES DE MÉTRICAS


def obter_probabilidade(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X)
        denominador = scores.max() - scores.min()

        if denominador == 0:
            return np.zeros_like(scores)

        return (scores - scores.min()) / denominador

    return None


def calcular_metricas(y_real, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()

    especificidade = tn / (tn + fp) if (tn + fp) else 0
    taxa_falso_negativo = fn / (fn + tp) if (fn + tp) else 0
    taxa_falso_positivo = fp / (fp + tn) if (fp + tn) else 0

    metricas = {
        "Acurácia": accuracy_score(y_real, y_pred),
        "Precisão": precision_score(y_real, y_pred, zero_division=0),
        "Recall/Sensibilidade": recall_score(y_real, y_pred, zero_division=0),
        "Especificidade": especificidade,
        "F1-score": f1_score(y_real, y_pred, zero_division=0),
        "Falsos Negativos": int(fn),
        "Taxa de Falsos Negativos": taxa_falso_negativo,
        "Falsos Positivos": int(fp),
        "Taxa de Falsos Positivos": taxa_falso_positivo,
    }

    if y_prob is not None:
        try:
            metricas["ROC AUC"] = roc_auc_score(y_real, y_prob)
            metricas["PR AUC"] = average_precision_score(y_real, y_prob)
        except ValueError:
            metricas["ROC AUC"] = np.nan
            metricas["PR AUC"] = np.nan
    else:
        metricas["ROC AUC"] = np.nan
        metricas["PR AUC"] = np.nan

    return metricas


# CAMADA DE AGENTE E EXPLICABILIDADE


def classificar_risco(probabilidade):
    if probabilidade is None:
        return "Indefinido", "Não foi possível calcular a probabilidade do risco."

    if probabilidade >= 0.70:
        return (
            "Alto",
            "Probabilidade elevada. Recomenda-se buscar avaliação profissional com prioridade.",
        )

    if probabilidade >= 0.40:
        return (
            "Moderado",
            "Há sinais de atenção. Recomenda-se acompanhamento e avaliação profissional.",
        )

    return (
        "Baixo",
        "Probabilidade menor pelo modelo, mas o resultado não elimina a necessidade de cuidados preventivos.",
    )


def gerar_recomendacoes_preventivas(paciente, probabilidade):
    p = paciente.iloc[0]

    fatores = []
    recomendacoes = []

    if p["Glucose"] >= 126:
        fatores.append("glicose elevada")
        recomendacoes.append(
            "A glicose informada está em faixa de atenção. Procure orientação profissional para interpretação adequada."
        )
    elif p["Glucose"] >= 100:
        fatores.append("glicose em faixa de atenção")
        recomendacoes.append(
            "A glicose está acima de uma faixa ideal em jejum. Vale acompanhar esse indicador com um profissional."
        )

    if p["BMI"] >= 30:
        fatores.append("IMC elevado")
        recomendacoes.append(
            "O IMC informado sugere obesidade, fator associado ao risco de diabetes tipo 2."
        )
    elif p["BMI"] >= 25:
        fatores.append("IMC acima da faixa ideal")
        recomendacoes.append(
            "O IMC está acima da faixa ideal e pode contribuir para risco metabólico."
        )

    if p["Age"] >= 45:
        fatores.append("idade mais elevada")
        recomendacoes.append(
            "A idade é um fator de risco importante. Acompanhamentos periódicos podem ajudar na prevenção."
        )

    if p["DiabetesPedigreeFunction"] >= 0.5:
        fatores.append("histórico familiar relevante")
        recomendacoes.append(
            "O histórico familiar informado aumenta a necessidade de atenção preventiva."
        )

    if p["BloodPressure"] >= 130:
        fatores.append("pressão arterial elevada")
        recomendacoes.append(
            "A pressão arterial informada está elevada. Ela deve ser acompanhada por profissional de saúde."
        )

    if not recomendacoes:
        recomendacoes.append(
            "Mantenha acompanhamento preventivo, hábitos saudáveis e exames periódicos conforme orientação profissional."
        )

    nivel, mensagem = classificar_risco(probabilidade)

    return {
        "nivel": nivel,
        "mensagem": mensagem,
        "fatores": fatores
        if fatores
        else ["nenhum fator isolado se destacou pelas regras clínicas simples"],
        "recomendacoes": recomendacoes,
    }


def obter_importancia_variaveis(modelo, nomes_colunas):
    etapa_modelo = (
        modelo.named_steps.get("model") if isinstance(modelo, Pipeline) else modelo
    )

    if hasattr(etapa_modelo, "feature_importances_"):
        importancias = etapa_modelo.feature_importances_
    elif hasattr(etapa_modelo, "coef_"):
        importancias = np.abs(etapa_modelo.coef_[0])
    else:
        return None

    df_importancias = pd.DataFrame(
        {"Variável": nomes_colunas, "Importância": importancias}
    ).sort_values("Importância", ascending=False)

    df_importancias["Variável"] = (
        df_importancias["Variável"]
        .map(NOMES_VARIAVEIS)
        .fillna(df_importancias["Variável"])
    )

    return df_importancias


# TREINAMENTO DOS MODELOS


@st.cache_resource
def treinar_modelos():
    df = carregar_dados()
    X, y, X_train, X_test, y_train, y_test = preparar_dados(df)

    modelos = {}

    modelos["Regressão Logística"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    modelos["Árvore de Decisão"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]
    )

    pipeline_arvore_grid = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )

    parametros_arvore = {
        "model__max_depth": [3, 4, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced"],
    }

    grid_arvore = GridSearchCV(
        estimator=pipeline_arvore_grid,
        param_grid=parametros_arvore,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )

    modelos["Random Forest Otimizado"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    modelos["KNN"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    modelos["SVC"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )

    modelos["Gradient Boosting Otimizado"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_leaf=2,
                    min_samples_split=10,
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    for modelo in modelos.values():
        modelo.fit(X_train, y_train)

    grid_arvore.fit(X_train, y_train)
    modelos["Árvore de Decisão Otimizada"] = grid_arvore.best_estimator_

    metricas = []

    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_prob = obter_probabilidade(modelo, X_test)

        resultado = calcular_metricas(y_test, y_pred, y_prob)
        resultado["Modelo"] = nome
        metricas.append(resultado)

    df_metricas = pd.DataFrame(metricas)

    ordem_colunas = [
        "Modelo",
        "Acurácia",
        "Precisão",
        "Recall/Sensibilidade",
        "Especificidade",
        "F1-score",
        "ROC AUC",
        "PR AUC",
        "Falsos Negativos",
        "Taxa de Falsos Negativos",
        "Falsos Positivos",
        "Taxa de Falsos Positivos",
    ]

    df_metricas = df_metricas[ordem_colunas].sort_values(
        by=["F1-score", "Recall/Sensibilidade", "Acurácia"], ascending=False
    )

    df_metricas.to_csv(REPORTS_DIR / "metricas_modelos.csv", index=False)

    return modelos, df_metricas, X, y, X_train, X_test, y_train, y_test


# FUNÇÕES DE VISUALIZAÇÃO - EDA


def plotar_correlacao(df):
    fig, ax = plt.subplots(figsize=(10, 7))

    corr = df.corr(numeric_only=True)
    imagem = ax.imshow(corr, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
            )

    ax.set_title("Matriz de Correlação")
    fig.colorbar(imagem, ax=ax)
    fig.tight_layout()

    return fig


def plotar_histograma(df, coluna):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df[coluna].dropna(), bins=25)
    ax.set_title(f"Distribuição de {NOMES_VARIAVEIS.get(coluna, coluna)}")
    ax.set_xlabel(NOMES_VARIAVEIS.get(coluna, coluna))
    ax.set_ylabel("Frequência")
    ax.grid(True, alpha=0.3)

    return fig


def plotar_boxplot_por_classe(df, coluna):
    sem_diabetes = df[df["Outcome"] == 0][coluna].dropna()
    com_diabetes = df[df["Outcome"] == 1][coluna].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.boxplot([sem_diabetes, com_diabetes], labels=["Sem Diabetes", "Com Diabetes"])

    ax.set_title(f"{NOMES_VARIAVEIS.get(coluna, coluna)} por classe")
    ax.set_ylabel(NOMES_VARIAVEIS.get(coluna, coluna))
    ax.grid(True, alpha=0.3)

    return fig


# EXECUÇÃO DO PIPELINE


df = carregar_dados()
X, y, X_train, X_test, y_train, y_test = preparar_dados(df)

modelos, df_metricas, X, y, X_train, X_test, y_train, y_test = treinar_modelos()


# INTERFACE PRINCIPAL


st.title("🩺 IA para Predição de Risco de Diabetes")

st.markdown(
    """
    Este sistema é um **protótipo acadêmico** de apoio à triagem e prevenção.
    Ele usa dados clínicos para estimar possível risco de diabetes com modelos de Machine Learning.

    **Importante:** o resultado não substitui avaliação médica profissional.
    """
)


# SIDEBAR


st.sidebar.header("Configurações")

modelo_escolhido_nome = st.sidebar.selectbox(
    "Escolha o modelo de IA:",
    list(modelos.keys()),
    index=list(modelos.keys()).index("Árvore de Decisão"),
)

modelo_escolhido = modelos[modelo_escolhido_nome]

st.sidebar.write("Modelo selecionado:")
st.sidebar.success(modelo_escolhido_nome)


# ABAS


aba_predicao, aba_modelos, aba_dataset, aba_eda, aba_etica = st.tabs(
    ["Predição", "Modelos e Métricas", "Dataset", "EDA", "Ética e Limitações"]
)


# ABA 1: PREDIÇÃO


with aba_predicao:
    st.header("Predição individual")

    st.write(
        "Preencha os dados abaixo para estimar o risco. "
        "Os campos representam as mesmas variáveis usadas no dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input(
            "Gestações", min_value=0, max_value=20, value=2, step=1
        )

        glucose = st.number_input(
            "Glicose", min_value=0, max_value=250, value=140, step=1
        )

    with col2:
        blood_pressure = st.number_input(
            "Pressão arterial", min_value=0, max_value=150, value=80, step=1
        )

        skin_thickness = st.number_input(
            "Espessura da pele", min_value=0, max_value=120, value=30, step=1
        )

    with col3:
        insulin = st.number_input(
            "Insulina", min_value=0, max_value=900, value=120, step=1
        )

        bmi = st.number_input(
            "IMC", min_value=0.0, max_value=80.0, value=32.5, step=0.1
        )

    with col4:
        diabetes_pedigree = st.number_input(
            "Histórico familiar",
            min_value=0.0,
            max_value=3.0,
            value=0.6,
            step=0.001,
            format="%.3f",
        )

        age = st.number_input("Idade", min_value=1, max_value=120, value=35, step=1)

    paciente = pd.DataFrame(
        {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree],
            "Age": [age],
        }
    )

    st.subheader("Dados informados")
    st.dataframe(paciente, use_container_width=True)

    if st.button("Realizar predição"):
        predicao = modelo_escolhido.predict(paciente)[0]
        y_prob_paciente = None

        if hasattr(modelo_escolhido, "predict_proba"):
            probabilidades = modelo_escolhido.predict_proba(paciente)[0]
            probabilidade_sem_diabetes = probabilidades[0]
            probabilidade_com_diabetes = probabilidades[1]
            y_prob_paciente = probabilidade_com_diabetes
        else:
            probabilidade_sem_diabetes = None
            probabilidade_com_diabetes = float(predicao)

        interpretacao = gerar_recomendacoes_preventivas(
            paciente, probabilidade_com_diabetes
        )

        st.markdown("---")
        st.subheader("Resultado da predição")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            if predicao == 1:
                st.error("Possível risco de diabetes")
            else:
                st.success("Baixo risco pelo modelo")

        with col_r2:
            st.metric("Nível de risco preventivo", interpretacao["nivel"])

        with col_r3:
            if y_prob_paciente is not None:
                st.metric(
                    "Probabilidade com diabetes",
                    f"{probabilidade_com_diabetes * 100:.2f}%",
                )
            else:
                st.metric("Classe prevista", int(predicao))

        if y_prob_paciente is not None:
            col_prob1, col_prob2 = st.columns(2)

            with col_prob1:
                st.metric(
                    "Probabilidade Sem Diabetes",
                    f"{probabilidade_sem_diabetes * 100:.2f}%",
                )

            with col_prob2:
                st.metric(
                    "Probabilidade Com Diabetes",
                    f"{probabilidade_com_diabetes * 100:.2f}%",
                )

        st.info(interpretacao["mensagem"])

        st.subheader("Explicação preventiva do resultado")

        st.write("Fatores que chamaram atenção nesta predição:")
        st.write(", ".join(interpretacao["fatores"]))

        st.write("Recomendações geradas pelo agente:")

        for recomendacao in interpretacao["recomendacoes"]:
            st.markdown(f"- {recomendacao}")

        importancia = obter_importancia_variaveis(modelo_escolhido, paciente.columns)

        if importancia is not None:
            st.subheader("Variáveis mais importantes para este modelo")
            st.dataframe(importancia.head(8), use_container_width=True)
            st.bar_chart(importancia.set_index("Variável")["Importância"])
        else:
            st.caption(
                "Este modelo não possui importância de variáveis diretamente interpretável."
            )

        st.warning(
            "Este resultado é apenas uma estimativa gerada por IA. "
            "Não use a ferramenta como diagnóstico, decisão médica, aprovação de benefício, "
            "plano de saúde ou substituição de consulta."
        )


# ABA 2: MODELOS E MÉTRICAS


with aba_modelos:
    st.header("Comparação dos modelos")

    st.write(
        "A tabela abaixo compara os modelos usando métricas importantes para saúde, "
        "como acurácia, precisão, recall/sensibilidade, especificidade, F1-score, "
        "ROC AUC, PR AUC, falsos negativos e falsos positivos."
    )

    st.dataframe(df_metricas, use_container_width=True)

    st.subheader("Melhor modelo pelo critério F1-score > Recall > Acurácia")

    melhor_modelo = df_metricas.iloc[0]

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric("Acurácia", f"{melhor_modelo['Acurácia']:.3f}")

    with col_m2:
        st.metric("Precisão", f"{melhor_modelo['Precisão']:.3f}")

    with col_m3:
        st.metric("Recall", f"{melhor_modelo['Recall/Sensibilidade']:.3f}")

    with col_m4:
        st.metric("F1-score", f"{melhor_modelo['F1-score']:.3f}")

    with col_m5:
        st.metric("ROC AUC", f"{melhor_modelo['ROC AUC']:.3f}")

    st.info(f"Modelo melhor classificado: **{melhor_modelo['Modelo']}**")

    st.subheader("Por que o recall é importante neste projeto?")

    st.markdown(
        """
        Em um sistema de triagem preventiva, o **falso negativo** é um erro crítico:
        ele ocorre quando o modelo classifica uma pessoa como baixo risco mesmo havendo sinais de risco.

        Por isso, além da acurácia, este projeto avalia **recall/sensibilidade**,
        matriz de confusão e taxa de falsos negativos.
        """
    )

    st.subheader("Matriz de confusão do modelo selecionado")

    y_pred_modelo = modelo_escolhido.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred_modelo)

    matriz_df = pd.DataFrame(
        matriz,
        index=["Real: Sem Diabetes", "Real: Com Diabetes"],
        columns=["Previsto: Sem Diabetes", "Previsto: Com Diabetes"],
    )

    st.dataframe(matriz_df, use_container_width=True)

    tn, fp, fn, tp = matriz.ravel()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Verdadeiros Negativos", tn)
    c2.metric("Falsos Positivos", fp)
    c3.metric("Falsos Negativos", fn)
    c4.metric("Verdadeiros Positivos", tp)

    st.subheader("Gráfico de comparação")

    st.bar_chart(
        df_metricas.set_index("Modelo")[
            ["Acurácia", "Precisão", "Recall/Sensibilidade", "F1-score", "ROC AUC"]
        ]
    )


# ABA 3: DATASET


with aba_dataset:
    st.header("Visualização do dataset")

    st.write("Primeiras linhas do dataset:")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Resumo estatístico")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribuição das classes")

    contagem_classes = (
        df["Outcome"]
        .value_counts()
        .rename(index={0: "Sem Diabetes", 1: "Com Diabetes"})
    )

    st.bar_chart(contagem_classes)

    total = len(df)
    sem_diabetes = int((df["Outcome"] == 0).sum())
    com_diabetes = int((df["Outcome"] == 1).sum())

    col_d1, col_d2, col_d3 = st.columns(3)

    col_d1.metric("Total de registros", total)
    col_d2.metric("Sem diabetes", sem_diabetes)
    col_d3.metric("Com diabetes", com_diabetes)

    st.subheader("Análise de zeros inválidos")

    zeros_invalidos = pd.DataFrame(
        {
            "Coluna": COLUNAS_ZERO_INVALIDO,
            "Quantidade de zeros": [
                int((df[col] == 0).sum()) for col in COLUNAS_ZERO_INVALIDO
            ],
            "Percentual": [
                float((df[col] == 0).mean() * 100) for col in COLUNAS_ZERO_INVALIDO
            ],
        }
    )

    st.dataframe(zeros_invalidos, use_container_width=True)

    st.info(
        "No pré-processamento, zeros clinicamente improváveis em glicose, "
        "pressão arterial, espessura da pele, insulina e IMC são tratados "
        "como dados ausentes e preenchidos pela mediana."
    )


# ABA 4: EDA


with aba_eda:
    st.header("Análise Exploratória de Dados - EDA")

    st.write(
        "Esta seção reforça a interpretação dos dados antes da modelagem, "
        "com gráficos de distribuição, correlação, comparação por classe "
        "e médias agrupadas."
    )

    st.subheader("Matriz de correlação")
    st.pyplot(plotar_correlacao(df))

    st.subheader("Comparação de médias por classe")

    medias = df.groupby("Outcome").mean(numeric_only=True)
    medias.index = ["Sem Diabetes", "Com Diabetes"]

    st.dataframe(medias, use_container_width=True)

    st.subheader("Histogramas")

    coluna_hist = st.selectbox(
        "Escolha uma variável para visualizar o histograma:",
        [
            "Glucose",
            "BMI",
            "Age",
            "Insulin",
            "BloodPressure",
            "DiabetesPedigreeFunction",
        ],
    )

    st.pyplot(plotar_histograma(df, coluna_hist))

    st.subheader("Boxplot por classe")

    coluna_box = st.selectbox(
        "Escolha uma variável para comparar por classe:",
        [
            "Glucose",
            "BMI",
            "Age",
            "Insulin",
            "BloodPressure",
            "DiabetesPedigreeFunction",
        ],
        index=1,
    )

    st.pyplot(plotar_boxplot_por_classe(df, coluna_box))

    st.markdown(
        """
        **Leitura crítica para o relatório:**  
        A EDA ajuda a identificar padrões, valores ausentes, distribuição das classes
        e variáveis com maior relação com o desfecho. Em saúde, essa etapa é essencial
        para evitar conclusões baseadas apenas na acurácia do modelo.
        """
    )


# ABA 5: ÉTICA E LIMITAÇÕES


with aba_etica:
    st.header("Ética, riscos e limitações")

    st.markdown(
        """
        Este protótipo deve ser usado apenas como **apoio educacional e acadêmico**.

        A proposta do sistema é auxiliar na **triagem preventiva** e na conscientização
        sobre possíveis fatores associados ao risco de diabetes. Ele não realiza diagnóstico
        médico, não substitui consulta profissional e não deve ser usado como única fonte
        para tomada de decisão clínica.

        ### Limitações do dataset

        - O dataset possui quantidade limitada de registros.
        - A base pode não representar a população brasileira.
        - Algumas variáveis possuem muitos valores ausentes ou valores 0 usados como ausência de informação.
        - O modelo aprende padrões do dataset utilizado e pode reproduzir vieses presentes nos dados.
        - O desempenho obtido neste projeto não garante o mesmo resultado em populações diferentes.
        - O dataset não contém todos os fatores clínicos, sociais e comportamentais relevantes para avaliação de diabetes.

        ### Riscos do modelo

        - **Falsos negativos:** podem atrasar a busca por orientação profissional.
        - **Falsos positivos:** podem gerar preocupação desnecessária.
        - O modelo pode apresentar desempenho diferente entre grupos populacionais.
        - A predição pode ser interpretada incorretamente por usuários sem orientação.
        - O sistema não deve ser usado para diagnóstico, prescrição, exclusão de atendimento,
          definição de plano de saúde ou decisões administrativas sobre pessoas.

        ### Uso responsável

        - A IA deve apoiar a triagem e a conscientização, nunca substituir profissionais de saúde.
        - A decisão final deve ser feita por médicos ou profissionais qualificados.
        - O usuário deve ser informado sobre as limitações do modelo.
        - Dados clínicos exigem cuidado com privacidade, armazenamento, consentimento e segurança.
        - O resultado deve ser apresentado como probabilidade estimada, não como certeza clínica.

        ### Transparência

        Sempre que possível, o sistema deve explicar quais variáveis influenciaram o risco,
        apresentar métricas de erro e deixar claro que a predição é probabilística.

        A interface apresenta:
        
        - probabilidade estimada;
        - matriz de confusão;
        - falsos positivos e falsos negativos;
        - importância das variáveis em modelos interpretáveis;
        - recomendações preventivas baseadas em regras simples;
        - aviso explícito de que o sistema não substitui avaliação médica.
        """
    )

    st.subheader("Recomendações concretas de mitigação")

    st.markdown(
        """
        Para reduzir riscos e aumentar a confiabilidade do sistema, recomenda-se:

        - Usar bases de dados maiores, mais diversas e mais próximas da população-alvo.
        - Validar o modelo com dados locais antes de qualquer uso real.
        - Monitorar falsos negativos e falsos positivos continuamente.
        - Priorizar recall/sensibilidade em cenários de triagem preventiva.
        - Avaliar desempenho por subgrupos, como faixa etária, sexo, etnia e contexto social, quando os dados estiverem disponíveis.
        - Manter mensagens claras de que o sistema não substitui avaliação médica.
        - Implementar política de privacidade se dados reais forem armazenados.
        - Registrar versões do modelo, métricas e dataset utilizados.
        - Revisar periodicamente o modelo com acompanhamento de profissionais de saúde.
        """
    )

    st.subheader("Conclusão ética")

    st.success(
        """
        A IA pode apoiar ações preventivas em saúde, mas deve ser usada com responsabilidade,
        transparência e supervisão humana. Neste projeto, o modelo é tratado como ferramenta
        educacional de apoio à triagem, não como diagnóstico definitivo.
        """
    )
