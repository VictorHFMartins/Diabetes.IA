# app.py
# ==========================================================
# IA para Predição de Risco de Diabetes
# Protótipo acadêmico com foco em triagem, prevenção,
# explicabilidade, métricas e reflexão ética.
# ==========================================================

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

st.set_page_config(
    page_title="IA para Predição de Diabetes", page_icon="🩺", layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "data" / "diabetes.csv"
MODELS_DIR = BASE_DIR.parent / "models"
REPORTS_DIR = BASE_DIR.parent / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

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


df = carregar_dados()
X, y, X_train, X_test, y_train, y_test = preparar_dados(df)

st.title("🩺 IA para Predição de Risco de Diabetes")

st.markdown(
    """
    Este sistema é um **protótipo acadêmico** de apoio à triagem e prevenção.
    Ele usa dados clínicos para estimar possível risco de diabetes com modelos de Machine Learning.

    **Importante:** o resultado não substitui avaliação médica profissional.
    """
)

st.sidebar.header("Configurações")

aba_predicao, aba_modelos, aba_dataset, aba_eda, aba_etica = st.tabs(
    ["Predição", "Modelos e Métricas", "Dataset", "EDA", "Ética e Limitações"]
)

with aba_predicao:
    st.header("Predição individual")
    st.info("A predição será adicionada nos próximos commits.")

with aba_modelos:
    st.header("Modelos e Métricas")
    st.info("As métricas serão adicionadas nos próximos commits.")

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

with aba_eda:
    st.header("EDA")
    st.info("A análise exploratória será adicionada nos próximos commits.")

with aba_etica:
    st.header("Ética e Limitações")
    st.info("A reflexão ética será adicionada nos próximos commits.")

@st.cache_resource
def treinar_modelos():
    df = carregar_dados()
    X, y, X_train, X_test, y_train, y_test = preparar_dados(df)

    modelos = {}

    modelos["Regressão Logística"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    modelos["Árvore de Decisão"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])

    pipeline_arvore_grid = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    parametros_arvore = {
        "model__max_depth": [3, 4, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced"]
    }

    grid_arvore = GridSearchCV(
        estimator=pipeline_arvore_grid,
        param_grid=parametros_arvore,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    modelos["Random Forest Otimizado"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    modelos["KNN"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])

    modelos["SVC"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    modelos["Gradient Boosting Otimizado"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=2,
            min_samples_split=10,
            n_estimators=200,
            random_state=42
        ))
    ])

    for modelo in modelos.values():
        modelo.fit(X_train, y_train)

    grid_arvore.fit(X_train, y_train)
    modelos["Árvore de Decisão Otimizada"] = grid_arvore.best_estimator_

    return modelos, X, y, X_train, X_test, y_train, y_test


modelos, X, y, X_train, X_test, y_train, y_test = treinar_modelos()

modelo_escolhido_nome = st.sidebar.selectbox(
    "Escolha o modelo de IA:",
    list(modelos.keys()),
    index=list(modelos.keys()).index("Árvore de Decisão")
)

modelo_escolhido = modelos[modelo_escolhido_nome]

st.sidebar.write("Modelo selecionado:")
st.sidebar.success(modelo_escolhido_nome)