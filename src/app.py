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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
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
    page_title="IA para Predição de Diabetes",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "data" / "diabetes.csv"
MODELS_DIR = BASE_DIR.parent / "models"
REPORTS_DIR = BASE_DIR.parent / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

COLUNAS_ZERO_INVALIDO = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

NOMES_VARIAVEIS = {
    "Pregnancies": "Gestações",
    "Glucose": "Glicose",
    "BloodPressure": "Pressão arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Histórico familiar",
    "Age": "Idade",
    "Outcome": "Resultado"
}

st.title("🩺 IA para Predição de Risco de Diabetes")

st.markdown(
    """
    Este sistema é um **protótipo acadêmico** de apoio à triagem e prevenção.
    Ele usa dados clínicos para estimar possível risco de diabetes com modelos de Machine Learning.

    **Importante:** o resultado não substitui avaliação médica profissional.
    """
)

st.sidebar.header("Configurações")
st.sidebar.info("As funcionalidades serão adicionadas nos próximos commits.")

aba_predicao, aba_modelos, aba_dataset, aba_eda, aba_etica = st.tabs([
    "Predição",
    "Modelos e Métricas",
    "Dataset",
    "EDA",
    "Ética e Limitações"
])

with aba_predicao:
    st.header("Predição individual")
    st.info("A predição será adicionada nos próximos commits.")

with aba_modelos:
    st.header("Modelos e Métricas")
    st.info("As métricas serão adicionadas nos próximos commits.")

with aba_dataset:
    st.header("Dataset")
    st.info("O carregamento dos dados será adicionado nos próximos commits.")

with aba_eda:
    st.header("EDA")
    st.info("A análise exploratória será adicionada nos próximos commits.")

with aba_etica:
    st.header("Ética e Limitações")
    st.info("A reflexão ética será adicionada nos próximos commits.")