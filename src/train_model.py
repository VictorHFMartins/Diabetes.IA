#%%
## 1. IMPORTAÇÃO DAS BIBLIOTECAS

from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# TensorFlow / Keras é opcional nesta versão.
# Se não estiver instalado, o script continua funcionando com os modelos clássicos.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False
# %%
##2. CONFIGURAÇÕES GERAIS E DIRETÓRIOS DO PROJETO

np.random.seed(42)

if HAS_TENSORFLOW:
    tf.random.set_seed(42)

# LOCALIZAÇÃO DA RAIZ DO PROJETO
#
#  Esta função procura a raiz real do projeto com base no arquivo diabetes.csv.
# Ela funciona nos dois cenários mais comuns:
# 1) projeto/src/train_model.py + projeto/data/diabetes.csv
# 2) projeto/train_model.py     + projeto/data/diabetes.csv

def encontrar_raiz_projeto():
    pasta_script = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    pasta_execucao = Path.cwd().resolve()

    candidatos = [
        pasta_script,
        pasta_script.parent,
        pasta_execucao,
        pasta_execucao.parent,
    ]

    for candidato in candidatos:
        if (candidato / "data" / "diabetes.csv").exists():
            return candidato, candidato / "data" / "diabetes.csv"

        if candidato.name.lower() == "data" and (candidato / "diabetes.csv").exists():
            return candidato.parent, candidato / "diabetes.csv"

        if (candidato / "diabetes.csv").exists():
            return candidato, candidato / "diabetes.csv"

    raise FileNotFoundError(
        "Não encontrei o arquivo diabetes.csv. "
        "Deixe o dataset em 'data/diabetes.csv' e execute o script a partir da raiz do projeto."
    )


PROJECT_ROOT, CAMINHO_DATASET = encontrar_raiz_projeto()

PASTA_MODELS = PROJECT_ROOT / "models"
PASTA_REPORTS = PROJECT_ROOT / "reports"
PASTA_FIGURAS = PASTA_REPORTS / "figures"
PASTA_TREINO = PROJECT_ROOT / "treino"

for pasta in [PASTA_MODELS, PASTA_REPORTS, PASTA_FIGURAS, PASTA_TREINO]:
    pasta.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DIRETÓRIOS DO PROJETO")
print("=" * 70)
print(f"Raiz do projeto: {PROJECT_ROOT}")
print(f"Dataset usado: {CAMINHO_DATASET}")
print(f"Pasta de modelos: {PASTA_MODELS}")
print(f"Pasta de relatórios: {PASTA_REPORTS}")
print(f"Pasta de figuras: {PASTA_FIGURAS}")
print(f"Pasta de treino: {PASTA_TREINO}")

LIMIAR_REGRESSAO = 0.4
LIMIAR_REDE = 0.5
LIMIAR_REDE_AJUSTADO = 0.4

COLUNAS_COM_ZERO_INVALIDO = [
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

# %%
##3. FUNÇÕES AUXILIARES

def salvar_figura(nome_arquivo):
    caminho = PASTA_FIGURAS / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    print(f"Figura salva em: {caminho}")


def obter_probabilidade(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

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
        metricas["ROC AUC"] = roc_auc_score(y_real, y_prob)
        metricas["PR AUC"] = average_precision_score(y_real, y_prob)
    else:
        metricas["ROC AUC"] = np.nan
        metricas["PR AUC"] = np.nan

    return metricas


def avaliar_modelo(nome_modelo, modelo, X_train, X_test, y_train, y_test):
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)
    y_prob_test = obter_probabilidade(modelo, X_test)

    metricas = calcular_metricas(y_test, y_pred_test, y_prob_test)
    metricas["Modelo"] = nome_modelo

    overfitting = {
        "Modelo": nome_modelo,
        "Acurácia Treino": accuracy_score(y_train, y_pred_train),
        "Acurácia Teste": accuracy_score(y_test, y_pred_test),
        "Diferença Acurácia": accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test),
        "Recall Treino": recall_score(y_train, y_pred_train, zero_division=0),
        "Recall Teste": recall_score(y_test, y_pred_test, zero_division=0),
        "Diferença Recall": recall_score(y_train, y_pred_train, zero_division=0) - recall_score(y_test, y_pred_test, zero_division=0),
    }

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO DO MODELO: {nome_modelo}")
    print("=" * 80)
    for chave, valor in metricas.items():
        if chave != "Modelo":
            print(f"{chave}: {valor}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_test))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    return metricas, overfitting, y_pred_test, y_prob_test


def plotar_matriz_confusao(nome_modelo, y_real, y_pred):
    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_pred,
        display_labels=["Sem Diabetes", "Com Diabetes"]
    )
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    salvar_figura(f"matriz_confusao_{nome_modelo.lower().replace(' ', '_').replace('/', '_')}.png")
    plt.show()


def classificar_risco(probabilidade):
    if probabilidade >= 0.70:
        return "Alto", "Probabilidade elevada. Recomenda-se buscar avaliação profissional com prioridade."
    if probabilidade >= 0.40:
        return "Moderado", "Há sinais de atenção. Recomenda-se acompanhamento e avaliação profissional."
    return "Baixo", "Probabilidade menor pelo modelo, mas o resultado não elimina a necessidade de cuidados preventivos."


def gerar_recomendacoes_preventivas(paciente, probabilidade):
    p = paciente.iloc[0]
    fatores = []
    recomendacoes = []

    if p["Glucose"] >= 126:
        fatores.append("glicose elevada")
        recomendacoes.append("A glicose informada está elevada. Recomenda-se avaliação médica.")
    elif p["Glucose"] >= 100:
        fatores.append("glicose em faixa de atenção")
        recomendacoes.append("A glicose está em faixa de atenção. Recomenda-se acompanhar com exames e orientação profissional.")

    if p["BMI"] >= 30:
        fatores.append("IMC elevado")
        recomendacoes.append("O IMC informado sugere obesidade, fator associado ao risco de diabetes tipo 2.")
    elif p["BMI"] >= 25:
        fatores.append("IMC acima da faixa ideal")
        recomendacoes.append("O IMC está acima da faixa ideal e pode contribuir para risco metabólico.")

    if p["Age"] >= 45:
        fatores.append("idade mais elevada")
        recomendacoes.append("A idade é um fator de risco relevante. Consultas preventivas podem ajudar no rastreamento.")

    if p["DiabetesPedigreeFunction"] >= 0.5:
        fatores.append("histórico familiar relevante")
        recomendacoes.append("O histórico familiar informado aumenta a necessidade de atenção preventiva.")

    if p["BloodPressure"] >= 130:
        fatores.append("pressão arterial elevada")
        recomendacoes.append("A pressão arterial informada está elevada e deve ser acompanhada por profissional de saúde.")

    if not fatores:
        fatores.append("nenhum fator isolado se destacou pelas regras simples do agente")
        recomendacoes.append("Mantenha hábitos saudáveis e exames preventivos conforme orientação profissional.")

    nivel, mensagem = classificar_risco(probabilidade)

    return {
        "Nivel de risco": nivel,
        "Mensagem": mensagem,
        "Fatores de atenção": fatores,
        "Recomendações": recomendacoes
    }


def obter_importancia_variaveis(modelo, nomes_colunas):
    etapa_modelo = modelo.named_steps.get("model") if isinstance(modelo, Pipeline) else modelo

    if hasattr(etapa_modelo, "feature_importances_"):
        importancias = etapa_modelo.feature_importances_
    elif hasattr(etapa_modelo, "coef_"):
        importancias = np.abs(etapa_modelo.coef_[0])
    else:
        return None

    df_importancias = pd.DataFrame({
        "Variável": nomes_colunas,
        "Importância": importancias
    }).sort_values("Importância", ascending=False)

    df_importancias["Nome em português"] = df_importancias["Variável"].map(NOMES_VARIAVEIS)

    return df_importancias
# %%
## 4. CARREGAMENTO E PREPARAÇÃO DOS DADOS

df = pd.read_csv(CAMINHO_DATASET)

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())

print("\nResumo estatístico:")
print(df.describe())

print("\nQuantidade de valores por classe:")
print(df["Outcome"].value_counts())

print("\nPercentual por classe:")
print(df["Outcome"].value_counts(normalize=True) * 100)
