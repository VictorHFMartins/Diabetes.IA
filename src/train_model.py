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
    print(f"TensorFlow carregado com sucesso. Versão: {tf.__version__}")

except Exception as erro:
    HAS_TENSORFLOW = False
    print("\nTensorFlow não pôde ser carregado nesta execução.")
    print("Erro encontrado:")
    print(erro)
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

#%%
## 5. EDA - ANÁLISE EXPLORATÓRIA VISUAL

print("\n" + "=" * 80)
print("EDA - ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 80)

# Distribuição da variável alvo
contagem_classes = df["Outcome"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Sem Diabetes", "Com Diabetes"], contagem_classes.values)
plt.title("Distribuição das Classes")
plt.ylabel("Quantidade de registros")
salvar_figura("eda_distribuicao_classes.png")
plt.show()

# Zeros inválidos antes do tratamento
zeros_invalidos = pd.DataFrame({
    "Coluna": COLUNAS_COM_ZERO_INVALIDO,
    "Quantidade de zeros": [(df[col] == 0).sum() for col in COLUNAS_COM_ZERO_INVALIDO],
    "Percentual (%)": [(df[col] == 0).mean() * 100 for col in COLUNAS_COM_ZERO_INVALIDO]
})

print("\nZeros inválidos antes do tratamento:")
print(zeros_invalidos)
zeros_invalidos.to_csv(PASTA_REPORTS / "zeros_invalidos.csv", index=False)

# Histogramas principais
for coluna in ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "DiabetesPedigreeFunction"]:
    plt.figure(figsize=(8, 5))
    plt.hist(df[coluna].dropna(), bins=25)
    plt.title(f"Distribuição de {NOMES_VARIAVEIS.get(coluna, coluna)}")
    plt.xlabel(NOMES_VARIAVEIS.get(coluna, coluna))
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    salvar_figura(f"eda_histograma_{coluna}.png")
    plt.show()

# Boxplots por classe
for coluna in ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "DiabetesPedigreeFunction"]:
    sem_diabetes = df[df["Outcome"] == 0][coluna].dropna()
    com_diabetes = df[df["Outcome"] == 1][coluna].dropna()

    plt.figure(figsize=(8, 5))
    plt.boxplot([sem_diabetes, com_diabetes], labels=["Sem Diabetes", "Com Diabetes"])
    plt.title(f"{NOMES_VARIAVEIS.get(coluna, coluna)} por classe")
    plt.ylabel(NOMES_VARIAVEIS.get(coluna, coluna))
    plt.grid(True, alpha=0.3)
    salvar_figura(f"eda_boxplot_{coluna}.png")
    plt.show()

# Matriz de correlação
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.title("Matriz de Correlação")
plt.colorbar()
salvar_figura("eda_matriz_correlacao.png")
plt.show()

# Comparação de médias por classe
medias_por_classe = df.groupby("Outcome").mean(numeric_only=True)
medias_por_classe.index = ["Sem Diabetes", "Com Diabetes"]

print("\nComparação de médias por classe:")
print(medias_por_classe)

medias_por_classe.to_csv(PASTA_REPORTS / "medias_por_classe.csv")

#%%
## 6. SEPARAR ENTRADAS E SAÍDA + TRATAR ZEROS INVÁLIDOS

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

X[COLUNAS_COM_ZERO_INVALIDO] = X[COLUNAS_COM_ZERO_INVALIDO].replace(0, np.nan)

print("\nQuantidade de valores ausentes depois da troca de 0 por NaN:")
print(X.isna().sum())

# %%
## 7. SEPARAR TREINO E TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)

print("\nDistribuição das classes no treino:")
print(y_train.value_counts())

print("\nDistribuição das classes no teste:")
print(y_test.value_counts())

# Salvar divisões de treino e teste para reprodutibilidade acadêmica.
# Esses arquivos ajudam a demonstrar exatamente quais dados foram usados
# no treinamento e na avaliação.
X_train.to_csv(PASTA_TREINO / "X_train.csv", index=False)
X_test.to_csv(PASTA_TREINO / "X_test.csv", index=False)
y_train.to_csv(PASTA_TREINO / "y_train.csv", index=False)
y_test.to_csv(PASTA_TREINO / "y_test.csv", index=False)
print(f"\nArquivos de treino/teste salvos em: {PASTA_TREINO}")


# %%
## 8. DEFINIÇÃO E TREINAMENTO DOS MODELOS

modelos = {}

modelos["Regressão Logística"] = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

modelos["Regressão Logística - Limiar 0.4"] = modelos["Regressão Logística"]

modelos["Árvore de Decisão"] = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(max_depth=4, random_state=42))
])

# Árvore otimizada
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

modelos["Random Forest"] = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
])

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

modelos["Gradient Boosting"] = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingClassifier(random_state=42))
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

# Treina modelos principais
for nome, modelo in modelos.items():
    if nome != "Regressão Logística - Limiar 0.4":
        modelo.fit(X_train, y_train)

# Treina árvore otimizada via GridSearchCV
grid_arvore.fit(X_train, y_train)
modelos["Árvore de Decisão Otimizada"] = grid_arvore.best_estimator_

print("\nMelhores parâmetros da Árvore de Decisão Otimizada:")
print(grid_arvore.best_params_)

print("\nMelhor recall médio na validação cruzada da árvore:")
print(grid_arvore.best_score_)

# %%
## 9. REDE NEURAL COM TENSORFLOW / KERAS - OPCIONAL

if HAS_TENSORFLOW:
    preprocessador_rede = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_rede = preprocessador_rede.fit_transform(X_train).astype("float32")
    X_test_rede = preprocessador_rede.transform(X_test).astype("float32")

    y_train_rede = y_train.astype("float32")
    y_test_rede = y_test.astype("float32")

    modelo_rede = keras.Sequential([
        layers.Input(shape=(X_train_rede.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo_rede.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    historico_rede = modelo_rede.fit(
        X_train_rede,
        y_train_rede,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    probabilidades_rede = modelo_rede.predict(X_test_rede).ravel()
    y_pred_rede = (probabilidades_rede >= LIMIAR_REDE).astype(int)
    y_pred_rede_limiar = (probabilidades_rede >= LIMIAR_REDE_AJUSTADO).astype(int)

    # Métricas da rede serão adicionadas manualmente na etapa de avaliação

    plt.figure(figsize=(10, 6))
    plt.plot(historico_rede.history["loss"], label="Loss Treino")
    plt.plot(historico_rede.history["val_loss"], label="Loss Validação")
    plt.title("Evolução da Perda - Rede Neural")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    salvar_figura("rede_neural_loss.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(historico_rede.history["accuracy"], label="Acurácia Treino")
    plt.plot(historico_rede.history["val_accuracy"], label="Acurácia Validação")
    plt.title("Evolução da Acurácia - Rede Neural")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    salvar_figura("rede_neural_acuracia.png")
    plt.show()

else:
    print("\nTensorFlow não está instalado. A rede neural será ignorada nesta execução.")

# %%
# 10. AVALIAR TODOS OS MODELOS

resultados_lista = []
overfitting_lista = {}
predicoes_teste = {}
probabilidades_teste = {}

for nome, modelo in modelos.items():
    if nome == "Regressão Logística - Limiar 0.4":
        probabilidades = modelos["Regressão Logística"].predict_proba(X_test)[:, 1]
        y_pred = (probabilidades >= LIMIAR_REGRESSAO).astype(int)

        metricas = calcular_metricas(y_test, y_pred, probabilidades)
        metricas["Modelo"] = nome
        resultados_lista.append(metricas)

        y_pred_train = (modelos["Regressão Logística"].predict_proba(X_train)[:, 1] >= LIMIAR_REGRESSAO).astype(int)
        overfitting_lista[nome] = {
            "Modelo": nome,
            "Acurácia Treino": accuracy_score(y_train, y_pred_train),
            "Acurácia Teste": accuracy_score(y_test, y_pred),
            "Diferença Acurácia": accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred),
            "Recall Treino": recall_score(y_train, y_pred_train, zero_division=0),
            "Recall Teste": recall_score(y_test, y_pred, zero_division=0),
            "Diferença Recall": recall_score(y_train, y_pred_train, zero_division=0) - recall_score(y_test, y_pred, zero_division=0),
        }

        predicoes_teste[nome] = y_pred
        probabilidades_teste[nome] = probabilidades
        continue

    metricas, overfitting, y_pred, y_prob = avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test)

    resultados_lista.append(metricas)
    overfitting_lista[nome] = overfitting
    predicoes_teste[nome] = y_pred
    probabilidades_teste[nome] = y_prob

# Adiciona rede neural se disponível
if HAS_TENSORFLOW:
    metricas_rede = calcular_metricas(y_test, y_pred_rede, probabilidades_rede)
    metricas_rede["Modelo"] = "Rede Neural - Keras"
    resultados_lista.append(metricas_rede)
    predicoes_teste["Rede Neural - Keras"] = y_pred_rede
    probabilidades_teste["Rede Neural - Keras"] = probabilidades_rede

    metricas_rede_limiar = calcular_metricas(y_test, y_pred_rede_limiar, probabilidades_rede)
    metricas_rede_limiar["Modelo"] = "Rede Neural - Keras - Limiar 0.4"
    resultados_lista.append(metricas_rede_limiar)
    predicoes_teste["Rede Neural - Keras - Limiar 0.4"] = y_pred_rede_limiar
    probabilidades_teste["Rede Neural - Keras - Limiar 0.4"] = probabilidades_rede

    # Overfitting simplificado da rede pela última época do histórico
    overfitting_lista["Rede Neural - Keras"] = {
        "Modelo": "Rede Neural - Keras",
        "Acurácia Treino": historico_rede.history["accuracy"][-1],
        "Acurácia Teste": accuracy_score(y_test, y_pred_rede),
        "Diferença Acurácia": historico_rede.history["accuracy"][-1] - accuracy_score(y_test, y_pred_rede),
        "Recall Treino": historico_rede.history["recall"][-1],
        "Recall Teste": recall_score(y_test, y_pred_rede, zero_division=0),
        "Diferença Recall": historico_rede.history["recall"][-1] - recall_score(y_test, y_pred_rede, zero_division=0),
    }

# %%
# 11. TABELA FINAL DE RESULTADOS

resultados = pd.DataFrame(resultados_lista)

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

resultados = resultados[ordem_colunas].sort_values(
    by=["F1-score", "Recall/Sensibilidade", "Acurácia"],
    ascending=False
)

print("\n" + "=" * 80)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("=" * 80)
print(resultados)

resultados.to_csv(PASTA_REPORTS / "metricas_modelos.csv", index=False)

melhor_linha = resultados.iloc[0]
melhor_modelo_nome = melhor_linha["Modelo"]

print("\n" + "=" * 80)
print("MELHOR MODELO PELO CRITÉRIO: F1-SCORE > RECALL > ACURÁCIA")
print("=" * 80)
print(melhor_linha)


# %%
# 12. ANÁLISE DE OVERFITTING

df_overfitting = pd.DataFrame(list(overfitting_lista.values()))
df_overfitting = df_overfitting.sort_values(by="Diferença Acurácia", ascending=False)

print("\n" + "=" * 80)
print("ANÁLISE DE OVERFITTING - TREINO X TESTE")
print("=" * 80)
print(df_overfitting)

df_overfitting.to_csv(PASTA_REPORTS / "analise_overfitting.csv", index=False)

plt.figure(figsize=(14, 6))
plt.bar(df_overfitting["Modelo"], df_overfitting["Diferença Acurácia"])
plt.title("Diferença de Acurácia entre Treino e Teste")
plt.ylabel("Acurácia Treino - Acurácia Teste")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
salvar_figura("analise_overfitting_diferenca_acuracia.png")
plt.show()

# %%
# 13. VALIDAÇÃO CRUZADA DO MELHOR MODELO

# Se o melhor for uma variação de limiar ou rede neural, usa a árvore simples como modelo reproduzível do sklearn.
modelo_para_cv_nome = melhor_modelo_nome
if modelo_para_cv_nome not in modelos or "Limiar" in modelo_para_cv_nome:
    modelo_para_cv_nome = "Árvore de Decisão"

modelo_para_cv = modelos[modelo_para_cv_nome]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_resultado = cross_validate(
    modelo_para_cv,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    },
    n_jobs=-1
)

df_cv = pd.DataFrame({
    "Métrica": ["Acurácia", "Precisão", "Recall", "F1-score", "ROC AUC"],
    "Média CV": [
        cv_resultado["test_accuracy"].mean(),
        cv_resultado["test_precision"].mean(),
        cv_resultado["test_recall"].mean(),
        cv_resultado["test_f1"].mean(),
        cv_resultado["test_roc_auc"].mean(),
    ],
    "Desvio padrão": [
        cv_resultado["test_accuracy"].std(),
        cv_resultado["test_precision"].std(),
        cv_resultado["test_recall"].std(),
        cv_resultado["test_f1"].std(),
        cv_resultado["test_roc_auc"].std(),
    ]
})

print("\n" + "=" * 80)
print(f"VALIDAÇÃO CRUZADA DO MODELO: {modelo_para_cv_nome}")
print("=" * 80)
print(df_cv)

df_cv.to_csv(PASTA_REPORTS / "validacao_cruzada_melhor_modelo.csv", index=False)


# %%
# 14. GRÁFICOS COMPARATIVOS E CURVAS

resultados_plot = resultados.set_index("Modelo")

plt.figure(figsize=(20, 8))
resultados_plot[["Acurácia", "Precisão", "Recall/Sensibilidade", "F1-score", "ROC AUC"]].plot(kind="bar", figsize=(20, 8))
plt.title("Comparação de Métricas entre os Modelos")
plt.ylabel("Valor da Métrica")
plt.xlabel("Modelo")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métricas")
salvar_figura("comparacao_metricas_modelos.png")
plt.show()

# Matrizes de confusão dos principais modelos
for nome in ["Regressão Logística", "Árvore de Decisão", "Árvore de Decisão Otimizada", "Random Forest Otimizado", "Gradient Boosting Otimizado"]:
    if nome in predicoes_teste:
        plotar_matriz_confusao(nome, y_test, predicoes_teste[nome])

# Curva ROC e Precision-Recall do melhor modelo disponível
if melhor_modelo_nome in probabilidades_teste and probabilidades_teste[melhor_modelo_nome] is not None:
    y_prob_melhor = probabilidades_teste[melhor_modelo_nome]
else:
    y_prob_melhor = probabilidades_teste.get(modelo_para_cv_nome)

if y_prob_melhor is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob_melhor)
    auc = roc_auc_score(y_test, y_prob_melhor)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
    plt.title(f"Curva ROC - {melhor_modelo_nome}")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos / Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura("curva_roc_melhor_modelo.png")
    plt.show()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_melhor)
    pr_auc = average_precision_score(y_test, y_prob_melhor)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")
    plt.title(f"Curva Precision-Recall - {melhor_modelo_nome}")
    plt.xlabel("Recall / Sensibilidade")
    plt.ylabel("Precisão")
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura("curva_precision_recall_melhor_modelo.png")
    plt.show()
# %%
# 15. EXPLICABILIDADE: ÁRVORE E IMPORTÂNCIA DE VARIÁVEIS

# Árvore simples
arvore_treinada = modelos["Árvore de Decisão"].named_steps["model"]

plt.figure(figsize=(22, 12))
plot_tree(
    arvore_treinada,
    feature_names=X.columns,
    class_names=["Sem Diabetes", "Com Diabetes"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Árvore de Decisão - Predição de Diabetes")
salvar_figura("arvore_decisao_simples.png")
plt.show()

# Importância das variáveis nos modelos interpretáveis
for nome in ["Árvore de Decisão", "Árvore de Decisão Otimizada", "Random Forest Otimizado", "Gradient Boosting Otimizado", "Regressão Logística"]:
    if nome in modelos:
        importancias = obter_importancia_variaveis(modelos[nome], X.columns)

        if importancias is not None:
            print(f"\nImportância das variáveis - {nome}:")
            print(importancias)

            importancias.to_csv(PASTA_REPORTS / f"importancias_{nome.lower().replace(' ', '_')}.csv", index=False)

            plt.figure(figsize=(10, 6))
            plt.barh(importancias["Nome em português"].fillna(importancias["Variável"]), importancias["Importância"])
            plt.title(f"Importância das Variáveis - {nome}")
            plt.xlabel("Importância")
            plt.ylabel("Variável")
            plt.gca().invert_yaxis()
            salvar_figura(f"importancia_variaveis_{nome.lower().replace(' ', '_')}.png")
            plt.show()
# %%
# 16. SALVAR MELHOR MODELO E ARTEFATOS

if melhor_modelo_nome in modelos:
    modelo_para_salvar = modelos[melhor_modelo_nome]
else:
    modelo_para_salvar = modelos[modelo_para_cv_nome]

caminho_modelo = PASTA_MODELS / "melhor_modelo_diabetes.joblib"
joblib.dump(modelo_para_salvar, caminho_modelo)

print("\n" + "=" * 80)
print("MODELO SALVO")
print("=" * 80)
print(f"Modelo salvo em: {caminho_modelo}")

metadata = {
    "melhor_modelo_criterio_equilibrio": melhor_modelo_nome,
    "modelo_salvo": modelo_para_cv_nome if melhor_modelo_nome not in modelos else melhor_modelo_nome,
    "criterio": "F1-score > Recall/Sensibilidade > Acurácia",
    "observacao": "Em saúde, recall e falsos negativos também devem ser analisados antes da escolha final."
}

pd.Series(metadata).to_csv(PASTA_REPORTS / "metadata_modelo.csv")

# %%
# 17. TESTE COM UM NOVO PACIENTE + CAMADA DE AGENTE

novo_paciente = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [140],
    "BloodPressure": [80],
    "SkinThickness": [30],
    "Insulin": [120],
    "BMI": [32.5],
    "DiabetesPedigreeFunction": [0.6],
    "Age": [35]
})

print("\n" + "=" * 80)
print("TESTE COM NOVO PACIENTE")
print("=" * 80)

print("\nDados do novo paciente:")
print(novo_paciente)

modelo_agente = modelo_para_salvar
predicao = modelo_agente.predict(novo_paciente)[0]

if hasattr(modelo_agente, "predict_proba"):
    probabilidade_com_diabetes = modelo_agente.predict_proba(novo_paciente)[0][1]
else:
    probabilidade_com_diabetes = float(predicao)

interpretacao = gerar_recomendacoes_preventivas(novo_paciente, probabilidade_com_diabetes)

print("\nResultado do agente:")
print(f"Classe prevista: {predicao}")
print(f"Probabilidade com diabetes: {probabilidade_com_diabetes:.4f}")
print(f"Nível de risco preventivo: {interpretacao['Nivel de risco']}")
print(f"Mensagem: {interpretacao['Mensagem']}")

print("\nFatores de atenção:")
for fator in interpretacao["Fatores de atenção"]:
    print(f"- {fator}")

print("\nRecomendações preventivas:")
for recomendacao in interpretacao["Recomendações"]:
    print(f"- {recomendacao}")

# %%


#%%
# 18. REFLEXÃO ÉTICA E LIMITAÇÕES

texto_etica = """
REFLEXÃO ÉTICA E LIMITAÇÕES

Este sistema é um protótipo acadêmico de apoio à triagem e prevenção.
Ele não realiza diagnóstico médico e não deve ser usado para tomada de decisão clínica definitiva.

Principais limitações:
1. O dataset pode não representar a população brasileira.
2. A base possui registros limitados e variáveis com valores ausentes.
3. Alguns valores 0 foram tratados como ausentes por serem clinicamente improváveis.
4. O modelo pode apresentar vieses de acordo com a população representada no dataset.
5. Falsos negativos podem atrasar a busca por atendimento profissional.
6. Falsos positivos podem gerar preocupação desnecessária.
7. Dados de saúde exigem privacidade, consentimento e proteção adequada.

Uso responsável:
- O sistema deve apoiar a triagem, não substituir médicos ou profissionais de saúde.
- A predição deve ser explicada de forma transparente ao usuário.
- Recomenda-se validar o modelo com dados locais antes de qualquer uso real.
- O modelo não deve ser usado para negar atendimento, plano de saúde, benefícios ou oportunidades.
"""

print("\n" + "=" * 80)
print(texto_etica)
print("=" * 80)

with open(PASTA_REPORTS / "reflexao_etica_limitacoes.txt", "w", encoding="utf-8") as arquivo:
    arquivo.write(texto_etica)

print(f"\nRelatório ético salvo em: {PASTA_REPORTS / 'reflexao_etica_limitacoes.txt'}")

# %%
