# Diabetes.IA — Protótipo de IA para Predição de Risco de Diabetes

## Sobre o projeto

Este projeto foi desenvolvido como parte de uma atividade acadêmica da disciplina de **Inteligência Artificial**, com foco em **IA aplicada à saúde** e **prevenção de doenças**.

A proposta principal foi construir um protótipo funcional em Python capaz de analisar dados clínicos e estimar o possível risco de diabetes, utilizando técnicas de **análise de dados**, **pré-processamento**, **modelagem preditiva**, **avaliação de métricas**, **explicabilidade** e **reflexão ética**.

O sistema não tem o objetivo de realizar diagnóstico médico. Ele foi desenvolvido como uma ferramenta acadêmica de apoio à triagem preventiva, demonstrando como modelos de Machine Learning podem ser aplicados em problemas reais da área da saúde, desde que utilizados com responsabilidade, transparência e senso crítico.

---

## Tema escolhido

O tema escolhido foi:

**Predição de risco de Diabetes Tipo 2**

A escolha desse tema se deu pela relevância do diabetes como problema de saúde pública e pela possibilidade de trabalhar com variáveis clínicas importantes, como:

* glicose;
* IMC;
* pressão arterial;
* idade;
* histórico familiar;
* insulina;
* espessura da pele;
* número de gestações.

A ideia central foi criar um sistema capaz de receber dados de um paciente e retornar uma estimativa de risco, acompanhada de explicações preventivas e recomendações de cuidado.

---

## Objetivo do projeto

O objetivo deste projeto é desenvolver um protótipo de Inteligência Artificial capaz de:

* carregar e analisar um dataset clínico;
* realizar análise exploratória dos dados;
* tratar valores ausentes e inconsistentes;
* treinar diferentes algoritmos de Machine Learning;
* comparar os modelos usando métricas adequadas;
* avaliar overfitting;
* aplicar validação cruzada;
* gerar predição individual;
* apresentar recomendações preventivas;
* oferecer uma interface interativa com Streamlit;
* discutir limitações, riscos, vieses e uso responsável da IA em saúde.

---

## Pergunta de pesquisa

A pergunta que norteou o desenvolvimento do projeto foi:

> A partir de variáveis clínicas simples, é possível construir um modelo de Inteligência Artificial capaz de estimar o risco de diabetes de forma útil para apoio à triagem preventiva?

---

## Tecnologias utilizadas

O projeto foi desenvolvido em Python, utilizando bibliotecas voltadas para análise de dados, Machine Learning, visualização e interface interativa.

### Linguagem

* Python 3.12

### Bibliotecas principais

* pandas;
* numpy;
* matplotlib;
* scikit-learn;
* streamlit;
* joblib;
* tensorflow/keras, usado no treinamento do agente, quando disponível.

---

## Estrutura do projeto

A estrutura principal do projeto ficou organizada da seguinte forma:

```text
Diabetes.IA/
├── data/
│   └── diabetes.csv
│
├── docs/
│   └── arquivos e documentos de apoio
│
├── models/
│   └── melhor_modelo_diabetes.joblib
│
├── notebooks/
│   └── experimentações em notebook, se necessário
│
├── reports/
│   ├── figures/
│   │   └── gráficos gerados pelo treinamento
│   │
│   ├── metricas_modelos.csv
│   ├── analise_overfitting.csv
│   ├── validacao_cruzada_melhor_modelo.csv
│   ├── zeros_invalidos.csv
│   ├── medias_por_classe.csv
│   └── reflexao_etica_limitacoes.txt
│
├── src/
│   ├── app.py
│   └── train_model.py
│
├── treino/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup_env.ps1
```

---

## Arquivos principais

### `src/train_model.py`

Este é o arquivo responsável por executar o pipeline principal de análise e treinamento.

Ele realiza:

* carregamento do dataset;
* análise exploratória dos dados;
* identificação de zeros inválidos;
* pré-processamento;
* separação entre treino e teste;
* treinamento de múltiplos modelos;
* comparação entre algoritmos;
* análise de overfitting;
* validação cruzada;
* geração de gráficos;
* salvamento de métricas;
* salvamento do melhor modelo;
* teste com novo paciente;
* geração de reflexão ética.

---

### `src/app.py`

Este arquivo executa a interface interativa em Streamlit.

A interface permite:

* preencher dados de um paciente;
* escolher o modelo de IA;
* realizar predição individual;
* visualizar probabilidade estimada;
* consultar fatores de atenção;
* receber recomendações preventivas;
* visualizar métricas dos modelos;
* analisar matriz de confusão;
* consultar overfitting;
* visualizar validação cruzada;
* acessar gráficos de EDA;
* ler a reflexão ética e limitações do sistema.

---

### `setup_env.ps1`

Script em PowerShell criado para facilitar a configuração do ambiente no Windows.

Ele automatiza:

* criação da `.venv`;
* ativação do ambiente virtual;
* atualização do `pip`;
* instalação das dependências;
* criação das pastas principais do projeto.

---

## Dataset utilizado

O dataset utilizado contém registros clínicos relacionados ao risco de diabetes.

As colunas utilizadas são:

| Coluna                     | Significado                                 |
| -------------------------- | ------------------------------------------- |
| `Pregnancies`              | Número de gestações                         |
| `Glucose`                  | Glicose                                     |
| `BloodPressure`            | Pressão arterial                            |
| `SkinThickness`            | Espessura da pele                           |
| `Insulin`                  | Insulina                                    |
| `BMI`                      | Índice de massa corporal                    |
| `DiabetesPedigreeFunction` | Indicador associado ao histórico familiar   |
| `Age`                      | Idade                                       |
| `Outcome`                  | Classe alvo: 0 sem diabetes, 1 com diabetes |

---

## Variável alvo

A variável alvo do projeto é:

```text
Outcome
```

Ela indica se o registro está associado ou não à presença de diabetes:

```text
0 = Sem Diabetes
1 = Com Diabetes
```

---

## Tratamento de dados

Durante a análise, identifiquei que algumas variáveis clínicas apresentavam valor `0`, o que em determinados casos não faz sentido do ponto de vista clínico.

As colunas tratadas foram:

* `Glucose`;
* `BloodPressure`;
* `SkinThickness`;
* `Insulin`;
* `BMI`.

Esses valores foram substituídos por `NaN` e posteriormente tratados com imputação pela mediana dentro dos pipelines de Machine Learning.

Esse tratamento é importante porque evita que o modelo interprete valores clinicamente improváveis como dados reais.

---

## Análise Exploratória de Dados

A etapa de EDA foi desenvolvida para entender melhor a distribuição dos dados antes da modelagem.

Foram gerados:

* distribuição da variável alvo;
* análise de zeros inválidos;
* histogramas das variáveis principais;
* boxplots por classe;
* matriz de correlação;
* comparação de médias entre pacientes classificados como sem diabetes e com diabetes.

Essa etapa foi importante para identificar padrões, possíveis limitações e diferenças entre os grupos.

---

## Modelos utilizados

Foram implementados e comparados diferentes algoritmos de classificação:

* Regressão Logística;
* Árvore de Decisão;
* Árvore de Decisão Otimizada com GridSearchCV;
* Random Forest Otimizado;
* KNN;
* SVC;
* Gradient Boosting Otimizado;
* Rede Neural com TensorFlow/Keras, quando disponível no ambiente.

A comparação entre diferentes modelos foi importante para avaliar qual abordagem apresentava melhor equilíbrio entre desempenho, interpretabilidade e utilidade para triagem preventiva.

---

## Otimização de modelo

A Árvore de Decisão foi otimizada com `GridSearchCV`.

Os hiperparâmetros avaliados incluíram:

* profundidade máxima da árvore;
* número mínimo de amostras para divisão;
* número mínimo de amostras por folha;
* critério de divisão;
* balanceamento de classes.

O objetivo da otimização foi melhorar especialmente o **recall**, pois em um problema de triagem em saúde é importante reduzir a chance de falsos negativos.

---

## Métricas de avaliação

Foram utilizadas métricas adequadas para classificação binária e para contexto de saúde.

As principais métricas foram:

* acurácia;
* precisão;
* recall/sensibilidade;
* especificidade;
* F1-score;
* ROC AUC;
* PR AUC;
* falsos negativos;
* taxa de falsos negativos;
* falsos positivos;
* taxa de falsos positivos.

---

## Por que o recall é importante?

Neste projeto, o recall/sensibilidade foi tratado como uma métrica muito importante.

Em um sistema de triagem preventiva, um **falso negativo** ocorre quando o modelo classifica uma pessoa como baixo risco, mesmo quando ela apresenta sinais associados ao risco de diabetes.

Esse tipo de erro pode ser mais preocupante do que um falso positivo, pois pode atrasar a busca por avaliação profissional.

Por isso, além da acurácia, analisei também:

* recall;
* falsos negativos;
* taxa de falsos negativos;
* matriz de confusão;
* overfitting;
* validação cruzada.

---

## Análise de overfitting

O projeto também realiza análise de overfitting comparando o desempenho dos modelos no conjunto de treino e no conjunto de teste.

Foram comparadas:

* acurácia no treino;
* acurácia no teste;
* diferença de acurácia;
* recall no treino;
* recall no teste;
* diferença de recall.

Essa análise ajuda a identificar se algum modelo está apenas “decorando” os dados de treino, sem generalizar bem para dados novos.

---

## Validação cruzada

Também implementei validação cruzada com `StratifiedKFold`.

A validação cruzada foi usada para avaliar a estabilidade do melhor modelo em diferentes divisões dos dados.

As métricas avaliadas na validação cruzada foram:

* acurácia;
* precisão;
* recall;
* F1-score;
* ROC AUC.

A validação cruzada torna a avaliação mais confiável do que depender apenas de uma única divisão entre treino e teste.

---

## Explicabilidade

A explicabilidade foi incluída para tornar o sistema mais transparente.

O projeto apresenta:

* importância das variáveis nos modelos interpretáveis;
* matriz de confusão;
* métricas de erro;
* fatores de atenção identificados no paciente;
* recomendações preventivas geradas com base em regras simples.

A proposta foi evitar que o sistema seja apenas uma “caixa preta”, principalmente por se tratar de uma aplicação em saúde.

---

## Camada de agente

Além da classificação feita pelos modelos, implementei uma camada de agente que interpreta o resultado e gera uma resposta mais acessível.

Essa camada retorna:

* nível de risco preventivo;
* mensagem explicativa;
* fatores de atenção;
* recomendações preventivas.

Os níveis de risco foram definidos com base na probabilidade estimada pelo modelo:

```text
Baixo: probabilidade menor que 40%
Moderado: probabilidade entre 40% e 70%
Alto: probabilidade igual ou maior que 70%
```

Esses limites foram usados apenas para fins acadêmicos e demonstrativos.

---

## Interface Streamlit

A interface foi construída com Streamlit para facilitar a demonstração do projeto.

Ela possui as seguintes abas:

### Predição

Permite inserir os dados de um paciente e gerar uma predição individual.

A interface mostra:

* classe prevista;
* probabilidade estimada;
* nível de risco;
* fatores de atenção;
* recomendações preventivas;
* importância das variáveis, quando disponível.

---

### Modelos e Métricas

Apresenta:

* tabela comparativa dos modelos;
* melhor modelo pelo critério F1-score > Recall > Acurácia;
* matriz de confusão do modelo selecionado;
* falsos positivos;
* falsos negativos;
* gráfico comparativo de métricas;
* análise de overfitting;
* validação cruzada;
* curva ROC;
* curva Precision-Recall.

---

### Dataset

Apresenta:

* primeiras linhas do dataset;
* resumo estatístico;
* distribuição das classes;
* quantidade de registros;
* análise de zeros inválidos.

---

### EDA

Apresenta:

* matriz de correlação;
* comparação de médias por classe;
* histogramas;
* boxplots por classe;
* leitura crítica para o relatório.

---

### Ética e Limitações

Apresenta:

* limitações do dataset;
* riscos do modelo;
* riscos de falsos positivos e falsos negativos;
* recomendações de mitigação;
* princípios de uso responsável;
* conclusão ética.

---

## Como executar o projeto

### 1. Clonar o repositório

```powershell
git clone https://github.com/VictorHFMartins/Diabetes.IA.git
cd Diabetes.IA
```

---

### 2. Criar e configurar o ambiente virtual

Caso esteja no Windows, é possível usar o script:

```powershell
.\setup_env.ps1
```

Se preferir fazer manualmente:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Executar o treinamento

```powershell
python src\train_model.py
```

Esse comando gera:

* modelos;
* relatórios;
* gráficos;
* métricas;
* arquivos de treino e teste;
* reflexão ética;
* artefatos em `models/` e `reports/`.

---

### 4. Executar a interface

```powershell
streamlit run src\app.py
```

Após executar o comando, o Streamlit abrirá a aplicação no navegador.

---

## Ambiente recomendado

O projeto foi desenvolvido utilizando:

```text
Python 3.12
Windows
VS Code
Streamlit
scikit-learn
pandas
numpy
matplotlib
```

Caso utilize Jupyter Notebook ou o modo interativo do VS Code, é importante selecionar o kernel correto da `.venv`.

---

## Observação sobre TensorFlow

O TensorFlow foi usado de forma opcional no treinamento.

Caso o ambiente não tenha TensorFlow instalado, o projeto ainda pode funcionar com os modelos clássicos de Machine Learning.

Para instalar TensorFlow:

```powershell
pip install tensorflow
```

Para testar:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Principais arquivos gerados

Ao executar `train_model.py` ou o app, alguns arquivos podem ser gerados automaticamente:

```text
reports/metricas_modelos.csv
reports/analise_overfitting.csv
reports/validacao_cruzada_melhor_modelo.csv
reports/zeros_invalidos.csv
reports/medias_por_classe.csv
reports/reflexao_etica_limitacoes.txt
models/melhor_modelo_diabetes.joblib
```

Esses arquivos ajudam a documentar os resultados do projeto e reforçam a reprodutibilidade.

---

## Versionamento com Git

O projeto foi versionado com commits incrementais para demonstrar a evolução do desenvolvimento.

A sequência de desenvolvimento foi organizada em etapas como:

```text
chore: reorganiza pastas e atualiza setup do ambiente
feat: adiciona base do pipeline de analise e treinamento
feat: adiciona analise exploratoria visual dos dados
feat: implementa preprocessamento inicial dos dados
feat: separa e salva dados de treino e teste
feat: define e treina modelos de classificacao
feat: adiciona rede neural opcional com tensorflow
feat: avalia modelos com metricas avancadas
feat: gera tabela final de resultados dos modelos
feat: adiciona analise de overfitting dos modelos
feat: adiciona validacao cruzada do melhor modelo
feat: adiciona graficos comparativos e curvas de avaliacao
feat: adiciona explicabilidade dos modelos
feat: salva melhor modelo e metadados
feat: adiciona camada de agente para recomendacoes preventivas
docs: adiciona reflexao etica e limitacoes
feat: cria estrutura inicial da interface streamlit
feat: adiciona carregamento e preparacao dos dados no app
feat: treina modelos para uso na interface
feat: adiciona predicao individual no streamlit
feat: adiciona metricas e comparacao de modelos na interface
feat: adiciona analise exploratoria na interface
feat: adiciona explicabilidade e recomendacoes preventivas
docs: adiciona etica e limitacoes na interface
feat: adiciona overfitting e validacao cruzada na interface
```

Essa organização ajuda a mostrar que o projeto foi construído de forma gradual, com cada etapa tendo uma função clara.

---

## Reflexão ética

Como este projeto envolve IA aplicada à saúde, a parte ética foi tratada como essencial.

O sistema não deve ser utilizado para:

* diagnóstico definitivo;
* prescrição médica;
* substituição de consulta;
* negação de atendimento;
* definição de plano de saúde;
* tomada de decisão administrativa sobre pessoas.

A proposta correta de uso é:

* apoio educacional;
* triagem preventiva;
* conscientização;
* demonstração acadêmica;
* apoio à interpretação inicial de risco.

---

## Limitações

O projeto possui limitações importantes:

* o dataset é limitado;
* a base pode não representar a população brasileira;
* algumas variáveis possuem valores ausentes;
* alguns registros usam `0` como ausência de informação;
* o modelo pode reproduzir vieses presentes nos dados;
* os resultados podem variar em outras populações;
* o sistema não considera todos os fatores clínicos, sociais e comportamentais associados ao diabetes.

---

## Recomendações futuras

Como melhorias futuras, eu poderia implementar:

* uso de bases brasileiras, como DATASUS ou OpenDataSUS;
* análise de fairness por grupos demográficos;
* uso de SHAP Values para explicabilidade mais avançada;
* deploy da aplicação em ambiente online;
* autenticação e controle de privacidade;
* histórico de predições;
* dashboard administrativo;
* comparação com modelos calibrados;
* otimização mais profunda de hiperparâmetros;
* testes automatizados;
* documentação técnica mais detalhada.

---

## Aviso importante

Este projeto é apenas um protótipo acadêmico.

A predição gerada pelo sistema é probabilística e não deve ser interpretada como diagnóstico.

Qualquer preocupação relacionada à saúde deve ser avaliada por profissionais qualificados.

---

## Autor

Desenvolvido por **Victor Hugo Faria Martins**.

Projeto acadêmico de Inteligência Artificial aplicada à saúde.

---

## Licença

Este projeto está disponível para fins acadêmicos e educacionais.

Caso seja utilizado como base para outros estudos ou projetos, recomenda-se manter os créditos e destacar que se trata de um protótipo experimental.
