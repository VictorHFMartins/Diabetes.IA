# ==========================================================
# setup_env.ps1
# Script para configurar o ambiente do projeto Diabetes.IA
# Python recomendado: 3.12
# ==========================================================

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " CONFIGURACAO DO AMBIENTE - Diabetes.IA" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------------------
# 1. Ajustar politica de execucao apenas nesta sessao
# ----------------------------------------------------------

Write-Host "Ajustando politica de execucao para esta sessao..." -ForegroundColor Green
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# ----------------------------------------------------------
# 2. Verificar se esta na raiz do projeto
# ----------------------------------------------------------

Write-Host ""
Write-Host "Verificando estrutura do projeto..." -ForegroundColor Green

if (-Not (Test-Path ".git")) {
    Write-Host "ATENCAO: Nao encontrei a pasta .git." -ForegroundColor Yellow
    Write-Host "Confirme se voce esta dentro da pasta clonada do repositorio Diabetes.IA." -ForegroundColor Yellow
    Write-Host "Exemplo:" -ForegroundColor Yellow
    Write-Host "cd C:\Users\navic\Desktop\Diabetes.IA" -ForegroundColor Yellow
    Write-Host ""
}

# ----------------------------------------------------------
# 3. Criar pastas do projeto
# ----------------------------------------------------------

Write-Host ""
Write-Host "Criando pastas necessarias..." -ForegroundColor Green

$pastas = @(
    "data",
    "models",
    "reports",
    "reports\figures",
    "treino",
    "notebooks",
    "docs",
    "src"
)

foreach ($pasta in $pastas) {
    if (-Not (Test-Path $pasta)) {
        New-Item -ItemType Directory -Path $pasta | Out-Null
        Write-Host "Criada: $pasta" -ForegroundColor Green
    }
    else {
        Write-Host "Ja existe: $pasta" -ForegroundColor Yellow
    }
}

# ----------------------------------------------------------
# 4. Verificar Python 3.12
# ----------------------------------------------------------

Write-Host ""
Write-Host "Verificando Python 3.12..." -ForegroundColor Green

try {
    py -3.12 --version
}
catch {
    Write-Host ""
    Write-Host "ERRO: Python 3.12 nao foi encontrado pelo comando py -3.12." -ForegroundColor Red
    Write-Host "Verifique se o Python 3.12 esta instalado e se o Python Launcher esta disponivel." -ForegroundColor Red
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERRO: Python 3.12 nao foi encontrado." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 5. Criar ambiente virtual
# ----------------------------------------------------------

Write-Host ""
Write-Host "Criando ambiente virtual .venv..." -ForegroundColor Green

if (Test-Path ".venv") {
    Write-Host "A pasta .venv ja existe. Pulando criacao do ambiente virtual." -ForegroundColor Yellow
}
else {
    py -3.12 -m venv .venv

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO: Falha ao criar o ambiente virtual." -ForegroundColor Red
        exit 1
    }
}

# ----------------------------------------------------------
# 6. Ativar ambiente virtual
# ----------------------------------------------------------

Write-Host ""
Write-Host "Ativando ambiente virtual..." -ForegroundColor Green

$activatePath = ".\.venv\Scripts\Activate.ps1"

if (Test-Path $activatePath) {
    & $activatePath
}
else {
    Write-Host "ERRO: Nao encontrei .venv\Scripts\Activate.ps1." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 7. Atualizar pip
# ----------------------------------------------------------

Write-Host ""
Write-Host "Atualizando pip..." -ForegroundColor Green
python -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: Falha ao atualizar o pip." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 8. Criar requirements.txt caso nao exista
# ----------------------------------------------------------

if (-Not (Test-Path "requirements.txt")) {
    Write-Host ""
    Write-Host "Criando requirements.txt padrao..." -ForegroundColor Green

@"
pandas
numpy
matplotlib
scikit-learn
streamlit
joblib
openpyxl
"@ | Set-Content -Path "requirements.txt" -Encoding UTF8
}

# ----------------------------------------------------------
# 9. Instalar dependencias
# ----------------------------------------------------------

Write-Host ""
Write-Host "Instalando dependencias do requirements.txt..." -ForegroundColor Green

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: Falha ao instalar dependencias." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 10. Testar bibliotecas principais
# ----------------------------------------------------------

Write-Host ""
Write-Host "Testando bibliotecas principais..." -ForegroundColor Green

python -c "import pandas, numpy, matplotlib, sklearn, streamlit, joblib; print('Bibliotecas principais OK')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: Alguma biblioteca principal apresentou erro na importacao." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 11. Verificar dataset
# ----------------------------------------------------------

Write-Host ""
Write-Host "Verificando dataset..." -ForegroundColor Green

if (Test-Path "data\diabetes.csv") {
    Write-Host "Dataset encontrado em data\diabetes.csv" -ForegroundColor Green
}
else {
    Write-Host "ATENCAO: Dataset ainda nao encontrado." -ForegroundColor Yellow
    Write-Host "Quando for adicionar o dataset, coloque em:" -ForegroundColor Yellow
    Write-Host "data\diabetes.csv" -ForegroundColor Yellow
}

# ----------------------------------------------------------
# 12. Finalizacao
# ----------------------------------------------------------

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " AMBIENTE CONFIGURADO COM SUCESSO!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para ativar o ambiente novamente em outro dia:" -ForegroundColor White
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para treinar o modelo:" -ForegroundColor White
Write-Host "python src\train.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para abrir a interface Streamlit:" -ForegroundColor White
Write-Host "streamlit run src\App.py" -ForegroundColor Yellow
Write-Host ""