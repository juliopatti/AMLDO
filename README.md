# AMLDO — Setup do Ambiente (mínimo)

Passos para **criar e configurar o ambiente virtual** exatamente como usado no projeto.

## Requisitos
- Python **3.11**
- `pip` atualizado

## Passo a passo

```bash
# Criar e ativar o ambiente virtual
python3.11 -m venv venv
source venv/bin/activate

# Atualizar o pip
pip install --upgrade pip

# Instalar bibliotecas do projeto
pip install -r requirements.txt

# Registrar kernel no Jupyter
python -m ipykernel install --user --name=venv --display-name "amldo_kernel"
```

### Observações
- Rodar versão demo no adk web: com o venv ativado, execute
```bash
adk web
```
e selecione o agent 'RAG'
- Se precisar de variáveis de ambiente (ex.: `GOOGLE_API_KEY`), crie um `.env` na raiz e carregue com `python-dotenv`.
