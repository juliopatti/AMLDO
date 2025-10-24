# AMLDO — Setup do Ambiente (mínimo)

Passos para **criar e configurar o ambiente virtual** exatamente como usado no projeto.

## Requisitos
- Python **3.11**
- `pip` atualizado

## Passo a passo

```bash
# 1) Criar e ativar o ambiente virtual
python3.11 -m venv venv
source venv/bin/activate

# 2) Atualizar o pip
pip install --upgrade pip

# 3) Registrar kernel no Jupyter
pip install ipykernel==6.30.1
python -m ipykernel install --user --name=venv --display-name "amldo_kernel"

# 4) Instalar bibliotecas do projeto
pip install pandas==2.3.2
pip install dotenv==0.9.9
pip install google-cloud-aiplatform==1.122.0
pip install sentencepiece==0.2.1
pip install pymupdf==1.26.5

pip install langchain==1.0.2
pip install langchain-core==1.0.1
pip install langchain-text-splitters==1.0.0
pip install langchain_community==0.4
pip install langchain_huggingface==1.0.0
pip install sentence-transformers==5.1.2
pip install faiss-cpu==1.12.0
pip install langchain-google-genai==3.0.0
pip install google-ai-generativelanguage==0.9.0
```

### Observações rápidas
- Use sempre o **mesmo venv** no terminal e no Jupyter (kernel **amldo_kernel**).
- Caso instale pacotes com o notebook aberto, **reinicie o kernel**.
- Se precisar de variáveis de ambiente (ex.: `GOOGLE_API_KEY`), crie um `.env` na raiz e carregue com `python-dotenv`.
