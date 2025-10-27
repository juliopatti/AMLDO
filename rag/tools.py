# /home/julio/Documentos/agents_pos/amldo/AMLDO/rag/tools.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# -------------------------------------------------
# Inicializações globais (carregam uma vez só)
# -------------------------------------------------

# LLM usado dentro do RAG
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Modelo de embedding
modelo_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# FAISS index já salvo em disco
vector_db = FAISS.load_local(
    "data/vector_db/v1_faiss_vector_db",
    embeddings=modelo_embedding,
    allow_dangerous_deserialization=True,
)

def _get_retriever(vector_db=vector_db, search_type: str = "mmr", k: int = 12):
    """
    Cria o retriever do FAISS.
    """
    return vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

def _rag_answer(question: str,
                search_type: str = "mmr",
                k: int = 12) -> str:
    """
    Pipeline RAG "bruto": busca contexto no FAISS e gera resposta com base SOMENTE nesse contexto.
    """
    retriever = _get_retriever(search_type=search_type, k=k)

    prompt = ChatPromptTemplate.from_template(
        "Use APENAS o contexto para responder.\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta:\n{question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    resposta = rag_chain.invoke(question)
    return resposta


def consultar_base_rag(pergunta: str) -> str:
    """
    Retorna uma resposta baseada EXCLUSIVAMENTE nos documentos internos indexados
    na base vetorial FAISS (com embeddings multilíngues).

    Use esta função quando a dúvida envolver:
    - licitação, dispensa de licitação por valor, contratos, governança, compliance,
      regulatório, normas internas, leis, decretos, portarias etc.;
    - perguntas do tipo "com base nos documentos", "na nossa base", "na legislação interna".

    Args:
        pergunta (str): Pergunta completa do usuário em linguagem natural.
                        Passe o texto inteiro, sem resumir.

    Returns:
        str: Texto de resposta gerado a partir do contexto recuperado.
             Se não existir contexto relevante, a resposta pode indicar
             que não encontrou informação na base.
    """
    return _rag_answer(pergunta, search_type="mmr", k=12)
