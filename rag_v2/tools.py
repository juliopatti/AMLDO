import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
import pandas as pd

load_dotenv()

# Parâmetros do RAG
K = 12
SEARCH_TYPE = "mmr"  # ou "similarity"
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
df_art_0 = pd.read_csv('data/processed/v1_artigos_0.csv')

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

def _get_retriever(vector_db=vector_db, search_type: str = SEARCH_TYPE, k: int = K):
    """
    Cria o retriever do FAISS.
    """
    return vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "filter": {
                "artigo": { "$nin": ['artigo_0.txt'] }
            } 
        }
    )

# Função para obter o texto do artigo 0 correspondente
def get_art_0(law, title, chapter, df_art_0):
    cond = (
        (df_art_0['lei'] == law) &
        (df_art_0['titulo'] == title) &
        (df_art_0['capitulo'] == chapter)
    )
    resultados = df_art_0[cond]
    if resultados.shape[0] > 0:
        resultados = '\n'.join(resultados['texto'].tolist())
        return resultados

def get_pos_processed_context(df_resultados, df_art_0):
    df_cap = df_resultados[['lei', 'titulo', 'capitulo']].drop_duplicates().reset_index(drop=True)
    context = ''
    for law in df_cap['lei'].unique():
        context += f'\n\n<LEI {law}>\n'
        df_law = df_cap[df_cap['lei']==law].copy()
        for title in df_law['titulo'].unique():
            art_0 = get_art_0(law, title, 'CAPITULO_0', df_art_0)
            if art_0:
                context += f'{art_0}\n'
            if title != 'TITULO_0':
                context += f'<TITULO: {title}>\n'
            df_title = df_law[df_law['titulo']==title].copy()
            for chapter in df_title['capitulo'].unique():
                art_0 = get_art_0(law, title, chapter, df_art_0)
                if art_0:
                    context += f'{art_0}\n'
                if chapter != 'CAPITULO_0':
                    context += f'<CAPITULO: {chapter}>\n'
                cond_lei = (df_resultados['lei']==law)
                cond_titulo = (df_resultados['titulo']==title)
                cond_capitulo = (df_resultados['capitulo']==chapter)
                mask = cond_lei & cond_titulo & cond_capitulo
                df_chapter = df_resultados[mask].copy()
                for artigo in df_chapter['artigo'].unique():
                    df_article = df_chapter[df_chapter['artigo']==artigo].copy()
                    context += f'<ARTIGO: {artigo.replace(".txt", "")}>\n'
                    for _, row in df_article.iterrows():
                        context += f"{row['texto']}\n"
                    context += f'</ARTIGO: {artigo.replace(".txt", "")}>\n'
                if chapter != 'CAPITULO_0':
                    context += f'</CAPITULO: {chapter}>\n'
            if title != 'TITULO_0':
                context += f'</TITULO: {title}>\n'
        context += f'</LEI {law}>\n'
    return context.replace('\n[[SECTION:', '[[SECTION:')


def _rag_answer(question: str,
                search_type: str = SEARCH_TYPE,
                k: int = K) -> str:
    """
    Pipeline RAG "bruto": busca contexto no FAISS e gera resposta com base SOMENTE nesse contexto.
    """
    retriever = _get_retriever(search_type=search_type, k=k)
    contexto = retriever.invoke(question)

    # separação dos resultados do novo retriever
    linhas = []
    for doc in contexto:
        linhas.append({
            "texto": doc.page_content,
            **doc.metadata
        })

    # Ordenação dos resultados
    df_resultados = pd.DataFrame(linhas).sort_values(
        ['lei', 'titulo', 'capitulo', 'artigo', 'chunk_idx']
    ).reset_index(drop=True)

    # Pós-processamento do contexto
    context = get_pos_processed_context(df_resultados, df_art_0)

    prompt = ChatPromptTemplate.from_template(
        "Use APENAS o contexto para responder.\n\n"
        f"<Contexto>:\n{context}\n</Contexto>\n\n"
        "<Pergunta>:\n{question}</Pergunta>\n\n"
    )

    rag_chain = (
        {"question": RunnablePassthrough()}
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
    return _rag_answer(pergunta, search_type=SEARCH_TYPE, k=K)
