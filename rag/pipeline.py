"""
FINAL PRODUCTION RAG PIPELINE (Simple + Stable)
Dense Retrieval + Cohere Reranker + format_docs
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

import cohere

load_dotenv()

# ----------------------------------------
# PATHS
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "db"
PROMPTS_DIR = BASE_DIR / "prompts"

# ----------------------------------------
# COHERE RERANK
# ----------------------------------------
_cohere_client = None

def get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Missing COHERE_API_KEY")
        _cohere_client = cohere.Client(api_key)
    return _cohere_client


def cohere_rerank(query, docs, top_k=5):
    if not docs:
        return docs

    try:
        response = get_cohere_client().rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_k
        )
        return [docs[r.index] for r in response.results]
    except Exception as e:
        print("⚠ Cohere rerank failed:", e)
        return docs


# ----------------------------------------
# VECTORSTORE + RETRIEVER
# ----------------------------------------
_vectorstore = None
_dense_retriever = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("🔍 Loading Chroma DB...")
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    return _vectorstore


vectorstore = get_vectorstore()


def get_dense_retriever():
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 25}
        )
    return _dense_retriever


# ----------------------------------------
# FORMAT DOCS
# ----------------------------------------
def format_docs(docs):
    output = []
    for d in docs:
        meta = f"(Source: {Path(d.metadata.get('source', 'N/A')).name})"
        output.append(meta + "\n" + d.page_content)
    return "\n\n".join(output)


# ----------------------------------------
# PROMPTS
# ----------------------------------------
def load_prompt(name):
    return (PROMPTS_DIR / name).read_text()

direct_prompt = ChatPromptTemplate.from_template(
    load_prompt("rag_prompts.txt")
)

socratic_prompt = ChatPromptTemplate.from_template(
    load_prompt("socratic_prompt.txt")
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
parser = StrOutputParser()


# ----------------------------------------
# RETRIEVAL FUNCTIONS
# ----------------------------------------
def retrieve_for_rag(question):
    dense_docs = get_dense_retriever().invoke(question)
    reranked_docs = cohere_rerank(question, dense_docs)
    context = format_docs(reranked_docs)
    return {"question": question, "context": context}


def retrieve_for_socratic(question):
    dense_docs = get_dense_retriever().invoke(question)
    reranked_docs = cohere_rerank(question, dense_docs)
    context = format_docs(reranked_docs)
    return {"question": question, "context": context}


# ----------------------------------------
# PIPELINES
# ----------------------------------------
rag_pipeline = (
    RunnableLambda(retrieve_for_rag)
    | direct_prompt
    | llm
    | parser
)

socratic_pipeline = (
    RunnableLambda(retrieve_for_socratic)
    | socratic_prompt
    | llm
    | parser
)


# ----------------------------------------
# DEBUG
# ----------------------------------------
if __name__ == "__main__":
    print("Vectorstore size:", get_vectorstore()._collection.count())
    print(rag_pipeline.invoke("What is sociology?"))