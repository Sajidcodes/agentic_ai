"""
IMPROVED RAG PIPELINE - Better Retrieval & Response Quality
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough
)
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


def cohere_rerank(query, docs, top_k=8):  # CHANGED: 5 → 8 for more context
    """Rerank docs using Cohere with better defaults."""
    if not docs:
        return docs

    try:
        response = get_cohere_client().rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_k,
            return_documents=False  # We already have the docs
        )
        # Return docs with relevance scores attached
        reranked = []
        for r in response.results:
            doc = docs[r.index]
            doc.metadata['relevance_score'] = r.relevance_score
            reranked.append(doc)
        return reranked
    except Exception as e:
        print("⚠ Cohere rerank failed:", e)
        return docs[:top_k]  # fallback with limit


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
            search_type="similarity",  # CHANGED: explicit similarity search
            search_kwargs={
                "k": 20,  # CHANGED: 25 → 20 (less noise)
                "fetch_k": 50  # ADDED: fetch more for MMR diversity
            }
        )
    return _dense_retriever


# ----------------------------------------
# IMPROVED FORMAT DOCS
# ----------------------------------------
def format_docs(docs):
    """Format docs with better structure and metadata."""
    if not docs:
        return "No relevant information found in the knowledge base."
    
    output = []
    for i, d in enumerate(docs, 1):
        # Extract metadata
        source = Path(d.metadata.get('source', 'Unknown')).name
        page = d.metadata.get('page', 'N/A')
        relevance = d.metadata.get('relevance_score', 0)
        
        # Format chunk with clear structure
        chunk_header = f"[Chunk {i}] Source: {source}"
        if page != 'N/A':
            chunk_header += f" | Page: {page}"
        if relevance > 0:
            chunk_header += f" | Relevance: {relevance:.2f}"
        
        formatted_chunk = f"{chunk_header}\n{d.page_content.strip()}"
        output.append(formatted_chunk)
    
    return "\n\n---\n\n".join(output)


# ----------------------------------------
# IMPROVED PROMPTS
# ----------------------------------------
IMPROVED_RAG_PROMPT = """You are an expert AI literacy tutor with access to comprehensive educational materials. Your role is to provide detailed, accurate, and pedagogically sound answers.

## CONTEXT FROM KNOWLEDGE BASE:
{context}

## STUDENT'S QUESTION:
{question}

## INSTRUCTIONS:
1. **Use ONLY the information from the context above** - never invent facts
2. **Be comprehensive and educational** - don't just state facts, explain concepts with examples
3. **Cite your sources** using inline citations like [Source: filename.pdf]
4. **If the context doesn't contain enough information**, say: "I don't have enough information in my knowledge base to fully answer this question. Here's what I can tell you based on available materials: [partial answer if applicable]"
5. **Structure your answer clearly** with:
   - A direct answer to the question
   - Detailed explanation with examples from the context
   - Key takeaways or implications (if relevant)
6. **Make it engaging** - imagine you're teaching a curious student who wants to truly understand

## YOUR RESPONSE:
"""

IMPROVED_SOCRATIC_PROMPT = """You are a Socratic tutor helping students learn about AI literacy through guided questioning and thoughtful discussion.

## CONTEXT FROM KNOWLEDGE BASE:
{context}

## CONVERSATION SO FAR:
{question}

## YOUR TEACHING APPROACH:
1. **First, guide their thinking** with 1-2 probing questions that help them reason through the concept
2. **Then, provide a clear explanation** using information from the context
3. **Use examples** from the context materials to illustrate points
4. **Encourage deeper thinking** by connecting concepts or highlighting implications
5. **Be supportive and encouraging** - learning is a journey

## IMPORTANT RULES:
- Use ONLY information from the context provided
- Don't make up facts or examples not in the materials
- If context is insufficient, acknowledge limitations while still being helpful
- Cite sources when providing factual information: [Source: filename.pdf]

## YOUR RESPONSE:
"""


# ----------------------------------------
# PROMPTS
# ----------------------------------------
direct_prompt = ChatPromptTemplate.from_template(IMPROVED_RAG_PROMPT)
socratic_prompt = ChatPromptTemplate.from_template(IMPROVED_SOCRATIC_PROMPT)

# CHANGED: GPT-4o-mini with better temperature for educational content
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Cheaper but better than gpt-3.5
    temperature=0.3,  # Slightly more creative for educational content
    max_tokens=1000  # Ensure complete responses
)
parser = StrOutputParser()


# ----------------------------------------
# IMPROVED RAG PIPELINE
# ----------------------------------------
def retrieve_and_rerank(query):
    """Combined retrieval + reranking step."""
    # Get initial candidates
    dense_docs = get_dense_retriever().invoke(query)
    
    # Rerank for relevance
    reranked_docs = cohere_rerank(query, dense_docs, top_k=8)
    
    # Debug logging (remove in production)
    print(f"\n🔍 Retrieved {len(dense_docs)} docs, reranked to top {len(reranked_docs)}")
    if reranked_docs:
        print(f"📊 Top relevance score: {reranked_docs[0].metadata.get('relevance_score', 0):.3f}")
    
    return reranked_docs


rag_pipeline = (
    RunnableMap({
        "question": RunnablePassthrough(),
        "context_docs": RunnableLambda(retrieve_and_rerank)
    })
    |
    RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: format_docs(x["context_docs"]),
    })
    | direct_prompt
    | llm
    | parser
)


# ----------------------------------------
# IMPROVED SOCRATIC PIPELINE
# ----------------------------------------
socratic_pipeline = (
    RunnableMap({
        "question": RunnablePassthrough(),
        "context": RunnableLambda(
            lambda q: format_docs(retrieve_and_rerank(q))
        ),
    })
    | socratic_prompt
    | llm
    | parser
)


# ----------------------------------------
# DEBUG & TEST
# ----------------------------------------
if __name__ == "__main__":
    print("Vectorstore size:", get_vectorstore()._collection.count())
    
    # Test query
    test_query = "What is machine learning and how does it work?"
    print(f"\nTesting with: {test_query}\n")
    print(rag_pipeline.invoke(test_query))