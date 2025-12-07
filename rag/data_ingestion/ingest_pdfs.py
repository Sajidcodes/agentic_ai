import os
import time
import shutil
from pathlib import Path
from typing import List
from collections import defaultdict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ================= CONFIG ================= #

PDF_DIR = Path("/data/pdfs")
CHROMA_DIR = Path("/data/chroma")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 32

# ========================================= #

def find_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted(pdf_dir.glob("*.pdf"))

def load_and_split(pdf_path: Path, splitter):
    print(f"\n📘 {pdf_path.name}")
    docs = PyPDFLoader(str(pdf_path)).load()
    chunks = splitter.split_documents(docs)

    for d in chunks:
        d.metadata = {
            "source": pdf_path.name,
            "domain": pdf_path.stem.split("_")[0].lower()
        }

    print(f"   pages={len(docs)} chunks={len(chunks)}")
    return chunks

def build_docs():
    if not PDF_DIR.exists():
        raise FileNotFoundError("/data/pdfs missing")

    pdfs = find_pdfs(PDF_DIR)
    if not pdfs:
        raise RuntimeError("No PDFs found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = []
    for pdf in pdfs:
        docs.extend(load_and_split(pdf, splitter))

    print(f"\n✅ Total chunks: {len(docs)}")
    return docs

def build_chroma(docs):
    if CHROMA_DIR.exists():
        print("⚠️ Removing old DB")
        shutil.rmtree(CHROMA_DIR)

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]

    print("🧠 Embedding...")
    embeddings = embedder.embed_documents(texts)

    print("💾 Saving Chroma DB...")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder,
        collection_metadata={"hnsw:space": "cosine"},
    )

    vectorstore._collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
        ids=[f"doc_{i}" for i in range(len(texts))],
    )

    print(f"✅ Stored {vectorstore._collection.count()} chunks")

def main():
    print("===== RAG INGESTION START =====")
    docs = build_docs()
    build_chroma(docs)
    print("🎉 DONE — RAG READY")

if __name__ == "__main__":
    main()
