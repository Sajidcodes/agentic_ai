import os
import time
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = Path("/data/pdfs")
CHROMA_DIR = Path("/data/chroma")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 50  # Smaller batches to avoid rate limits

# ---------------------------------------- #

def find_pdfs(pdf_dir: Path) -> List[Path]:
    return list(pdf_dir.rglob("*.pdf"))


def load_and_split_pdf(pdf_path: Path, splitter):
    print(f"\n📖 Loading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    print(f"   - Pages: {len(docs)}")
    split_docs = splitter.split_documents(docs)
    print(f"   - Chunks: {len(split_docs)}")

    # ✅ Normalize metadata so it's portable & clean
    doc_id = pdf_path.stem.lower().replace(" ", "_")

    for d in split_docs:
        d.metadata = {
            "source": doc_id,          # e.g. "introduction_to_business_op_8d04gaa"
            "file_name": pdf_path.name # original filename, if you want to show it
        }


    return split_docs


def build_corpus():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF folder missing: {PDF_DIR}")

    pdf_files = find_pdfs(PDF_DIR)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    print(f"🔎 Found {len(pdf_files)} PDF(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    all_docs = []
    for pdf_path in pdf_files:
        all_docs.extend(load_and_split_pdf(pdf_path, splitter))

    print(f"\n📚 Total chunks: {len(all_docs)}")
    return all_docs


def embed_with_retry(embedder, texts, batch_num, total_batches):
    """Embed a batch with retry logic for rate limits"""
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            result = embedder.embed_documents(texts)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "quota" in error_str:
                wait_time = (2 ** attempt) + 1  # 2, 3, 5, 9, 17 seconds
                print(f"   ⚠️ Rate limited on batch {batch_num}. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    
    raise Exception(f"Failed batch {batch_num} after {max_retries} retries")


def build_chroma(docs):
    print(f"\n🧠 Building Chroma DB at: {CHROMA_DIR}")

    # Delete existing db
    if CHROMA_DIR.exists():
        print("   ⚠️ Removing old database...")
        shutil.rmtree(CHROMA_DIR)
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    
    # Create embedder
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_minutes = total_batches * 0.5  # ~30 seconds per batch
    
    print(f"\n📊 Total chunks to embed: {len(texts)}")
    print(f"   Batches: {total_batches}")
    print(f"   Estimated time: ~{estimated_minutes:.0f} minutes\n")

    # Step 1: Embed all texts in batches (ONCE)
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_texts = texts[i:i + BATCH_SIZE]
        
        print(f"   📤 Batch {batch_num}/{total_batches} ({i + len(batch_texts)}/{len(texts)})")
        
        batch_embeddings = embed_with_retry(embedder, batch_texts, batch_num, total_batches)
        all_embeddings.extend(batch_embeddings)
        
        # Small delay to avoid rate limits
        time.sleep(0.3)
    
    print(f"\n✅ Embeddings complete: {len(all_embeddings)} vectors")

    # Step 2: Create Chroma and add with pre-computed embeddings
    print("\n💾 Storing in ChromaDB...")
    
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    # Add in batches to avoid memory issues
    store_batch_size = 5000
    for i in range(0, len(texts), store_batch_size):
        end_idx = min(i + store_batch_size, len(texts))
        
        vectorstore._collection.add(
            documents=texts[i:end_idx],
            embeddings=all_embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=[f"doc_{j}" for j in range(i, end_idx)]
        )
        print(f"   💾 Stored {end_idx}/{len(texts)} chunks")

    print("\n✅ Chroma DB saved.")
    print(f"   → Total chunks: {len(texts)}")
    print(f"   → DB count: {vectorstore._collection.count()}")


def main():
    print("==================================================")
    print("📥 Multi-Book PDF Ingestion for RAG")
    print("==================================================")

    print(f"PDF directory:   {PDF_DIR}")
    print(f"Chroma DB dir:   {CHROMA_DIR}")

    docs = build_corpus()
    build_chroma(docs)

    print("\n🎉 Done. Your RAG system can now load this DB.")


if __name__ == "__main__":
    main()
