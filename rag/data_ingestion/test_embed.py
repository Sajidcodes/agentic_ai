"""
Test embedding with 10 chunks before running full ingestion
"""
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Test with just 10 texts
test_texts = [
    "Economics is the study of scarcity",
    "Supply and demand determine prices",
    "Anatomy studies body structures",
    "Machine learning uses algorithms",
    "Sociology examines society",
]

print("Testing embedding with 5 texts...")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    embeddings = embedder.embed_documents(test_texts)
    print(f"✅ Success! Got {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
except Exception as e:
    print(f"❌ Failed: {e}")