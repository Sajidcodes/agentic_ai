import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from rag.pipeline import vectorstore  # now works everywhere

print(vectorstore._collection.count())  # verify count

# Modern LangChain retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is RAG in AI?")  # NEW API

for d in docs:
    print(len(d.page_content))
    print(d.page_content[:500])
