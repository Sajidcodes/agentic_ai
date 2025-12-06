# qualitative validation only ->
# which documents did each method return
from rag.pipeline import retriever, dense_retriever, bm25_retriever, format_docs

queries = [
    "anatomy",
    "GDP growth",
    "oligarchy",
    "linear algebra",
    "opportunity cost",
]

for q in queries:
    print("\n==============================")
    print(f"🔎 Query: {q}")

    print("\n--- Dense Only ---")
    d = dense_retriever.invoke(q)
    for x in d[:2]:
        print(" •", x.metadata.get("source"))

    print("\n--- BM25 Only ---")
    b = bm25_retriever.invoke(q)
    for x in b[:2]:
        print(" •", x.metadata.get("source"))

    print("\n--- HYBRID (Final Retriever) ---")
    h = retriever.invoke(q)
    for x in h[:2]:
        print(" •", x.metadata.get("source"))
