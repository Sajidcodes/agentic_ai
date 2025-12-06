"""
Direct load test for RAG pipeline (bypasses Streamlit)
Tests your actual bottleneck: retrieval + LLM
"""

import time
import concurrent.futures
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag.pipeline import retriever, format_docs, llm
from rag.prompts.query_rewriter import rewrite_query

# Test questions across all 11 PDFs
QUESTIONS = [
    "What is economics?",
    "What is human anatomy?",
    "Explain machine learning",
    "What is political science?",
    "What is a vector space?",
    "What is sociology?",
    "Explain neural networks",
    "What is supply and demand?",
    "Describe the muscular system",
    "What is data science?",
]

def simulate_chat(user_id: int, question: str):
    """Simulate one user chat request"""
    start = time.time()
    
    try:
        # Step 1: Rewrite query
        rewrite_start = time.time()
        rewritten = rewrite_query(question, [])
        rewrite_time = time.time() - rewrite_start
        
        # Step 2: Retrieve
        retrieve_start = time.time()
        docs = retriever.invoke(rewritten)
        context = format_docs(docs)
        retrieve_time = time.time() - retrieve_start
        
        # Step 3: Generate response
        llm_start = time.time()
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = llm.invoke(prompt)
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start
        
        return {
            "user_id": user_id,
            "question": question[:30],
            "success": True,
            "rewrite_time": round(rewrite_time, 2),
            "retrieve_time": round(retrieve_time, 2),
            "llm_time": round(llm_time, 2),
            "total_time": round(total_time, 2),
        }
    
    except Exception as e:
        return {
            "user_id": user_id,
            "question": question[:30],
            "success": False,
            "error": str(e),
            "total_time": round(time.time() - start, 2),
        }


def run_load_test(num_users: int = 10):
    """Run concurrent user simulation"""
    print(f"\n{'='*60}")
    print(f"🚀 Load Test: {num_users} Concurrent Users")
    print(f"{'='*60}\n")
    
    # Each user asks a random question
    tasks = [
        (i, random.choice(QUESTIONS)) 
        for i in range(num_users)
    ]
    
    start_time = time.time()
    results = []
    
    # Run concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(simulate_chat, user_id, question)
            for user_id, question in tasks
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            status = "✅" if result["success"] else "❌"
            print(f"{status} User {result['user_id']}: {result['question']}... | Total: {result['total_time']}s")
    
    total_time = time.time() - start_time
    
    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Users:      {num_users}")
    print(f"Successful:       {len(successful)}")
    print(f"Failed:           {len(failed)}")
    print(f"Success Rate:     {len(successful)/num_users*100:.1f}%")
    print(f"Total Test Time:  {total_time:.2f}s")
    
    if successful:
        avg_total = sum(r["total_time"] for r in successful) / len(successful)
        avg_retrieve = sum(r["retrieve_time"] for r in successful) / len(successful)
        avg_llm = sum(r["llm_time"] for r in successful) / len(successful)
        
        print(f"\n⏱️  LATENCY BREAKDOWN (Avg)")
        print(f"Retrieval:        {avg_retrieve:.2f}s")
        print(f"LLM Generation:   {avg_llm:.2f}s")
        print(f"Total:            {avg_total:.2f}s")
        
        # Percentiles
        times = sorted([r["total_time"] for r in successful])
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)] if len(times) > 1 else times[-1]
        p99 = times[int(len(times)*0.99)] if len(times) > 1 else times[-1]
        
        print(f"\n📈 PERCENTILES")
        print(f"p50:              {p50:.2f}s")
        print(f"p95:              {p95:.2f}s")
        print(f"p99:              {p99:.2f}s")
    
    if failed:
        print(f"\n❌ FAILURES:")
        for f in failed:
            print(f"   User {f['user_id']}: {f.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Test with increasing load
    for num_users in [5, 10, 20, 30 ,40 ,50]:
        run_load_test(num_users)
        print("\n⏳ Waiting 5 seconds before next test...\n")
        time.sleep(5)