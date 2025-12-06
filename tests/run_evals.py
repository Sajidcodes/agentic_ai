"""
Run Evaluations Script
Execute: python tests/run_evals.py
Outputs: eval_results.json
"""

import sys
import os
import json
from datetime import datetime

# ========= Ensure project root on PATH =========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# Import your pipelines
try:
    from rag.pipeline import rag_pipeline, socratic_pipeline
    PIPELINES_AVAILABLE = True
except ImportError:
    print("⚠️ Could not import pipelines. Running in demo mode.")
    PIPELINES_AVAILABLE = False

# Import evaluation modules
from fend.evaluation import (
    RAGEvaluator,
    run_socratic_tests,
    generate_synthetic_eval_dataset,
    save_eval_results
)

# ======================== CONFIG ========================

OUTPUT_FILE = "eval_results.json"
LLM_MODEL = "gpt-4.1-mini"


def main():
    print("=" * 50)
    print("🧪 RAG Evaluation Suite")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "prompt_tests": None,
        "rag_eval": None,
        "socratic_eval": None
    }
    
    # Initialize LLM for eval
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    evaluator = RAGEvaluator(llm=llm)
    
    # ==================== PROMPT UNIT TESTS ====================
    print("\n📝 Running Prompt Unit Tests...")
    
    if PIPELINES_AVAILABLE:
        test_results = run_socratic_tests(socratic_pipeline)
        results["prompt_tests"] = test_results
        
        print(f"   ✅ Passed: {test_results['passed']}")
        print(f"   ❌ Failed: {test_results['failed']}")
        print(f"   📊 Pass Rate: {test_results['pass_rate']*100:.0f}%")

        for test in test_results['results']:
            if not test['passed']:
                print(f"   ⚠️ FAILED: {test['test_name']}")
                print(f"      Expected: {test['expected']}")
                print(f"      Got: {test['response'][:120]}...")
    else:
        print("   ⏭️ Skipped (pipelines unavailable)")
    
    # ==================== RAG QUALITY EVAL ====================
    print("\n🔍 Running RAG Quality Evaluations...")
    
    eval_dataset = generate_synthetic_eval_dataset()

    if PIPELINES_AVAILABLE:
        try:
            for item in eval_dataset[:3]:
                print(f"   Evaluating: {item['question'][:50]}...")

                response = ""
                for chunk in rag_pipeline.stream(item["question"]):
                    response += chunk.content if hasattr(chunk, "content") else str(chunk)

                contexts = [response[:500]]

                eval_result = evaluator.evaluate(
                    question=item["question"],
                    answer=response,
                    contexts=contexts
                )

                print(f"      Faithfulness: {eval_result['faithfulness'].get('score')}")
                print(f"      Relevance: {eval_result['answer_relevance'].get('score')}")

            results["rag_eval"] = evaluator.get_summary()

        except Exception as e:
            print(f"   ❌ Error in RAG evals: {e}")
    else:
        print("   ⏭️ Skipped")
    
    # ==================== SOCRATIC QUALITY EVAL ====================
    print("\n🎓 Running Socratic Quality Evaluations...")
    
    if PIPELINES_AVAILABLE:
        socratic_questions = [
            "What is machine learning?",
            "How does a neural network work?",
            "Explain overfitting."
        ]

        scores = []

        for q in socratic_questions:
            print(f"   Testing: {q}")

            response = ""
            for chunk in socratic_pipeline.stream(q):
                response += chunk.content if hasattr(chunk, "content") else str(chunk)

            eval_result = evaluator.evaluate_socratic_quality(q, response)

            if eval_result.get("overall_score"):
                scores.append(eval_result["overall_score"])
                print(f"      Score: {eval_result['overall_score']}")
        
        if scores:
            results["socratic_eval"] = {
                "avg_score": sum(scores)/len(scores),
                "num_evaluated": len(scores)
            }
    else:
        print("   ⏭️ Skipped")
    
    # ==================== SAVE RESULTS ====================
    print("\n💾 Saving results...")
    save_eval_results(results, OUTPUT_FILE)
    print(f"   ✅ Saved to {OUTPUT_FILE}")
    
    # ======================== SUMMARY ========================
    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)

    if results["prompt_tests"]:
        print(f"Prompt Tests: {results['prompt_tests']['pass_rate']*100:.0f}% pass")
    if results["rag_eval"]:
        print(f"RAG Quality (faithfulness): {results['rag_eval'].get('faithfulness_avg',0)*100:.0f}%")
    if results["socratic_eval"]:
        print(f"Socratic Quality: {results['socratic_eval']['avg_score']*100:.0f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()
