"""
RAG Evaluation Suite
Implements: RAGAS metrics, prompt unit tests, synthetic eval datasets
"""

import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os

# ======================== EVALUATION METRICS ========================

class RAGEvaluator:
    """
    Evaluates RAG system quality using RAGAS-inspired metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevance: Does the answer address the question?
    - Context Precision: Are retrieved docs relevant?
    - Context Recall: Did we retrieve all needed info?
    """
    
    def __init__(self, llm=None):
        """Initialize with an LLM for LLM-as-judge evaluations"""
        self.llm = llm
        self.eval_results = []
    
    # ==================== FAITHFULNESS ====================
    
    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Check if answer is grounded in context (no hallucinations)
        Uses LLM-as-judge approach
        """
        if not self.llm:
            return {"score": None, "error": "No LLM configured"}
        
        eval_prompt = f"""You are evaluating whether an AI answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Evaluate faithfulness:
1. Is every claim in the answer supported by the context?
2. Does the answer contain any information NOT in the context?

Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<brief explanation>", "hallucinations": ["<list any unsupported claims>"]}}
"""
        
        try:
            response = self.llm.invoke(eval_prompt)
            result = json.loads(response.content)
            result["metric"] = "faithfulness"
            return result
        except Exception as e:
            return {"score": None, "error": str(e), "metric": "faithfulness"}
    
    # ==================== ANSWER RELEVANCE ====================
    
    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """Check if answer actually addresses the question"""
        if not self.llm:
            return {"score": None, "error": "No LLM configured"}
        
        eval_prompt = f"""You are evaluating whether an answer is relevant to the question asked.

QUESTION: {question}

ANSWER: {answer}

Evaluate relevance:
1. Does the answer directly address what was asked?
2. Is the answer complete or partial?
3. Does the answer go off-topic?

Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<brief explanation>", "completeness": "<complete/partial/off-topic>"}}
"""
        
        try:
            response = self.llm.invoke(eval_prompt)
            result = json.loads(response.content)
            result["metric"] = "answer_relevance"
            return result
        except Exception as e:
            return {"score": None, "error": str(e), "metric": "answer_relevance"}
    
    # ==================== CONTEXT PRECISION ====================
    
    def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Check if retrieved contexts are relevant to the question"""
        if not self.llm:
            return {"score": None, "error": "No LLM configured"}
        
        context_evals = []
        
        for i, ctx in enumerate(contexts):
            eval_prompt = f"""Is this context relevant to answering the question?

QUESTION: {question}

CONTEXT {i+1}:
{ctx[:500]}

Respond with JSON only:
{{"relevant": true/false, "reasoning": "<brief explanation>"}}
"""
            try:
                response = self.llm.invoke(eval_prompt)
                result = json.loads(response.content)
                context_evals.append(result)
            except:
                context_evals.append({"relevant": None, "error": "eval failed"})
        
        # Calculate precision: relevant / total
        relevant_count = sum(1 for e in context_evals if e.get("relevant") == True)
        precision = relevant_count / len(contexts) if contexts else 0
        
        return {
            "score": precision,
            "metric": "context_precision",
            "relevant_count": relevant_count,
            "total_contexts": len(contexts),
            "details": context_evals
        }
    
    # ==================== SOCRATIC QUALITY (Custom) ====================
    
    def evaluate_socratic_quality(
        self,
        question: str,
        response: str
    ) -> Dict[str, Any]:
        """
        Custom eval for Socratic tutoring quality:
        - Does it ask a guiding question?
        - Does it avoid giving direct answers?
        - Is it encouraging?
        """
        if not self.llm:
            return {"score": None, "error": "No LLM configured"}
        
        eval_prompt = f"""You are evaluating a Socratic tutor's response.

STUDENT QUESTION: {question}

TUTOR RESPONSE: {response}

Evaluate on these criteria:
1. Does the tutor ask a guiding question (not just answer directly)?
2. Does the tutor encourage thinking rather than memorization?
3. Is the tone warm and non-judgmental?
4. Does the response help the student discover the answer themselves?

Respond with JSON only:
{{
    "asks_guiding_question": true/false,
    "avoids_direct_answer": true/false,
    "encouraging_tone": true/false,
    "promotes_discovery": true/false,
    "overall_score": <0.0 to 1.0>,
    "feedback": "<what could be improved>"
}}
"""
        
        try:
            response_obj = self.llm.invoke(eval_prompt)
            result = json.loads(response_obj.content)
            result["metric"] = "socratic_quality"
            return result
        except Exception as e:
            return {"score": None, "error": str(e), "metric": "socratic_quality"}
    
    # ==================== FULL EVALUATION ====================
    
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Run full evaluation suite"""
        context_combined = "\n\n".join(contexts)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200],
            "faithfulness": self.evaluate_faithfulness(question, answer, context_combined),
            "answer_relevance": self.evaluate_answer_relevance(question, answer),
            "context_precision": self.evaluate_context_precision(question, contexts)
        }
        
        # Calculate aggregate score
        scores = []
        for key in ["faithfulness", "answer_relevance", "context_precision"]:
            if results[key].get("score") is not None:
                scores.append(results[key]["score"])
        
        results["aggregate_score"] = sum(scores) / len(scores) if scores else None
        
        self.eval_results.append(results)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.eval_results:
            return {"message": "No evaluations run yet"}
        
        metrics = ["faithfulness", "answer_relevance", "context_precision"]
        summary = {"total_evals": len(self.eval_results)}
        
        for metric in metrics:
            scores = [
                r[metric]["score"] 
                for r in self.eval_results 
                if r[metric].get("score") is not None
            ]
            if scores:
                summary[f"{metric}_avg"] = sum(scores) / len(scores)
                summary[f"{metric}_min"] = min(scores)
                summary[f"{metric}_max"] = max(scores)
        
        return summary


# ======================== PROMPT UNIT TESTS ========================

class PromptTestSuite:
    """Unit tests for prompt behavior"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.test_results = []
    
    def run_test(
        self,
        test_name: str,
        input_query: str,
        expected_behavior: str,
        check_function: callable
    ) -> Dict[str, Any]:
        """Run a single prompt test"""
        try:
            # Get response from pipeline
            response = ""
            for chunk in self.pipeline.stream(input_query):
                if hasattr(chunk, 'content'):
                    response += chunk.content
                else:
                    response += str(chunk)
            
            # Check if response meets expected behavior
            passed = check_function(response)
            
            result = {
                "test_name": test_name,
                "input": input_query,
                "expected": expected_behavior,
                "response": response[:300],
                "passed": passed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            result = {
                "test_name": test_name,
                "input": input_query,
                "expected": expected_behavior,
                "error": str(e),
                "passed": False,
                "timestamp": datetime.now().isoformat()
            }
        
        self.test_results.append(result)
        return result
    
    def get_results(self) -> Dict[str, Any]:
        """Get test results summary"""
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        return {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "results": self.test_results
        }


# ======================== SOCRATIC PROMPT TESTS ========================

def get_socratic_test_cases() -> List[Dict]:
    """Pre-defined test cases for Socratic tutor"""
    return [
        {
            "name": "asks_question_not_answer",
            "input": "What is machine learning?",
            "expected": "Should ask a guiding question, not give definition directly",
            "check": lambda r: "?" in r and not r.lower().startswith("machine learning is")
        },
        {
            "name": "handles_i_dont_know",
            "input": "I don't know",
            "expected": "Should provide help, not ask more questions",
            "check": lambda r: "?" not in r or "let me" in r.lower() or "here's" in r.lower()
        },
        {
            "name": "warm_tone",
            "input": "I'm confused about neural networks",
            "expected": "Should be encouraging, not condescending",
            "check": lambda r: any(word in r.lower() for word in ["great", "good", "let's", "no problem", "that's okay"])
        },
        {
            "name": "no_lecture",
            "input": "Explain deep learning",
            "expected": "Should be concise, not a long lecture",
            "check": lambda r: len(r.split()) < 150
        },
        {
            "name": "single_question",
            "input": "How does RAG work?",
            "expected": "Should ask only ONE question",
            "check": lambda r: r.count("?") <= 2  # Allow 1-2 question marks
        }
    ]


def run_socratic_tests(pipeline) -> Dict[str, Any]:
    """Run all Socratic prompt tests"""
    suite = PromptTestSuite(pipeline)
    
    for test_case in get_socratic_test_cases():
        suite.run_test(
            test_name=test_case["name"],
            input_query=test_case["input"],
            expected_behavior=test_case["expected"],
            check_function=test_case["check"]
        )
    
    return suite.get_results()


# ======================== SYNTHETIC EVAL DATASET ========================

def generate_synthetic_eval_dataset() -> List[Dict]:
    """
    Synthetic dataset for RAG evaluation
    Contains: question, expected_answer_contains, topic
    """
    return [
        {
            "question": "What is RAG in AI?",
            "expected_contains": ["retrieval", "generation", "knowledge"],
            "topic": "RAG basics"
        },
        {
            "question": "How does vector search work?",
            "expected_contains": ["embedding", "similarity", "vector"],
            "topic": "retrieval"
        },
        {
            "question": "What is prompt engineering?",
            "expected_contains": ["prompt", "instruction", "output"],
            "topic": "prompting"
        },
        {
            "question": "Explain model drift",
            "expected_contains": ["performance", "data", "change", "time"],
            "topic": "MLOps"
        },
        {
            "question": "What is few-shot learning?",
            "expected_contains": ["example", "few", "learn"],
            "topic": "ML concepts"
        }
    ]


# ======================== SAVE/LOAD RESULTS ========================

def save_eval_results(results: Dict, filename: str = "eval_results.json"):
    """Save evaluation results to file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_eval_results(filename: str = "eval_results.json") -> Dict:
    """Load evaluation results from file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}