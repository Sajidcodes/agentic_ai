"""
Structured Logging & Observability Module
Tracks: agent steps, retrieval hits, tool inputs/outputs, errors
observability/
"""
from pathlib import Path

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps
import traceback

# ======================== LOGGING SETUP ========================

logger = logging.getLogger("rag_observability")
logger.setLevel(logging.DEBUG)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(JSONFormatter())
logger.addHandler(console_handler)

file_handler = logging.FileHandler("observability.log")
file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)
# ======================== META-DATA FORMATTER ========================
def format_source_info(docs):
    """
    Takes retrieved LangChain Document objects and formats metadata for display.
    """
    out = []
    for d in docs:
        meta = d.metadata or {}
        
        src = meta.get("source", "Unknown source")
        page = meta.get("page", meta.get("page_number", "N/A"))
        chunk = meta.get("chunk_id", "N/A")
        
        out.append(f"- **Source:** {src} | **Page:** {page} | **Chunk:** {chunk}")
    
    return "\n".join(out)

# ======================== LOG FUNCTIONS ========================

def log_event(event_type: str, data: Dict[str, Any], level: str = "INFO"):
    extra = {"extra_data": {"event_type": event_type, **data}}
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(event_type, extra=extra)

# ======================== AGENT STEP LOGGING ========================

def log_agent_step(step_name: str, session_id: str, input_data: Any,
                   output_data: Any = None, duration_ms: float = None,
                   metadata: Dict = None):
    log_event("AGENT_STEP", {
        "step_name": step_name,
        "session_id": session_id,
        "input": str(input_data)[:500],
        "output": str(output_data)[:500] if output_data else None,
        "duration_ms": duration_ms,
        "metadata": metadata or {}
    })

# ======================== RETRIEVAL LOGGING ========================

def log_retrieval(session_id: str, query: str, num_docs_retrieved: int,
                  doc_sources: List[str], relevance_scores: List[float] = None,
                  duration_ms: float = None):
    log_event("RETRIEVAL", {
        "session_id": session_id,
        "query": query[:200],
        "num_docs_retrieved": num_docs_retrieved,
        "doc_sources": doc_sources,
        "relevance_scores": relevance_scores,
        "duration_ms": duration_ms
    })

def log_retrieval_hit(session_id: str, query: str, chunk_id: str,
                      chunk_preview: str, score: float):
    log_event("RETRIEVAL_HIT", {
        "session_id": session_id,
        "query": query[:100],
        "chunk_id": chunk_id,
        "chunk_preview": chunk_preview[:200],
        "relevance_score": score
    })

# ======================== LLM CALL LOGGING ========================

def log_llm_call(session_id: str, model: str, prompt_template: str,
                 input_tokens: int = None, output_tokens: int = None,
                 duration_ms: float = None, temperature: float = None):
    log_event("LLM_CALL", {
        "session_id": session_id,
        "model": model,
        "prompt_template": prompt_template[:100],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_ms": duration_ms,
        "temperature": temperature
    })

# ======================== ERROR LOGGING ========================

class ErrorCategory:
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_CONTENT_FILTER = "LLM_CONTENT_FILTER"
    INVALID_INPUT = "INVALID_INPUT"
    CONTEXT_TOO_LONG = "CONTEXT_TOO_LONG"
    UNKNOWN = "UNKNOWN"

def categorize_error(exception: Exception) -> str:
    error_str = str(exception).lower()
    if "timeout" in error_str:
        return ErrorCategory.LLM_TIMEOUT
    elif "rate" in error_str and "limit" in error_str:
        return ErrorCategory.LLM_RATE_LIMIT
    elif "content" in error_str and ("filter" in error_str or "policy" in error_str):
        return ErrorCategory.LLM_CONTENT_FILTER
    elif "context" in error_str and "long" in error_str:
        return ErrorCategory.CONTEXT_TOO_LONG
    elif "retrieval" in error_str or "vector" in error_str:
        return ErrorCategory.RETRIEVAL_FAILURE
    return ErrorCategory.UNKNOWN

def log_error(session_id: str, error: Exception, context: str = None, recoverable: bool = True):
    category = categorize_error(error)
    log_event("ERROR", {
        "session_id": session_id,
        "error_category": category,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "recoverable": recoverable,
        "stack_trace": traceback.format_exc()
    }, level="ERROR")

# ======================== SESSION LOGGING ========================

def log_session_start(session_id: str, mode: str, user_agent: str = None):
    log_event("SESSION_START", {
        "session_id": session_id,
        "mode": mode,
        "user_agent": user_agent
    })

def log_session_end(session_id: str, duration_seconds: float,
                    num_interactions: int, feedback_score: float = None):
    log_event("SESSION_END", {
        "session_id": session_id,
        "duration_seconds": duration_seconds,
        "num_interactions": num_interactions,
        "feedback_score": feedback_score
    })

# ======================== FEEDBACK LOGGING (FIXED!) ========================

def log_feedback(session_id: str, run_id: str, sentiment: str, query: str = None):
    """
    sentiment must be: 'positive' or 'negative'
    """
    score = 1 if sentiment == "positive" else -1

    log_event("FEEDBACK", {
        "session_id": session_id,
        "run_id": run_id,
        "score": score,
        "sentiment": sentiment,
        "query": query[:100] if query else None
    })

    if sentiment == "positive":
        metrics.increment("feedback_positive")
    else:
        metrics.increment("feedback_negative")


# ======================== DECORATOR ========================

def trace_function(func_name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration_ms = (datetime.now() - start).total_seconds() * 1000
                log_event("FUNCTION_CALL", {
                    "function": name,
                    "duration_ms": round(duration_ms, 2),
                    "success": True
                })
                return result
            except Exception as e:
                duration_ms = (datetime.now() - start).total_seconds() * 1000
                log_event("FUNCTION_CALL", {
                    "function": name,
                    "duration_ms": round(duration_ms, 2),
                    "success": False,
                    "error": str(e)
                }, level="ERROR")
                raise
        return wrapper
    return decorator

# ======================== METRICS AGGREGATION ========================

METRICS_FILE = Path("metrics.json")

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_latency_ms": 0,
            "feedback_positive": 0,
            "feedback_negative": 0,
            "retrieval_calls": 0,
            "llm_calls": 0,
        }
        self._load()

    def _load(self):
        if METRICS_FILE.exists():
            try:
                self.metrics.update(json.loads(METRICS_FILE.read_text()))
            except Exception:
                pass

    def _save(self):
        METRICS_FILE.write_text(json.dumps(self.metrics))

    def increment(self, metric: str, value: int = 1):
        if metric in self.metrics:
            self.metrics[metric] += value
            self._save()

    def get_metrics(self) -> Dict:
        total_feedback = (
            self.metrics["feedback_positive"]
            + self.metrics["feedback_negative"]
        )
        return {
            **self.metrics,
            "error_rate": self.metrics["errors"] / max(self.metrics["requests"], 1),
            "avg_latency_ms": self.metrics["total_latency_ms"] / max(self.metrics["requests"], 1),
            "satisfaction_score": self.metrics["feedback_positive"] / max(total_feedback, 1),
        }

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = 0
        self._save()


metrics = MetricsCollector()
