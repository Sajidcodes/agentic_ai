"""
Query Rewriter for RAG
Expands vague follow-up questions into standalone queries for better retrieval.

Example:
- History: "What is economics?" -> "Economics is the study of..."
- User: "How does micro differ from macro?"
- Rewritten: "How does microeconomics differ from macroeconomics?"
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Use a fast, cheap model for rewriting
rewriter_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template(
    """You are a query rewriter for a RAG system. Your job is to rewrite vague or context-dependent queries into standalone, searchable questions.

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER QUERY:
{query}

RULES:
1. If the query is already clear and standalone, return it unchanged.
2. If the query contains pronouns (it, this, that, they) or vague references (the concept, this topic), replace them with the actual subject from conversation history.
3. If the query is a follow-up (tell me more, explain further, what about X), expand it to be specific.
4. Keep the rewritten query concise - just the question, no explanation.
5. Do NOT answer the question, just rewrite it.

REWRITTEN QUERY:"""
)

rewrite_chain = rewrite_prompt | rewriter_llm


def rewrite_query(query: str, conversation_history: list) -> str:
    """
    Rewrite a query to be standalone based on conversation history.
    
    Args:
        query: The user's current query
        conversation_history: List of {"role": "user"/"assistant", "content": "..."}
    
    Returns:
        Rewritten standalone query
    """
    # If no history, return query as-is
    if not conversation_history:
        return query
    
    # Format conversation history
    history_str = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}"  # Truncate long messages
        for msg in conversation_history[-6:]  # Last 3 exchanges max
    ])
    
    try:
        result = rewrite_chain.invoke({
            "conversation_history": history_str,
            "query": query
        })
        
        rewritten = result.content.strip()
        
        # Sanity check - if rewriter returns something weird, use original
        if len(rewritten) > 500 or len(rewritten) < 2:
            return query
        
        print(f"Query rewrite: '{query}' -> '{rewritten}'")
        return rewritten
        
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return query


# Quick test
if __name__ == "__main__":
    test_history = [
        {"role": "user", "content": "What is economics?"},
        {"role": "assistant", "content": "Economics is the study of how society uses scarce resources..."},
    ]
    
    test_queries = [
        "Tell me more",
        "How does it relate to business?",
        "What about microeconomics?",
        "What is anatomy?",  # Should stay unchanged
    ]
    
    print("Testing query rewriter:\n")
    for q in test_queries:
        rewritten = rewrite_query(q, test_history)
        print(f"  '{q}' -> '{rewritten}'\n")