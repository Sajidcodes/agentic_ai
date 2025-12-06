import os
import sys
import asyncio
import uuid
import time
import hashlib

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from user_tracking import track_user

# ---------------- LangSmith / Tracing ----------------- #
from langsmith import traceable, Client
from langchain.callbacks import LangChainTracer
from langchain_core.messages import HumanMessage

# ---------------- Environment ---------------- #
load_dotenv()

# ---------------- Backend / Pipelines ---------------- #
from bend.langgraph_backend import chatbot
from rag.pipeline import vectorstore, rag_pipeline, socratic_pipeline
from rag.prompts.query_rewriter import rewrite_query

# ---------------- Observability ---------------- #
from observability.observability import (
    log_agent_step,
    log_retrieval,
    log_feedback as obs_log_feedback,
    log_error,
    metrics,
)

# ---------------- RATE LIMITER ---------------- #

RATE_LIMIT_REQUESTS = 20         # allowed requests
RATE_LIMIT_WINDOW = 60 * 60      # per hour


def get_user_id():
    ip = (
        st.context.headers.get("X-Forwarded-For")
        or st.context.headers.get("REMOTE_ADDR")
    )

    if ip:
        return hashlib.sha256(ip.encode()).hexdigest()[:12]

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = hashlib.sha256(
            str(uuid.uuid4()).encode()
        ).hexdigest()[:12]
    return st.session_state["user_id"]


def rate_limiter():
    """Rate limit by session_state-backed sliding window per user_id."""
    user_id = get_user_id()
    now = time.time()

    if "rate_limiter" not in st.session_state:
        st.session_state["rate_limiter"] = {}

    bucket = st.session_state["rate_limiter"].get(user_id, {"timestamps": []})

    # keep only timestamps in the active window
    bucket["timestamps"] = [t for t in bucket["timestamps"] if now - t < RATE_LIMIT_WINDOW]

    if len(bucket["timestamps"]) >= RATE_LIMIT_REQUESTS:
        st.error("⛔ Rate limit exceeded. Try again later.")
        st.stop()

    bucket["timestamps"].append(now)
    st.session_state["rate_limiter"][user_id] = bucket


# ======================== LANGSMITH SETUP ========================

try:
    langsmith_client = Client()
    LANGSMITH_ENABLED = True
except Exception:
    langsmith_client = None
    LANGSMITH_ENABLED = False


def get_tracer():
    """Return a LangChainTracer instance or None if LangSmith isn't available."""
    if not LANGSMITH_ENABLED:
        return None
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    return LangChainTracer(project_name=project)


def save_to_dataset(question: str, response: str, mode: str):
    """Auto-save every interaction to a LangSmith dataset for later inspection."""
    if not LANGSMITH_ENABLED or langsmith_client is None:
        return

    dataset_name = "production_interactions"
    try:
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Auto-collected from production",
        )

    try:
        langsmith_client.create_example(
            inputs={"question": question, "mode": mode},
            outputs={"response": response},
            dataset_id=dataset.id,
        )
    except Exception as e:
        # don't break user flow for telemetry failures
        print(f"LangSmith save error: {e}")


# ======================== TRACEABLE HELPERS (STREAM-PRESERVING) ========================

@traceable(name="rewrite_query")
def traced_rewrite(query: str, history):
    """Wrap rewrite_query in a traceable function. Returns rewritten string."""
    return rewrite_query(query=query, conversation_history=history)


@traceable(name="rag_pipeline_stream")
def traced_rag_stream(pipeline, query, callbacks, run_id):
    """
    Traceable generator wrapper around pipeline.stream for RAG.
    It yields the same chunks the underlying pipeline yields so streaming is preserved.
    """
    for chunk in pipeline.stream(query, config={"callbacks": callbacks, "run_id": run_id}):
        yield chunk


@traceable(name="socratic_pipeline_stream")
def traced_socratic_stream(pipeline, query, callbacks, run_id):
    """
    Traceable generator wrapper around socratic pipeline.stream.
    Yields underlying chunks to preserve incremental streaming.
    """
    for chunk in pipeline.stream(query, config={"callbacks": callbacks, "run_id": run_id}):
        yield chunk


# ======================== STREAMLIT PAGE CONFIG ========================

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

# quick sanity print for operator
try:
    if "vectorstore_checked" not in st.session_state:
        print("FRONTEND VECTORSTORE COUNT:", vectorstore._collection.count())
        st.session_state["vectorstore_checked"] = True

except Exception:
    print("Vectorstore count unavailable.")

# ======================== SESSION DEFAULTS & HELPERS ========================

defaults = {
    "rag_message_history": [],
    "rag_pipeline": rag_pipeline,
    "socratic_pipeline": socratic_pipeline,
    "message_history": [],
    "thread_id": str(uuid.uuid4()),
    "chat_threads": [],
    "chat_names": {},
    "rag_mode": False,
    "socratic_mode": False,
    "run_ids": [],
}


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str, name: str = "Untitled"):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
        st.session_state["chat_names"] = {}

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["chat_names"][thread_id] = name


def reset_chat(reset_rag: bool = True, reset_socratic: bool = True):
    
    """
    Reset conversation state. Uses the defaults dict for consistent behavior.
    Keeps streaming & UI flow intact.
    """
    # set defaults where missing and update essential fields
    for k, v in defaults.items():
        st.session_state.setdefault(k, v if not isinstance(v, list) else list(v))

    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id, "Untitled")
    st.session_state["message_history"] = []
    st.session_state["rag_message_history"] = []
    st.session_state["run_ids"] = []

    if reset_rag:
        st.session_state["rag_mode"] = False
    if reset_socratic:
        st.session_state["socratic_mode"] = False


def load_conversation(thread_id: str):
    """Load conversation state from the chatbot backend (if available)."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", []) or []
    except Exception:
        return []


def get_current_mode() -> str:
    if st.session_state.get("socratic_mode"):
        return "socratic"
    if st.session_state.get("rag_mode"):
        return "rag"
    return "chat"


def log_feedback(run_id: str, is_positive: bool):
    user_id = get_user_id()
    sentiment = "positive" if is_positive else "negative"

    try:
        obs_log_feedback(
            session_id=user_id,
            run_id=run_id,
            sentiment=sentiment,
        )
    except Exception as e:
        print(f"Observability feedback error: {e}")

    print(f"📊 FEEDBACK | user={user_id} | run={run_id} | {sentiment}")



# ======================== SESSION STATE INIT ========================

for key, val in defaults.items():
    if key not in st.session_state:
        # copy mutable defaults
        st.session_state[key] = val if not isinstance(val, (list, dict)) else (list(val) if isinstance(val, list) else dict(val))

add_thread(st.session_state["thread_id"])

# ======================== SIDEBAR ========================

st.sidebar.title("🤖 AI Learning Assistant")
st.sidebar.markdown("---")

st.sidebar.subheader("Select Mode")

if st.sidebar.button("💬 Normal Chat", use_container_width=True):
    reset_chat(reset_rag=True, reset_socratic=True)
    st.rerun()

if st.sidebar.button("📚 RAG Mode (Grounded Answers)", use_container_width=True):
    reset_chat(reset_rag=False, reset_socratic=True)
    st.session_state["rag_mode"] = True
    st.session_state["socratic_mode"] = False
    st.rerun()

if st.sidebar.button("🎓 Study Mode ", use_container_width=True):
    reset_chat(reset_rag=True, reset_socratic=False)
    st.session_state["rag_mode"] = False
    st.session_state["socratic_mode"] = True
    st.rerun()

st.sidebar.markdown("---")

# Mode badges
if st.session_state.get("socratic_mode", False):
    st.sidebar.markdown(
        """<div style='color:white; background-color:#6B46C1; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>🎓 STUDY MODE<br><small>I'll guide your thinking</small></div>""",
        unsafe_allow_html=True,
    )
elif st.session_state.get("rag_mode", False):
    st.sidebar.markdown(
        """<div style='color:white; background-color:#2563EB; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>📚 RAG MODE<br><small>Direct answers from docs</small></div>""",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """<div style='color:white; background-color:#059669; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>💬 CHAT MODE<br><small>General conversation</small></div>""",
        unsafe_allow_html=True,
    )


# ======================== USER STATS ========================

stats = metrics.get_metrics()
total_feedback = stats["feedback_positive"] + stats["feedback_negative"]

if total_feedback > 0:
    love_pct = int(stats["satisfaction_score"] * 100)
    st.sidebar.markdown(f"❤️ **Loved by {love_pct}% of users**")
else:
    st.sidebar.markdown("❤️ Loved by —")


# Chat history in sidebar (recent 5)
st.sidebar.subheader("Chat History")

for i, thread_id in enumerate(st.session_state["chat_threads"][::-1][:5]):
    name = st.session_state["chat_names"].get(thread_id, "Untitled")

    if st.sidebar.button(name, key=f"thread_btn_{thread_id}_{i}"):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = st.session_state.get(
            f"history_{thread_id}", []
        )
        st.rerun()


# ======================== MAIN HEADER ========================

if st.session_state.get("socratic_mode", False):
    st.title("🎓 Study Mode")
    st.caption("I'll ask questions to help you think through concepts, then give you clear answers.")
elif st.session_state.get("rag_mode", False):
    st.title("📚 RAG Mode")
    st.caption("Ask questions from the following topics:\n " \
    "1. Anatomy and Physiology\n" \
    "2. Machine Learning\n" \
    "3. Economics\n" \
    "4. Data Science\n" \
    "5. Political Science\n" \
    "6. Sociology\n" \
    "7. Business\n" \
    "> I'll answer your questions directly from this knowledge base.\n")
else:
    st.title("💬 Chat")
    st.caption("General conversation mode.")

# ======================== DISPLAY HISTORY + FEEDBACK ========================

for idx, message in enumerate(st.session_state["message_history"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            assistant_idx = len([m for m in st.session_state["message_history"][: idx + 1] if m["role"] == "assistant"]) - 1
            run_id = st.session_state["run_ids"][assistant_idx] if assistant_idx < len(st.session_state["run_ids"]) else None

            feedback_key = f"feedback_{idx}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None

            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{idx}", disabled=st.session_state[feedback_key] is not None):
                    log_feedback(run_id, True)
                    st.session_state[feedback_key] = "up"
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{idx}", disabled=st.session_state[feedback_key] is not None):
                    log_feedback(run_id, False)
                    st.session_state[feedback_key] = "down"
                    st.rerun()
            if st.session_state[feedback_key]:
                with col3:
                    st.caption("Thanks for your feedback!")

# ======================== CHAT INPUT ========================

user_input = st.chat_input("Ready when you are...")

if user_input:
    metrics.increment("requests")   
    rate_limiter()
    st.session_state["message_history"].append({"role": "user", "content": user_input})


    # Auto-generate chat name (from first user message of thread)
    if st.session_state["chat_names"].get(st.session_state["thread_id"], "Untitled") == "Untitled":
        short_name = user_input.strip().split("?")[0][:40]
        st.session_state["chat_names"][st.session_state["thread_id"]] = (
            short_name + ("..." if len(user_input) > 40 else "")
        )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    ai_message = ""
    current_mode = get_current_mode()
    run_id = str(uuid.uuid4())

    tracer = get_tracer()
    callbacks = [tracer] if tracer else []

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # ==================== SOCRATIC MODE ====================
        if current_mode == "socratic":
            rag_context = "\n".join(
                [
                    f"Student: {msg['content']}" if msg["role"] == "user" else f"Tutor: {msg['content']}"
                    for msg in st.session_state["rag_message_history"]
                ]
            )
            query = (rag_context + "\nStudent: " + user_input) if rag_context else user_input

            # streaming via traced_socratic_stream preserves token-by-token behavior
            ai_chunks = []
            for chunk in traced_socratic_stream(
                st.session_state["socratic_pipeline"],
                query,
                callbacks,
                run_id,
            ):
                token = getattr(chunk, "content", str(chunk))
                ai_chunks.append(token)
                # incremental update -> streams to the UI exactly like your base code
                placeholder.markdown("".join(ai_chunks))

            ai_message = "".join(ai_chunks)

            # Persist conversation for future Socratic context
            st.session_state["rag_message_history"].append({"role": "user", "content": user_input})
            st.session_state["rag_message_history"].append({"role": "assistant", "content": ai_message})

            # Save eval-example
            save_to_dataset(user_input, ai_message, "socratic")

        # ==================== RAG MODE ====================
        elif current_mode == "rag":
            # Query rewriting for better retrieval (traced)
            rewritten = traced_rewrite(user_input, st.session_state["rag_message_history"])
            print(f"Original: '{user_input}' → Rewritten: '{rewritten}'")

            ai_chunks = []
            for chunk in traced_rag_stream(
                st.session_state["rag_pipeline"],
                rewritten,
                callbacks,
                run_id,
            ):
                token = getattr(chunk, "content", str(chunk))
                ai_chunks.append(token)
                placeholder.markdown("".join(ai_chunks))

            ai_message = "".join(ai_chunks)

            st.session_state["rag_message_history"].append({"role": "user", "content": user_input})
            st.session_state["rag_message_history"].append({"role": "assistant", "content": ai_message})

            save_to_dataset(user_input, ai_message, "rag")

        # ==================== NORMAL CHAT MODE ====================
        else:
            CONFIG = {
                "configurable": {"thread_id": st.session_state["thread_id"]},
                "callbacks": callbacks,
                "run_id": run_id,
            }

            # Keep your async streaming approach intact — we only add safe tracing & telemetry.
            async def stream_response():
                collected = ""
                async for message_chunk, _ in chatbot.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    token = message_chunk.content or ""
                    collected += token
                    placeholder.markdown(collected)
                return collected

            try:
                # If already in an event loop (e.g., Jupyter), patch it (keeps your base behavior)
                try:
                    asyncio.get_running_loop()
                    import nest_asyncio  # local-only fallback for embedded loops
                    nest_asyncio.apply()
                except RuntimeError:
                    pass

                ai_message = asyncio.run(stream_response())
            except Exception:
                # Fallback to non-streaming invoke (unchanged)
                result = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
                ai_message = result["messages"][-1].content
                placeholder.markdown(ai_message)

            save_to_dataset(user_input, ai_message, "chat")

    # Store run_id for feedback mapping
    st.session_state["run_ids"].append(run_id)

    # Save assistant message to main history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    
    # Persist history per thread
    st.session_state[f"history_{st.session_state['thread_id']}"] = list(
    st.session_state["message_history"]
)

    # Rerun to update UI (keeps your original behavior)
    st.rerun()
