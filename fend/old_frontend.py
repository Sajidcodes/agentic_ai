import os
import sys
import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv

from user_tracking import track_user
import time
import hashlib
from langsmith import traceable

# ---------------- RATE LIMITER ---------------- #

RATE_LIMIT_REQUESTS = 20          # max requests
RATE_LIMIT_WINDOW = 60 * 60        # 60 minutes

def get_user_id():
    """Identify a user by IP or session."""
    ip = st.context.headers.get("X-Forwarded-For") \
         or st.context.headers.get("REMOTE_ADDR") \
         or str(uuid.uuid4())  # fallback

    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def rate_limiter():
    user_id = get_user_id()

    if "rate_limiter" not in st.session_state:
        st.session_state["rate_limiter"] = {}

    user_bucket = st.session_state["rate_limiter"].get(user_id, {
        "timestamps": []
    })

    # Remove old timestamps
    now = time.time()
    user_bucket["timestamps"] = [
        t for t in user_bucket["timestamps"]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(user_bucket["timestamps"]) >= RATE_LIMIT_REQUESTS:
        st.error("⛔ Rate limit exceeded. Please wait before trying again.")
        st.stop()

    # Allow request → record timestamp
    user_bucket["timestamps"].append(now)
    st.session_state["rate_limiter"][user_id] = user_bucket


# ======================== PATH SETUP ========================

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================== ENV & IMPORTS ========================

load_dotenv()

from bend.langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
from langchain.callbacks import LangChainTracer
from langsmith import Client

from observability.observability import (
    log_agent_step,
    log_retrieval,
    log_feedback as obs_log_feedback,
    log_error,
    metrics,
)

# RAG / pipelines
from rag.pipeline import (
    vectorstore,
    rag_pipeline,
    socratic_pipeline,
)

from rag.prompts.query_rewriter import rewrite_query

# ======================== LANGSMITH SETUP ========================

try:
    langsmith_client = Client()
    LANGSMITH_ENABLED = True
except Exception:
    langsmith_client = None
    LANGSMITH_ENABLED = False


def get_tracer():
    """Get tracer for current project."""
    if not LANGSMITH_ENABLED:
        return None
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    return LangChainTracer(project_name=project)


def save_to_dataset(question: str, response: str, mode: str):
    """Auto-save every interaction to a LangSmith dataset."""
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

    langsmith_client.create_example(
        inputs={"question": question, "mode": mode},
        outputs={"response": response},
        dataset_id=dataset.id,
    )


print("FRONTEND VECTORSTORE COUNT:", vectorstore._collection.count())

# ======================== STREAMLIT PAGE CONFIG ========================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

# ======================== UTILITY FUNCTIONS ========================


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat(reset_rag: bool = True, reset_socratic: bool = True):
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


def add_thread(thread_id: str, name: str = "Untitled"):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
        st.session_state["chat_names"] = {}

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["chat_names"][thread_id] = name


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_current_mode() -> str:
    if st.session_state.get("socratic_mode"):
        return "socratic"
    if st.session_state.get("rag_mode"):
        return "rag"
    return "chat"


def log_feedback(run_id: str, is_positive: bool):
    """Log thumbs up/down to LangSmith."""
    if not LANGSMITH_ENABLED or langsmith_client is None or not run_id:
        return False
    try:
        langsmith_client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=1 if is_positive else 0,
            comment="thumbs_up" if is_positive else "thumbs_down",
        )
        # optional: keep observability hook
        obs_log_feedback(run_id, "thumbs_up" if is_positive else "thumbs_down")
        return True
    except Exception as e:
        print(f"Feedback error: {e}")
        return False


# ======================== SESSION STATE INIT ========================

if "rag_message_history" not in st.session_state:
    st.session_state["rag_message_history"] = []

if "rag_pipeline" not in st.session_state:
    st.session_state["rag_pipeline"] = rag_pipeline

if "socratic_pipeline" not in st.session_state:
    st.session_state["socratic_pipeline"] = socratic_pipeline

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = {}

if "rag_mode" not in st.session_state:
    st.session_state["rag_mode"] = False

if "socratic_mode" not in st.session_state:
    st.session_state["socratic_mode"] = False

if "run_ids" not in st.session_state:
    st.session_state["run_ids"] = []

add_thread(st.session_state["thread_id"])

# ======================== SIDEBAR ========================

st.sidebar.title("🤖 AI Learning Assistant")
st.sidebar.markdown("---")

st.sidebar.subheader("Select Mode")

if st.sidebar.button("💬 New Chat", use_container_width=True):
    reset_chat(reset_rag=True, reset_socratic=True)

if st.sidebar.button("📚 RAG Mode (Grounded Answers)", use_container_width=True):
    reset_chat(reset_rag=False, reset_socratic=True)
    st.session_state["rag_mode"] = True
    st.session_state["socratic_mode"] = False

if st.sidebar.button("🎓 Study Mode ", use_container_width=True):
    reset_chat(reset_rag=True, reset_socratic=False)
    st.session_state["rag_mode"] = False
    st.session_state["socratic_mode"] = True

st.sidebar.markdown("---")

# Mode indicator
if st.session_state.get("socratic_mode", False):
    st.sidebar.markdown(
        """<div style='color:white; background-color:#6B46C1; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>
        🎓 STUDY MODE<br><small>I'll guide your thinking</small></div>""",
        unsafe_allow_html=True,
    )
elif st.session_state.get("rag_mode", False):
    st.sidebar.markdown(
        """<div style='color:white; background-color:#2563EB; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>
        📚 RAG MODE<br><small>Direct answers from docs</small></div>""",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """<div style='color:white; background-color:#059669; text-align:center;
        padding:10px; border-radius:8px; font-weight:bold;'>
        💬 CHAT MODE<br><small>General conversation</small></div>""",
        unsafe_allow_html=True,
    )

# Chat history in sidebar
st.sidebar.subheader("Chat History")
for i, thread_id in enumerate(st.session_state["chat_threads"][::-1][:5]):
    name = st.session_state["chat_names"].get(thread_id, "Untitled")
    if st.sidebar.button(name, key=f"thread_btn_{thread_id}_{i}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ======================== MAIN HEADER ========================

if st.session_state.get("socratic_mode", False):
    st.title("🎓 Study Mode")
    st.caption("I'll ask questions to help you think through concepts, then give you clear answers.")
elif st.session_state.get("rag_mode", False):
    st.title("📚 RAG Mode")
    st.caption("I'll answer your questions directly from the knowledge base.")
else:
    st.title("💬 Chat")
    st.caption("General conversation mode.")

# ======================== DISPLAY HISTORY + FEEDBACK ========================

for idx, message in enumerate(st.session_state["message_history"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            # Map index of assistant messages to run_ids
            assistant_idx = len(
                [m for m in st.session_state["message_history"][: idx + 1] if m["role"] == "assistant"]
            ) - 1
            run_id = (
                st.session_state["run_ids"][assistant_idx]
                if assistant_idx < len(st.session_state["run_ids"])
                else None
            )

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
    rate_limiter()
    # Save user message to main history
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

            ai_chunks = []
            for chunk in st.session_state["socratic_pipeline"].stream(
                query, config={"callbacks": callbacks, "run_id": run_id}
            ):
                token = getattr(chunk, "content", str(chunk))
                ai_chunks.append(token)
                placeholder.markdown("".join(ai_chunks))

            ai_message = "".join(ai_chunks)

            # Persist conversation for future Socratic context
            st.session_state["rag_message_history"].append({"role": "user", "content": user_input})
            st.session_state["rag_message_history"].append({"role": "assistant", "content": ai_message})

            # Save eval-example
            save_to_dataset(user_input, ai_message, "socratic")

        # ==================== RAG MODE ====================
        elif current_mode == "rag":
            # Query rewriting for better retrieval
            rewritten_query = rewrite_query(
                query=user_input,
                conversation_history=st.session_state["rag_message_history"],
            )
            print(f"Original: '{user_input}' → Rewritten: '{rewritten_query}'")

            ai_chunks = []
            for chunk in st.session_state["rag_pipeline"].stream(
                rewritten_query, config={"callbacks": callbacks, "run_id": run_id}
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
                # If already in an event loop (e.g., Jupyter), patch it
                try:
                    asyncio.get_running_loop()
                    import nest_asyncio

                    nest_asyncio.apply()
                except RuntimeError:
                    pass

                ai_message = asyncio.run(stream_response())
            except Exception:
                # Fallback to non-streaming invoke
                result = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
                ai_message = result["messages"][-1].content
                placeholder.markdown(ai_message)

            save_to_dataset(user_input, ai_message, "chat")

    # Store run_id for feedback mapping
    st.session_state["run_ids"].append(run_id)

    # Save assistant message to main history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    st.rerun()
