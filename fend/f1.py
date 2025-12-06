import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.rag import vectorstore, ChatOpenAI, ChatPromptTemplate, RunnablePassthrough, rag_pipeline

import streamlit as st
from bend.langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
import asyncio

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat(reset_rag=True):
    """
    Resets a normal chat. RAG history is preserved unless explicitly cleared.
    """
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, "Untitled")
    st.session_state['message_history'] = []
    if reset_rag:
        st.session_state['rag_mode'] = False  # normal chatbot

def add_thread(thread_id, name="Untitled"):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_names'][thread_id] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************

if 'rag_message_history' not in st.session_state:
    st.session_state['rag_message_history'] = []

if 'rag_pipeline' not in st.session_state:
    st.session_state['rag_pipeline'] = rag_pipeline

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'chat_names' not in st.session_state:
    st.session_state['chat_names'] = {}

if 'rag_mode' not in st.session_state:
    st.session_state['rag_mode'] = False    

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat(reset_rag=True)
    st.session_state['rag_mode'] = False

if st.sidebar.button('RAG Workflow'):
    reset_chat(reset_rag=False)
    st.session_state['rag_mode'] = True

if st.session_state.get('rag_mode', False):
    st.sidebar.markdown(
        "<div style='color:white; background-color:red; text-align:center; padding:5px; border-radius:5px'>RAG MODE ON</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<div style='color:white; background-color:green; text-align:center; padding:5px; border-radius:5px'>NORMAL CHATBOT</div>",
        unsafe_allow_html=True
    )

st.sidebar.header('Chat history')
for i, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    name = st.session_state['chat_names'].get(thread_id, "Untitled")
    if st.sidebar.button(name, key=f"thread_btn_{thread_id}_{i}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages

# **************************************** Main UI ************************************

# Show conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # Add user message to chat history immediately
    st.session_state['message_history'].append({'role':'user','content':user_input})

    # Auto-generate chat name if still Untitled
    if st.session_state['chat_names'][st.session_state['thread_id']] == "Untitled":
        short_name = user_input.strip().split('?')[0][:40]
        st.session_state['chat_names'][st.session_state['thread_id']] = short_name + ("..." if len(user_input) > 40 else "")

    with st.chat_message('user'):
        st.text(user_input)

    # Create placeholder for assistant message
    placeholder = st.chat_message("assistant")
    msg_container = placeholder.empty()

    if st.session_state.get('rag_mode', False):
        # RAG context
        rag_context = "\n".join([
            f"User: {msg['content']}" if msg['role']=='user' else f"Assistant: {msg['content']}"
            for msg in st.session_state['rag_message_history']
        ])
        query = (rag_context + "\nUser: " + user_input) if rag_context else user_input

        # Invoke RAG
        results = st.session_state['rag_pipeline'].invoke(query)

        # Format answer
        if isinstance(results, list):
            ai_message = "\n\n".join(
                [
                    f"{doc.content}\n\n*(Source: {getattr(doc, 'metadata', {}).get('source', 'Unknown')})*"
                    if hasattr(doc, 'content') else str(doc)
                    for doc in results
                ]
            )
        else:
            ai_message = str(results)

        # Display immediately
        msg_container.text(ai_message)

        # Save to RAG history
        st.session_state['rag_message_history'].append({'role':'user','content':user_input})
        st.session_state['rag_message_history'].append({'role':'assistant','content':ai_message})

        # Save to normal chat history
        st.session_state['message_history'].append({'role':'assistant','content':ai_message})

    else:
        # Normal streaming
        CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

        async def run_stream():
            ai_message = ""
            async for message_chunk, _ in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                ai_message += message_chunk.content
                msg_container.markdown(ai_message)
            return ai_message

        ai_message = asyncio.run(run_stream())

        # Save to normal chat history
        st.session_state['message_history'].append({'role':'assistant','content':ai_message})
