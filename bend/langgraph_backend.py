"""Your LangGraph backend for normal chat mode, independent from RAG.

Memory

Planning

Tools

Graph streaming

chatbot.invoke() and chatbot.astream()"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import asyncio

load_dotenv()

llm = ChatOpenAI(model='gpt-4.1-mini', streaming=True)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def chat_node(state: ChatState):
    messages = state['messages']
    output = []
    async for event in llm.astream_events(messages):
        if event['event'] == "on_chat_model_stream":
            token = event['data']['chunk'].content
            print(token, end='', flush=True)
            output.append(token)
    return {"messages": [AIMessage(content="".join(output))]}



# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

