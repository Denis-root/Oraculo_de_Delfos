import time
import traceback

from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from icecream import ic
import ast
from langgraph.types import Command
from dataclasses import asdict
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import SQLiteVec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import sqlite3
import inspect


class State(MessagesState):
    secuencia: list[str]
    index: int


def nodo_clasificador(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")
    secuencia =  ['nodo_b', 'nodo_e', 'nodo_d']
    return Command(
        goto=secuencia[0],
        update={
        "secuencia": secuencia,
        "index": 1
    }
    )


def nodo_a(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")

    if len(state['secuencia']) <= state['index']:
        return Command(goto=END)

    return Command(
        goto=state['secuencia'][state['index']],
        update={'index': state['index'] + 1}
    )


def nodo_b(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")

    ic(state)

    if len(state['secuencia']) <= state['index']:
        return {}

    return Command(
        goto=state['secuencia'][state['index']],
        update={'index': state['index'] + 1}
    )


def nodo_c(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")
    if len(state['secuencia']) <= state['index']:
        return {}

    return Command(
        goto=state['secuencia'][state['index']],
        update={'index': state['index'] + 1}
    )


def nodo_d(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")
    if len(state['secuencia']) <= state['index']:
        return {}

    return Command(
        goto=state['secuencia'][state['index']],
        update={'index': state['index'] + 1}
    )


def nodo_e(state: State):
    ic(f"➡️FX: {inspect.currentframe().f_code.co_name}")
    if len(state['secuencia']) <= state['index']:
        return {}
    return Command(
        goto=state['secuencia'][state['index']],
        update={'index': state['index'] + 1}
    )


workflow = StateGraph(State)

workflow.add_node(nodo_clasificador)
workflow.add_node(nodo_a)
workflow.add_node(nodo_b)
workflow.add_node(nodo_c)
workflow.add_node(nodo_d)
workflow.add_node(nodo_e)

workflow.add_edge(START, "nodo_clasificador")
workflow.add_edge("nodo_clasificador", END)
workflow.add_edge("nodo_a", END)
workflow.add_edge("nodo_b", END)
workflow.add_edge("nodo_c", END)
workflow.add_edge("nodo_d", END)
workflow.add_edge("nodo_e", END)

graph = workflow.compile()


config = {"configurable":
    {
        "thread_id": "2",
    }
}
while True:
    question = input('🧢 | User: ')
    logos = []
    for chunk in graph.stream(
            {"messages": [("human", question)]}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    ic('👾 | iA: ', logos)

