from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.constants import Agents, Nodes
from app.agents.guide import build_guide_node
from app.agents.products import build_product_search_node
from app.agents.states import ChatState
from app.agents.supervisor import build_supervisor_node


def build_agent(checkpointer: BaseCheckpointSaver[Any] | None = None):
    builder = StateGraph(state_schema=ChatState)

    builder.add_node(Nodes.SUPERVISOR, build_supervisor_node())
    builder.add_node(Nodes.GUIDE, build_guide_node())
    builder.add_node(Nodes.PRODUCTS, build_product_search_node())

    builder.add_edge(START, Nodes.SUPERVISOR)

    def get_next(state: ChatState) -> str:
        return state.get('next', '')

    builder.add_conditional_edges(Nodes.SUPERVISOR, get_next, {k: k for k in Agents})
    builder.add_edge([Nodes.GUIDE, Nodes.PRODUCTS], END)

    return builder.compile(checkpointer=checkpointer)


def make_graph():
    # Entrypoint usado pelo langgraph.json / LangGraph Studio. O platform gerencia a
    # persistência (durabilidade dos interrupts), então compilamos sem checkpointer.
    # É uma factory de 0 argumentos de propósito: o langgraph-api classifica factories
    # pela assinatura e injetaria um RunnableConfig (dict) em qualquer parâmetro extra.
    return build_agent()
