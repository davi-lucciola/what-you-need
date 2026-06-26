from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.constants import Agents, AllowedAgents
from app.agents.guide import guide_agent
from app.agents.product_search import product_search_agent
from app.agents.states import ChatState
from app.agents.supervisor import supervisor_agent


def build_agent(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(state_schema=ChatState)

    builder.add_node(Agents.SUPERVISOR, supervisor_agent)
    builder.add_node(Agents.GUIDE, guide_agent)
    builder.add_node(Agents.PRODUCT_SEARCH, product_search_agent)

    builder.add_edge(START, Agents.SUPERVISOR)

    def get_next(state: ChatState):
        return state['next']

    builder.add_conditional_edges(
        Agents.SUPERVISOR, get_next, {k: k for k in AllowedAgents}
    )
    builder.add_edge([Agents.GUIDE, Agents.PRODUCT_SEARCH], END)

    return builder.compile(checkpointer=checkpointer)


def make_graph():
    # Entrypoint usado pelo langgraph.json / LangGraph Studio. O platform gerencia a
    # persistência (durabilidade dos interrupts), então compilamos sem checkpointer.
    # É uma factory de 0 argumentos de propósito: o langgraph-api classifica factories
    # pela assinatura e injetaria um RunnableConfig (dict) em qualquer parâmetro extra.
    return build_agent()
