from langgraph.graph import END, START, StateGraph

from app.agents.products.constants import Nodes
from app.agents.products.nodes import (
    collect_requirements_node,
    present_recommendations_node,
    route_after_collect,
    route_after_validate,
    search_products_node,
    search_purchase_links_node,
    validate_products_node,
)
from app.agents.products.state import ProductSearchState


def build_product_search_agent():
    """Constrói e compila o sub-grafo de busca de produtos.

    Como retorna um grafo compilado (e não uma função), o LangGraph o trata como
    sub-grafo e o checkpointer do grafo pai propaga, permitindo que os interrupt()
    funcionem entre os turnos. Compilado sem checkpointer de propósito.
    """
    builder = StateGraph(ProductSearchState)

    builder.add_node(Nodes.COLLECT, collect_requirements_node)
    builder.add_node(Nodes.SEARCH, search_products_node)
    builder.add_node(Nodes.VALIDATE, validate_products_node)
    builder.add_node(Nodes.PRESENT, present_recommendations_node)
    builder.add_node(Nodes.LINKS, search_purchase_links_node)

    builder.add_edge(START, Nodes.COLLECT)
    builder.add_conditional_edges(
        Nodes.COLLECT, route_after_collect, [Nodes.COLLECT, Nodes.SEARCH]
    )
    builder.add_edge(Nodes.SEARCH, Nodes.VALIDATE)
    builder.add_conditional_edges(
        Nodes.VALIDATE, route_after_validate, [Nodes.SEARCH, Nodes.PRESENT]
    )
    builder.add_edge(Nodes.PRESENT, Nodes.LINKS)
    builder.add_edge(Nodes.LINKS, END)

    return builder.compile()
