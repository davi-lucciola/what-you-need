from langgraph.graph import END, START, StateGraph

from app.core.agents.products.constants import Nodes
from app.core.agents.products.nodes import (
    collect_requirements_node,
    execute_search_node,
    plan_search_node,
    present_recommendations_node,
    route_after_collect,
    route_after_validate,
    validate_products_node,
)
from app.core.agents.products.state import ProductSearchState


def build_product_search_agent():
    """Constrói e compila o sub-grafo de busca de produtos (turn-based).

    Sem `interrupt()`: a coleta (next-best-question) encerra o turno via `phase`, e
    quando os requisitos estão completos o fluxo segue planner → executor → validação
    → apresentação no mesmo turno. Compilado sem checkpointer (o do pai propaga).
    """
    builder = StateGraph(ProductSearchState)

    builder.add_node(Nodes.COLLECT, collect_requirements_node)
    builder.add_node(Nodes.PLAN, plan_search_node)
    builder.add_node(Nodes.EXECUTE, execute_search_node)
    builder.add_node(Nodes.VALIDATE, validate_products_node)
    builder.add_node(Nodes.PRESENT, present_recommendations_node)

    builder.add_edge(START, Nodes.COLLECT)
    builder.add_conditional_edges(
        Nodes.COLLECT, route_after_collect, [END, Nodes.PLAN]
    )
    builder.add_edge(Nodes.PLAN, Nodes.EXECUTE)
    builder.add_edge(Nodes.EXECUTE, Nodes.VALIDATE)
    builder.add_conditional_edges(
        Nodes.VALIDATE, route_after_validate, [Nodes.PLAN, Nodes.PRESENT]
    )
    builder.add_edge(Nodes.PRESENT, END)

    return builder.compile()
