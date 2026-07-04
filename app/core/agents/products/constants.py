from enum import StrEnum


class Nodes(StrEnum):
    """Nós do sub-grafo de busca de produtos.

    Os valores também aparecem assim no LangGraph Studio.
    """

    COLLECT = 'collect_requirements'
    PLAN = 'plan_search'
    EXECUTE = 'execute_search'
    VALIDATE = 'validate_products'
    PRESENT = 'present_recommendations'
