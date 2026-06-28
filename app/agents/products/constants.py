from enum import StrEnum


class Nodes(StrEnum):
    """Nós do sub-grafo de busca de produtos.

    Os valores também aparecem assim no LangGraph Studio.
    """

    COLLECT = 'collect_requirements'
    SEARCH = 'search_products'
    VALIDATE = 'validate_products'
    PRESENT = 'present_recommendations'
    LINKS = 'search_purchase_links'
