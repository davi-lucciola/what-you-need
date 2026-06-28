from enum import StrEnum


class Agents(StrEnum):
    """Agentes para os quais o supervisor pode rotear a conversa."""

    GUIDE = 'guide'
    PRODUCTS = 'products'


class Nodes(StrEnum):
    """Todos os nós do grafo principal (inclui o supervisor, que não é roteável)."""

    SUPERVISOR = 'supervisor'
    GUIDE = 'guide'
    PRODUCTS = 'products'


AGENTS_DESCRIPTION: dict[Agents, str] = {
    Agents.GUIDE: (
        'Agente que irá recepcionar e guiar o usuário, conduzirá a conversa '
        'explicando sobre o bot até que o deseje realizar alguma operação.'
    ),
    Agents.PRODUCTS: (
        'Agente que irá guiar o fluxo de busca de um produto, ele pedirá '
        'para o usuário principalmente informações como orçamento disponivel '
        'e o problema que ele deseja solucionar. Aqui o usuário também pode '
        'achar os links de compra do produto.'
    ),
}
