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


class Phase(StrEnum):
    """Posição durável dentro de um fluxo multi-turno (turn-based, sem interrupt).

    O valor é prefixado pelo agente dono (ver `PHASE_OWNER`) para que o supervisor
    saiba, como dado, para onde re-rotear quando uma fase está ativa. O `phase`
    referencia o agente; o passo interno é decisão do subgrafo.
    """

    PRODUCTS_ACTIVE = 'products:active'


# Ownership explícito: a qual agente cada fase pertence. O override do supervisor
# lê este mapa em vez de um `if` hardcoded — novo fluxo multi-turno = uma linha aqui.
PHASE_OWNER: dict[Phase, Agents] = {
    Phase.PRODUCTS_ACTIVE: Agents.PRODUCTS,
}


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
