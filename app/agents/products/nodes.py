from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.agents.products.constants import Nodes
from app.agents.products.prompt import (
    ASK_REQUIREMENTS_PROMPT,
    EXTRACT_REQUIREMENTS_PROMPT,
)
from app.agents.products.schemas import (
    CollectedInfo,
    Product,
    ProductChoice,
    Requirements,
)
from app.agents.products.state import ProductSearchState
from app.agents.products.tools import (
    deep_search_purchase_links,
    search_candidates,
)
from app.llm import get_llm


# Limite de re-buscas no loop de validação para não cair em loop infinito.
MAX_SEARCH_ATTEMPTS = 2
# Quantidade de produtos apresentados ao usuário.
TOP_PRODUCTS = 3


# --------------------------------------------------------------------------- #
# Helpers de (de)serialização: o estado guarda dicts JSON-native (TypedDict);
# os nós reconstroem os modelos pydantic só quando precisam da lógica/validators.
# --------------------------------------------------------------------------- #
def _requirements_from_state(state: ProductSearchState) -> Requirements | None:
    req = state.get('requirements')
    return Requirements.model_validate(req) if req is not None else None


def _products_from_state(state: ProductSearchState) -> list[Product]:
    return [Product.model_validate(p) for p in state.get('products', [])]


# --------------------------------------------------------------------------- #
# Coleta de requisitos
# --------------------------------------------------------------------------- #
async def _extract_info(messages: list) -> CollectedInfo:
    llm = get_llm().with_structured_output(CollectedInfo, method='function_calling')
    info = await llm.ainvoke([SystemMessage(EXTRACT_REQUIREMENTS_PROMPT), *messages])
    assert isinstance(info, CollectedInfo)
    return info


async def _next_question(info: CollectedInfo, messages: list) -> str | None:
    """Gera a próxima pergunta com base no contexto, ou None se já há dados suficientes."""
    if not info.to_requirements().is_complete:
        missing = (
            'que tipo de produto ele procura, para que vai usar e quais características '
            'são mais importantes'
        )
    elif info.budget is None:
        missing = 'o orçamento máximo em reais (BRL)'
    else:
        return None

    llm = get_llm()
    ai_message = await llm.ainvoke(
        [SystemMessage(ASK_REQUIREMENTS_PROMPT.format(missing=missing)), *messages]
    )
    return ai_message.text.strip()


async def collect_requirements_node(state: ProductSearchState):
    info = await _extract_info(list(state['messages']))
    question = await _next_question(info, list(state['messages']))

    new_messages: list = []

    if question is not None:
        answer = interrupt({'type': 'collect', 'message': '', 'question': question})
        human = HumanMessage(str(answer))
        new_messages.append(human)
        # Re-extrai já considerando a resposta recém-dada.
        info = await _extract_info([*state['messages'], human])

    return {
        'messages': new_messages,
        'requirements': info.to_requirements().model_dump(mode='json'),
        'budget': info.budget,
    }


def route_after_collect(state: ProductSearchState):
    requirements = _requirements_from_state(state)
    if (
        requirements is not None
        and requirements.is_complete
        and state.get('budget') is not None
    ):
        return Nodes.SEARCH
    return Nodes.COLLECT


# --------------------------------------------------------------------------- #
# Busca e validação
# --------------------------------------------------------------------------- #
async def search_products_node(state: ProductSearchState):
    attempts = state.get('search_attempts', 0)
    requirements = _requirements_from_state(state)
    assert requirements is not None

    refine_hint = None
    if attempts > 0:
        # No re-loop, amplia o leque buscando alternativas e melhores avaliações.
        refine_hint = f'{requirements.product_type} alternativas bem avaliadas custo-benefício 2026'

    products = await search_candidates(requirements, state.get('budget'), refine_hint)
    return {
        'products': [p.model_dump(mode='json') for p in products],
        'search_attempts': attempts + 1,
    }


async def validate_products_node(state: ProductSearchState):
    """Reflexão: descarta itens fora do orçamento; o roteamento decide se re-busca."""
    products = _products_from_state(state)
    budget = state.get('budget')
    if budget is not None:
        products = [
            p
            for p in products
            if p.estimated_price is None or float(p.estimated_price) <= budget
        ]
    return {'products': [p.model_dump(mode='json') for p in products]}


def route_after_validate(state: ProductSearchState):
    products = state.get('products', [])
    attempts = state.get('search_attempts', 0)
    if len(products) >= TOP_PRODUCTS or attempts >= MAX_SEARCH_ATTEMPTS:
        return Nodes.PRESENT
    return Nodes.SEARCH


# --------------------------------------------------------------------------- #
# Apresentação e escolha
# --------------------------------------------------------------------------- #
def _format_recommendations(products: list[Product]) -> str:
    lines = ['Encontrei estas opções com melhor custo-benefício para você:\n']
    for i, product in enumerate(products[:TOP_PRODUCTS], start=1):
        price = (
            f'R${product.estimated_price}'
            if product.estimated_price is not None
            else 'preço não informado'
        )
        features = (
            f'\n   Destaques: {", ".join(product.key_features)}'
            if product.key_features
            else ''
        )
        lines.append(
            f'{i}. {product.name}'
            f'{f" ({product.brand})" if product.brand else ""} — {price}\n'
            f'   Por quê: {product.reason}{features}'
        )
    return '\n\n'.join(lines)


async def _resolve_choice(answer: str, products: list[Product]) -> Product:
    options = '\n'.join(
        f'{i}. {p.name}' for i, p in enumerate(products[:TOP_PRODUCTS], start=1)
    )
    llm = get_llm().with_structured_output(ProductChoice, method='function_calling')
    choice = await llm.ainvoke(
        [
            SystemMessage(
                'Identifique qual produto o usuário escolheu a partir da resposta dele.'
            ),
            HumanMessage(f'Opções:\n{options}\n\nResposta do usuário: {answer}'),
        ]
    )
    assert isinstance(choice, ProductChoice)
    index = max(1, min(choice.index, len(products[:TOP_PRODUCTS]))) - 1
    return products[index]


async def present_recommendations_node(state: ProductSearchState):
    products = _products_from_state(state)
    presentation = _format_recommendations(products)

    answer = interrupt(
        {
            'type': 'choice',
            'message': presentation,
            'question': 'Qual modelo te interessou? (responda o número ou o nome)',
        }
    )
    chosen = await _resolve_choice(str(answer), products)

    return {
        'messages': [AIMessage(presentation), HumanMessage(str(answer))],
        'chosen_product': chosen.model_dump(mode='json'),
    }


# --------------------------------------------------------------------------- #
# Pesquisa profunda dos links de compra (após a escolha)
# --------------------------------------------------------------------------- #
def _format_links_message(product: Product, links) -> str:
    if not links:
        return (
            f'Não encontrei links de compra confiáveis para o {product.name} agora. '
            'Quer que eu busque outro modelo?'
        )
    lines = [f'Onde comprar o {product.name}:\n']
    for link in links:
        price = f' — R${link.price}' if link.price is not None else ''
        lines.append(f'- {link.store}{price}: {link.url}')
    return '\n'.join(lines)


async def search_purchase_links_node(state: ProductSearchState):
    chosen = state.get('chosen_product')
    assert chosen is not None
    product = Product.model_validate(chosen)
    links = await deep_search_purchase_links(product)
    return {'messages': [AIMessage(_format_links_message(product, links))]}
