import asyncio
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END

from app.core.agents.constants import Phase
from app.core.agents.products.constants import Nodes
from app.core.agents.products.prompt import (
    ASK_REQUIREMENTS_PROMPT,
    EXTRACT_REQUIREMENTS_PROMPT,
    PLANNER_PROMPT,
    REVIEW_VALIDATION_PROMPT,
)
from app.core.agents.products.schemas import (
    CollectedInfo,
    Product,
    PurchaseLink,
    Requirements,
    ReviewVerdict,
    SearchPlan,
)
from app.core.agents.products.state import ProductSearchState
from app.core.agents.products.tools import (
    extract_candidates,
    fetch_purchase_links,
    run_planned_searches,
    search_product_reviews,
)
from app.infra.llm import get_llm

# Limite de re-buscas (replan) para não cair em loop infinito.
MAX_SEARCH_ATTEMPTS = 2
# Quantidade de produtos apresentados ao usuário.
TOP_PRODUCTS = 5


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
# Coleta de requisitos (next-best-question, turn-based)
# --------------------------------------------------------------------------- #
async def _extract_info(messages: list[AnyMessage]) -> CollectedInfo:
    llm = get_llm().with_structured_output(CollectedInfo, method='function_calling')
    info = await llm.ainvoke([SystemMessage(EXTRACT_REQUIREMENTS_PROMPT), *messages])
    assert isinstance(info, CollectedInfo)
    return info


def _missing_field(info: CollectedInfo) -> str | None:
    """A lacuna de maior valor a perguntar agora (next-best-question).

    Ordem: tipo de produto → uso/prioridades → orçamento. Retorna None se completo.
    """
    req = info.to_requirements()
    if not req.product_type:
        return 'que tipo de produto ele procura'
    if not (req.use_case or req.priorities):
        return (
            'para que ele vai usar o produto e quais características são mais '
            'importantes'
        )
    if info.budget is None:
        return 'o orçamento máximo em reais (BRL)'
    return None


async def _next_question(missing: str, messages: list[AnyMessage]) -> str:
    llm = get_llm()
    ai_message = await llm.ainvoke(
        [SystemMessage(ASK_REQUIREMENTS_PROMPT.format(missing=missing)), *messages]
    )
    return ai_message.text.strip()


async def collect_requirements_node(state: ProductSearchState) -> dict[str, Any]:
    messages = list(state['messages'])
    info = await _extract_info(messages)
    requirements = info.to_requirements().model_dump(mode='json')

    missing = _missing_field(info)

    if missing is not None:
        question = await _next_question(missing, messages)

        return {
            'messages': [AIMessage(question)],
            'requirements': requirements,
            'budget': info.budget,
            'phase': Phase.PRODUCTS_COLLECTING.value,
        }

    return {
        'requirements': requirements,
        'budget': info.budget,
        'phase': '',
    }


def route_after_collect(state: ProductSearchState):
    # Fase ainda ativa = perguntamos algo neste turno → encerra o turno.
    if state.get('phase'):
        return END
    return Nodes.PLAN


# --------------------------------------------------------------------------- #
# Planner + Executor
# --------------------------------------------------------------------------- #
async def plan_search_node(state: ProductSearchState) -> dict[str, Any]:
    requirements = _requirements_from_state(state)
    assert requirements is not None
    budget = state.get('budget')
    attempts = state.get('search_attempts', 0)

    priorities = ', '.join(requirements.priorities) or 'não informadas'
    must_haves = ', '.join(requirements.must_haves) or 'nenhum'
    budget_text = f'R${budget:.0f}' if budget else 'não informado'
    context = (
        f'Tipo de produto: {requirements.product_type}\n'
        f'Uso pretendido: {requirements.use_case}\n'
        f'Prioridades: {priorities}\n'
        f'Requisitos obrigatórios: {must_haves}\n'
        f'Orçamento máximo: {budget_text}'
    )
    if attempts > 0:
        # Replan: a busca anterior não rendeu produtos suficientes.
        context += (
            '\n\nDica de replanejamento: a busca anterior foi fraca; amplie o leque '
            'com ângulos novos (alternativas, outras marcas, melhores avaliações).'
        )

    llm = get_llm().with_structured_output(SearchPlan, method='function_calling')
    plan = await llm.ainvoke([SystemMessage(PLANNER_PROMPT), HumanMessage(context)])
    assert isinstance(plan, SearchPlan)
    return {'plan': plan.queries}


async def execute_search_node(state: ProductSearchState) -> dict[str, Any]:
    requirements = _requirements_from_state(state)
    assert requirements is not None
    queries = state.get('plan') or []

    results = await run_planned_searches(queries)
    products = await extract_candidates(results, requirements, state.get('budget'))
    return {
        'products': [p.model_dump(mode='json') for p in products],
        'search_attempts': state.get('search_attempts', 0) + 1,
    }


# --------------------------------------------------------------------------- #
# Validação: orçamento + avaliações online
# --------------------------------------------------------------------------- #
async def _review_verdict(product: Product) -> ReviewVerdict:
    context = await search_product_reviews(product)
    llm = get_llm().with_structured_output(ReviewVerdict, method='function_calling')
    verdict = await llm.ainvoke(
        [
            SystemMessage(REVIEW_VALIDATION_PROMPT),
            HumanMessage(f'Produto: {product.name}\n\nResultados de busca:\n{context}'),
        ]
    )
    assert isinstance(verdict, ReviewVerdict)
    return verdict


async def validate_products_node(state: ProductSearchState) -> dict[str, Any]:
    """Reflexão: descarta itens fora do orçamento e mal avaliados online."""
    products = _products_from_state(state)
    budget = state.get('budget')
    if budget is not None:
        products = [
            p
            for p in products
            if p.estimated_price is None or float(p.estimated_price) <= budget
        ]

    # Avaliações online (concorrente): mantém só os bem avaliados, anexa o resumo.
    verdicts = await asyncio.gather(*(_review_verdict(p) for p in products))
    validated = [
        product.model_copy(update={'review_summary': verdict.summary})
        for product, verdict in zip(products, verdicts)
        if verdict.well_rated
    ]
    return {'products': [p.model_dump(mode='json') for p in validated]}


def route_after_validate(state: ProductSearchState):
    products = state.get('products', [])
    attempts = state.get('search_attempts', 0)
    if len(products) >= TOP_PRODUCTS or attempts >= MAX_SEARCH_ATTEMPTS:
        return Nodes.PRESENT
    return Nodes.PLAN


# --------------------------------------------------------------------------- #
# Apresentação (com links de compra anexados)
# --------------------------------------------------------------------------- #
def _format_links(links: list[PurchaseLink]) -> str:
    if not links:
        return '   Onde comprar: não encontrei links confiáveis agora.'
    lines = ['   Onde comprar:']
    for link in links:
        price = f' — R${link.price}' if link.price is not None else ''
        lines.append(f'     - {link.store}{price}: {link.url}')
    return '\n'.join(lines)


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
        review = (
            f'\n   Avaliações: {product.review_summary}'
            if product.review_summary
            else ''
        )
        block = (
            f'{i}. {product.name}'
            f'{f" ({product.brand})" if product.brand else ""} — {price}\n'
            f'   Por quê: {product.reason}{features}{review}\n'
            f'{_format_links(product.purchase_links)}'
        )
        lines.append(block)
    return '\n\n'.join(lines)


async def present_recommendations_node(state: ProductSearchState) -> dict[str, Any]:
    products = _products_from_state(state)[:TOP_PRODUCTS]

    # Anexa os links de compra de cada produto (concorrente).
    links_per_product = await asyncio.gather(
        *(fetch_purchase_links(p) for p in products)
    )
    enriched = [
        product.model_copy(update={'purchase_links': links})
        for product, links in zip(products, links_per_product)
    ]

    presentation = _format_recommendations(enriched)
    return {
        'messages': [AIMessage(presentation)],
        'products': [p.model_dump(mode='json') for p in enriched],
    }
