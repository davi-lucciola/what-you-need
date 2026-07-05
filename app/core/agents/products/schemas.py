import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field, field_validator


def parse_brl_price(value: object) -> Decimal | None:
    """Converte um preço informado pelo LLM (ex. "R$3.950,00") em Decimal.

    Aceita formato brasileiro (ponto de milhar e vírgula decimal), símbolo de moeda e
    valores já numéricos. Retorna None quando o preço não estiver disponível.
    """
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))

    text = str(value).strip()
    if not text:
        return None

    # Mantém apenas dígitos e separadores, removendo "R$", espaços, etc.
    text = re.sub(r'[^\d.,]', '', text)
    if not text:
        return None

    if ',' in text and '.' in text:
        # Formato brasileiro: ponto de milhar, vírgula decimal -> 3.950,00 -> 3950.00
        text = text.replace('.', '').replace(',', '.')
    elif ',' in text:
        text = text.replace(',', '.')

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


class PurchaseLink(BaseModel):
    store: str = Field(
        description='Loja do link, ex. "Amazon", "Mercado Livre", "Shopee".'
    )
    url: str = Field(description='URL direta para comprar o produto na loja.')
    price: float | None = Field(
        default=None, description='Preço anunciado em reais (BRL), se disponível.'
    )

    @field_validator('price', mode='before')
    @classmethod
    def _coerce_price(cls, value: object) -> float | None:
        parsed = parse_brl_price(value)
        return float(parsed) if parsed is not None else None


class Product(BaseModel):
    name: str = Field(
        description='Nome/modelo do produto, ex. "Samsung Galaxy A17 5G".'
    )
    brand: str | None = Field(default=None, description='Marca/fabricante do produto.')
    estimated_price: Decimal | None = Field(
        default=None,
        description=(
            'Preço aproximado em reais (BRL) como número decimal simples, sem símbolo '
            'de moeda nem separador de milhar (ex.: 3950.00). Use null se desconhecido.'
        ),
    )
    reason: str = Field(
        description=(
            'Por que é uma boa opção para a necessidade e o orçamento informados.'
        )
    )
    key_features: list[str] = Field(
        default_factory=list,
        description='Principais características do produto (ex. RAM, câmera, bateria).',
    )
    unmet_requirements: list[str] = Field(
        default_factory=list,
        description=(
            'Requisitos do usuário que este produto NÃO atende '
            '(ex. ["à prova d\'água"]). Preencha quando não houver opção que cumpra '
            'todos os requisitos, para a recomendação nunca ficar vazia. Deixe [] se '
            'o produto atende a tudo.'
        ),
    )
    # Preenchidos pelo sistema (não pelo extrator): review/disponibilidade na validação,
    # links na apresentação. Deixe vazios/None na extração.
    review_summary: str | None = Field(
        default=None,
        description='NÃO preencher na extração — o sistema preenche na validação.',
    )
    available: bool | None = Field(
        default=None,
        description='NÃO preencher na extração — o sistema confirma na validação.',
    )
    purchase_links: list[PurchaseLink] = Field(
        default_factory=list,
        description='NÃO preencher na extração — o sistema preenche na apresentação.',
    )

    @field_validator('unmet_requirements', mode='before')
    @classmethod
    def _coerce_unmet_none_to_empty(cls, value: object) -> object:
        return value if value is not None else []

    @field_validator('estimated_price', mode='before')
    @classmethod
    def _coerce_estimated_price(cls, value: object) -> Decimal | None:
        return parse_brl_price(value)


class ProductRecommendations(BaseModel):
    products: list[Product] = Field(
        description=(
            'Lista com os 5 produtos mais adequados, ordenados por custo-benefício.'
        )
    )


class ReviewVerdict(BaseModel):
    well_rated: bool = Field(
        description='True se o produto é, no geral, bem avaliado na internet.'
    )
    summary: str = Field(
        description='Resumo curto das avaliações online (nota geral e pontos citados).'
    )


class PurchaseLinks(BaseModel):
    links: list[PurchaseLink] = Field(
        description='Até 2 links de compra, priorizando Amazon, Mercado Livre e Shopee.'
    )


class ListingVerdict(BaseModel):
    available: bool = Field(
        description=(
            'True se o anúncio está no ar e o produto disponível para compra '
            '(não "indisponível", "esgotado" nem página de erro/404).'
        )
    )
    price: float | None = Field(
        default=None,
        description='Preço anunciado em reais (BRL) extraído da página, se disponível.',
    )

    @field_validator('price', mode='before')
    @classmethod
    def _coerce_price(cls, value: object) -> float | None:
        parsed = parse_brl_price(value)
        return float(parsed) if parsed is not None else None
