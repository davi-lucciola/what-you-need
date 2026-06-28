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

    @field_validator('estimated_price', mode='before')
    @classmethod
    def _coerce_estimated_price(cls, value: object) -> Decimal | None:
        return parse_brl_price(value)


class ProductRecommendations(BaseModel):
    products: list[Product] = Field(
        description=(
            'Lista com os 3 produtos mais adequados, ordenados por custo-benefício.'
        )
    )


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


class PurchaseLinks(BaseModel):
    links: list[PurchaseLink] = Field(
        description='Até 2 links de compra, priorizando Amazon, Mercado Livre e Shopee.'
    )


class Requirements(BaseModel):
    product_type: str | None = Field(
        default=None,
        description='Tipo/categoria do produto procurado, ex. "celular", "notebook".',
    )
    use_case: str | None = Field(
        default=None,
        description=(
            'Para que o usuário vai usar o produto, ex. "tirar fotos", "jogar".'
        ),
    )
    priorities: list[str] = Field(
        default_factory=list,
        description=(
            'Características mais importantes para o usuário, '
            'ex. ["câmera", "bateria"].'
        ),
    )
    brand_preferences: list[str] = Field(
        default_factory=list,
        description='Marcas preferidas ou a evitar informadas pelo usuário.',
    )
    must_haves: list[str] = Field(
        default_factory=list,
        description=(
            'Requisitos obrigatórios/inegociáveis, ex. ["5G", "à prova d\'água"].'
        ),
    )

    @field_validator('priorities', 'brand_preferences', 'must_haves', mode='before')
    @classmethod
    def _coerce_none_to_empty_list(cls, value: object) -> object:
        # O LLM costuma emitir null em vez de [] quando o campo está vazio.
        return value if value is not None else []

    @property
    def is_complete(self) -> bool:
        """Há dados suficientes para iniciar a busca.

        Requer tipo de produto + (uso ou prioridade).
        """
        return bool(self.product_type) and bool(self.use_case or self.priorities)


class CollectedInfo(BaseModel):
    """Modelo achatado de extração (modelos menores erram com schema aninhado)."""

    product_type: str | None = Field(
        default=None, description='Tipo/categoria do produto, ex. "celular".'
    )
    use_case: str | None = Field(
        default=None, description='Para que o usuário vai usar o produto.'
    )
    priorities: list[str] = Field(
        default_factory=list, description='Características mais importantes.'
    )
    brand_preferences: list[str] = Field(
        default_factory=list, description='Marcas preferidas ou a evitar.'
    )
    must_haves: list[str] = Field(
        default_factory=list, description='Requisitos obrigatórios.'
    )
    budget: float | None = Field(
        default=None,
        description=(
            'Orçamento máximo do usuário em reais (BRL), se informado. Null se não.'
        ),
    )

    @field_validator('priorities', 'brand_preferences', 'must_haves', mode='before')
    @classmethod
    def _coerce_none_to_empty_list(cls, value: object) -> object:
        return value if value is not None else []

    def to_requirements(self) -> Requirements:
        return Requirements(
            product_type=self.product_type,
            use_case=self.use_case,
            priorities=self.priorities,
            brand_preferences=self.brand_preferences,
            must_haves=self.must_haves,
        )


class ProductChoice(BaseModel):
    index: int = Field(
        description=(
            'Número (1 a N) do produto escolhido pelo usuário. Use 1 se ele '
            'pedir o mais recomendado ou não especificar claramente.'
        )
    )
