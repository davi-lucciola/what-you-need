from typing import Any

from langchain_core.language_models import BaseChatModel
from pytest_mock import MockerFixture


def patch_structured_llm(
    mocker: MockerFixture,
    module: Any,
    *,
    structured_return: Any | None = None,
):
    """Mocka `get_llm` de um módulo de agente.

    Retorna (llm, ainvoke, with_structured_output):
      - llm: AsyncMock(BaseChatModel) — use `llm.ainvoke` para chamadas planas.
      - ainvoke: AsyncMock do `with_structured_output(...).ainvoke`.
      - with_structured_output: mock de `llm.with_structured_output`.
    """
    structured_llm = mocker.AsyncMock()
    ainvoke = mocker.patch.object(
        structured_llm, 'ainvoke', return_value=structured_return
    )
    llm = mocker.AsyncMock(BaseChatModel)
    with_structured_output = mocker.patch.object(
        llm, 'with_structured_output', return_value=structured_llm
    )
    mocker.patch.object(module, 'get_llm', return_value=llm)
    return llm, ainvoke, with_structured_output
