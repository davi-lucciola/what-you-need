import json
from typing import Any, AsyncGenerator, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

from app.api.deps import Agent
from app.core.agents.constants import Nodes


def _interrupt_value(mode: str, payload: Any) -> Any:
    """Extrai o payload do interrupt de um item do astream, ou None.

    Espelha o `_pending_interrupt` do `main.py`, mas lendo do stream em vez do
    resultado final: no `stream_mode='updates'` o chunk de pausa vem com a chave
    `__interrupt__`.
    """
    if mode == 'updates' and isinstance(payload, dict) and '__interrupt__' in payload:
        interrupts = payload['__interrupt__']

        if interrupts:
            return interrupts[0].value

    return None


async def _final_message(agent: Agent, config: RunnableConfig) -> str | None:
    """Texto autoritativo do turno: a última AIMessage com conteúdo.

    Mesma regra do `_render` do `main.py`. Garante a entrega de mensagens
    montadas sem LLM (ex.: links de compra), que não geram tokens no stream.
    """
    snapshot = await agent.aget_state(config)
    messages = snapshot.values.get('messages', [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    return last_ai.text if last_ai is not None else None


async def event_stream(
    agent: Agent, graph_input: Any, config: RunnableConfig
) -> AsyncGenerator[dict[str, str]]:
    interrupted = False

    async for mode, payload in agent.astream(
        graph_input, config, stream_mode=['messages', 'updates']
    ):
        interrupt = _interrupt_value(mode, payload)
        if interrupt is not None:
            # Turno pausou esperando input: emite a pergunta e encerra o stream.
            interrupted = True
            yield {'event': 'interrupt', 'data': json.dumps(interrupt)}
            break

        if mode == 'messages':
            chunk = payload[0]
            metadata = cast(dict[str, Any], payload[1])
            # `token` = texto incremental (preview ao vivo). Pula o supervisor,
            # cujo structured-output de roteamento sairia como JSON cru.
            if (
                isinstance(chunk, AIMessageChunk)
                and chunk.text
                and metadata.get('langgraph_node') != Nodes.SUPERVISOR
            ):
                yield {'event': 'token', 'data': chunk.text}

    if not interrupted:
        # `message` = texto final autoritativo do turno concluído (o cliente pode
        # substituir os tokens incrementais por ele).
        text = await _final_message(agent, config)

        if text:
            yield {'event': 'message', 'data': text}

    yield {'event': 'done', 'data': ''}
