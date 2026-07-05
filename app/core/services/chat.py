from typing import Any, AsyncGenerator, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

from app.api.deps import Agent
from app.core.agents.constants import Nodes


async def _final_message(agent: Agent, config: RunnableConfig) -> str | None:
    """Texto autoritativo do turno: a última AIMessage com conteúdo.

    Garante a entrega de mensagens montadas sem LLM (ex.: apresentação com links),
    que não geram tokens no stream.
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
    # Turn-based: o grafo roda START→END sem pausas. Só emitimos os tokens
    # incrementais (preview ao vivo) e, no fim, a mensagem final autoritativa.
    async for chunk, metadata in agent.astream(
        graph_input, config, stream_mode='messages'
    ):
        meta = cast(dict[str, Any], metadata)
        # `token` = texto incremental. Pula o supervisor, cujo structured-output de
        # roteamento sairia como JSON cru.
        if (
            isinstance(chunk, AIMessageChunk)
            and chunk.text
            and meta.get('langgraph_node') != Nodes.SUPERVISOR
        ):
            yield {'event': 'token', 'data': chunk.text}

    # `message` = texto final autoritativo (o cliente pode substituir os tokens
    # incrementais por ele).
    text = await _final_message(agent, config)

    if text:
        yield {'event': 'message', 'data': text}

    yield {'event': 'done', 'data': ''}
