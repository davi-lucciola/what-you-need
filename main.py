import asyncio
import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich import print

from app.agents import build_agent


def _render(result: dict[str, Any], last_id: str | None) -> str | None:
    """Mostra a última mensagem do agente, se ainda não foi exibida.

    O checkpointer devolve o histórico acumulado a cada turno; renderizar só a
    última AIMessage (deduplicada por id) evita reimprimir mensagens anteriores.
    """
    messages = result.get('messages', [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    if last_ai is None or last_ai.id == last_id:
        return last_id
    print(f'[bold green]Assistente:[/bold green] {last_ai.content}')
    return last_ai.id


def _pending_interrupt(result: dict[str, Any]) -> Any:
    """Retorna o payload do interrupt pendente, ou None."""
    interrupts = result.get('__interrupt__')

    if interrupts:
        return interrupts[0].value

    return None


async def main():
    # MemorySaver é em memória; para persistência real, troque por um saver
    # Sqlite/Postgres — build_agent aceita qualquer BaseCheckpointSaver.
    agent = build_agent(MemorySaver())
    config: RunnableConfig = {'configurable': {'thread_id': str(uuid.uuid4())}}

    print(
        '[bold]Assistente de busca de produtos[/bold] (digite "sair" para encerrar)\n'
    )

    result: dict[str, Any] = {}
    last_rendered_id: str | None = None

    while True:
        interrupt_payload = _pending_interrupt(result)

        if interrupt_payload is not None:
            # O nó pausou esperando input (coleta de requisitos ou escolha do produto).
            if isinstance(interrupt_payload, dict):
                payload = cast(dict[str, Any], interrupt_payload)
                if payload.get('message'):
                    print(f'[bold green]Assistente:[/bold green] {payload["message"]}')
                question = payload.get('question', '')
                print(f'[bold yellow]{question}[/bold yellow]')

            answer = input('Você: ')

            if answer.strip().lower() == 'sair':
                break

            result = await agent.ainvoke(Command(resume=answer), config)
            if _pending_interrupt(result) is None:
                last_rendered_id = _render(result, last_rendered_id)
            continue

        # Sem interrupt: turno concluído, aguarda nova mensagem do usuário.
        user_input = input('Você: ')

        if user_input.strip().lower() == 'sair':
            break

        result = await agent.ainvoke(
            {'messages': [HumanMessage(user_input)], 'next': ''}, config
        )

        if _pending_interrupt(result) is None:
            last_rendered_id = _render(result, last_rendered_id)

    print(await agent.aget_state(config=config))


if __name__ == '__main__':
    asyncio.run(main())
