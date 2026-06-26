import asyncio
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich import print

from app.agents import build_agent


def _render(result: dict) -> None:
    """Mostra a última mensagem do agente, se houver."""
    messages = result.get('messages') or []
    for message in messages:
        if isinstance(message, AIMessage) and message.content:
            print(f'[bold green]Assistente:[/bold green] {message.content}')


def _pending_interrupt(result: dict):
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

    result = {}

    while True:
        interrupt_payload = _pending_interrupt(result)

        if interrupt_payload is not None:
            # O nó pausou esperando input (coleta de requisitos ou escolha do produto).
            if isinstance(interrupt_payload, dict):
                if interrupt_payload.get('message'):
                    print(
                        f'[bold green]Assistente:[/bold green] '
                        f'{interrupt_payload["message"]}'
                    )
                print(
                    f'[bold yellow]{interrupt_payload.get("question", "")}[/bold yellow]'
                )

            answer = input('Você: ')

            if answer.strip().lower() == 'sair':
                break

            result = await agent.ainvoke(Command(resume=answer), config)
            continue

        # Sem interrupt: turno concluído, aguarda nova mensagem do usuário.
        user_input = input('Você: ')

        if user_input.strip().lower() == 'sair':
            break

        result = await agent.ainvoke(
            {'messages': [HumanMessage(user_input)], 'next': ''}, config
        )

        _render(result)

    print(await agent.aget_state(config=config))


if __name__ == '__main__':
    asyncio.run(main())
