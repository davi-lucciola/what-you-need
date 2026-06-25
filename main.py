import asyncio

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import HumanMessage

from rich import print

from app.agent import build_agent


async def main():
    agent = build_agent()

    entry = InputAgentState(messages=[HumanMessage('Quanto é 10 + 2 e o resultado disso multiplicado por 2?')])
    response = await agent.ainvoke(entry)

    print(response)


if __name__ == '__main__':
    asyncio.run(main())
