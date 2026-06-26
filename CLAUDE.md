# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A LangGraph-based conversational agent (in Brazilian Portuguese) that helps users find the best cost-benefit product for their budget and need. It uses a supervisor/router pattern to dispatch a conversation to specialized sub-agents, with Tavily for web search.

## Commands

Tasks are defined in `pyproject.toml` under `[tool.taskipy.tasks]` and run via `uv`:

```bash
uv run task main        # Run the agent once via main.py (one-shot ainvoke)
uv run task langsmith   # Run LangGraph dev server (langgraph dev) with hot reload + LangSmith Studio
uv sync                 # Install dependencies (including the dev group)
uv run ruff check .     # Lint
uv run ruff format .    # Format
uv run pyright          # Type-check
```

Both tasks load env vars from `.env` (copy `.env.example` to `.env`). There is no test suite yet.

## Architecture

The compiled graph is built in `app/agents/__init__.py:build_agent` and is the entrypoint exported to both `main.py` and `langgraph.json` (graph id `grafo`). Flow:

```
START → supervisor → (conditional: guide | search) → END
```

- **Supervisor** (`app/agents/supervisor.py`): an LLM with `with_structured_output(Router)` that reads the message history and `AGENTS_DESCRIPTION`, then writes the chosen agent key into `state['next']`. It does not produce user-facing messages — only routing.
- **Guide** (`app/agents/guide.py`): the welcome/reception agent. Explains the bot and identifies intent; explicitly must NOT collect budget/requirements or run searches (that's the search agent's job). Behavior is driven entirely by `GUIDE_SYSTEM_PROMPT`.
- **Search** (`app/agents/search.py`): currently a stub returning a "not available" message. This is where the product-search flow (budget, requirements, web search) is meant to be built out.

### Key conventions

- **State** (`app/agents/states.py`): `ChatState` extends LangChain's `AgentState` (so it carries `messages`) and adds `next`, a custom `Annotated` reducer (`take_latest_nonempty`) that keeps the latest non-empty router decision. `ProductSearch` is defined for the eventual search flow.
- **Two enums** in `app/agents/constants.py`: `AllowedAgents` (routable targets: guide, search) vs `Agents` (all nodes, including supervisor). The supervisor can only route to `AllowedAgents`; the conditional edges and `END` edge are built from these sets. Keep them in sync when adding an agent.
- **LLM** (`app/llm.py`): `get_llm()` is an `lru_cache`'d `init_chat_model(settings.AGENT_CHAT_MODEL)`. The model is provider-prefixed (e.g. `openai:gpt-4.1-nano`, `google_genai:gemini-2.5-flash`) and swapped purely via env, so don't hardcode providers.
- **Tavily** (`app/tavily.py`, `app/agents/tools.py`): `web_search` is a LangChain `@tool` wrapping the cached `AsyncTavilyClient`. The `TavilySearchResponse` TypedDict mirrors the Tavily API shape. Note this tool is not yet wired into the search agent.
- **Config** (`app/config.py`): `Settings` is `pydantic-settings` reading from env. Calls use `Settings()  # type: ignore` because fields are populated from the environment, not constructor args.

### Conventions to follow

- All user-facing text and prompts are in Brazilian Portuguese.
- Ruff format uses single quotes, space indentation, LF endings. The pydocstyle (`D`) rules and several others are ignored — see `[tool.ruff.lint]`.
- Agents are `async` node functions taking `ChatState` and returning a partial state dict.
