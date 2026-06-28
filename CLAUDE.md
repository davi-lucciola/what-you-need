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
uv run task lint        # Lint + type-check (pyright && ruff check)
uv run task format      # Format (ruff format)
```

Both tasks load env vars from `.env` (copy `.env.example` to `.env`). There is no test suite yet.

## Architecture

The compiled graph is built in `app/agents/__init__.py:build_agent` and is the entrypoint exported to both `main.py` and `langgraph.json` (graph id `grafo`). Flow:

```
START → supervisor → (conditional: guide | product-search) → END
```

Each agent is its own package under `app/agents/` exposing a `build_*_node()` (in the package `__init__.py`) that `build_agent` passes to `builder.add_node(...)`. The package layout is uniform: `prompt.py` (prompts), `schemas.py` (structured-output pydantic models), `agent.py` (the `build_*_agent` function), and — for the product-search subgraph — `state.py`, `tools.py`, and `nodes.py`.

- **Supervisor** (`app/agents/supervisor/`): `build_supervisor_agent` is an LLM with `with_structured_output(Router)` that reads the message history and `AGENTS_DESCRIPTION`, then writes the chosen agent key into `state['next']`. It does not produce user-facing messages — only routing.
- **Guide** (`app/agents/guide/`): the welcome/reception agent (`build_guide_agent`). Explains the bot and identifies intent; explicitly must NOT collect budget/requirements or run searches (that's the search agent's job). Behavior is driven entirely by `GUIDE_SYSTEM_PROMPT`.
- **Product search** (`app/agents/product_search/`): `build_product_search_agent` compiles a subgraph (`collect_requirements → search_products → validate_products → present_recommendations → search_purchase_links`) with `interrupt()`-based human-in-the-loop steps. `tools.py` runs the Tavily searches; `nodes.py` holds the subgraph steps + routers.

### Naming convention

- **`*_agent`** = a *routable* sub-agent the supervisor dispatches to; **`*_node`** = a step inside the main graph or a subgraph. The supervisor follows the package pattern but is a node, not a routable agent.
- **`build_*_node()`** lives in each package `__init__.py` and is the factory passed to `builder.add_node(...)` — used for **main-graph nodes only**. Internal subgraph steps are added as plain `*_node` functions. `build_guide_node()` / `build_supervisor_node()` return the `build_*_agent` function reference (single async nodes); `build_product_search_node()` returns `build_product_search_agent()` called (it compiles a subgraph).
- **`state.py` vs `schemas.py`**: `state.py` holds graph state (`AgentState`/`TypedDict`); `schemas.py` holds the pydantic models passed to `with_structured_output(...)`.

### Key conventions

- **State** (`app/agents/states.py`): shared `ChatState` extends LangChain's `AgentState` (so it carries `messages`) and adds `next`, a custom `Annotated` reducer (`take_latest_nonempty`) that keeps the latest non-empty router decision. The product-search state (`ProductSearchState`, `RequirementsDict`, `ProductDict`) lives in `app/agents/product_search/state.py`.
- **Two enums** in `app/agents/constants.py`: `Agents` (routable targets: guide, product-search) vs `Nodes` (all nodes, including supervisor). The supervisor can only route to `Agents`; the conditional edges and `END` edge are built from these sets. Keep them in sync when adding an agent.
- **LLM** (`app/llm.py`): `get_llm()` is an `lru_cache`'d `init_chat_model(settings.AGENT_CHAT_MODEL)`. The model is provider-prefixed (e.g. `openai:gpt-4.1-nano`, `google_genai:gemini-2.5-flash`) and swapped purely via env, so don't hardcode providers.
- **Tavily** (`app/tavily.py`, `app/agents/product_search/tools.py`): the cached `AsyncTavilyClient` (`get_tavily_client`) backs `search_candidates` / `deep_search_purchase_links`. The `TavilyResult` TypedDict mirrors the Tavily API shape.
- **Config** (`app/config.py`): `Settings` is `pydantic-settings` reading from env. Calls use `Settings()  # type: ignore` because fields are populated from the environment, not constructor args.

### Conventions to follow

- All user-facing text and prompts are in Brazilian Portuguese.
- Ruff format uses single quotes, space indentation, LF endings. The pydocstyle (`D`) rules and several others are ignored — see `[tool.ruff.lint]`.
- Agents are `async` node functions taking `ChatState` and returning a partial state dict.
