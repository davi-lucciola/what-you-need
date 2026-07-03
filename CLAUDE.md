# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`what-you-need` is a LangGraph-based conversational agent (in Brazilian Portuguese) that helps users find the best cost-benefit product for their budget and need. It uses a supervisor/router pattern to dispatch a conversation to specialized sub-agents, with Tavily for web search. The agent is exposed as a **FastAPI service that streams responses over SSE**, and conversation state is persisted in **PostgreSQL** via a LangGraph checkpointer.

## Commands

Tasks are defined in `pyproject.toml` under `[tool.taskipy.tasks]` and run via `uv`:

```bash
uv run task dev         # Run the API with hot reload (uvicorn app.main:app --reload)
uv run task start       # Run the API (uvicorn on 0.0.0.0:8000)
uv run task langsmith   # Run LangGraph dev server (langgraph dev) with hot reload + LangSmith Studio
uv sync                 # Install dependencies (including the dev group)
uv run task lint        # Lint + type-check (pyright && ruff check)
uv run task format      # Format (ruff format)
uv run task test        # Run the tests with coverage (pytest --cov)
```

Tasks load env vars from `.env` (copy `.env.example` to `.env`). Postgres (the checkpointer DB) is provided by `docker compose up -d`. Tests live under `tests/unit/`, mirroring the `app/` package structure.

## Architecture

The code is organized in layers under `app/`: **`api/`** (FastAPI HTTP layer), **`core/`** (domain logic — agents and services), and **`infra/`** (external integrations — LLM and Tavily).

- **Web entrypoint**: `app.main:app`, built by `create_app()` in `app/__init__.py`. The FastAPI lifespan opens the Postgres checkpointer pool and compiles the agent onto `app.state.agent`.
- **Graph entrypoint**: the compiled graph is built in `app/core/agents/__init__.py:build_agent` (also exposed via `make_graph` to `langgraph.json`, graph id `grafo`). Flow:

```
START → supervisor → (conditional: guide | products) → END
```

Each agent is its own package under `app/core/agents/` exposing a `build_*_node()` (in the package `__init__.py`) that `build_agent` passes to `builder.add_node(...)`. The package layout is uniform: `prompt.py` (prompts), `schemas.py` (structured-output pydantic models), `agent.py` (the `build_*_agent` function), and — for the products subgraph — `state.py`, `tools.py`, and `nodes.py`.

- **Supervisor** (`app/core/agents/supervisor/`): `build_supervisor_agent` is an LLM with `with_structured_output(Router)` that reads the message history and `AGENTS_DESCRIPTION`, then writes the chosen agent key into `state['next']`. It does not produce user-facing messages — only routing.
- **Guide** (`app/core/agents/guide/`): the welcome/reception agent (`build_guide_agent`). Explains the bot and identifies intent; explicitly must NOT collect budget/requirements or run searches (that's the search agent's job). Behavior is driven entirely by `GUIDE_SYSTEM_PROMPT`.
- **Products** (`app/core/agents/products/`): `build_product_search_agent` compiles a subgraph (`collect_requirements → search_products → validate_products → present_recommendations → search_purchase_links`) with `interrupt()`-based human-in-the-loop steps. `tools.py` runs the Tavily searches; `nodes.py` holds the subgraph steps + routers.

### API and services layer

- **API** (`app/api/`): the single endpoint is `POST /threads/{thread_id}/chat` (`routers/chat.py`), returning an `EventSourceResponse` (SSE). `schemas/chat.py` holds `ChatRequest` (`{ message }`); `deps.py` exposes `get_agent`, which reads `app.state.agent`. Before running, the router inspects the thread state: if a pending interrupt exists, the message is fed as `Command(resume=message)`; otherwise as a fresh `HumanMessage`.
- **Service** (`app/core/services/chat.py`): `event_stream` runs the graph with `astream(stream_mode=['messages', 'updates'])` and yields SSE events: `token` (incremental live text, supervisor node skipped), `message` (final authoritative text — the last non-empty `AIMessage`, also covers non-LLM replies like purchase links), `interrupt` (human-in-the-loop pause payload, ends the turn early), and always `done` last.
- **Checkpointer** (`app/core/agents/checkpointer.py`): `get_postgres_checkpointer_pool` yields an `AsyncPostgresSaver` over an `AsyncConnectionPool`, set up in the FastAPI lifespan. The connection string comes from `CHECKPOINTER_DATABASE_URL`.

### Naming convention

- **`*_agent`** = a *routable* sub-agent the supervisor dispatches to; **`*_node`** = a step inside the main graph or a subgraph. The supervisor follows the package pattern but is a node, not a routable agent.
- **`build_*_node()`** lives in each package `__init__.py` and is the factory passed to `builder.add_node(...)` — used for **main-graph nodes only**. Internal subgraph steps are added as plain `*_node` functions. `build_guide_node()` / `build_supervisor_node()` return the `build_*_agent` function reference (single async nodes); `build_product_search_node()` returns `build_product_search_agent()` called (it compiles a subgraph).
- **`state.py` vs `schemas.py`**: `state.py` holds graph state (`AgentState`/`TypedDict`); `schemas.py` holds the pydantic models passed to `with_structured_output(...)`.

### Key conventions

- **State** (`app/core/agents/states.py`): shared `ChatState` extends LangChain's `AgentState` (so it carries `messages`) and adds `next`, a custom `Annotated` reducer (`take_latest_nonempty`) that keeps the latest non-empty router decision. The product-search state (`ProductSearchState`, `RequirementsDict`, `ProductDict`) lives in `app/core/agents/products/state.py`.
- **Two enums** in `app/core/agents/constants.py`: `Agents` (routable targets: guide, products) vs `Nodes` (all nodes, including supervisor). The supervisor can only route to `Agents`; the conditional edges and `END` edge are built from these sets. Keep them in sync when adding an agent.
- **LLM** (`app/infra/llm.py`): `get_llm()` is an `lru_cache`'d `init_chat_model(settings.AGENT_CHAT_MODEL)`. The model is provider-prefixed (e.g. `openai:gpt-4.1-nano`, `google_genai:gemini-2.5-flash`) and swapped purely via env, so don't hardcode providers.
- **Tavily** (`app/infra/tavily.py`, `app/core/agents/products/tools.py`): the cached `AsyncTavilyClient` (`get_tavily_client`) backs `search_candidates` / `deep_search_purchase_links`. The `TavilyResult` TypedDict mirrors the Tavily API shape.
- **Config** (`app/config.py`): `Settings` is `pydantic-settings` reading from env (`AGENT_CHAT_MODEL`, `TAVILY_API_KEY`, `CHECKPOINTER_DATABASE_URL`). Calls use `Settings()  # type: ignore` because fields are populated from the environment, not constructor args.

### Conventions to follow

- All user-facing text and prompts are in Brazilian Portuguese.
- Ruff format uses single quotes, space indentation, LF endings. The pydocstyle (`D`) rules and several others are ignored — see `[tool.ruff.lint]`.
- Agents are `async` node functions taking `ChatState` and returning a partial state dict.
