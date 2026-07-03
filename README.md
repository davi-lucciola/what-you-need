# What you Need?

## 🔭 Overview

A conversational agent built with **LangGraph** (in Brazilian Portuguese) that helps the
user find the product with the **best cost-benefit** for their budget and need.

It follows a **supervisor/router** pattern: a supervisor reads the conversation and
dispatches the flow to specialized sub-agents. Web search is powered by **Tavily**.

The agent is exposed as a **FastAPI web service** that **streams responses over SSE**
(Server-Sent Events), and conversation state is persisted in **PostgreSQL** through a
LangGraph checkpointer, so each thread survives across requests.

## 🏗️ Architecture

The project is organized in layers under `app/`:

```
app/
  __init__.py         # create_app() FastAPI factory + lifespan (builds the Postgres
                      #   checkpointer pool and compiles the agent onto app.state.agent)
  main.py             # app = create_app()   (uvicorn entrypoint: app.main:app)
  config.py           # Settings (env): AGENT_CHAT_MODEL, TAVILY_API_KEY,
                      #   CHECKPOINTER_DATABASE_URL
  api/                # HTTP layer (FastAPI)
    routers/chat.py   #   POST /threads/{thread_id}/chat -> SSE stream
    schemas/chat.py   #   ChatRequest { message }
    deps.py           #   get_agent (reads app.state.agent)
  core/               # domain logic
    agents/           #   the LangGraph graph (supervisor / guide / products)
      checkpointer.py #   AsyncPostgresSaver over an async connection pool
    services/
      chat.py         #   event_stream(): turns the graph run into SSE events
  infra/              # external integrations
    llm.py            #   get_llm()
    tavily.py         #   get_tavily_client() + Tavily response types
```

- **Web entrypoint**: `app.main:app`, built by `create_app()` in `app/__init__.py`. The
  FastAPI lifespan opens the Postgres checkpointer pool and compiles the agent onto
  `app.state.agent`.
- **Graph entrypoint**: the graph is built in `app/core/agents/__init__.py`
  (`build_agent` / `make_graph`) and is also exposed to LangGraph Studio via
  `langgraph.json` (graph id `grafo`).

Main graph flow:

```
START → supervisor → (conditional: guide | products) → END
```

### Nodes and sub-agents

- **Supervisor** (`app/core/agents/supervisor/`): an LLM with
  `with_structured_output(Router)` that reads the message history and writes the chosen
  agent into `state['next']`. It **only routes** — it does not produce user-facing
  messages.
- **Guide** (`app/core/agents/guide/`): the welcome/reception agent. Explains the bot and
  identifies the user's intent. It does **not** collect budget/requirements or run
  searches (that is the products agent's responsibility).
- **Products** (`app/core/agents/products/`): compiles a subgraph with human-in-the-loop
  steps via `interrupt()`:

  ```
  collect_requirements → search_products → validate_products → present_recommendations → search_purchase_links
  ```

  - `tools.py` runs the Tavily searches.
  - `nodes.py` holds the subgraph steps and the routers.

### API and SSE streaming

A single endpoint drives a conversation thread:

```
POST /threads/{thread_id}/chat
Content-Type: application/json

{ "message": "..." }
```

The response is an **SSE stream** (`text/event-stream`) produced by
`event_stream` in `app/core/services/chat.py`, which runs the graph with
`astream(stream_mode=['messages', 'updates'])` and emits these events:

| Event | Meaning |
| --- | --- |
| `token` | Incremental live text of the assistant's reply (the supervisor node is skipped, since its structured-output routing would leak as raw JSON). |
| `message` | The final, authoritative text of the turn (the last non-empty `AIMessage`). Clients can replace the streamed tokens with it. It also delivers replies assembled without the LLM (e.g. purchase links), which produce no stream tokens. |
| `interrupt` | The turn paused waiting for user input (human-in-the-loop). Carries the question payload and ends the stream early. |
| `done` | Always emitted last to signal the turn is complete. |

**Human-in-the-loop resume** (`app/api/routers/chat.py`): before running, the router
checks the thread state — if there is a pending interrupt, the incoming message is fed as
`Command(resume=message)`; otherwise it starts a fresh turn with a `HumanMessage`.

### Persistence (Postgres checkpointer)

Conversation state is persisted with `AsyncPostgresSaver` over an
`AsyncConnectionPool` (`app/core/agents/checkpointer.py`), set up in the FastAPI lifespan.
A local Postgres is provided by `docker-compose.yml` (`postgres:17-alpine`), with schema
bootstrap SQL under `docker/initdb/`. The connection string comes from
`CHECKPOINTER_DATABASE_URL`.

### Shared state

`ChatState` (in `app/core/agents/states.py`) extends LangChain's `AgentState` — so it
carries `messages` — and adds the `next` field, with a custom `take_latest_nonempty`
reducer that keeps the latest non-empty routing decision.

## 🧰 Stack

- **Python 3.13**
- **FastAPI** + **Uvicorn** (web server)
- **sse-starlette** (SSE streaming)
- **LangGraph / LangChain**
- **PostgreSQL** via **langgraph-checkpoint-postgres** / **psycopg** (state persistence)
- **Tavily** (web search)
- **pydantic-settings** (configuration via environment)
- **uv** (package manager / runner)
- **Ruff** (lint + format)
- **Pyright** (type checking, strict mode)
- **Taskipy** (tasks)

## ⚙️ Setup

1. Install the dependencies (includes the dev group):

   ```bash
   uv sync
   ```

2. Create the `.env` file from the example and fill in the keys:

   ```bash
   cp .env.example .env
   ```

3. Start PostgreSQL (checkpointer database):

   ```bash
   docker compose up -d
   ```

### Environment variables

| Variable | Description |
| --- | --- |
| `AGENT_CHAT_MODEL` | Agent model, **provider-prefixed** (e.g. `openai:gpt-4.1-nano` or `google_genai:gemini-2.5-flash`). |
| `OPENAI_API_KEY` | OpenAI key (if using an `openai:` model). |
| `GOOGLE_API_KEY` | Google key (if using a `google_genai:` model). |
| `TAVILY_API_KEY` | Tavily API key for web search. |
| `CHECKPOINTER_DATABASE_URL` | PostgreSQL URL for the LangGraph checkpointer (e.g. `postgresql://postgres:postgres@localhost:5432/checkpointer`). |
| `LANGSMITH_TRACING` | Enables LangSmith tracing (`true`/`false`). |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint (e.g. `https://api.smith.langchain.com`). |
| `LANGSMITH_API_KEY` | LangSmith API key. |
| `LANGSMITH_PROJECT` | LangSmith project name. |

## ▶️ How to run

Tasks live under `[tool.taskipy.tasks]` in `pyproject.toml` and run via `uv`:

```bash
uv run task dev       # Run the API with hot reload (uvicorn app.main:app --reload)
uv run task start     # Run the API (uvicorn on 0.0.0.0:8000)
uv run task lint      # Type-check + lint (pyright && ruff check)
uv run task format    # Format the code (ruff format)
uv run task test      # Run the tests with coverage (pytest --cov)
```

Once the server is up, talk to a thread over SSE:

```bash
curl -N -X POST http://localhost:8000/threads/my-thread/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Preciso de um notebook até 4 mil reais"}'
```

## 🧪 LangGraph Studio + LangSmith

```bash
uv run task langsmith
```

This command runs `langgraph dev --config ./langgraph.json` with **hot reload** and opens
**LangGraph Studio**. **LangSmith** tracing is enabled by the `LANGSMITH_*` variables in
`.env`.

## 🔎 Type checking

We use **Pyright** (the engine behind Pylance) both in the editor and in the
`uv run task lint` CLI. A single engine keeps the editor and CLI in agreement. The config
lives in `[tool.pyright]` in `pyproject.toml`, and Pylance reads it automatically.

The mode is `typeCheckingMode = "strict"`, with **three rules disabled**:

- `reportUnknownMemberType`
- `reportUnknownVariableType`
- `reportUnknownArgumentType`

They are disabled because they fire on `Unknown` types that **leak out of the incomplete
LangChain/LangGraph stubs** — not out of our code. Keeping them on would produce dozens of
errors on every LLM call and bury the real ones. Our own signatures stay strict via
`reportUnknownParameterType` / `reportMissingParameterType`, which remain **on**.

Where the `Unknown`s come from:

- `BaseChatModel.with_structured_output()` is declared as
  `Runnable[..., dict[str, Any] | BaseModel]`; the `dict[str, Any]` arm is an `Any` leak,
  so any member access on the result becomes `Unknown`.
- `init_chat_model` can return the private `_ConfigurableModel` class, whose methods
  (`.ainvoke`, `.with_structured_output`) are loosely typed — which is why the result of
  `get_llm()` is flagged.
- Passing the `dict[str, Any] | BaseModel` result of `.ainvoke()` onward (e.g. into a
  partial-state dict like `{"next": router.next}`) propagates the `Any` to the call site.

### `StateGraph` + `TypedDict`

`StateGraph.__init__(state_schema: type[StateT])` takes a `TypedDict` (`ChatState`,
`ProductSearchState`). The old type checker (`ty`) rejected this with
`invalid-argument-type`, which motivated the old `# ty: ignore[...]` comments. **Pyright
models `type[SomeTypedDict]` correctly**, so those ignores were unnecessary and have been
removed. Reducer state — `Annotated[list[AnyMessage], add_messages]` and the custom
`take_latest_nonempty` — is also understood by Pyright, with no ignores needed.

### A real error strict mode caught

Strict mode is not only stub noise. Enabling it surfaced
`reportTypedDictNotRequiredAccess` in `app/core/agents/__init__.py`: `next` is declared
`NotRequired[NextNode]` on `ChatState`, so subscripting `state["next"]` can raise at
runtime if the key is absent. The supervisor always writes `next` before the conditional
edge runs, so the fix uses `state.get("next", "")` — type-safe and behavior-preserving
(the default mirrors the `take_latest_nonempty` reducer's own `current or ""` fallback).
