# web-search-tool

## 🔭 Overview

A conversational agent built with **LangGraph** (in Brazilian Portuguese) that helps
the user find the product with the **best cost-benefit** for their budget and need.

It follows a **supervisor/router** pattern: a supervisor reads the conversation and
dispatches the flow to specialized sub-agents. Web search is powered by **Tavily**.

## 🏗️ Architecture

The main graph is compiled in `app/agents/__init__.py` and exposed as the entrypoint
`app.agents:make_graph` (id `grafo` in `langgraph.json`), as well as being exported to
`main.py`.

Main graph flow:

```
START → supervisor → (conditional: guide | products) → END
```

### Nodes and sub-agents

- **Supervisor** (`app/agents/supervisor/`): an LLM with
  `with_structured_output(Router)` that reads the message history and writes the chosen
  agent into `state['next']`. It **only routes** — it does not produce user-facing
  messages.
- **Guide** (`app/agents/guide/`): the welcome/reception agent. Explains the bot and
  identifies the user's intent. It does **not** collect budget/requirements or run
  searches (that is the products agent's responsibility).
- **Products** (`app/agents/products/`): compiles a subgraph with human-in-the-loop
  steps via `interrupt()`:

  ```
  collect_requirements → search_products → validate_products → present_recommendations → search_purchase_links
  ```

  - `tools.py` runs the Tavily searches.
  - `nodes.py` holds the subgraph steps and the routers.

### Package convention

Each agent is a package under `app/agents/` with a uniform layout:

- `prompt.py` — prompts.
- `schemas.py` — pydantic structured-output models.
- `agent.py` — the `build_*_agent` function.
- The **products** subgraph also has `state.py`, `tools.py` and `nodes.py`.

Each package's `__init__.py` exposes a `build_*_node()` that is passed to
`builder.add_node(...)`.

### Shared state

`ChatState` (in `app/agents/states.py`) extends LangChain's `AgentState` — so it carries
`messages` — and adds the `next` field, with a custom `take_latest_nonempty` reducer that
keeps the latest non-empty routing decision.

## 🧰 Stack

- **Python 3.13**
- **LangGraph / LangChain**
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

### Environment variables

| Variable | Description |
| --- | --- |
| `AGENT_CHAT_MODEL` | Agent model, **provider-prefixed** (e.g. `openai:gpt-4.1-nano` or `google_genai:gemini-2.5-flash`). |
| `OPENAI_API_KEY` | OpenAI key (if using an `openai:` model). |
| `GOOGLE_API_KEY` | Google key (if using a `google_genai:` model). |
| `TAVILY_API_KEY` | Tavily API key for web search. |
| `LANGSMITH_TRACING` | Enables LangSmith tracing (`true`/`false`). |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint (e.g. `https://api.smith.langchain.com`). |
| `LANGSMITH_API_KEY` | LangSmith API key. |
| `LANGSMITH_PROJECT` | LangSmith project name. |

## ▶️ How to run

Tasks live under `[tool.taskipy.tasks]` in `pyproject.toml` and run via `uv`:

```bash
uv run task main      # Run the agent once via main.py (one-shot ainvoke)
uv run task lint      # Type-check + lint (pyright && ruff check)
uv run task format    # Format the code (ruff format)
uv run task test      # Run the tests with coverage (pytest --cov)
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
`reportTypedDictNotRequiredAccess` in `app/agents/__init__.py`: `next` is declared
`NotRequired[NextNode]` on `ChatState`, so subscripting `state["next"]` can raise at
runtime if the key is absent. The supervisor always writes `next` before the conditional
edge runs, so the fix uses `state.get("next", "")` — type-safe and behavior-preserving
(the default mirrors the `take_latest_nonempty` reducer's own `current or ""` fallback).
