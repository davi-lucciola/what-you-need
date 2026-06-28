# Type checking

We use **Pyright** (the engine behind Pylance) for both the editor and the CLI
`uv run task lint`. One engine → editor and CLI agree. Config lives in
`[tool.pyright]` in `pyproject.toml`; Pylance reads it automatically.

Mode: `typeCheckingMode = "strict"` with three rules disabled. They are disabled
because they fire on `Unknown` types that **leak out of LangChain/LangGraph's
incomplete stubs**, not out of our code. Keeping them on produces dozens of errors
on every LLM call and buries the real ones. Our own signatures stay strict via
`reportUnknownParameterType` / `reportMissingParameterType`, which remain ON.

## Disabled rules

### `reportUnknownMemberType`

Fires when you access an attribute/method on a value whose type is partially `Unknown`.

- **Class `langchain_core.language_models.chat_models.BaseChatModel` — method
  `.with_structured_output()`**: declared as
  `-> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]`. The `dict[str, Any]`
  arm is a raw `Any`-leak; any member access on the result is `Unknown`.
- **Class `langchain.chat_models.base._ConfigurableModel`** (private, underscore):
  `init_chat_model` can return this, and its methods (`.ainvoke`,
  `.with_structured_output`) are loosely typed, so member access on a `get_llm()`
  result is flagged.
- Concrete site: `app/llm.py` `get_llm()` and every supervisor/guide call that does
  `llm.with_structured_output(Router).ainvoke(...)`.

### `reportUnknownVariableType`

Fires when a local variable's *inferred* type contains `Unknown`.

- **Function `langchain.chat_models.base.init_chat_model`**: the resolved overload
  returns `BaseChatModel | _ConfigurableModel` (`chat_models/base.py:210`). `get_llm()`
  has no return annotation, so `llm = get_llm()` inherits that partially-unknown union.
- **Method `Runnable.ainvoke()`** on the structured-output chain returns
  `dict[str, Any] | BaseModel`; `result = await chain.ainvoke(...)` is therefore
  partially `Unknown`.

### `reportUnknownArgumentType`

Fires when you pass a partially-`Unknown` value as an argument.

- Passing the `dict[str, Any] | BaseModel` result of `.ainvoke()` onward — e.g. into a
  returned partial-state dict (`{"next": router.next}`) or a helper — propagates the
  `Any` from `dict[str, Any]` into the call site.

## The LangGraph `StateGraph` + `TypedDict` problem

- **Class `langgraph.graph.StateGraph` — `__init__(state_schema: type[StateT])`** where
  `StateT = TypeVar("StateT", bound=StateLike)` (`langgraph/typing.py:16`).
- `ty` rejected passing a `TypedDict` subclass (`ChatState`, `ProductSearchState`) here
  with `invalid-argument-type`, which is why the old `# ty: ignore[...]` comments
  existed. **Pyright models `type[SomeTypedDict]` correctly**, so the ignores were
  unnecessary under Pyright and have been removed.
- Reducer state — `Annotated[list[AnyMessage], add_messages]` and the custom
  `take_latest_nonempty` (`app/agents/states.py`) — is understood by Pyright; no
  ignores needed.

## A real error strict mode caught

Strict mode is not only stub noise. Enabling it surfaced
`reportTypedDictNotRequiredAccess` on `get_next` in `app/agents/__init__.py`:
`next` is declared `NotRequired[NextNode]` on `ChatState`, so subscripting
`state["next"]` can raise at runtime if the key is absent. The supervisor always
writes `next` before the conditional edge runs, so the fix uses
`state.get("next", "")` — type-safe and behavior-preserving (the default mirrors the
`take_latest_nonempty` reducer's own `current or ""` fallback).
