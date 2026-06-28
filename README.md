# web-search-tool

## 🔭 Visão geral

Agente conversacional construído com **LangGraph** (em português do Brasil) que
ajuda o usuário a encontrar o produto com o **melhor custo-benefício** para o seu
orçamento e necessidade.

Ele segue um padrão **supervisor/router**: um supervisor lê a conversa e despacha
o fluxo para sub-agentes especializados. A busca na web é feita com **Tavily**.

## 🏗️ Arquitetura

O grafo principal é compilado em `app/agents/__init__.py` e exposto como
entrypoint `app.agents:make_graph` (id `grafo` no `langgraph.json`), além de ser
exportado para `main.py`.

Fluxo do grafo principal:

```
START → supervisor → (condicional: guide | products) → END
```

### Nós e sub-agentes

- **Supervisor** (`app/agents/supervisor/`): LLM com
  `with_structured_output(Router)` que lê o histórico de mensagens e escreve a
  escolha em `state['next']`. Ele **apenas roteia** — não produz mensagem
  voltada ao usuário.
- **Guide** (`app/agents/guide/`): agente de recepção/boas-vindas. Explica o bot
  e identifica a intenção do usuário. **Não** coleta orçamento/requisitos nem
  executa buscas (isso é responsabilidade do agente de produtos).
- **Products** (`app/agents/products/`): compila um subgrafo com etapas de
  human-in-the-loop via `interrupt()`:

  ```
  collect_requirements → search_products → validate_products → present_recommendations → search_purchase_links
  ```

  - `tools.py` roda as buscas no Tavily.
  - `nodes.py` contém os passos do subgrafo e os routers.

### Convenção de pacotes

Cada agente é um pacote sob `app/agents/` com layout uniforme:

- `prompt.py` — prompts.
- `schemas.py` — modelos pydantic de structured-output.
- `agent.py` — a função `build_*_agent`.
- No subgrafo **products** também há `state.py`, `tools.py` e `nodes.py`.

O `__init__.py` de cada pacote expõe um `build_*_node()` que é passado para
`builder.add_node(...)`.

> ⚠️ O pacote real do agente de produtos é `app/agents/products/`
> (**não** `product_search`).

### Estado compartilhado

`ChatState` (em `app/agents/states.py`) estende o `AgentState` do LangChain — por
isso carrega `messages` — e adiciona o campo `next`, com um reducer customizado
`take_latest_nonempty` que mantém a última decisão de roteamento não vazia.

## 🧰 Stack

- **Python 3.13**
- **LangGraph / LangChain**
- **Tavily** (busca web)
- **pydantic-settings** (configuração via ambiente)
- **uv** (gerenciador de pacotes/execução)
- **Ruff** (lint + format)
- **Pyright** (type checking em modo strict)
- **Taskipy** (tasks)

## ⚙️ Setup

1. Instale as dependências (inclui o grupo de dev):

   ```bash
   uv sync
   ```

2. Crie o arquivo `.env` a partir do exemplo e preencha as chaves:

   ```bash
   cp .env.example .env
   ```

### Variáveis de ambiente

| Variável | Descrição |
| --- | --- |
| `AGENT_CHAT_MODEL` | Modelo do agente, **prefixado pelo provider** (ex.: `openai:gpt-4.1-nano` ou `google_genai:gemini-2.5-flash`). |
| `OPENAI_API_KEY` | Chave da OpenAI (se usar um modelo `openai:`). |
| `GOOGLE_API_KEY` | Chave do Google (se usar um modelo `google_genai:`). |
| `TAVILY_API_KEY` | Chave da API do Tavily para a busca web. |
| `LANGSMITH_TRACING` | Ativa o tracing no LangSmith (`true`/`false`). |
| `LANGSMITH_ENDPOINT` | Endpoint do LangSmith (ex.: `https://api.smith.langchain.com`). |
| `LANGSMITH_API_KEY` | Chave da API do LangSmith. |
| `LANGSMITH_PROJECT` | Nome do projeto no LangSmith. |

## ▶️ Como rodar

As tasks ficam em `[tool.taskipy.tasks]` no `pyproject.toml` e são executadas via
`uv`:

```bash
uv run task main      # Executa o agente uma vez via main.py (ainvoke one-shot)
uv run task lint      # Type-check + lint (pyright && ruff check)
uv run task format    # Formata o código (ruff format)
uv run task test      # Roda os testes com cobertura (pytest --cov)
```

## 🧪 LangGraph Studio + LangSmith

```bash
uv run task langsmith
```

Esse comando roda `langgraph dev --config ./langgraph.json` com **hot reload** e
abre o **LangGraph Studio**. O tracing no **LangSmith** é ativado pelas variáveis
`LANGSMITH_*` no `.env`.

## 🔎 Type checking

Usamos **Pyright** (o engine por trás do Pylance) tanto no editor quanto no CLI
`uv run task lint`. Um único engine garante que editor e CLI concordem. A config
fica em `[tool.pyright]` no `pyproject.toml`, e o Pylance a lê automaticamente.

O modo é `typeCheckingMode = "strict"`, com **três regras desabilitadas**:

- `reportUnknownMemberType`
- `reportUnknownVariableType`
- `reportUnknownArgumentType`

Elas são desabilitadas porque disparam em tipos `Unknown` que **vazam dos stubs
incompletos do LangChain/LangGraph** — não do nosso código. Mantê-las ligadas
geraria dezenas de erros a cada chamada de LLM e enterraria os erros reais. As
nossas próprias assinaturas continuam estritas via `reportUnknownParameterType` /
`reportMissingParameterType`, que permanecem **ligadas**.

De onde vêm os `Unknown`:

- `BaseChatModel.with_structured_output()` é declarado como
  `Runnable[..., dict[str, Any] | BaseModel]`; o braço `dict[str, Any]` é um
  vazamento de `Any`, então qualquer acesso a membro no resultado vira `Unknown`.
- `init_chat_model` pode retornar a classe privada `_ConfigurableModel`, cujos
  métodos (`.ainvoke`, `.with_structured_output`) são frouxamente tipados — por
  isso o resultado de `get_llm()` é sinalizado.
- Passar adiante o resultado `dict[str, Any] | BaseModel` de `.ainvoke()` (ex.:
  para um dict de estado parcial como `{"next": router.next}`) propaga o `Any`
  para o call site.

### `StateGraph` + `TypedDict`

`StateGraph.__init__(state_schema: type[StateT])` recebe um `TypedDict`
(`ChatState`, `ProductSearchState`). O type checker antigo (`ty`) rejeitava isso
com `invalid-argument-type`, o que motivava os antigos comentários
`# ty: ignore[...]`. O **Pyright modela `type[SomeTypedDict]` corretamente**,
então esses ignores eram desnecessários e foram removidos. O estado com reducers
— `Annotated[list[AnyMessage], add_messages]` e o `take_latest_nonempty`
customizado — também é entendido pelo Pyright, sem necessidade de ignores.

### Um erro real que o strict pegou

O strict não é só ruído de stub. Ao ligá-lo, surgiu
`reportTypedDictNotRequiredAccess` em `app/agents/__init__.py`: `next` é declarado
`NotRequired[NextNode]` em `ChatState`, então subscrever `state["next"]` pode
levantar em runtime se a chave estiver ausente. O supervisor sempre escreve
`next` antes da edge condicional rodar, então a correção usa
`state.get("next", "")` — type-safe e preservando o comportamento (o default
espelha o fallback `current or ""` do próprio reducer `take_latest_nonempty`).
