# AI Usage

I used AI as a development assistant while building this project, mainly for second opinions, boilerplate, and documentation cleanup. The important runtime choices still live in the code: the harness validates planner output, tools validate their inputs and outputs, and approval checks are enforced before risky actions can run.

The delivered app can run with a live LLM planner, but it is not an open-ended agent. The model is only allowed to choose the next typed step. It cannot execute tools directly, skip approval, write to the database, or decide the final API shape on its own.

## Where It Helped

AI was useful for a few practical parts of the work:

- turning the assignment requirements into a small harness design;
- sketching the first pass of the Pydantic schemas, tool registry, and tests;
- checking edge cases around approval flow, prompt injection, and malformed planner output;
- building the single-file demo UI in `app/static/index.html`;
- tightening wording in the README and architecture notes.

I also used Context7 near the end to check current FastAPI documentation behavior around startup, generated `/docs`, and response-model validation.

## What I Kept Manual

I did not treat AI output as trusted business logic. The procurement behavior is intentionally boring and inspectable:

- policy flags are computed in `app/services/procurement.py`;
- tools only run through `ToolRegistry.execute`;
- `submit_to_erp` is blocked unless approval is complete;
- final responses are validated as `AgentRunResponse`;
- prompt-injection style requests are classified as unsafe input, not followed as instructions.

The optional LLM path has the same boundaries. If it returns malformed JSON, multiple tool calls, an unexpected tool, or a bypass attempt, the harness fails closed instead of trying to recover creatively.

## Checks I Ran

I verified the project with API tests, parser tests, fake-LLM guardrail tests, fixture demo scenarios, and Ruff linting. The main commands are:

```bash
uv --cache-dir .uv-cache run --extra dev ruff check .
uv --cache-dir .uv-cache run --extra dev pytest -q
uv --cache-dir .uv-cache run --extra dev python scripts/demo.py
```

The tests cover the main paths I cared about for this assignment: low-risk purchase, approval-required purchase, missing information, prompt injection, persisted run reads, parser behavior, and bad LLM outputs.

## Runtime Modes

The current default is:

```bash
AGENT_PLANNER=llm
```

That mode uses LiteLLM and needs `LLM_MODEL` plus the provider credentials in the environment.

For local deterministic runs, the project also supports:

- `AGENT_PLANNER=mock`: deterministic mock adapter with the same planner interface.
- `AGENT_PLANNER=rule`: fully local rule-based planner with no model call.

In all three modes, the local harness remains the source of truth for approval, tool execution, trace recording, and final structured output.
