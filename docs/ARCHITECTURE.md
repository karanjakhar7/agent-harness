# Architecture

This project is a small procurement backend built to make the agent harness easy to inspect. FastAPI handles HTTP and OpenAPI documentation; `AgentHarness` handles the agent run.

## Delivery Summary

The implementation satisfies the required agent-harness boundaries:

- one run is managed by `app.harness.runtime.AgentHarness`;
- planning is swappable through `RuleBasedPlanner`, `MockLLMPlanner`, or `LLMPlanner`;
- tools are registered and executed only through `ToolRegistry`;
- tool input and output are validated with Pydantic models;
- approval-required operations are blocked until approval is complete;
- every tool attempt records a `ToolCallTrace`;
- final API output validates as `AgentRunResponse`;
- runs, traces, draft POs, approvals, and pending state are persisted in SQLite.

## Request Flow

```text
HTTP route
  -> AgentHarness.run()
  -> planner.parse()
  -> bounded observe-decide-act loop
  -> ToolRegistry.execute() for each tool step
  -> final policy cross-check
  -> AgentRunResponse validation
  -> AgentRunRepository.save()
```

The route layer is intentionally thin. `app/api/app.py` creates the harness and exposes these endpoints:

- `GET /` — demo UI (see [Frontend](#frontend))
- `POST /agent/run`
- `GET /agent/runs/{run_id}`
- `GET /agent/runs/{run_id}/details`
- `POST /agent/runs/{run_id}/clarify`
- `POST /agent/runs/{run_id}/approve`
- `GET /health`

## Agent Loop

`AgentHarness` owns the state of a single run:

1. Create a `run_id` and timestamp.
2. Ask the planner to parse the request into `ParsedAgentRequest`.
3. Build `AgentState` with request, parsed fields, observations, approval flag, and safe metadata.
4. Ask the planner for exactly one next `AgentStep`.
5. Validate the step as one of:
   - `ToolStep`
   - `ClarifyStep`
   - `FinalStep`
6. Normalize trusted tool inputs before execution.
7. Execute tool steps through `ToolRegistry`.
8. Append each trace to state observations.
9. Pause for clarification, pause for approval, reject, fail closed, or complete.
10. Validate and persist the final `AgentRunResponse`.

The loop is bounded by `MAX_STEPS = 6` to avoid unbounded planner behavior.

## Planner Boundary

Planner contract:

```python
parse(request: AgentRunRequest) -> ParsedAgentRequest
next_step(state: AgentState) -> AgentStep
```

Planner implementations:

- `RuleBasedPlanner`: default deterministic planner.
- `MockLLMPlanner`: deterministic adapter with the same shape as the live planner.
- `LLMPlanner`: optional LiteLLM-based planner that chooses one function call per loop turn.

The planner chooses steps. It does not execute tools, write to the database, or override approval checks.

## Tool Boundary

Tools are declared in `app/harness/tools.py` as `ToolSpec` objects. Each spec includes:

- tool name;
- Pydantic input model;
- Pydantic output model;
- handler function;
- risk level;
- approval requirement flag;
- description for LLM tool schemas.

Registered tools:

- `lookup_catalog`
- `check_policy`
- `create_draft_po`
- `submit_to_erp`

`ToolRegistry.execute()` is the only execution path. It performs:

1. tool existence check;
2. approval-required tool blocking;
3. input schema validation;
4. handler execution;
5. output schema validation;
6. trace creation.

Tool traces use status `success`, `blocked`, or `error`, and blocked/error traces name the boundary, such as `ApprovalBoundary`, `InputSchema`, `OutputSchema`, or `ToolRuntime`.

## Approval Boundary

Policy is checked by `check_policy`, using `fixtures/policies.json` and `fixtures/budgets.json`.

The policy flags these cases:

- amount above the configured approval threshold, currently USD 5000;
- restricted category, including hardware and enterprise software;
- department budget overrun;
- user-stated budget overrun;
- prompt injection or approval bypass request.

Outcomes:

- Low-risk allowed request: create a draft PO.
- Requires human approval: return `AWAITING_HUMAN_APPROVAL`, store pending draft input, and record a blocked `create_draft_po` trace.
- User budget conflict: return `NEEDS_CLARIFICATION` so the user can lower quantity or change budget.
- Prompt injection or bypass attempt: return `REJECT`.

`submit_to_erp` is registered as a high-risk tool with `requires_approval=True`. If any planner tries to call it before approval is complete, `ToolRegistry` blocks the call at `ApprovalBoundary`.

## Human Approval Path

`POST /agent/runs/{run_id}/approve` resumes a run only when the stored response is `AWAITING_HUMAN_APPROVAL`.

If approved:

- the repository records the approval;
- the harness adds an approval reference;
- `create_draft_po` runs with `approval_completed=True`;
- the run becomes `COMPLETED`.

If rejected:

- the approval is recorded;
- no draft PO is created;
- the run becomes `REJECTED`.

Pending approval can survive process restart because the repository stores the pending draft PO input and the previous response snapshot.

## Clarification Path

`POST /agent/runs/{run_id}/clarify` resumes a run only when the stored response is `NEEDS_CLARIFICATION`.

The harness:

- records the clarification answer in response metadata;
- parses active missing fields locally when possible;
- optionally lets the live LLM resolve unresolved answers in `AGENT_PLANNER=llm`;
- invalidates stale observations when parsed fields change;
- re-enters the same guarded loop on the same `run_id`.

The original request remains unchanged for auditability.

## Schema Validation

Validation happens at these layers:

- FastAPI validates request bodies and route `response_model` values.
- Pydantic validates planner state and discriminated `AgentStep` values.
- `ToolRegistry` validates every tool input and output.
- `AgentHarness` validates the final `AgentRunResponse`.
- Fixture loaders validate catalog, policy, budget, and sample request data.
- The repository validates persisted snapshots back into Pydantic models when reading.

All main schemas live in `app/schemas/agents.py`.

## Persistence

`app/db/repositories.py` uses SQLAlchemy ORM with SQLite. `AGENT_DB_PATH` defaults to:

```text
sqlite+pysqlite:///procurement.sqlite3
```

The database stores:

- users, auto-created from incoming `user_id` and `approver_id`;
- agent runs with request, plan, response, status, risk, and pending PO input;
- ordered tool traces;
- draft POs;
- approval records.

There is no authentication system in this MVP. User IDs are accepted as request data and stored for traceability.

## LLM Mode Safety

The live LLM planner is optional. It does not replace the harness.

In `AGENT_PLANNER=llm`:

- the initial parse is seeded locally;
- each model response must contain exactly one OpenAI-compatible function call;
- provider errors, malformed calls, multiple calls, or missing calls fail closed;
- LLM-selected tool inputs are normalized by the harness before execution;
- catalog, policy, and budget facts are learned through tool observations, not trusted from the prompt;
- final decisions are cross-checked against the latest successful `check_policy` result.

This keeps the model as a step chooser while the backend enforces business and safety boundaries.

## Frontend

`app/static/index.html` is a self-contained demo UI served at `GET /`. It requires no build step and has no external dependencies — all CSS and JavaScript are inline in the file.

The UI has two panels:

- **Left** — a request form (user ID, department, free-text message) and four pre-filled demo scenario buttons that cover the main decision paths.
- **Right** — the agent response, rendered after each API call: status badge, decision card with policy flags, draft PO table, collapsible tool call trace, and an inline approval or clarification form that appears contextually based on run status.

All data fetching is done client-side with `fetch()` against the same origin. The `GET /` route reads the file from `app/static/index.html` using a path anchored on `__file__`, which resolves correctly in local development, Docker, and Vercel.

Live URL: coming soon

## Fixtures

The fixture bundle is intentionally small:

- `fixtures/catalog.json`: catalog items, aliases, unit price, category, currency.
- `fixtures/budgets.json`: remaining budget per department.
- `fixtures/policies.json`: approval threshold, restricted categories, policy rules.
- `fixtures/sample_requests.json`: demo scenarios and expected behavior.

Important examples:

- `Figma Enterprise Seat` is standard software and can create a draft PO when within budget and policy.
- `MacBook Pro` is hardware and requires human approval.
- `Oracle License` is enterprise software and requires human approval.

## Verification Coverage

Tests cover:

- low-risk draft PO creation;
- hardware approval blocking;
- amount over USD 5000;
- department budget overrun;
- enterprise software approval;
- missing quantity clarification;
- clarification continuation after restart;
- approval continuation after restart;
- prompt injection rejection and ERP blocking;
- persisted details endpoint;
- live LLM planner failure modes with fake completion functions.
