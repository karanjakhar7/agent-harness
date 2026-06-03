# AGENTS.md

Concise onboarding reference for contributors and coding agents working on this project.

## Project purpose

Build a minimal procurement approval backend that shows an explicit agent harness:

- manage one agent run,
- plan from a user request,
- execute registered tools,
- validate all inputs/outputs,
- enforce approval boundaries,
- record tool traces,
- return structured Pydantic output.

Sample fixtures implement a purchase-approval workflow; swap catalog and policy tools for other domains.

## Architecture map

- `main.py`: ASGI entrypoint exporting `app`.
- `app/api/app.py`: FastAPI app and routes only; keep handlers thin.
- `app/harness/runtime.py`: central `AgentHarness`; owns run state, decisions, approval flow, final validation.
- `app/harness/planner.py`: deterministic planner plus mock LLM adapter contract.
- `app/harness/clarification.py`: internal structured schema for clarification-answer resolution.
- `app/harness/tools.py`: `ToolRegistry`, tool specs, guarded tool execution.
- `app/services/procurement.py`: procurement operations used by registered tools.
- `app/db/models.py`: SQLAlchemy ORM table models.
- `app/db/repositories.py`: database-backed run repository.
- `app/schemas/agents.py`: Pydantic request, response, tool, decision, and trace schemas.
- `app/services/fixtures.py`: fixture loading helpers.
- `app/templates/`: reserved for future server-rendered HTML templates.
- `fixtures/catalog.json`: mock catalog for the sample workflow.
- `fixtures/sample_requests.json`: demo scenarios.
- `tests/test_agent_api.py`: required behavior coverage.
- `docs/ARCHITECTURE.md`: fuller design explanation.
- `docs/AI_USAGE.md`: implementation AI usage notes.

## Safety invariants

- Do not put decision logic directly in FastAPI handlers.
- Do not execute tool handlers without `ToolRegistry.execute`.
- Every tool needs a Pydantic input model and output model.
- High-risk operations must be blocked unless approval is completed.
- `submit_to_erp` must never run before approval.
- Prompt-injection or policy-bypass requests must fail closed as `REJECT` or human-review safe state.
- Final API responses must validate as `AgentRunResponse`.

## Main API

- `POST /agent/run`: starts an agent run.
- `GET /agent/runs/{run_id}`: reads stored run output.
- `GET /agent/runs/{run_id}/details`: full persisted run view (request, plan, response, tool-call traces, draft PO, approvals, pending PO input).
- `POST /agent/runs/{run_id}/approve`: optional human approval path.
- `GET /health`: service health.

## Decision outcomes

- `CREATE_DRAFT_PO`: low-risk request, draft PO created.
- `NEED_HUMAN_APPROVAL`: policy requires human review.
- `ASK_CLARIFICATION`: missing quantity/item or budget conflict.
- `REJECT`: bypass attempt or unsafe request.

## Run commands

```bash
uv sync --extra dev
uvicorn main:app --reload
```

Codex sandbox-friendly variants:

```bash
uv --cache-dir .uv-cache run --extra dev ruff check .
uv --cache-dir .uv-cache run --extra dev pytest -q
uv --cache-dir .uv-cache run --extra dev python scripts/demo.py
```

## Planner switch

```bash
AGENT_PLANNER=rule uvicorn main:app --reload
AGENT_PLANNER=mock uvicorn main:app --reload
```

Both planners are local and deterministic. A real LLM planner should keep the same `plan(request) -> PlannerPlan` contract.

## When changing behavior

- Add or update fixture scenarios first when possible.
- Add/adjust tests for every policy or boundary change.
- Keep trace output useful: blocked/error tool calls should name the boundary.
- Update `docs/ARCHITECTURE.md` if the harness/tool/approval boundary changes.

## Notes
Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.
