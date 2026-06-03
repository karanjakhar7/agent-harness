# Procurement Approval Agent

Minimal FastAPI backend that demonstrates a safe procurement agent harness. The API accepts natural-language purchase requests, routes them through a planner, executes registered tools with Pydantic validation, enforces approval boundaries, persists the run, and returns a structured response.

## Live Demo

| | |
|---|---|
| **UI** | [https://agent-harness-xi.vercel.app](https://agent-harness-xi.vercel.app) |
| **API docs** | [https://agent-harness-xi.vercel.app/docs](https://agent-harness-xi.vercel.app/docs) |

Open the UI to submit purchase requests interactively and watch the agent decide, approve, or ask for clarification in real time. The API docs cover every endpoint in detail.

## What This Delivers

- `GET /` interactive demo UI.
- `POST /agent/run` starts one procurement agent run.
- `GET /agent/runs/{run_id}` returns the stored structured response.
- `GET /agent/runs/{run_id}/details` returns the persisted request, plan, traces, draft PO, approvals, and pending PO input.
- `POST /agent/runs/{run_id}/clarify` continues a run that needs missing information.
- `POST /agent/runs/{run_id}/approve` completes the human approval path and creates the draft PO when approved.
- `GET /health` returns service health.

The default behavior is deterministic and offline. A mock planner and an opt-in live LLM planner are also available behind `AGENT_PLANNER`.

## Stack

- Python 3.12
- FastAPI
- Pydantic v2
- SQLAlchemy 2.0 with SQLite
- uv for local dependency management
- Optional LiteLLM planner path

## Project Map

- `main.py`: ASGI entrypoint exporting `app`.
- `app/api/app.py`: FastAPI routes only; handlers delegate to the harness.
- `app/static/index.html`: self-contained demo UI served at `GET /`.
- `app/harness/runtime.py`: `AgentHarness`, run loop, approval flow, final response validation.
- `app/harness/planner.py`: deterministic rule planner and mock LLM adapter.
- `app/harness/llm_planner.py`: optional live LLM step chooser.
- `app/harness/tools.py`: `ToolRegistry`, tool specs, schema validation, approval-required tool blocking.
- `app/services/procurement.py`: catalog lookup, policy check, draft PO creation, ERP submission stub.
- `app/db/`: SQLAlchemy models and repository.
- `app/schemas/agents.py`: Pydantic API, planner, tool, decision, trace, and persistence schemas.
- `fixtures/`: catalog, budgets, policies, and sample requests.
- `tests/`: API, parser, and LLM guardrail coverage.

## Quick Start

Install dependencies:

```bash
uv sync --extra dev
```

Start the API:

```bash
uvicorn main:app --reload
```

Open the demo UI:

```text
http://127.0.0.1:8000/
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

By default, runs are stored in `./procurement.sqlite3`. To use another SQLite path or SQLAlchemy URL:

```bash
AGENT_DB_PATH=/tmp/procurement.sqlite3 uvicorn main:app --reload
```

## Run A Request

```bash
curl -X POST http://127.0.0.1:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "u_001",
    "department": "marketing",
    "message": "Please buy 3 Figma Enterprise seats for marketing, keeping budget under USD 3000."
  }'
```

Expected result: `COMPLETED` with decision `CREATE_DRAFT_PO`, three successful tool traces, and a draft PO total of USD 2400.

## Human Approval Flow

Requests that exceed policy boundaries return `AWAITING_HUMAN_APPROVAL` with a blocked draft-creation trace. Approve the same run with:

```bash
curl -X POST http://127.0.0.1:8000/agent/runs/{run_id}/approve \
  -H 'Content-Type: application/json' \
  -d '{
    "approver_id": "manager_001",
    "approved": true,
    "comment": "Approved."
  }'
```

If approved, the harness creates the draft PO with an approval reference. If rejected, the run becomes `REJECTED`.

## Clarification Flow

Requests with missing item or quantity return `NEEDS_CLARIFICATION`. Continue the same run with:

```bash
curl -X POST http://127.0.0.1:8000/agent/runs/{run_id}/clarify \
  -H 'Content-Type: application/json' \
  -d '{"answer": "3"}'
```

The harness parses the answer, invalidates stale observations if needed, and re-enters the same guarded loop.

## Demo And Tests

Run all fixture scenarios:

```bash
uv --cache-dir .uv-cache run --extra dev python scripts/demo.py
```

Run the test suite:

```bash
uv --cache-dir .uv-cache run --extra dev pytest -q
```

Run linting:

```bash
uv --cache-dir .uv-cache run --extra dev ruff check .
```

Fixture scenarios cover:

- low-risk software purchase -> `CREATE_DRAFT_PO`
- hardware purchase -> `NEED_HUMAN_APPROVAL`
- amount over USD 5000 -> `NEED_HUMAN_APPROVAL`
- missing information -> `ASK_CLARIFICATION`
- prompt injection or approval bypass -> `REJECT`

## Planner Modes

Deterministic rule planner, used by default:

```bash
AGENT_PLANNER=rule uvicorn main:app --reload
```

Mock LLM adapter shape, still deterministic:

```bash
AGENT_PLANNER=mock uvicorn main:app --reload
```

Live LLM planner through LiteLLM:

```bash
uv sync --extra dev

AGENT_PLANNER=llm \
LLM_MODEL=gemini-3.1-flash-lite \
uvicorn main:app --reload
```

The app does not load `.env` files. Provider credentials must already exist in the process environment.

Live LLM settings:

- `LLM_MODEL`: required for `AGENT_PLANNER=llm`; this project uses `gemini-3.1-flash-lite`.
- `LLM_TIMEOUT_SECONDS`: defaults to `20`.
- `LLM_MAX_RETRIES`: defaults to `1`.
- `LLM_TEMPERATURE`: defaults to `0`.
- `LLM_MAX_TOKENS`: optional provider response limit.

The live model chooses one structured step at a time. The local harness still owns tool execution, trusted-field normalization, approval checks, traces, persistence, and final response validation.

## Delivery Notes

- FastAPI handlers stay thin; decision logic is in `AgentHarness`.
- All tool calls go through `ToolRegistry.execute`.
- Every tool has a Pydantic input and output model.
- `submit_to_erp` is registered but cannot run without completed approval.
- Final responses validate as `AgentRunResponse`.
- `docs/ARCHITECTURE.md` explains the agent loop and boundaries.
- `docs/AI_USAGE.md` explains AI-assisted implementation and verification.
