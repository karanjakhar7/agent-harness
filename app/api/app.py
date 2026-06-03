from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.db.repositories import AgentRunRepository
from app.harness.llm_planner import LLMPlanner
from app.harness.planner import MockLLMPlanner, RuleBasedPlanner
from app.harness.runtime import AgentHarness
from app.harness.tools import build_default_tool_registry
from app.logging_config import configure_logging
from app.schemas.agents import (
    AgentRunDetail,
    AgentRunRequest,
    AgentRunResponse,
    ApproveRunRequest,
    ClarifyRunRequest,
)
from app.services.fixtures import load_procurement_fixtures


def build_harness() -> AgentHarness:
    configure_logging()
    planner_name = os.getenv("AGENT_PLANNER", "rule").lower()
    fixtures = load_procurement_fixtures()
    tool_registry = build_default_tool_registry(fixtures)
    if planner_name == "rule":
        planner = RuleBasedPlanner()
    elif planner_name == "mock":
        planner = MockLLMPlanner()
    elif planner_name == "llm":
        planner = LLMPlanner.from_env(tool_registry.llm_tool_schemas())
    else:
        raise RuntimeError("AGENT_PLANNER must be one of: rule, mock, llm.")

    database_url = os.getenv("AGENT_DB_PATH") or _default_database_url()
    return AgentHarness(
        planner=planner,
        tool_registry=tool_registry,
        run_repository=AgentRunRepository(database_url),
        catalog=fixtures.catalog,
    )


def _default_database_url() -> str:
    if os.getenv("VERCEL"):
        return "sqlite+pysqlite:////tmp/procurement.sqlite3"
    return "sqlite+pysqlite:///procurement.sqlite3"


harness = build_harness()

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Agent Harness API",
    version="0.1.0",
    description="Minimal agent harness with guarded tools, planning, and approval boundaries.",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=(_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    return harness.run(request)


@app.get("/agent/runs/{run_id}", response_model=AgentRunResponse)
def get_run(run_id: str) -> AgentRunResponse:
    response = harness.get_run(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return response


@app.get("/agent/runs/{run_id}/details", response_model=AgentRunDetail)
def get_run_details(run_id: str) -> AgentRunDetail:
    detail = harness.get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return detail


@app.post("/agent/runs/{run_id}/approve", response_model=AgentRunResponse)
def approve_run(run_id: str, request: ApproveRunRequest) -> AgentRunResponse:
    try:
        return harness.approve_run(run_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/agent/runs/{run_id}/clarify", response_model=AgentRunResponse)
def clarify_run(run_id: str, request: ClarifyRunRequest) -> AgentRunResponse:
    try:
        return harness.continue_run(run_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
