from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    AgentRun,
    ApprovalRecord,
    Base,
    DraftPORecord,
    ToolCallTraceRecord,
    User,
)
from app.schemas.agents import (
    AgentRunDetail,
    AgentRunPlan,
    AgentRunRequest,
    AgentRunResponse,
    ApproveRunRequest,
    DraftPO,
    StoredApproval,
    StoredDraftPO,
    StoredToolCall,
    ToolCallStatus,
    ToolCallTrace,
)


@dataclass
class AgentRunRecord:
    request: AgentRunRequest
    response: AgentRunResponse
    plan: AgentRunPlan
    pending_po_input: dict[str, Any] | None = None
    approval_completed: bool = False


class AgentRunRepository:
    def __init__(self, database_url: str | Path) -> None:
        self.database_url = _normalize_database_url(database_url)
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False},
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self._initialize()

    def save(self, record: AgentRunRecord) -> None:
        response = record.response
        now = _now()

        with self.session_factory.begin() as session:
            self._ensure_user(
                session,
                user_id=record.request.user_id,
                department=record.request.department,
                now=now,
            )

            run = session.get(AgentRun, response.run_id)
            is_new_run = run is None
            if run is None:
                run = AgentRun(
                    run_id=response.run_id,
                    user_id=record.request.user_id,
                    department=record.request.department,
                    status=response.status.value,
                    decision_action=response.decision.action.value,
                    risk_level=response.decision.risk_level.value,
                    requires_human_approval=response.decision.requires_human_approval,
                    request_json=record.request.model_dump(mode="json"),
                    plan_json=record.plan.model_dump(mode="json"),
                    response_json=response.model_dump(mode="json"),
                    pending_po_input_json=record.pending_po_input,
                    approval_completed=record.approval_completed,
                    created_at=response.created_at,
                    updated_at=now,
                )
                session.add(run)
            else:
                run.user_id = record.request.user_id
                run.department = record.request.department
                run.status = response.status.value
                run.decision_action = response.decision.action.value
                run.risk_level = response.decision.risk_level.value
                run.requires_human_approval = response.decision.requires_human_approval
                run.request_json = record.request.model_dump(mode="json")
                run.plan_json = record.plan.model_dump(mode="json")
                run.response_json = response.model_dump(mode="json")
                run.pending_po_input_json = record.pending_po_input
                run.approval_completed = record.approval_completed
                run.updated_at = now

            if not is_new_run:
                run.tool_call_traces.clear()
                session.flush()
            run.tool_call_traces.extend(
                [
                    ToolCallTraceRecord(
                        run_id=response.run_id,
                        sequence=sequence,
                        tool=trace.tool,
                        status=trace.status.value,
                        boundary=trace.boundary,
                        trace_json=trace.model_dump(mode="json"),
                        created_at=now,
                    )
                    for sequence, trace in enumerate(response.tool_calls, start=1)
                ]
            )

            if response.draft_po is None:
                run.draft_po = None
            elif run.draft_po is None:
                run.draft_po = DraftPORecord(
                    po_id=response.draft_po.po_id,
                    run_id=response.run_id,
                    status=response.draft_po.status,
                    draft_po_json=response.draft_po.model_dump(mode="json"),
                    created_at=now,
                    updated_at=now,
                )
            else:
                run.draft_po.po_id = response.draft_po.po_id
                run.draft_po.status = response.draft_po.status
                run.draft_po.draft_po_json = response.draft_po.model_dump(mode="json")
                run.draft_po.updated_at = now

    def get(self, run_id: str) -> AgentRunRecord | None:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if run is None:
                return None

            return AgentRunRecord(
                request=AgentRunRequest.model_validate(run.request_json),
                plan=_load_agent_run_plan(run.plan_json),
                response=AgentRunResponse.model_validate(run.response_json),
                pending_po_input=run.pending_po_input_json,
                approval_completed=run.approval_completed,
            )

    def get_detail(self, run_id: str) -> AgentRunDetail | None:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if run is None:
                return None

            tool_calls = [
                StoredToolCall(
                    sequence=trace.sequence,
                    tool=trace.tool,
                    status=ToolCallStatus(trace.status),
                    boundary=trace.boundary,
                    trace=ToolCallTrace.model_validate(trace.trace_json),
                    created_at=trace.created_at,
                )
                for trace in run.tool_call_traces
            ]

            draft_po = None
            if run.draft_po is not None:
                draft_po = StoredDraftPO(
                    po_id=run.draft_po.po_id,
                    status=run.draft_po.status,
                    draft_po=DraftPO.model_validate(run.draft_po.draft_po_json),
                    created_at=run.draft_po.created_at,
                    updated_at=run.draft_po.updated_at,
                )

            approvals = [
                StoredApproval(
                    id=approval.id,
                    approver_id=approval.approver_id,
                    approved=approval.approved,
                    comment=approval.comment,
                    created_at=approval.created_at,
                )
                for approval in run.approvals
            ]

            return AgentRunDetail(
                run_id=run.run_id,
                user_id=run.user_id,
                department=run.department,
                status=run.status,
                decision_action=run.decision_action,
                risk_level=run.risk_level,
                requires_human_approval=run.requires_human_approval,
                approval_completed=run.approval_completed,
                request=AgentRunRequest.model_validate(run.request_json),
                plan=_load_agent_run_plan(run.plan_json),
                response=AgentRunResponse.model_validate(run.response_json),
                tool_calls=tool_calls,
                draft_po=draft_po,
                approvals=approvals,
                pending_po_input=run.pending_po_input_json,
                created_at=run.created_at,
                updated_at=run.updated_at,
            )

    def record_approval(self, run_id: str, request: ApproveRunRequest) -> None:
        now = _now()
        with self.session_factory.begin() as session:
            self._ensure_user(session, user_id=request.approver_id, department=None, now=now)
            session.add(
                ApprovalRecord(
                    run_id=run_id,
                    approver_id=request.approver_id,
                    approved=request.approved,
                    comment=request.comment,
                    approval_json={"run_id": run_id, **request.model_dump(mode="json")},
                    created_at=now,
                )
            )

    def _initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _ensure_user(
        session: Session,
        *,
        user_id: str,
        department: str | None,
        now: datetime,
    ) -> None:
        user = session.get(User, user_id)
        if user is None:
            session.add(
                User(
                    id=user_id,
                    display_name=None,
                    department=department,
                    created_at=now,
                    updated_at=now,
                )
            )
            return

        if user.department is None and department is not None:
            user.department = department
        user.updated_at = now


def _normalize_database_url(database_url: str | Path) -> str:
    value = str(database_url)
    if "://" in value:
        return value
    if value == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{path}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_agent_run_plan(plan_json: dict[str, Any]) -> AgentRunPlan:
    try:
        return AgentRunPlan.model_validate(plan_json)
    except ValidationError:
        return AgentRunPlan.model_validate(
            {
                "planner": plan_json.get("planner", "unknown"),
                "parsed_request": plan_json.get("parsed_request"),
                "observations": plan_json.get("observations", []),
                "metadata": {
                    **plan_json.get("metadata", {}),
                    "legacy_tool_plan": plan_json.get("tool_plan", []),
                    "legacy_rationale": plan_json.get("rationale"),
                },
            }
        )
