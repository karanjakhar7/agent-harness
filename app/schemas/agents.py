from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"
    REJECTED = "REJECTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"


class DecisionAction(StrEnum):
    CREATE_DRAFT_PO = "CREATE_DRAFT_PO"
    NEED_HUMAN_APPROVAL = "NEED_HUMAN_APPROVAL"
    REJECT = "REJECT"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ToolCallStatus(StrEnum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    ERROR = "error"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentRunRequest(StrictModel):
    user_id: str = Field(min_length=1)
    department: str = Field(min_length=1)
    message: str = Field(min_length=3)


class ApproveRunRequest(StrictModel):
    approver_id: str = Field(min_length=1)
    approved: bool = True
    comment: str | None = None


class ClarifyRunRequest(StrictModel):
    answer: str = Field(min_length=1, max_length=1000)


class Decision(StrictModel):
    action: DecisionAction
    risk_level: RiskLevel
    requires_human_approval: bool
    reason: str
    policy_flags: list[str] = Field(default_factory=list)


class ToolCallTrace(StrictModel):
    tool: str
    status: ToolCallStatus
    risk_level: RiskLevel | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    boundary: str | None = None


class CatalogItem(StrictModel):
    id: str
    name: str
    category: str
    unit_price: int = Field(ge=0)
    aliases: list[str] = Field(default_factory=list)
    currency: Literal["USD"] = "USD"


class DepartmentBudget(StrictModel):
    remaining_budget_usd: int = Field(ge=0)


class PolicyRule(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ProcurementPolicy(StrictModel):
    approval_threshold_usd: int = Field(gt=0)
    restricted_categories: list[str] = Field(default_factory=list)
    rules: list[PolicyRule] = Field(default_factory=list)


class ProcurementFixtures(StrictModel):
    catalog: list[CatalogItem] = Field(min_length=1)
    budgets: dict[str, DepartmentBudget]
    policies: ProcurementPolicy


class SampleAgentRequest(AgentRunRequest):
    id: str
    expected_behavior: str


class ParsedAgentRequest(StrictModel):
    item_query: str
    quantity: int | None = Field(default=None, gt=0)
    budget_limit: int | None = Field(default=None, gt=0)
    currency: Literal["USD"] = "USD"
    department: str
    direct_order_requested: bool = False


class ToolStep(StrictModel):
    kind: Literal["tool"] = "tool"
    tool: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class FinalStep(StrictModel):
    kind: Literal["final"] = "final"
    decision: Decision
    draft_po: "DraftPO | None" = None
    pending_tool: ToolStep | None = None


class ClarifyStep(StrictModel):
    kind: Literal["clarify"] = "clarify"
    question: str = Field(min_length=1)
    missing_fields: list[str] = Field(min_length=1)
    answer_hint: str = "Reply with the requested information."


AgentStep = Annotated[ToolStep | FinalStep | ClarifyStep, Field(discriminator="kind")]


class AgentState(StrictModel):
    run_id: str
    request: AgentRunRequest
    parsed: ParsedAgentRequest
    observations: list[ToolCallTrace] = Field(default_factory=list)
    approval_completed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_run(self, tool: str) -> bool:
        invalidated_at = _invalidated_at(self.metadata, tool)
        return any(
            trace.tool == tool and index > invalidated_at
            for index, trace in enumerate(self.observations, start=1)
        )

    def output_of(self, tool: str) -> dict[str, Any] | None:
        invalidated_at = _invalidated_at(self.metadata, tool)
        for index, trace in reversed(list(enumerate(self.observations, start=1))):
            if index <= invalidated_at:
                continue
            if trace.tool == tool and trace.status == ToolCallStatus.SUCCESS:
                return trace.output
        return None

    def latest_trace(self, tool: str) -> ToolCallTrace | None:
        invalidated_at = _invalidated_at(self.metadata, tool)
        for index, trace in reversed(list(enumerate(self.observations, start=1))):
            if index <= invalidated_at:
                continue
            if trace.tool == tool:
                return trace
        return None


class AgentRunPlan(StrictModel):
    planner: str
    parsed_request: ParsedAgentRequest
    observations: list[ToolCallTrace] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _invalidated_at(metadata: dict[str, Any], tool: str) -> int:
    invalidated_tools = metadata.get("invalidated_tools", {})
    if not isinstance(invalidated_tools, dict):
        return 0
    value = invalidated_tools.get(tool, 0)
    return value if isinstance(value, int) else 0


class LookupCatalogInput(StrictModel):
    query: str = Field(min_length=1)


class LookupCatalogOutput(StrictModel):
    match_found: bool
    item: CatalogItem | None = None
    alternatives: list[str] = Field(default_factory=list)


class CheckPolicyInput(StrictModel):
    user_id: str
    department: str
    item_id: str
    item_name: str
    category: str
    quantity: int = Field(gt=0)
    unit_price: int = Field(ge=0)
    estimated_total: int = Field(ge=0)
    budget_limit: int | None = Field(default=None, gt=0)
    currency: Literal["USD"] = "USD"
    direct_order_requested: bool = False


class CheckPolicyOutput(StrictModel):
    allowed: bool
    rejected: bool
    requires_human_approval: bool
    risk_level: RiskLevel
    flags: list[str] = Field(default_factory=list)
    reason: str


class CreateDraftPOInput(StrictModel):
    user_id: str
    department: str
    item_id: str
    item_name: str
    quantity: int = Field(gt=0)
    unit_price: int = Field(ge=0)
    estimated_total: int = Field(ge=0)
    currency: Literal["USD"] = "USD"
    approval_reference: str | None = None


class DraftPO(StrictModel):
    po_id: str
    item_id: str
    item: str
    quantity: int
    unit_price: int
    estimated_total: int
    department: str
    currency: Literal["USD"] = "USD"
    status: Literal["DRAFT"] = "DRAFT"
    approval_reference: str | None = None


class SubmitToERPInput(StrictModel):
    draft_po_id: str = Field(min_length=1)
    approval_reference: str = Field(min_length=1)


class SubmitToERPOutput(StrictModel):
    erp_submission_id: str
    draft_po_id: str
    status: Literal["SUBMITTED"] = "SUBMITTED"


class ApprovalRequestInfo(StrictModel):
    run_id: str
    reason: str
    required_for: list[str] = Field(default_factory=list)
    approver_hint: str = "Human approval required."


class ClarificationRequestInfo(StrictModel):
    run_id: str
    question: str
    missing_fields: list[str] = Field(min_length=1)
    answer_hint: str


class ClarificationAnswerInfo(StrictModel):
    answer: str
    answered_at: datetime


class AgentRunResponse(StrictModel):
    run_id: str
    status: RunStatus
    decision: Decision
    draft_po: DraftPO | None = None
    tool_calls: list[ToolCallTrace]
    approval_request: ApprovalRequestInfo | None = None
    clarification_request: ClarificationRequestInfo | None = None
    clarification_answer: ClarificationAnswerInfo | None = None
    created_at: datetime


class StoredToolCall(StrictModel):
    """A single persisted tool-call trace, including its execution order and timestamp."""

    sequence: int
    tool: str
    status: ToolCallStatus
    boundary: str | None = None
    trace: ToolCallTrace
    created_at: datetime


class StoredDraftPO(StrictModel):
    """The persisted draft PO row for a run, if one was created."""

    po_id: str
    status: str
    draft_po: DraftPO
    created_at: datetime
    updated_at: datetime


class StoredApproval(StrictModel):
    """A persisted human-approval action recorded against a run."""

    id: int
    approver_id: str
    approved: bool
    comment: str | None = None
    created_at: datetime


class AgentRunDetail(StrictModel):
    """Full view of everything persisted for a run: state, plan, traces, and actions."""

    run_id: str
    user_id: str
    department: str
    status: RunStatus
    decision_action: DecisionAction
    risk_level: RiskLevel
    requires_human_approval: bool
    approval_completed: bool
    request: AgentRunRequest
    plan: AgentRunPlan
    response: AgentRunResponse
    tool_calls: list[StoredToolCall]
    draft_po: StoredDraftPO | None = None
    approvals: list[StoredApproval] = Field(default_factory=list)
    pending_po_input: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
