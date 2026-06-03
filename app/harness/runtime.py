from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import TypeAdapter

from app.db.repositories import AgentRunRecord, AgentRunRepository
from app.harness.clarification import ClarificationResolution
from app.harness.planner import (
    Planner,
    PlannerError,
    _detect_direct_order_request,
    _extract_budget,
    _extract_quantity,
)
from app.harness.tools import ToolExecutionResult, ToolRegistry, blocked_tool_trace
from app.schemas.agents import (
    AgentRunDetail,
    AgentRunPlan,
    AgentRunRequest,
    AgentRunResponse,
    AgentState,
    AgentStep,
    ApprovalRequestInfo,
    ApproveRunRequest,
    CatalogItem,
    CheckPolicyOutput,
    ClarificationAnswerInfo,
    ClarificationRequestInfo,
    ClarifyRunRequest,
    ClarifyStep,
    Decision,
    DecisionAction,
    DraftPO,
    FinalStep,
    LookupCatalogOutput,
    ParsedAgentRequest,
    RiskLevel,
    RunStatus,
    ToolCallStatus,
    ToolCallTrace,
    ToolStep,
)


logger = logging.getLogger("app.harness.runtime")
AGENT_STEP_ADAPTER = TypeAdapter(AgentStep)


class ClarificationResolver(Protocol):
    name: str

    def resolve_clarification(
        self,
        *,
        request: AgentRunRequest,
        parsed: ParsedAgentRequest,
        clarification_request: ClarificationRequestInfo,
        answer: ClarificationAnswerInfo,
    ) -> ClarificationResolution:
        """Extract structured updates from a clarification answer."""
        ...


class AgentHarness:
    """Small observe-decide-act runtime responsible for loop state and guardrails."""

    MAX_STEPS = 6

    def __init__(
        self,
        *,
        planner: Planner,
        tool_registry: ToolRegistry,
        run_repository: AgentRunRepository,
        clarification_resolver: ClarificationResolver | None = None,
        catalog: list[CatalogItem] | None = None,
    ) -> None:
        self.planner = planner
        self.tool_registry = tool_registry
        self.run_repository = run_repository
        self.clarification_resolver = clarification_resolver
        self.catalog = catalog or []

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc)
        logger.info(
            "Run %s started: planner=%s user=%s department=%s message=%r",
            run_id,
            getattr(self.planner, "name", self.planner.__class__.__name__),
            request.user_id,
            request.department,
            request.message,
        )
        try:
            parsed = self.planner.parse(request)
        except PlannerError as exc:
            logger.error("Run %s parse failed: %s", run_id, exc)
            state = self._fallback_state(run_id, request, error=exc)
            return self._finalize_and_store(
                state=state,
                response=self._failed_response(
                    run_id,
                    created_at,
                    reason="Agent planning failed before guarded tool execution.",
                ),
            )

        state = AgentState(
            run_id=run_id,
            request=request,
            parsed=parsed,
            metadata={"planner": getattr(self.planner, "name", self.planner.__class__.__name__)},
        )
        logger.info(
            "Run %s parsed: item_query=%r quantity=%s budget=%s direct_order=%s",
            run_id,
            parsed.item_query,
            parsed.quantity,
            parsed.budget_limit,
            parsed.direct_order_requested,
        )
        return self._loop(state=state, created_at=created_at)

    def continue_run(self, run_id: str, request: ClarifyRunRequest) -> AgentRunResponse:
        record = self.run_repository.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.response.status != RunStatus.NEEDS_CLARIFICATION:
            raise ValueError("Run is not awaiting clarification.")
        if record.response.clarification_request is None:
            raise ValueError("Run has no active clarification request.")

        answer = ClarificationAnswerInfo(
            answer=request.answer,
            answered_at=datetime.now(timezone.utc),
        )
        logger.info("Run %s received clarification answer: %r", run_id, request.answer)
        state = self._state_from_record(record)
        state = self._apply_clarification_answer(
            state=state,
            clarification_request=record.response.clarification_request,
            answer=answer,
        )
        return self._loop(
            state=state,
            created_at=record.response.created_at,
            clarification_answer=answer,
        )

    def approve_run(self, run_id: str, request: ApproveRunRequest) -> AgentRunResponse:
        record = self.run_repository.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.response.status != RunStatus.AWAITING_HUMAN_APPROVAL:
            raise ValueError("Run is not awaiting human approval.")

        state = self._state_from_record(record)
        state.observations[:] = list(record.response.tool_calls)
        logger.info(
            "Run %s approval received: approver=%s approved=%s",
            run_id,
            request.approver_id,
            request.approved,
        )
        self.run_repository.record_approval(run_id, request)
        if not request.approved:
            response = AgentRunResponse(
                run_id=run_id,
                status=RunStatus.REJECTED,
                decision=Decision(
                    action=DecisionAction.REJECT,
                    risk_level=record.response.decision.risk_level,
                    requires_human_approval=False,
                    reason=request.comment or "Human approver rejected the request.",
                    policy_flags=record.response.decision.policy_flags,
                ),
                tool_calls=state.observations,
                clarification_answer=record.response.clarification_answer,
                created_at=record.response.created_at,
            )
            return self._finalize_and_store(state=state, response=response)

        if record.pending_po_input is None:
            raise ValueError("Run has no pending draft PO input.")

        approval_reference = f"approval_{request.approver_id}_{run_id}"
        draft_input = {
            **record.pending_po_input,
            "approval_reference": approval_reference,
        }
        draft_result = self.tool_registry.execute(
            "create_draft_po",
            draft_input,
            approval_completed=True,
        )
        state.observations.append(draft_result.trace)
        if draft_result.output is None:
            return self._finalize_and_store(
                state=state,
                response=self._failed_response(
                    run_id,
                    record.response.created_at,
                    tool_calls=state.observations,
                    clarification_answer=record.response.clarification_answer,
                ),
                approval_completed=True,
            )

        draft_po = DraftPO.model_validate(draft_result.output)
        response = AgentRunResponse(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            decision=Decision(
                action=DecisionAction.CREATE_DRAFT_PO,
                risk_level=record.response.decision.risk_level,
                requires_human_approval=False,
                reason=f"Human approval completed by {request.approver_id}; draft PO created.",
                policy_flags=record.response.decision.policy_flags,
            ),
            draft_po=draft_po,
            tool_calls=state.observations,
            clarification_answer=record.response.clarification_answer,
            created_at=record.response.created_at,
        )
        return self._finalize_and_store(
            state=state,
            response=response,
            approval_completed=True,
        )

    def get_run(self, run_id: str) -> AgentRunResponse | None:
        record = self.run_repository.get(run_id)
        return record.response if record else None

    def get_run_detail(self, run_id: str) -> AgentRunDetail | None:
        return self.run_repository.get_detail(run_id)

    def _loop(
        self,
        *,
        state: AgentState,
        created_at: datetime,
        clarification_answer: ClarificationAnswerInfo | None = None,
    ) -> AgentRunResponse:
        for step_number in range(1, self.MAX_STEPS + 1):
            try:
                raw_step = self.planner.next_step(state)
                step = AGENT_STEP_ADAPTER.validate_python(raw_step)
            except Exception as exc:
                logger.warning("Run %s planner step failed: %s", state.run_id, exc)
                return self._finalize_and_store(
                    state=state,
                    response=self._failed_response(
                        state.run_id,
                        created_at,
                        tool_calls=state.observations,
                        reason="Planner failed while choosing the next step.",
                        clarification_answer=clarification_answer,
                    ),
                )

            logger.info(
                "Run %s step %d/%d: %s",
                state.run_id,
                step_number,
                self.MAX_STEPS,
                step.kind,
            )
            if isinstance(step, ToolStep):
                try:
                    step = _normalized_tool_step(state, step)
                except PlannerError as exc:
                    logger.warning("Run %s tool normalization failed: %s", state.run_id, exc)
                    return self._finalize_and_store(
                        state=state,
                        response=self._failed_response(
                            state.run_id,
                            created_at,
                            tool_calls=state.observations,
                            reason=str(exc),
                            clarification_answer=clarification_answer,
                        ),
                    )

                guarded_response = self._guard_tool_step(
                    state=state,
                    step=step,
                    created_at=created_at,
                    clarification_answer=clarification_answer,
                )
                if guarded_response is not None:
                    return guarded_response

                result = self.tool_registry.execute(
                    step.tool,
                    step.input,
                    approval_completed=state.approval_completed,
                )
                state.observations.append(result.trace)
                blocked_response = self._handle_tool_result(
                    state=state,
                    result=result,
                    created_at=created_at,
                    clarification_answer=clarification_answer,
                )
                if blocked_response is not None:
                    return blocked_response
                continue

            if isinstance(step, ClarifyStep):
                return self._pause_for_clarification(
                    state=state,
                    step=step,
                    created_at=created_at,
                    clarification_answer=clarification_answer,
                )

            if isinstance(step, FinalStep):
                return self._finalize_step(
                    state=state,
                    step=step,
                    created_at=created_at,
                    clarification_answer=clarification_answer,
                )

        return self._finalize_and_store(
            state=state,
            response=self._failed_response(
                state.run_id,
                created_at,
                tool_calls=state.observations,
                reason="Agent step budget exceeded.",
                clarification_answer=clarification_answer,
            ),
        )

    def _guard_tool_step(
        self,
        *,
        state: AgentState,
        step: ToolStep,
        created_at: datetime,
        clarification_answer: ClarificationAnswerInfo | None,
    ) -> AgentRunResponse | None:
        if step.tool != "create_draft_po":
            return None

        policy = _policy_output(state)
        if policy is None:
            state.observations.append(
                blocked_tool_trace(
                    tool=step.tool,
                    raw_input=step.input,
                    risk_level=RiskLevel.MEDIUM,
                    error="Policy must be checked before draft PO creation.",
                    boundary="PolicyBoundary",
                )
            )
            return self._finalize_and_store(
                state=state,
                response=self._failed_response(
                    state.run_id,
                    created_at,
                    tool_calls=state.observations,
                    reason="Policy must be checked before draft PO creation.",
                    clarification_answer=clarification_answer,
                ),
            )

        if policy.rejected:
            return self._finalize_step(
                state=state,
                step=FinalStep(
                    decision=Decision(
                        action=DecisionAction.REJECT,
                        risk_level=policy.risk_level,
                        requires_human_approval=False,
                        reason=policy.reason,
                        policy_flags=policy.flags,
                    )
                ),
                created_at=created_at,
                clarification_answer=clarification_answer,
            )

        if "budget_exceeded" in policy.flags:
            budget_limit = _latest_policy_budget_limit(state) or state.parsed.budget_limit or 0
            estimated_total = _latest_policy_estimated_total(state) or 0
            return self._pause_for_clarification(
                state=state,
                step=ClarifyStep(
                    question=(
                        f"The estimated total is USD {estimated_total}, which exceeds the stated "
                        f"budget of USD {budget_limit}. What updated quantity or budget limit "
                        "should I use?"
                    ),
                    missing_fields=["quantity", "budget_limit"],
                    answer_hint=(
                        "Reply with a lower quantity like 3, or an updated budget like USD 9000."
                    ),
                ),
                created_at=created_at,
                clarification_answer=clarification_answer,
            )

        if policy.requires_human_approval:
            return self._finalize_step(
                state=state,
                step=FinalStep(
                    decision=Decision(
                        action=DecisionAction.NEED_HUMAN_APPROVAL,
                        risk_level=policy.risk_level,
                        requires_human_approval=True,
                        reason=policy.reason,
                        policy_flags=policy.flags,
                    ),
                    pending_tool=step,
                ),
                created_at=created_at,
                clarification_answer=clarification_answer,
            )

        return None

    def _handle_tool_result(
        self,
        *,
        state: AgentState,
        result: ToolExecutionResult,
        created_at: datetime,
        clarification_answer: ClarificationAnswerInfo | None,
    ) -> AgentRunResponse | None:
        trace = result.trace
        if trace.status == ToolCallStatus.SUCCESS:
            return None

        if trace.status == ToolCallStatus.BLOCKED and trace.boundary == "ApprovalBoundary":
            policy = _policy_output(state)
            if policy is not None:
                return None
            return self._finalize_and_store(
                state=state,
                response=self._failed_response(
                    state.run_id,
                    created_at,
                    tool_calls=state.observations,
                    reason="Tool was blocked before policy approval state was established.",
                    clarification_answer=clarification_answer,
                ),
            )

        return self._finalize_and_store(
            state=state,
            response=self._failed_response(
                state.run_id,
                created_at,
                tool_calls=state.observations,
                reason=trace.error or "Tool execution failed.",
                clarification_answer=clarification_answer,
            ),
        )

    def _finalize_step(
        self,
        *,
        state: AgentState,
        step: FinalStep,
        created_at: datetime,
        clarification_answer: ClarificationAnswerInfo | None,
    ) -> AgentRunResponse:
        policy = _policy_output(state)
        decision = _cross_checked_decision(step.decision, policy)

        if decision.action == DecisionAction.NEED_HUMAN_APPROVAL:
            pending_tool = step.pending_tool
            if pending_tool is not None:
                try:
                    pending_tool = _normalized_tool_step(state, pending_tool)
                except PlannerError as exc:
                    return self._finalize_and_store(
                        state=state,
                        response=self._failed_response(
                            state.run_id,
                            created_at,
                            tool_calls=state.observations,
                            reason=str(exc),
                            clarification_answer=clarification_answer,
                        ),
                    )
            pending_po_input = pending_tool.input if pending_tool is not None else None
            if pending_tool is not None and not _has_blocked_trace(state, pending_tool.tool):
                state.observations.append(
                    blocked_tool_trace(
                        tool=pending_tool.tool,
                        raw_input=pending_tool.input,
                        risk_level=decision.risk_level,
                        error="Policy requires human approval before this tool can execute.",
                    )
                )
            response = AgentRunResponse(
                run_id=state.run_id,
                status=RunStatus.AWAITING_HUMAN_APPROVAL,
                decision=decision,
                tool_calls=state.observations,
                approval_request=ApprovalRequestInfo(
                    run_id=state.run_id,
                    reason=decision.reason,
                    required_for=decision.policy_flags,
                ),
                clarification_answer=clarification_answer,
                created_at=created_at,
            )
            return self._finalize_and_store(
                state=state,
                response=response,
                pending_po_input=pending_po_input,
            )

        if decision.action == DecisionAction.REJECT:
            response = AgentRunResponse(
                run_id=state.run_id,
                status=RunStatus.REJECTED,
                decision=decision,
                tool_calls=state.observations,
                clarification_answer=clarification_answer,
                created_at=created_at,
            )
            return self._finalize_and_store(state=state, response=response)

        if decision.action == DecisionAction.CREATE_DRAFT_PO:
            draft_po = step.draft_po or _draft_output(state)
            if draft_po is None:
                return self._finalize_and_store(
                    state=state,
                    response=self._failed_response(
                        state.run_id,
                        created_at,
                        tool_calls=state.observations,
                        reason="Planner finalized draft creation before a draft PO existed.",
                        clarification_answer=clarification_answer,
                    ),
                )
            response = AgentRunResponse(
                run_id=state.run_id,
                status=RunStatus.COMPLETED,
                decision=decision,
                draft_po=draft_po,
                tool_calls=state.observations,
                clarification_answer=clarification_answer,
                created_at=created_at,
            )
            return self._finalize_and_store(state=state, response=response)

        return self._pause_for_clarification(
            state=state,
            step=ClarifyStep(
                question=decision.reason,
                missing_fields=["unknown"],
                answer_hint="Reply with the missing information.",
            ),
            created_at=created_at,
            clarification_answer=clarification_answer,
        )

    def _pause_for_clarification(
        self,
        *,
        state: AgentState,
        step: ClarifyStep,
        created_at: datetime,
        clarification_answer: ClarificationAnswerInfo | None,
    ) -> AgentRunResponse:
        clarification = ClarificationRequestInfo(
            run_id=state.run_id,
            question=step.question,
            missing_fields=step.missing_fields,
            answer_hint=step.answer_hint,
        )
        decision = Decision(
            action=DecisionAction.ASK_CLARIFICATION,
            risk_level=_policy_output(state).risk_level if _policy_output(state) else RiskLevel.LOW,
            requires_human_approval=False,
            reason=_clarification_reason(step),
            policy_flags=_policy_output(state).flags if _policy_output(state) else [],
        )
        response = AgentRunResponse(
            run_id=state.run_id,
            status=RunStatus.NEEDS_CLARIFICATION,
            decision=decision,
            tool_calls=state.observations,
            clarification_request=clarification,
            clarification_answer=clarification_answer,
            created_at=created_at,
        )
        return self._finalize_and_store(state=state, response=response)

    def _apply_clarification_answer(
        self,
        *,
        state: AgentState,
        clarification_request: ClarificationRequestInfo,
        answer: ClarificationAnswerInfo,
    ) -> AgentState:
        local_resolution = _resolve_clarification_locally(
            clarification_request=clarification_request,
            answer=answer,
            catalog=self.catalog,
        )
        local_updates = _updates_from_resolution(
            parsed=state.parsed,
            clarification_request=clarification_request,
            resolution=local_resolution,
        )
        llm_resolution: ClarificationResolution | None = None
        llm_error: str | None = None
        llm_updates: dict[str, Any] = {}

        if (
            not _has_requested_field_update(local_updates, clarification_request)
            and self.clarification_resolver is not None
        ):
            try:
                llm_resolution = self.clarification_resolver.resolve_clarification(
                    request=state.request,
                    parsed=state.parsed,
                    clarification_request=clarification_request,
                    answer=answer,
                )
                llm_updates = _updates_from_resolution(
                    parsed=state.parsed,
                    clarification_request=clarification_request,
                    resolution=llm_resolution,
                )
            except Exception as exc:  # pragma: no cover - provider-specific defensive guard
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Clarification resolver failed for run %s: %s", state.run_id, exc)

        parsed_updates = {**local_updates, **llm_updates}
        parsed = ParsedAgentRequest.model_validate(
            {**state.parsed.model_dump(mode="python"), **parsed_updates}
        )
        metadata = _clarification_metadata(
            state=state,
            clarification_request=clarification_request,
            answer=answer,
            parsed_updates=parsed_updates,
            local_resolution=local_resolution,
            local_updates=local_updates,
            llm_resolution=llm_resolution,
            llm_updates=llm_updates,
            llm_error=llm_error,
            resolver_name=(
                getattr(self.clarification_resolver, "name", None)
                if self.clarification_resolver is not None
                else None
            ),
        )
        return state.model_copy(update={"parsed": parsed, "metadata": metadata})

    def _finalize_and_store(
        self,
        *,
        state: AgentState,
        response: AgentRunResponse,
        pending_po_input: dict[str, Any] | None = None,
        approval_completed: bool | None = None,
    ) -> AgentRunResponse:
        validated = AgentRunResponse.model_validate(response.model_dump(mode="python"))
        state = state.model_copy(
            update={
                "observations": validated.tool_calls,
                "approval_completed": (
                    state.approval_completed if approval_completed is None else approval_completed
                ),
            }
        )
        logger.info(
            "Run %s finalized: status=%s action=%s risk=%s flags=%s tool_calls=%d reason=%r",
            validated.run_id,
            validated.status.value,
            validated.decision.action.value,
            validated.decision.risk_level.value,
            validated.decision.policy_flags,
            len(validated.tool_calls),
            validated.decision.reason,
        )
        self.run_repository.save(
            AgentRunRecord(
                request=state.request,
                response=validated,
                plan=self._plan_from_state(state),
                pending_po_input=pending_po_input,
                approval_completed=state.approval_completed,
            )
        )
        return validated

    def _plan_from_state(self, state: AgentState) -> AgentRunPlan:
        return AgentRunPlan(
            planner=getattr(self.planner, "name", self.planner.__class__.__name__),
            parsed_request=state.parsed,
            observations=state.observations,
            metadata=state.metadata,
        )

    @staticmethod
    def _failed_response(
        run_id: str,
        created_at: datetime,
        tool_calls: list[ToolCallTrace] | None = None,
        *,
        reason: str = "Agent run failed during guarded tool execution.",
        clarification_answer: ClarificationAnswerInfo | None = None,
    ) -> AgentRunResponse:
        return AgentRunResponse(
            run_id=run_id,
            status=RunStatus.FAILED,
            decision=Decision(
                action=DecisionAction.REJECT,
                risk_level=RiskLevel.HIGH,
                requires_human_approval=False,
                reason=reason,
            ),
            tool_calls=tool_calls or [],
            clarification_answer=clarification_answer,
            created_at=created_at,
        )

    def _state_from_record(self, record: AgentRunRecord) -> AgentState:
        return AgentState(
            run_id=record.response.run_id,
            request=record.request,
            parsed=record.plan.parsed_request,
            observations=list(record.response.tool_calls),
            approval_completed=record.approval_completed,
            metadata=dict(record.plan.metadata),
        )

    def _fallback_state(
        self,
        run_id: str,
        request: AgentRunRequest,
        *,
        error: Exception,
    ) -> AgentState:
        return AgentState(
            run_id=run_id,
            request=request,
            parsed=ParsedAgentRequest(
                item_query=request.message,
                quantity=None,
                budget_limit=None,
                department=request.department.lower(),
                direct_order_requested=False,
            ),
            metadata={
                "planner": getattr(self.planner, "name", self.planner.__class__.__name__),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def _policy_output(state: AgentState) -> CheckPolicyOutput | None:
    output = state.output_of("check_policy")
    return CheckPolicyOutput.model_validate(output) if output is not None else None


def _catalog_output(state: AgentState) -> LookupCatalogOutput | None:
    output = state.output_of("lookup_catalog")
    return LookupCatalogOutput.model_validate(output) if output is not None else None


def _draft_output(state: AgentState) -> DraftPO | None:
    output = state.output_of("create_draft_po")
    return DraftPO.model_validate(output) if output is not None else None


def _normalized_tool_step(state: AgentState, step: ToolStep) -> ToolStep:
    if step.tool == "lookup_catalog":
        query = step.input.get("query") or state.parsed.item_query
        return ToolStep(tool=step.tool, input={"query": str(query)})

    if step.tool == "check_policy":
        return ToolStep(tool=step.tool, input=_trusted_policy_input(state, step.input))

    if step.tool == "create_draft_po":
        return ToolStep(tool=step.tool, input=_trusted_draft_input(state, step.input))

    return step


def _trusted_policy_input(state: AgentState, raw_input: dict[str, Any]) -> dict[str, Any]:
    catalog_output = _catalog_output(state)
    if catalog_output is None or catalog_output.item is None:
        raise PlannerError("Policy input requires a successful catalog lookup.")

    quantity = _positive_int(raw_input.get("quantity")) or state.parsed.quantity
    if quantity is None:
        raise PlannerError("Policy input requires a positive quantity.")

    item = catalog_output.item
    estimated_total = quantity * item.unit_price
    return {
        "user_id": state.request.user_id,
        "department": state.parsed.department,
        "item_id": item.id,
        "item_name": item.name,
        "category": item.category,
        "quantity": quantity,
        "unit_price": item.unit_price,
        "estimated_total": estimated_total,
        "budget_limit": _positive_int(raw_input.get("budget_limit")) or state.parsed.budget_limit,
        "currency": item.currency,
        "direct_order_requested": state.parsed.direct_order_requested,
    }


def _trusted_draft_input(state: AgentState, raw_input: dict[str, Any]) -> dict[str, Any]:
    catalog_output = _catalog_output(state)
    if catalog_output is None or catalog_output.item is None:
        raise PlannerError("Draft PO input requires a successful catalog lookup.")

    quantity = (
        _positive_int(raw_input.get("quantity"))
        or _latest_policy_quantity(state)
        or state.parsed.quantity
    )
    if quantity is None:
        raise PlannerError("Draft PO input requires a positive quantity.")

    item = catalog_output.item
    estimated_total = quantity * item.unit_price
    return {
        "user_id": state.request.user_id,
        "department": state.parsed.department,
        "item_id": item.id,
        "item_name": item.name,
        "quantity": quantity,
        "unit_price": item.unit_price,
        "estimated_total": estimated_total,
        "currency": item.currency,
        "approval_reference": raw_input.get("approval_reference"),
    }


def _latest_policy_input(state: AgentState) -> dict[str, Any] | None:
    trace = state.latest_trace("check_policy")
    if trace is None or trace.status != ToolCallStatus.SUCCESS or trace.input is None:
        return None
    return trace.input


def _latest_policy_quantity(state: AgentState) -> int | None:
    policy_input = _latest_policy_input(state)
    if policy_input is None:
        return None
    return _positive_int(policy_input.get("quantity"))


def _latest_policy_budget_limit(state: AgentState) -> int | None:
    policy_input = _latest_policy_input(state)
    if policy_input is None:
        return None
    return _positive_int(policy_input.get("budget_limit"))


def _latest_policy_estimated_total(state: AgentState) -> int | None:
    policy_input = _latest_policy_input(state)
    if policy_input is None:
        return None
    return _positive_int(policy_input.get("estimated_total"))


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _cross_checked_decision(
    decision: Decision,
    policy: CheckPolicyOutput | None,
) -> Decision:
    if policy is None:
        return decision
    if policy.rejected:
        return Decision(
            action=DecisionAction.REJECT,
            risk_level=policy.risk_level,
            requires_human_approval=False,
            reason=policy.reason,
            policy_flags=policy.flags,
        )
    if policy.requires_human_approval and decision.action == DecisionAction.CREATE_DRAFT_PO:
        return Decision(
            action=DecisionAction.NEED_HUMAN_APPROVAL,
            risk_level=policy.risk_level,
            requires_human_approval=True,
            reason=policy.reason,
            policy_flags=policy.flags,
        )
    return decision


def _has_blocked_trace(state: AgentState, tool: str) -> bool:
    trace = state.latest_trace(tool)
    return trace is not None and trace.status == ToolCallStatus.BLOCKED


def _clarification_reason(step: ClarifyStep) -> str:
    missing = set(step.missing_fields)
    if missing == {"quantity"}:
        return "Quantity is required before policy can be checked."
    if "item_query" in missing:
        return "I could not match the requested item to the catalog."
    if "budget_limit" in missing:
        return "The estimated total exceeds the user's stated budget."
    return "More information is required before the agent can continue."


def _resolve_clarification_locally(
    *,
    clarification_request: ClarificationRequestInfo,
    answer: ClarificationAnswerInfo,
    catalog: list[CatalogItem],
) -> ClarificationResolution:
    missing_fields = set(clarification_request.missing_fields)
    budget_limit, budget_spans = _extract_budget(answer.answer)
    quantity = _extract_quantity(answer.answer, budget_spans)
    item_query: str | None = None

    if quantity is None and "quantity" in missing_fields:
        quantity = _extract_bare_positive_int(answer.answer)

    if "item_query" in missing_fields:
        item_query = _match_catalog_item_name(answer.answer, catalog)

    return ClarificationResolution(
        item_query=item_query,
        quantity=quantity,
        budget_limit=budget_limit,
        direct_order_requested=_detect_direct_order_request(answer.answer),
        rationale="Resolved with deterministic local clarification parsing.",
    )


def _updates_from_resolution(
    *,
    parsed: ParsedAgentRequest,
    clarification_request: ClarificationRequestInfo,
    resolution: ClarificationResolution,
) -> dict[str, Any]:
    parsed_updates: dict[str, Any] = {}
    missing_fields = set(clarification_request.missing_fields)
    if "item_query" in missing_fields and resolution.item_query is not None:
        parsed_updates["item_query"] = resolution.item_query
    if "quantity" in missing_fields and resolution.quantity is not None:
        parsed_updates["quantity"] = resolution.quantity
    if "budget_limit" in missing_fields and resolution.budget_limit is not None:
        parsed_updates["budget_limit"] = resolution.budget_limit
    if resolution.direct_order_requested and not parsed.direct_order_requested:
        parsed_updates["direct_order_requested"] = True

    if not parsed_updates:
        return {}

    ParsedAgentRequest.model_validate({**parsed.model_dump(mode="python"), **parsed_updates})
    return parsed_updates


def _has_requested_field_update(
    parsed_updates: dict[str, Any],
    clarification_request: ClarificationRequestInfo,
) -> bool:
    requested_fields = set(clarification_request.missing_fields)
    return any(field in parsed_updates for field in requested_fields)


def _clarification_metadata(
    *,
    state: AgentState,
    clarification_request: ClarificationRequestInfo,
    answer: ClarificationAnswerInfo,
    parsed_updates: dict[str, Any],
    local_resolution: ClarificationResolution,
    local_updates: dict[str, Any],
    llm_resolution: ClarificationResolution | None,
    llm_updates: dict[str, Any],
    llm_error: str | None,
    resolver_name: str | None,
) -> dict[str, Any]:
    metadata = dict(state.metadata)
    history = list(metadata.get("clarification_history", []))
    history.append(
        {
            "question": clarification_request.question,
            "missing_fields": clarification_request.missing_fields,
            "answer": answer.answer,
            "answered_at": answer.answered_at.isoformat(),
            "local_resolution": local_resolution.model_dump(mode="json"),
            "local_updated_fields": sorted(local_updates),
            "llm_resolution": (
                llm_resolution.model_dump(mode="json") if llm_resolution is not None else None
            ),
            "llm_updated_fields": sorted(llm_updates),
            "llm_error": llm_error,
        }
    )
    metadata["clarification_history"] = history
    metadata["last_clarification_answer"] = {
        "answer": answer.answer,
        "answered_at": answer.answered_at.isoformat(),
        "updated_fields": sorted(parsed_updates),
        "resolver": resolver_name,
        "llm_attempted": llm_resolution is not None or llm_error is not None,
    }

    if parsed_updates:
        invalidated_tools = dict(metadata.get("invalidated_tools", {}))
        invalidated_at = len(state.observations)
        fields = set(parsed_updates)
        if "item_query" in fields:
            for tool in ("lookup_catalog", "check_policy", "create_draft_po", "submit_to_erp"):
                invalidated_tools[tool] = invalidated_at
        if fields & {"quantity", "budget_limit", "direct_order_requested"}:
            for tool in ("check_policy", "create_draft_po", "submit_to_erp"):
                invalidated_tools[tool] = invalidated_at
        metadata["invalidated_tools"] = invalidated_tools

    return metadata


def _extract_bare_positive_int(answer: str) -> int | None:
    if not re.fullmatch(r"\s*[0-9]{1,6}\s*", answer):
        return None
    value = int(answer.strip())
    return value if value > 0 else None


def _match_catalog_item_name(answer: str, catalog: list[CatalogItem]) -> str | None:
    normalized_answer = _normalize_catalog_text(answer)
    if not normalized_answer:
        return None

    for item in catalog:
        terms = [item.name, *item.aliases]
        if any(normalized_answer == _normalize_catalog_text(term) for term in terms):
            return item.name
    return None


def _normalize_catalog_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()
