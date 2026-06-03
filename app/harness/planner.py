from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.agents import (
    AgentRunRequest,
    AgentState,
    AgentStep,
    CheckPolicyOutput,
    ClarifyStep,
    Decision,
    DecisionAction,
    DraftPO,
    FinalStep,
    LookupCatalogOutput,
    ParsedAgentRequest,
    ToolStep,
)


BUDGET_PATTERNS = [
    re.compile(r"(?:usd|us\$|\$)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:usd|dollars?)", re.IGNORECASE),
    re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:美元|美金)"),
    re.compile(
        r"(?:budget|under|below|less than|not exceed|within)"
        r"[^0-9]{0,24}([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:預算|预算)[^0-9]{0,24}([0-9][0-9,]*(?:\.[0-9]+)?)"),
]

QUANTITY_PATTERNS = [
    re.compile(
        r"(?<![\d.])([0-9]{1,4})\s*(?:x\s*)?"
        r"(?:seats?|licenses?|units?|laptops?|monitors?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\d.])([0-9]{1,4})\s+"
        r"(?:figma|notion|slack|macbook|oracle|salesforce|dell)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\d.])([0-9]{1,4})\s*(?:個|个|台|席次|張|份|套)"),
]

BYPASS_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:company\s+)?polic", re.IGNORECASE),
    re.compile(r"bypass\s+approval", re.IGNORECASE),
    re.compile(r"no\s+approval\s+needed", re.IGNORECASE),
    re.compile(r"without\s+approval", re.IGNORECASE),
    re.compile(r"place\s+the\s+order\s+directly", re.IGNORECASE),
    re.compile(r"submit\s+(?:it\s+)?(?:to\s+)?erp", re.IGNORECASE),
    re.compile(r"忽略.*(?:公司)?政策"),
    re.compile(r"(?:不|無|免)(?:要|需|需要)?核准"),
    re.compile(r"跳過.*核准"),
    re.compile(r"繞過.*核准"),
]


class Planner(Protocol):
    name: str

    def parse(self, request: AgentRunRequest) -> ParsedAgentRequest:
        """Extract durable request fields before the step loop starts."""
        ...

    def next_step(self, state: AgentState) -> AgentStep:
        """Choose exactly one next action from the current observed state."""
        ...


class PlannerError(RuntimeError):
    """Raised when planning fails before tool execution is safe."""


@dataclass(frozen=True)
class RuleBasedPlanner:
    """Deterministic observe-decide-act planner for the procurement demo domain."""

    name: str = "rule_based"

    def parse(self, request: AgentRunRequest) -> ParsedAgentRequest:
        budget_limit, budget_spans = _extract_budget(request.message)
        quantity = _extract_quantity(request.message, budget_spans)
        return ParsedAgentRequest(
            item_query=request.message,
            quantity=quantity,
            budget_limit=budget_limit,
            department=request.department.lower(),
            direct_order_requested=_detect_direct_order_request(request.message),
        )

    def next_step(self, state: AgentState) -> AgentStep:
        parsed = state.parsed

        catalog_output = _catalog_output(state)
        if catalog_output is None:
            return ToolStep(tool="lookup_catalog", input={"query": parsed.item_query})

        if not catalog_output.match_found or catalog_output.item is None:
            alternatives = ", ".join(catalog_output.alternatives)
            question = "Which catalog item should I use for this purchase?"
            if alternatives:
                question += f" Options include: {alternatives}."
            return ClarifyStep(
                question=question,
                missing_fields=["item_query"],
                answer_hint="Reply with the catalog item name, for example: Figma Enterprise Seat.",
            )

        item = catalog_output.item
        if parsed.quantity is None:
            return ClarifyStep(
                question=f"How many units, seats, or licenses of {item.name} should I buy?",
                missing_fields=["quantity"],
                answer_hint="Reply with a positive number, for example: 3.",
            )

        policy_output = _policy_output(state)
        if policy_output is None:
            return ToolStep(tool="check_policy", input=_policy_input(state, catalog_output))

        if parsed.direct_order_requested and not state.has_run("submit_to_erp"):
            return ToolStep(
                tool="submit_to_erp",
                input={
                    "draft_po_id": "planner_requested_direct_submission",
                    "approval_reference": "",
                },
            )

        if policy_output.rejected:
            return FinalStep(decision=_reject_decision(policy_output))

        if "budget_exceeded" in policy_output.flags:
            budget_limit = parsed.budget_limit or 0
            estimated_total = _estimated_total(state, catalog_output)
            return ClarifyStep(
                question=(
                    f"The estimated total is USD {estimated_total}, which exceeds the stated "
                    f"budget of USD {budget_limit}. What updated quantity or budget limit "
                    "should I use?"
                ),
                missing_fields=["quantity", "budget_limit"],
                answer_hint="Reply with a lower quantity like 3, or an updated budget like USD 9000.",
            )

        if policy_output.requires_human_approval:
            return FinalStep(
                decision=_approval_decision(policy_output),
                pending_tool=ToolStep(
                    tool="create_draft_po",
                    input=_draft_input(state, catalog_output),
                ),
            )

        draft_output = state.output_of("create_draft_po")
        if draft_output is None:
            return ToolStep(tool="create_draft_po", input=_draft_input(state, catalog_output))

        return FinalStep(
            decision=_create_po_decision(policy_output),
            draft_po=DraftPO.model_validate(draft_output),
        )


@dataclass(frozen=True)
class MockLLMPlanner(RuleBasedPlanner):
    """Mock LLM adapter with the same parse/next_step contract as the live planner."""

    name: str = "mock_llm"


def _catalog_output(state: AgentState) -> LookupCatalogOutput | None:
    output = state.output_of("lookup_catalog")
    return LookupCatalogOutput.model_validate(output) if output is not None else None


def _policy_output(state: AgentState) -> CheckPolicyOutput | None:
    output = state.output_of("check_policy")
    return CheckPolicyOutput.model_validate(output) if output is not None else None


def _policy_input(state: AgentState, catalog_output: LookupCatalogOutput) -> dict[str, object]:
    if catalog_output.item is None or state.parsed.quantity is None:
        raise PlannerError("Policy input requires a matched catalog item and quantity.")

    item = catalog_output.item
    estimated_total = state.parsed.quantity * item.unit_price
    return {
        "user_id": state.request.user_id,
        "department": state.parsed.department,
        "item_id": item.id,
        "item_name": item.name,
        "category": item.category,
        "quantity": state.parsed.quantity,
        "unit_price": item.unit_price,
        "estimated_total": estimated_total,
        "budget_limit": state.parsed.budget_limit,
        "currency": item.currency,
        "direct_order_requested": state.parsed.direct_order_requested,
    }


def _draft_input(state: AgentState, catalog_output: LookupCatalogOutput) -> dict[str, object]:
    if catalog_output.item is None or state.parsed.quantity is None:
        raise PlannerError("Draft PO input requires a matched catalog item and quantity.")

    item = catalog_output.item
    estimated_total = state.parsed.quantity * item.unit_price
    return {
        "user_id": state.request.user_id,
        "department": state.parsed.department,
        "item_id": item.id,
        "item_name": item.name,
        "quantity": state.parsed.quantity,
        "unit_price": item.unit_price,
        "estimated_total": estimated_total,
        "currency": item.currency,
    }


def _estimated_total(state: AgentState, catalog_output: LookupCatalogOutput) -> int:
    if catalog_output.item is None or state.parsed.quantity is None:
        return 0
    return state.parsed.quantity * catalog_output.item.unit_price


def _reject_decision(policy: CheckPolicyOutput) -> Decision:
    return Decision(
        action=DecisionAction.REJECT,
        risk_level=policy.risk_level,
        requires_human_approval=False,
        reason=policy.reason,
        policy_flags=policy.flags,
    )


def _approval_decision(policy: CheckPolicyOutput) -> Decision:
    return Decision(
        action=DecisionAction.NEED_HUMAN_APPROVAL,
        risk_level=policy.risk_level,
        requires_human_approval=True,
        reason=policy.reason,
        policy_flags=policy.flags,
    )


def _create_po_decision(policy: CheckPolicyOutput) -> Decision:
    return Decision(
        action=DecisionAction.CREATE_DRAFT_PO,
        risk_level=policy.risk_level,
        requires_human_approval=False,
        reason=policy.reason,
        policy_flags=policy.flags,
    )


def _extract_budget(message: str) -> tuple[int | None, list[tuple[int, int]]]:
    for pattern in BUDGET_PATTERNS:
        match = pattern.search(message)
        if match:
            return _to_int(match.group(1)), [match.span(1)]
    return None, []


def _extract_quantity(message: str, ignored_spans: list[tuple[int, int]]) -> int | None:
    for pattern in QUANTITY_PATTERNS:
        for match in pattern.finditer(message):
            if _overlaps_any(match.span(1), ignored_spans):
                continue
            value = _to_int(match.group(1))
            if value > 0:
                return value
    return None


def _detect_direct_order_request(message: str) -> bool:
    return any(pattern.search(message) for pattern in BYPASS_PATTERNS)


def _to_int(value: str) -> int:
    return int(float(value.replace(",", "")))


def _overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)
