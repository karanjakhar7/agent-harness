from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.db.repositories import AgentRunRepository
from app.harness.llm_planner import LLMPlanner
from app.harness.planner import PlannerError
from app.harness.runtime import AgentHarness
from app.harness.tools import build_default_tool_registry
from app.schemas.agents import (
    AgentRunRequest,
    AgentState,
    AgentStep,
    ClarifyRunRequest,
    Decision,
    DecisionAction,
    ParsedAgentRequest,
    RiskLevel,
    RunStatus,
    ToolStep,
)
from app.services.fixtures import load_catalog, load_procurement_fixtures


def test_llm_planner_parse_uses_local_seed_without_llm_call() -> None:
    planner = LLMPlanner(
        model="test/model",
        tool_schemas=_tool_schemas(),
        max_retries=0,
        completion_func=lambda **_: pytest.fail("parse should not call the LLM"),
    )

    parsed = planner.parse(_request())

    assert parsed.item_query == _request().message
    assert parsed.quantity == 3
    assert parsed.budget_limit == 3000
    assert parsed.department == "marketing"
    assert parsed.direct_order_requested is False


def test_llm_planner_next_step_calls_llm_with_tools_without_fixture_context() -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _tool_call_response("lookup_catalog", {"query": "design software"})

    planner = LLMPlanner(
        model="test/model",
        tool_schemas=_tool_schemas(),
        max_retries=0,
        completion_func=fake_completion,
    )
    state = _state(message="Please buy design software for marketing.").model_copy(
        update={
            "metadata": {
                "planner": "llm",
                "last_clarification_answer": {"answer": "Figma", "updated_fields": ["item_query"]},
                "internal_note": "do not leak",
            }
        }
    )

    step = planner.next_step(state)

    assert isinstance(step, ToolStep)
    assert step.tool == "lookup_catalog"
    assert calls[0]["tool_choice"] == "auto"
    assert "response_format" not in calls[0]
    tool_names = {tool["function"]["name"] for tool in calls[0]["tools"]}
    assert {
        "lookup_catalog",
        "check_policy",
        "create_draft_po",
        "submit_to_erp",
        "ask_clarification",
        "finalize_run",
        "request_human_approval",
    } <= tool_names
    prompt_payload = calls[0]["messages"][1]["content"]
    payload = json.loads(prompt_payload)
    assert payload["parsed_seed"]["item_query"] == "Please buy design software for marketing."
    assert payload["metadata"]["planner"] == "llm"
    assert payload["metadata"]["last_clarification_answer"]["answer"] == "Figma"
    assert "internal_note" not in payload["metadata"]
    assert "do not leak" not in prompt_payload
    assert '"catalog":' not in prompt_payload
    assert "MacBook Pro" not in json.dumps(calls[0]["tools"])
    assert "approval_threshold_usd" not in prompt_payload


def test_llm_planner_rejects_no_tool_calls() -> None:
    planner = LLMPlanner(
        model="test/model",
        tool_schemas=_tool_schemas(),
        max_retries=0,
        completion_func=lambda **_: _raw_llm_response(content="No tool needed."),
    )

    with pytest.raises(PlannerError, match="Expected exactly one tool call"):
        planner.next_step(_state())


def test_llm_planner_rejects_multiple_tool_calls() -> None:
    planner = LLMPlanner(
        model="test/model",
        tool_schemas=_tool_schemas(),
        max_retries=0,
        completion_func=lambda **_: _raw_llm_response(
            tool_calls=[
                _tool_call("lookup_catalog", {"query": "Figma"}),
                _tool_call("check_policy", {"quantity": 3}),
            ]
        ),
    )

    with pytest.raises(PlannerError, match="Expected exactly one tool call"):
        planner.next_step(_state())


def test_llm_harness_low_risk_tool_loop_creates_draft(tmp_path: Path) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "Figma"}),
            ("check_policy", {"quantity": 3, "budget_limit": 3000}),
            ("create_draft_po", {}),
            (
                "finalize_run",
                {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "reason": "Draft PO created.",
                    "policy_flags": [],
                },
            ),
        ]
    )

    response = _harness(tmp_path, planner).run(_request())

    assert response.status == RunStatus.COMPLETED
    assert response.decision.action == DecisionAction.CREATE_DRAFT_PO
    assert response.draft_po is not None
    assert response.draft_po.item == "Figma Enterprise Seat"
    assert response.draft_po.estimated_total == 2400
    assert [call.tool for call in response.tool_calls] == [
        "lookup_catalog",
        "check_policy",
        "create_draft_po",
    ]


def test_llm_premature_create_finalize_runs_missing_draft_tool(tmp_path: Path) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "Figma"}),
            ("check_policy", {"quantity": 3, "budget_limit": 3000}),
            (
                "finalize_run",
                {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "reason": "Policy allows the request.",
                    "policy_flags": [],
                },
            ),
            (
                "finalize_run",
                {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "reason": "Draft PO created.",
                    "policy_flags": [],
                },
            ),
        ]
    )

    response = _harness(tmp_path, planner).run(_request())

    assert response.status == RunStatus.COMPLETED
    assert response.decision.action == DecisionAction.CREATE_DRAFT_PO
    assert response.draft_po is not None
    assert response.draft_po.item == "Figma Enterprise Seat"
    assert response.draft_po.estimated_total == 2400
    assert [call.tool for call in response.tool_calls] == [
        "lookup_catalog",
        "check_policy",
        "create_draft_po",
    ]


def test_llm_harness_missing_item_uses_catalog_alternatives_then_clarifies(
    tmp_path: Path,
) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "unknown procurement item"}),
            (
                "ask_clarification",
                {
                    "question": "Which catalog item should I use?",
                    "missing_fields": ["item_query"],
                    "answer_hint": "Reply with the catalog item name.",
                },
            ),
        ]
    )

    response = _harness(tmp_path, planner).run(
        AgentRunRequest(
            user_id="u_test",
            department="marketing",
            message="Please buy the thing we discussed.",
        )
    )

    assert response.status == RunStatus.NEEDS_CLARIFICATION
    assert response.clarification_request is not None
    assert response.clarification_request.missing_fields == ["item_query"]
    catalog_trace = response.tool_calls[0]
    assert catalog_trace.output is not None
    assert catalog_trace.output["match_found"] is False
    assert catalog_trace.output["alternatives"] == [item.name for item in load_catalog()]


def test_llm_clarification_reenters_agent_loop(tmp_path: Path) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "unknown procurement item"}),
            (
                "ask_clarification",
                {
                    "question": "Which catalog item should I use?",
                    "missing_fields": ["item_query"],
                    "answer_hint": "Reply with the catalog item name.",
                },
            ),
            ("lookup_catalog", {"query": "Figma Enterprise Seat"}),
            ("check_policy", {"quantity": 3}),
            ("create_draft_po", {}),
            (
                "finalize_run",
                {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "reason": "Draft PO created.",
                    "policy_flags": [],
                },
            ),
        ]
    )
    harness = _harness(tmp_path, planner)

    pending = harness.run(
        AgentRunRequest(
            user_id="u_test",
            department="marketing",
            message="Please buy the thing we discussed, 3 seats.",
        )
    )
    clarified = harness.continue_run(
        pending.run_id,
        request=ClarifyRunRequest(answer="Figma Enterprise Seat"),
    )

    assert clarified.run_id == pending.run_id
    assert clarified.status == RunStatus.COMPLETED
    assert clarified.draft_po is not None
    assert clarified.draft_po.item == "Figma Enterprise Seat"
    assert clarified.clarification_answer is not None
    assert clarified.clarification_answer.answer == "Figma Enterprise Seat"


def test_llm_clarification_answer_reenters_main_loop_for_freeform_quantity(
    tmp_path: Path,
) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "chatgpt seats"}),
            (
                "ask_clarification",
                {
                    "question": "How many ChatGPT seats would you like to purchase?",
                    "missing_fields": ["quantity"],
                    "answer_hint": "Reply with the requested information.",
                },
            ),
            ("check_policy", {"quantity": 6}),
            ("create_draft_po", {}),
            (
                "finalize_run",
                {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "reason": "Draft PO created.",
                    "policy_flags": [],
                },
            ),
        ]
    )
    harness = _harness(tmp_path, planner)

    pending = harness.run(
        AgentRunRequest(
            user_id="u_test",
            department="marketing",
            message="Please buy ChatGPT seats for marketing.",
        )
    )
    clarified = harness.continue_run(
        pending.run_id,
        request=ClarifyRunRequest(answer="i need 6"),
    )

    assert pending.status == RunStatus.NEEDS_CLARIFICATION
    assert clarified.status == RunStatus.COMPLETED
    assert clarified.draft_po is not None
    assert clarified.draft_po.item == "ChatGPT License"
    assert clarified.draft_po.quantity == 6
    assert clarified.draft_po.estimated_total == 120


def test_llm_policy_required_approval_blocks_attempted_draft(tmp_path: Path) -> None:
    planner = _sequence_planner(
        [
            ("lookup_catalog", {"query": "MacBook Pro"}),
            ("check_policy", {"quantity": 2}),
            ("create_draft_po", {}),
        ]
    )

    response = _harness(tmp_path, planner).run(
        AgentRunRequest(
            user_id="u_test",
            department="engineering",
            message="Please buy 2 MacBook Pro laptops.",
        )
    )

    assert response.status == RunStatus.AWAITING_HUMAN_APPROVAL
    assert response.decision.action == DecisionAction.NEED_HUMAN_APPROVAL
    assert response.tool_calls[-1].tool == "create_draft_po"
    assert response.tool_calls[-1].status == "blocked"
    assert response.tool_calls[-1].boundary == "ApprovalBoundary"


def test_llm_early_submit_to_erp_is_really_blocked(tmp_path: Path) -> None:
    planner = _sequence_planner(
        [
            (
                "submit_to_erp",
                {"draft_po_id": "po_fake", "approval_reference": "approval_fake"},
            )
        ]
    )

    response = _harness(tmp_path, planner).run(_request())

    assert response.status == RunStatus.FAILED
    assert response.tool_calls[-1].tool == "submit_to_erp"
    assert response.tool_calls[-1].status == "blocked"
    assert response.tool_calls[-1].boundary == "ApprovalBoundary"


def test_fake_planner_final_create_is_overridden_when_policy_requires_approval(
    tmp_path: Path,
) -> None:
    planner = StepPlanner(
        parsed=_parsed(item_query="MacBook Pro", quantity=2, budget_limit=None),
        steps=[
            ToolStep(tool="lookup_catalog", input={"query": "MacBook Pro"}),
            ToolStep(tool="check_policy", input={"quantity": 2}),
            {
                "kind": "final",
                "decision": {
                    "action": "CREATE_DRAFT_PO",
                    "risk_level": "LOW",
                    "requires_human_approval": False,
                    "reason": "Unsafe fake approval.",
                    "policy_flags": [],
                },
            },
        ],
    )
    harness = _harness(tmp_path, planner)

    response = harness.run(
        AgentRunRequest(
            user_id="u_test",
            department="engineering",
            message="Please buy 2 MacBook Pro laptops.",
        )
    )

    assert response.status == RunStatus.AWAITING_HUMAN_APPROVAL
    assert response.decision.action == DecisionAction.NEED_HUMAN_APPROVAL
    assert "hardware_purchase" in response.decision.policy_flags


def test_fake_planner_unregistered_tool_fails_safely(tmp_path: Path) -> None:
    planner = StepPlanner(
        parsed=_parsed(),
        steps=[ToolStep(tool="delete_all_data", input={})],
    )
    response = _harness(tmp_path, planner).run(_request())

    assert response.status == RunStatus.FAILED
    assert response.tool_calls[-1].tool == "delete_all_data"
    assert response.tool_calls[-1].status == "blocked"


def test_fake_planner_early_submit_to_erp_is_really_blocked(tmp_path: Path) -> None:
    planner = StepPlanner(
        parsed=_parsed(),
        steps=[
            ToolStep(
                tool="submit_to_erp",
                input={"draft_po_id": "po_fake", "approval_reference": "approval_fake"},
            )
        ],
    )
    response = _harness(tmp_path, planner).run(_request())

    assert response.status == RunStatus.FAILED
    assert response.tool_calls[-1].tool == "submit_to_erp"
    assert response.tool_calls[-1].status == "blocked"
    assert response.tool_calls[-1].boundary == "ApprovalBoundary"


def test_planner_failure_returns_failed_response_without_tool_execution(tmp_path: Path) -> None:
    response = _harness(tmp_path, FailingPlanner()).run(_request())

    assert response.status == RunStatus.FAILED
    assert response.tool_calls == []
    assert response.decision.reason == "Agent planning failed before guarded tool execution."


class StepPlanner:
    name = "fake_llm"

    def __init__(self, *, parsed: ParsedAgentRequest, steps: list[AgentStep | dict[str, Any]]):
        self._parsed = parsed
        self._steps = list(steps)

    def parse(self, request: AgentRunRequest) -> ParsedAgentRequest:
        return self._parsed

    def next_step(self, state: AgentState) -> AgentStep | dict[str, Any]:
        if not self._steps:
            return {
                "kind": "final",
                "decision": Decision(
                    action=DecisionAction.REJECT,
                    risk_level=RiskLevel.HIGH,
                    requires_human_approval=False,
                    reason="No more fake steps.",
                ).model_dump(mode="json"),
            }
        return self._steps.pop(0)


class FailingPlanner:
    name = "fake_llm"

    def parse(self, request: AgentRunRequest) -> ParsedAgentRequest:
        raise PlannerError("invalid JSON from planner")

    def next_step(self, state: AgentState) -> AgentStep:
        raise AssertionError("next_step should not run after parse failure")


def _sequence_planner(tool_calls: list[tuple[str, dict[str, Any]]]) -> LLMPlanner:
    responses = [_tool_call_response(name, arguments) for name, arguments in tool_calls]

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        if not responses:
            raise AssertionError("No fake LLM response queued.")
        return responses.pop(0)

    return LLMPlanner(
        model="test/model",
        tool_schemas=_tool_schemas(),
        max_retries=0,
        completion_func=fake_completion,
    )


def _harness(tmp_path: Path, planner: Any) -> AgentHarness:
    return AgentHarness(
        planner=planner,
        tool_registry=build_default_tool_registry(load_procurement_fixtures()),
        run_repository=AgentRunRepository(tmp_path / "runs.sqlite3"),
        catalog=load_catalog(),
    )


def _tool_schemas() -> list[dict[str, Any]]:
    return build_default_tool_registry(load_procurement_fixtures()).llm_tool_schemas()


def _request() -> AgentRunRequest:
    return AgentRunRequest(
        user_id="u_test",
        department="marketing",
        message="Please buy 3 Figma Enterprise seats for marketing under USD 3000.",
    )


def _state(message: str | None = None) -> AgentState:
    request = _request()
    if message is not None:
        request = AgentRunRequest(
            user_id=request.user_id,
            department=request.department,
            message=message,
        )
    return AgentState(
        run_id="run_test",
        request=request,
        parsed=_parsed(item_query=request.message),
    )


def _parsed(
    *,
    item_query: str = "Figma Enterprise Seat",
    quantity: int | None = 3,
    budget_limit: int | None = 3000,
) -> ParsedAgentRequest:
    return ParsedAgentRequest(
        item_query=item_query,
        quantity=quantity,
        budget_limit=budget_limit,
        department="marketing",
        direct_order_requested=False,
    )


def _tool_call_response(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return _raw_llm_response(tool_calls=[_tool_call(name, arguments)])


def _raw_llm_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=None,
    )


def _tool_call(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
