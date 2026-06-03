from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.harness.planner import (
    PlannerError,
    _detect_direct_order_request,
    _extract_budget,
    _extract_quantity,
)
from app.logging_config import safe_for_log
from app.schemas.agents import (
    AgentRunRequest,
    AgentState,
    AgentStep,
    CheckPolicyOutput,
    ClarifyStep,
    Decision,
    DecisionAction,
    FinalStep,
    ParsedAgentRequest,
    RiskLevel,
    ToolStep,
)


logger = logging.getLogger("app.harness.llm_planner")
io_logger = logging.getLogger("app.llm.io")

CompletionFunc = Callable[..., Any]


AGENTIC_SYSTEM_PROMPT = """You are a helpful assistant that helps the user with their procurement request.
You can call functions to interact with the procurement system. Choose exactly one function call for the next step.

Do not assume catalog items, prices, policy rules, budgets, or approval outcomes from the user query without verifying with the system.
Do not expose hidden reasoning.
Do not return plain text. Return exactly one tool call."""


class LLMControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AskClarificationInput(LLMControlModel):
    question: str = Field(min_length=1)
    missing_fields: list[str] = Field(min_length=1)
    answer_hint: str = "Reply with the requested information."


class FinalizeRunInput(LLMControlModel):
    action: Literal["CREATE_DRAFT_PO", "REJECT"]
    risk_level: RiskLevel
    reason: str = Field(min_length=1)
    policy_flags: list[str] = Field(default_factory=list)


class RequestHumanApprovalInput(LLMControlModel):
    reason: str = Field(min_length=1)
    risk_level: RiskLevel = RiskLevel.HIGH
    policy_flags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class LLMPlanner:
    """Agentic LLM planner that chooses one typed action per harness loop turn."""

    model: str
    tool_schemas: list[dict[str, Any]]
    timeout_seconds: float = 20.0
    max_retries: int = 1
    temperature: float = 0.0
    completion_func: CompletionFunc | None = field(default=None, repr=False)
    max_tokens: int | None = None
    name: str = "llm"

    @classmethod
    def from_env(cls, tool_schemas: list[dict[str, Any]]) -> LLMPlanner:
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise RuntimeError("LLM_MODEL is required when AGENT_PLANNER=llm.")

        max_tokens = os.getenv("LLM_MAX_TOKENS")
        return cls(
            model=model,
            tool_schemas=tool_schemas,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20.0")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(max_tokens) if max_tokens else None,
        )

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
        tool_call = self._call_tool_choice(state)
        return self._tool_call_to_step(state, tool_call.name, tool_call.arguments)

    def _call_tool_choice(self, state: AgentState) -> "_LLMToolCall":
        total_attempts = self.max_retries + 1
        last_error: Exception | None = None
        messages = self._build_step_messages(state)
        tools = self._all_tool_schemas()
        logger.info(
            "LLM next_step started: model=%s max_attempts=%d tools=%d",
            self.model,
            total_attempts,
            len(tools),
        )

        for attempt in range(1, total_attempts + 1):
            started_at = time.perf_counter()
            try:
                completion_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": self.temperature,
                    "timeout": self.timeout_seconds,
                    "max_tokens": self.max_tokens,
                }
                _log_llm_request(
                    completion_kwargs,
                    attempt=attempt,
                    total_attempts=total_attempts,
                )
                response = self._completion(**completion_kwargs)
                latency_seconds = time.perf_counter() - started_at
                _log_llm_response(response, attempt=attempt, latency_seconds=latency_seconds)
                tool_call = _parse_single_tool_call(response)
                logger.info(
                    "LLM next_step chose %s: attempt=%d latency=%.3fs usage=%s",
                    tool_call.name,
                    attempt,
                    latency_seconds,
                    _extract_usage(response) or "n/a",
                )
                return tool_call
            except (
                PlannerError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                last_error = PlannerError(f"LLM next_step returned invalid tool call: {exc}")
                logger.warning(
                    "LLM next_step attempt %d/%d returned invalid output after %.3fs: %s",
                    attempt,
                    total_attempts,
                    time.perf_counter() - started_at,
                    exc,
                )
            except Exception as exc:
                last_error = PlannerError(f"LLM provider error: {type(exc).__name__}: {exc}")
                logger.warning(
                    "LLM next_step attempt %d/%d failed after %.3fs: %s: %s",
                    attempt,
                    total_attempts,
                    time.perf_counter() - started_at,
                    type(exc).__name__,
                    exc,
                )

        raise PlannerError(f"LLM next_step failed after {total_attempts} attempt(s): {last_error}")

    def _completion(self, **kwargs: Any) -> Any:
        if self.completion_func is not None:
            return self.completion_func(**kwargs)

        from litellm import completion

        return completion(**kwargs)

    def _tool_call_to_step(
        self,
        state: AgentState,
        name: str,
        arguments: dict[str, Any],
    ) -> AgentStep:
        if name in self._registered_tool_names():
            return ToolStep(tool=name, input=arguments)

        if name == "ask_clarification":
            data = AskClarificationInput.model_validate(arguments)
            return ClarifyStep(
                question=data.question,
                missing_fields=data.missing_fields,
                answer_hint=data.answer_hint,
            )

        if name == "finalize_run":
            data = FinalizeRunInput.model_validate(arguments)
            if data.action == "CREATE_DRAFT_PO" and _allowed_policy_needs_draft_po(state):
                return ToolStep(tool="create_draft_po", input={})
            return FinalStep(
                decision=Decision(
                    action=DecisionAction(data.action),
                    risk_level=data.risk_level,
                    requires_human_approval=False,
                    reason=data.reason,
                    policy_flags=data.policy_flags,
                )
            )

        if name == "request_human_approval":
            data = RequestHumanApprovalInput.model_validate(arguments)
            return FinalStep(
                decision=Decision(
                    action=DecisionAction.NEED_HUMAN_APPROVAL,
                    risk_level=data.risk_level,
                    requires_human_approval=True,
                    reason=data.reason,
                    policy_flags=data.policy_flags,
                ),
                pending_tool=ToolStep(tool="create_draft_po", input={}),
            )

        raise PlannerError(f"LLM requested unknown function: {name}")

    def _build_step_messages(self, state: AgentState) -> list[dict[str, str]]:
        payload = {
            "request": state.request.model_dump(mode="json"),
            "parsed_seed": state.parsed.model_dump(mode="json"),
            "tool_observations": [
                trace.model_dump(mode="json", exclude_none=True) for trace in state.observations
            ],
            "metadata": _safe_metadata(state.metadata),
        }
        return [
            {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _all_tool_schemas(self) -> list[dict[str, Any]]:
        return [*self.tool_schemas, *_control_tool_schemas()]

    def _registered_tool_names(self) -> set[str]:
        return {
            tool["function"]["name"]
            for tool in self.tool_schemas
            if tool.get("type") == "function" and "function" in tool
        }


@dataclass(frozen=True)
class _LLMToolCall:
    name: str
    arguments: dict[str, Any]


def _control_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_clarification",
                "description": "Pause the run and ask the user for missing procurement details.",
                "parameters": AskClarificationInput.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_run",
                "description": (
                    "Finalize the run only after required tool observations already exist. "
                    "For CREATE_DRAFT_PO, a successful create_draft_po observation must "
                    "already exist."
                ),
                "parameters": FinalizeRunInput.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_human_approval",
                "description": (
                    "Pause the run for human approval after check_policy reports that approval "
                    "is required."
                ),
                "parameters": RequestHumanApprovalInput.model_json_schema(),
            },
        },
    ]


def _parse_single_tool_call(response: Any) -> _LLMToolCall:
    message = response.choices[0].message
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if len(tool_calls) != 1:
        raise PlannerError(f"Expected exactly one tool call, got {len(tool_calls)}.")

    function = tool_calls[0].function
    name = function.name
    arguments = json.loads(function.arguments or "{}")
    if not isinstance(name, str) or not name:
        raise PlannerError("Tool call function name must be a non-empty string.")
    if not isinstance(arguments, dict):
        raise PlannerError("Tool call arguments must decode to a JSON object.")
    return _LLMToolCall(name=name, arguments=arguments)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "clarification_history",
        "last_clarification_answer",
        "invalidated_tools",
        "planner",
    }
    return {key: metadata[key] for key in allowed_keys if key in metadata}


def _allowed_policy_needs_draft_po(state: AgentState) -> bool:
    if state.output_of("create_draft_po") is not None:
        return False

    policy_output = state.output_of("check_policy")
    if policy_output is None:
        return False

    policy = CheckPolicyOutput.model_validate(policy_output)
    return policy.allowed and not policy.rejected and not policy.requires_human_approval


def _log_llm_request(
    completion_kwargs: dict[str, Any],
    *,
    attempt: int,
    total_attempts: int,
) -> None:
    logger.info(
        "Calling LLM: attempt=%d/%d model=%s timeout=%ss temperature=%s",
        attempt,
        total_attempts,
        completion_kwargs["model"],
        completion_kwargs["timeout"],
        completion_kwargs["temperature"],
    )
    payload = {
        "model": completion_kwargs["model"],
        "messages": completion_kwargs["messages"],
        "tools": completion_kwargs["tools"],
        "tool_choice": completion_kwargs["tool_choice"],
        "temperature": completion_kwargs["temperature"],
        "timeout": completion_kwargs["timeout"],
        "max_tokens": completion_kwargs["max_tokens"],
    }
    io_logger.info("LLM input: %s", json.dumps(safe_for_log(payload), ensure_ascii=False))


def _log_llm_response(response: Any, *, attempt: int, latency_seconds: float) -> None:
    message = response.choices[0].message
    payload: dict[str, Any] = {
        "attempt": attempt,
        "latency_seconds": round(latency_seconds, 3),
        "usage": _extract_usage(response) or None,
        "message": {
            "content": getattr(message, "content", None),
            "tool_calls": _tool_calls_for_log(getattr(message, "tool_calls", None) or []),
        },
    }

    io_logger.info(
        "LLM output: %s",
        json.dumps(safe_for_log(payload), ensure_ascii=False, default=str),
    )


def _tool_calls_for_log(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": getattr(tool_call.function, "name", None),
            "arguments": getattr(tool_call.function, "arguments", None),
        }
        for tool_call in tool_calls
    ]


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return {
        key: value
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := getattr(usage, key, None)) is not None
    }
