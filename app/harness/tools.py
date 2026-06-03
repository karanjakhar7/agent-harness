from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.schemas.agents import (
    CatalogItem,
    CheckPolicyInput,
    CheckPolicyOutput,
    CreateDraftPOInput,
    DepartmentBudget,
    DraftPO,
    LookupCatalogInput,
    LookupCatalogOutput,
    ProcurementFixtures,
    ProcurementPolicy,
    RiskLevel,
    SubmitToERPInput,
    SubmitToERPOutput,
    ToolCallStatus,
    ToolCallTrace,
)
from app.services.procurement import (
    check_policy_handler,
    create_draft_po,
    lookup_catalog_handler,
    submit_to_erp,
)


logger = logging.getLogger("app.harness.tools")

ToolHandler = Callable[[BaseModel], BaseModel | dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    risk_level: RiskLevel
    requires_approval: bool = False
    description: str = ""


@dataclass(frozen=True)
class ToolExecutionResult:
    output: BaseModel | None
    trace: ToolCallTrace


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def llm_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas without handler internals."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description or f"Run the {spec.name} procurement tool.",
                    "parameters": spec.input_model.model_json_schema(),
                },
            }
            for spec in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        raw_input: dict[str, Any],
        *,
        approval_completed: bool = False,
    ) -> ToolExecutionResult:
        logger.debug("Tool %s validating input; approval_completed=%s", name, approval_completed)
        spec = self.get(name)
        if spec is None:
            logger.warning("Tool %s BLOCKED: not registered.", name)
            return ToolExecutionResult(
                output=None,
                trace=ToolCallTrace(
                    tool=name,
                    status=ToolCallStatus.BLOCKED,
                    input=raw_input,
                    error="Tool is not registered.",
                    boundary="ToolRegistry",
                ),
            )

        if spec.requires_approval and not approval_completed:
            logger.warning(
                "Tool %s BLOCKED at ApprovalBoundary: requires completed human approval.", name
            )
            return ToolExecutionResult(
                output=None,
                trace=ToolCallTrace(
                    tool=name,
                    status=ToolCallStatus.BLOCKED,
                    risk_level=spec.risk_level,
                    input=raw_input,
                    error="Tool requires completed human approval before execution.",
                    boundary="ApprovalBoundary",
                ),
            )

        try:
            tool_input = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            logger.warning("Tool %s ERROR at InputSchema: %s", name, _validation_summary(exc))
            return ToolExecutionResult(
                output=None,
                trace=ToolCallTrace(
                    tool=name,
                    status=ToolCallStatus.ERROR,
                    risk_level=spec.risk_level,
                    input=raw_input,
                    error=_validation_summary(exc),
                    boundary="InputSchema",
                ),
            )

        try:
            raw_output = spec.handler(tool_input)
            tool_output = spec.output_model.model_validate(raw_output)
        except ValidationError as exc:
            logger.warning("Tool %s ERROR at OutputSchema: %s", name, _validation_summary(exc))
            return ToolExecutionResult(
                output=None,
                trace=ToolCallTrace(
                    tool=name,
                    status=ToolCallStatus.ERROR,
                    risk_level=spec.risk_level,
                    input=tool_input.model_dump(mode="json"),
                    error=_validation_summary(exc),
                    boundary="OutputSchema",
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Tool %s ERROR at ToolRuntime: %s", name, exc)
            return ToolExecutionResult(
                output=None,
                trace=ToolCallTrace(
                    tool=name,
                    status=ToolCallStatus.ERROR,
                    risk_level=spec.risk_level,
                    input=tool_input.model_dump(mode="json"),
                    error=str(exc),
                    boundary="ToolRuntime",
                ),
            )

        logger.debug("Tool %s succeeded (risk=%s)", name, spec.risk_level.value)
        return ToolExecutionResult(
            output=tool_output,
            trace=ToolCallTrace(
                tool=name,
                status=ToolCallStatus.SUCCESS,
                risk_level=spec.risk_level,
                input=tool_input.model_dump(mode="json"),
                output=tool_output.model_dump(mode="json"),
            ),
        )


def build_default_tool_registry(
    fixtures: ProcurementFixtures | None = None,
    *,
    catalog: list[CatalogItem] | None = None,
    policies: ProcurementPolicy | None = None,
    budgets: dict[str, DepartmentBudget] | None = None,
) -> ToolRegistry:
    if fixtures is not None:
        catalog = fixtures.catalog
        policies = fixtures.policies
        budgets = fixtures.budgets
    if catalog is None or policies is None or budgets is None:
        raise ValueError("Catalog, policies, and budgets are required to build the tool registry.")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_catalog",
            input_model=LookupCatalogInput,
            output_model=LookupCatalogOutput,
            handler=lookup_catalog_handler(catalog),
            risk_level=RiskLevel.LOW,
            description=(
                "Find a catalog item by user wording, product name, or alias. "
                "Call this before policy checks or draft creation."
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="check_policy",
            input_model=CheckPolicyInput,
            output_model=CheckPolicyOutput,
            handler=check_policy_handler(policies, budgets),
            risk_level=RiskLevel.LOW,
            description=(
                "Check company procurement policy for one matched catalog item, quantity, "
                "estimated total, and budget context."
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="create_draft_po",
            input_model=CreateDraftPOInput,
            output_model=DraftPO,
            handler=create_draft_po,
            risk_level=RiskLevel.MEDIUM,
            description=(
                "Create a draft purchase order only after catalog lookup and policy check allow it."
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="submit_to_erp",
            input_model=SubmitToERPInput,
            output_model=SubmitToERPOutput,
            handler=submit_to_erp,
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            description="Submit an approved draft PO to ERP. This requires completed approval.",
        )
    )
    return registry


def blocked_tool_trace(
    *,
    tool: str,
    raw_input: dict[str, Any],
    risk_level: RiskLevel,
    error: str,
    boundary: str = "ApprovalBoundary",
) -> ToolCallTrace:
    return ToolCallTrace(
        tool=tool,
        status=ToolCallStatus.BLOCKED,
        risk_level=risk_level,
        input=raw_input,
        error=error,
        boundary=boundary,
    )


def _validation_summary(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}"
