from __future__ import annotations

import re
import uuid

from pydantic import BaseModel

from app.schemas.agents import (
    CatalogItem,
    CheckPolicyInput,
    CheckPolicyOutput,
    CreateDraftPOInput,
    DepartmentBudget,
    DraftPO,
    LookupCatalogInput,
    LookupCatalogOutput,
    ProcurementPolicy,
    RiskLevel,
    SubmitToERPInput,
    SubmitToERPOutput,
)


def lookup_catalog_handler(catalog: list[CatalogItem]):
    def lookup(tool_input: BaseModel) -> LookupCatalogOutput:
        lookup_input = LookupCatalogInput.model_validate(tool_input)
        query = _normalize(lookup_input.query)
        best_item: CatalogItem | None = None
        best_score = 0

        for item in catalog:
            terms = [item.name, *item.aliases]
            for term in terms:
                normalized_term = _normalize(term)
                if normalized_term and normalized_term in query:
                    score = len(normalized_term)
                    if score > best_score:
                        best_item = item
                        best_score = score

        if best_item is not None:
            return LookupCatalogOutput(match_found=True, item=best_item, alternatives=[])

        alternatives = _catalog_alternatives(query, catalog)
        return LookupCatalogOutput(match_found=False, alternatives=alternatives)

    return lookup


def check_policy_handler(
    policies: ProcurementPolicy,
    budgets: dict[str, DepartmentBudget],
):
    restricted_categories = {category.lower() for category in policies.restricted_categories}

    def check_policy(tool_input: BaseModel) -> CheckPolicyOutput:
        policy_input = CheckPolicyInput.model_validate(tool_input)
        department = policy_input.department.lower()
        category = policy_input.category.lower()
        flags: list[str] = []

        if policy_input.direct_order_requested:
            flags.append("prompt_injection_or_bypass_attempt")

        if (
            policy_input.budget_limit is not None
            and policy_input.estimated_total > policy_input.budget_limit
        ):
            flags.append("budget_exceeded")

        if policy_input.estimated_total > policies.approval_threshold_usd:
            flags.append(f"amount_exceeds_{policies.approval_threshold_usd}")

        if category in restricted_categories:
            flags.append(_restricted_category_flag(category))

        department_budget = budgets.get(department)
        if (
            department_budget is not None
            and policy_input.estimated_total > department_budget.remaining_budget_usd
        ):
            flags.append("department_budget_exceeded")

        rejected = "prompt_injection_or_bypass_attempt" in flags
        budget_exceeded = "budget_exceeded" in flags
        approval_flags = [
            flag
            for flag in flags
            if flag
            in {
                f"amount_exceeds_{policies.approval_threshold_usd}",
                "hardware_purchase",
                "enterprise_software_license",
                "department_budget_exceeded",
            }
        ]
        requires_human_approval = bool(approval_flags) and not rejected and not budget_exceeded
        allowed = not rejected and not budget_exceeded

        if rejected:
            reason = "The request attempted to bypass approval or company policy."
            risk_level = RiskLevel.HIGH
        elif budget_exceeded:
            reason = "The estimated total exceeds the user's stated budget."
            risk_level = RiskLevel.MEDIUM
        elif requires_human_approval:
            reason = "Human approval is required by policy."
            risk_level = RiskLevel.HIGH
        else:
            reason = "The request matches policy and can proceed to draft PO creation."
            risk_level = RiskLevel.LOW if policy_input.estimated_total < 3000 else RiskLevel.MEDIUM

        return CheckPolicyOutput(
            allowed=allowed,
            rejected=rejected,
            requires_human_approval=requires_human_approval,
            risk_level=risk_level,
            flags=flags,
            reason=reason,
        )

    return check_policy


def create_draft_po(tool_input: BaseModel) -> DraftPO:
    draft_input = CreateDraftPOInput.model_validate(tool_input)
    return DraftPO(
        po_id=f"po_{uuid.uuid4().hex[:12]}",
        item_id=draft_input.item_id,
        item=draft_input.item_name,
        quantity=draft_input.quantity,
        unit_price=draft_input.unit_price,
        estimated_total=draft_input.estimated_total,
        department=draft_input.department,
        currency=draft_input.currency,
        approval_reference=draft_input.approval_reference,
    )


def submit_to_erp(tool_input: BaseModel) -> SubmitToERPOutput:
    erp_input = SubmitToERPInput.model_validate(tool_input)
    return SubmitToERPOutput(
        erp_submission_id=f"erp_{uuid.uuid4().hex[:12]}",
        draft_po_id=erp_input.draft_po_id,
    )


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _catalog_alternatives(query: str, catalog: list[CatalogItem]) -> list[str]:
    return [item.name for item in catalog]


def _restricted_category_flag(category: str) -> str:
    if category == "hardware":
        return "hardware_purchase"
    if category == "enterprise_software":
        return "enterprise_software_license"
    return f"restricted_category_{category}"
