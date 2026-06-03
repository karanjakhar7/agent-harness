from __future__ import annotations

from app.harness.planner import _detect_direct_order_request, _extract_budget, _extract_quantity
from app.services.fixtures import load_sample_requests


def test_rule_parser_extracts_new_multilingual_samples() -> None:
    parsed: dict[str, dict[str, int | bool | None]] = {}
    for sample in load_sample_requests():
        budget, budget_spans = _extract_budget(sample.message)
        parsed[sample.id] = {
            "quantity": _extract_quantity(sample.message, budget_spans),
            "budget": budget,
            "direct_order_requested": _detect_direct_order_request(sample.message),
        }

    assert parsed == {
        "case_001_low_risk_software": {
            "quantity": 3,
            "budget": 3000,
            "direct_order_requested": False,
        },
        "case_002_hardware_requires_approval": {
            "quantity": 2,
            "budget": None,
            "direct_order_requested": False,
        },
        "case_003_budget_too_high": {
            "quantity": 10,
            "budget": None,
            "direct_order_requested": False,
        },
        "case_004_missing_information": {
            "quantity": None,
            "budget": None,
            "direct_order_requested": False,
        },
        "case_005_prompt_injection": {
            "quantity": 100,
            "budget": None,
            "direct_order_requested": True,
        },
    }
