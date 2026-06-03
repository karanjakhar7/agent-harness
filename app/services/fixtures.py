from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from app.schemas.agents import (
    CatalogItem,
    DepartmentBudget,
    ProcurementFixtures,
    ProcurementPolicy,
    SampleAgentRequest,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "fixtures"


def load_catalog(path: Path | None = None) -> list[CatalogItem]:
    fixture_path = path or FIXTURES_DIR / "catalog.json"
    payload = _loads_fixture_json(fixture_path)
    return TypeAdapter(list[CatalogItem]).validate_python(payload)


def load_budgets(path: Path | None = None) -> dict[str, DepartmentBudget]:
    fixture_path = path or FIXTURES_DIR / "budgets.json"
    payload = _loads_fixture_json(fixture_path)
    return TypeAdapter(dict[str, DepartmentBudget]).validate_python(payload)


def load_policies(path: Path | None = None) -> ProcurementPolicy:
    fixture_path = path or FIXTURES_DIR / "policies.json"
    payload = _loads_fixture_json(fixture_path)
    return ProcurementPolicy.model_validate(payload)


def load_sample_requests(path: Path | None = None) -> list[SampleAgentRequest]:
    fixture_path = path or FIXTURES_DIR / "sample_requests.json"
    payload = _loads_fixture_json(fixture_path)
    return TypeAdapter(list[SampleAgentRequest]).validate_python(payload)


def load_procurement_fixtures(fixtures_dir: Path | None = None) -> ProcurementFixtures:
    base = fixtures_dir or FIXTURES_DIR
    return ProcurementFixtures(
        catalog=load_catalog(base / "catalog.json"),
        budgets=load_budgets(base / "budgets.json"),
        policies=load_policies(base / "policies.json"),
    )


def _loads_fixture_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    json_text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(json_text)
