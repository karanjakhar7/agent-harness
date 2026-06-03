from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.app import build_harness  # noqa: E402
from app.schemas.agents import AgentRunRequest  # noqa: E402
from app.services.fixtures import load_sample_requests  # noqa: E402


def main() -> None:
    harness = build_harness()
    scenarios = load_sample_requests()

    for scenario in scenarios:
        request = AgentRunRequest(
            user_id=scenario.user_id,
            department=scenario.department,
            message=scenario.message,
        )
        response = harness.run(request)
        payload = response.model_dump(mode="json")
        print(f"\n=== {scenario.id} ===")
        print(f"expected_behavior: {scenario.expected_behavior}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
