from __future__ import annotations

from pydantic import Field

from app.schemas.agents import StrictModel


class ClarificationResolution(StrictModel):
    """Structured answer extraction for one active clarification turn."""

    item_query: str | None = Field(default=None, min_length=1)
    quantity: int | None = Field(default=None, gt=0)
    budget_limit: int | None = Field(default=None, gt=0)
    direct_order_requested: bool = False
    rationale: str = Field(min_length=1)
