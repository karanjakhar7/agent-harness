from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    department: Mapped[str | None] = mapped_column(String, nullable=True)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    runs: Mapped[list[AgentRun]] = relationship(back_populates="user")
    approvals: Mapped[list[ApprovalRecord]] = relationship(back_populates="approver")


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("idx_agent_runs_user_id", "user_id"),
        Index("idx_agent_runs_status", "status"),
        Index("idx_agent_runs_created_at", "created_at"),
    )

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    department: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    decision_action: Mapped[str] = mapped_column(String, nullable=False)
    risk_level: Mapped[str] = mapped_column(String, nullable=False)
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    pending_po_input_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    approval_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped[User] = relationship(back_populates="runs")
    tool_call_traces: Mapped[list[ToolCallTraceRecord]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ToolCallTraceRecord.sequence",
    )
    draft_po: Mapped[DraftPORecord | None] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
    )
    approvals: Mapped[list[ApprovalRecord]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class ToolCallTraceRecord(Base):
    __tablename__ = "tool_call_traces"
    __table_args__ = (Index("idx_tool_call_traces_run_sequence", "run_id", "sequence", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    tool: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    boundary: Mapped[str | None] = mapped_column(String, nullable=True)
    trace_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    run: Mapped[AgentRun] = relationship(back_populates="tool_call_traces")


class DraftPORecord(Base):
    __tablename__ = "draft_pos"

    po_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    draft_po_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    run: Mapped[AgentRun] = relationship(back_populates="draft_po")


class ApprovalRecord(Base):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False)
    approver_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    approval_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    run: Mapped[AgentRun] = relationship(back_populates="approvals")
    approver: Mapped[User] = relationship(back_populates="approvals")
