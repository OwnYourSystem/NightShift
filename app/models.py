from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class MetricDirection(str, Enum):
    LOWER_IS_BETTER = "lower"
    HIGHER_IS_BETTER = "higher"

class OptimizationConfig(BaseModel):
    metric_name: str = Field(default="val_bpb")
    metric_direction: MetricDirection = MetricDirection.LOWER_IS_BETTER
    max_experiments: int = Field(default=100, ge=1, le=500)
    time_budget_minutes_per_run: int = Field(default=5, ge=1, le=60)
    notes: str = ""

class ExperimentResult(BaseModel):
    commit: str
    metric_value: float
    memory_gb: float
    status: str  # keep | discard | crash
    description: str

class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    customer_email: str
    model_filename: str
    config: OptimizationConfig
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    results: list[ExperimentResult] = []
    baseline_metric: float | None = None
    best_metric: float | None = None
