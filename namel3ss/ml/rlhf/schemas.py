"""
RLHF API Schemas - Pydantic models for request/response validation.

Defines schemas for:
- Feedback submission
- Task creation and updates
- Dataset export
- Query filters

Uses Pydantic v2 for validation and serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


class FeedbackTypeSchema(str, Enum):
    """Type of feedback."""
    PREFERENCE = "preference"
    SCORE = "score"
    BINARY = "binary"
    RANKING = "ranking"


class TaskStatusSchema(str, Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class DatasetStatusSchema(str, Enum):
    """Dataset status."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatusSchema(str, Enum):
    """Job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ============================================================================
# Feedback Schemas
# ============================================================================

class FeedbackBase(BaseModel):
    """Base feedback fields."""
    feedback_type: FeedbackTypeSchema
    prompt: str = Field(..., min_length=1)
    prompt_metadata: Optional[Dict[str, Any]] = None
    annotator_id: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    justification: Optional[str] = None
    tags: Optional[List[str]] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None


class PreferenceFeedbackCreate(FeedbackBase):
    """Create preference feedback (chosen vs rejected)."""
    feedback_type: FeedbackTypeSchema = FeedbackTypeSchema.PREFERENCE
    response_chosen: str = Field(..., min_length=1)
    response_rejected: str = Field(..., min_length=1)


class ScoreFeedbackCreate(FeedbackBase):
    """Create score feedback."""
    feedback_type: FeedbackTypeSchema = FeedbackTypeSchema.SCORE
    response_text: str = Field(..., min_length=1)
    score: float = Field(..., description="Numeric score")


class BinaryFeedbackCreate(FeedbackBase):
    """Create binary feedback (thumbs up/down)."""
    feedback_type: FeedbackTypeSchema = FeedbackTypeSchema.BINARY
    response_text: str = Field(..., min_length=1)
    is_preferred: bool


class RankingFeedbackCreate(FeedbackBase):
    """Create ranking feedback."""
    feedback_type: FeedbackTypeSchema = FeedbackTypeSchema.RANKING
    ranking: List[Dict[str, Any]] = Field(..., min_items=2)


class FeedbackResponse(BaseModel):
    """Feedback response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    feedback_type: str
    created_at: datetime
    annotator_id: Optional[str] = None
    prompt: str
    prompt_metadata: Optional[Dict[str, Any]] = None
    response_chosen: Optional[str] = None
    response_rejected: Optional[str] = None
    response_text: Optional[str] = None
    score: Optional[float] = None
    is_preferred: Optional[bool] = None
    ranking: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    justification: Optional[str] = None
    tags: Optional[List[str]] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    task_id: Optional[int] = None
    dataset_id: Optional[int] = None


class FeedbackListResponse(BaseModel):
    """Paginated feedback list."""
    items: List[FeedbackResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============================================================================
# Task Schemas
# ============================================================================

class TaskCreate(BaseModel):
    """Create annotation task."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    feedback_type: FeedbackTypeSchema
    instructions: Optional[str] = None
    num_prompts: int = Field(..., ge=1)
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskUpdate(BaseModel):
    """Update task."""
    status: Optional[TaskStatusSchema] = None
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskResponse(BaseModel):
    """Task response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    description: Optional[str] = None
    status: str
    assigned_to: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    feedback_type: str
    instructions: Optional[str] = None
    num_prompts: int
    num_completed: int
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskListResponse(BaseModel):
    """Paginated task list."""
    items: List[TaskResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetCreate(BaseModel):
    """Create dataset export."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    feedback_type: FeedbackTypeSchema
    export_format: str = Field(..., pattern="^(parquet|jsonl|hf)$")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    annotator_filter: Optional[List[str]] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    train_split: float = Field(0.8, ge=0.0, le=1.0)
    val_split: float = Field(0.1, ge=0.0, le=1.0)
    test_split: float = Field(0.1, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def validate_splits(cls, v, info):
        """Validate split values."""
        if hasattr(info, 'data'):
            # Pydantic v2
            data = info.data
            if 'train_split' in data and 'val_split' in data and 'test_split' in data:
                total = data['train_split'] + data['val_split'] + data['test_split']
                if abs(total - 1.0) > 0.001:
                    raise ValueError("Splits must sum to 1.0")
        return v


class DatasetResponse(BaseModel):
    """Dataset response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    description: Optional[str] = None
    status: str
    feedback_type: str
    num_samples: int
    created_at: datetime
    updated_at: datetime
    export_format: Optional[str] = None
    export_path: Optional[str] = None
    train_samples: int
    val_samples: int
    test_samples: int
    min_confidence: Optional[float] = None
    annotator_filter: Optional[List[str]] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetListResponse(BaseModel):
    """Paginated dataset list."""
    items: List[DatasetResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============================================================================
# Job Schemas
# ============================================================================

class JobCreate(BaseModel):
    """Create training job."""
    job_name: str = Field(..., min_length=1, max_length=255)
    algorithm: str = Field(..., pattern="^(ppo|dpo|ipo|orpo|kto|sft|reward)$")
    base_model: str = Field(..., min_length=1)
    dataset_id: int = Field(..., ge=1)
    reward_model: Optional[str] = None
    config: Dict[str, Any] = Field(..., description="RLHFConfig as dict")
    metadata: Optional[Dict[str, Any]] = None


class JobUpdate(BaseModel):
    """Update job."""
    status: Optional[JobStatusSchema] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    duration_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None
    output_model: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


class JobResponse(BaseModel):
    """Job response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    job_name: str
    algorithm: str
    status: str
    base_model: str
    output_model: Optional[str] = None
    reward_model: Optional[str] = None
    dataset_id: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    duration_seconds: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """Paginated job list."""
    items: List[JobResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============================================================================
# Filter Schemas
# ============================================================================

class FeedbackFilters(BaseModel):
    """Feedback query filters."""
    feedback_type: Optional[FeedbackTypeSchema] = None
    annotator_id: Optional[str] = None
    task_id: Optional[int] = None
    dataset_id: Optional[int] = None
    model_id: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)


class TaskFilters(BaseModel):
    """Task query filters."""
    status: Optional[TaskStatusSchema] = None
    assigned_to: Optional[str] = None
    feedback_type: Optional[FeedbackTypeSchema] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)


class DatasetFilters(BaseModel):
    """Dataset query filters."""
    status: Optional[DatasetStatusSchema] = None
    feedback_type: Optional[FeedbackTypeSchema] = None
    export_format: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)


class JobFilters(BaseModel):
    """Job query filters."""
    status: Optional[JobStatusSchema] = None
    algorithm: Optional[str] = None
    dataset_id: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)
