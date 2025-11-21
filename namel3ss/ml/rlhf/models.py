"""
RLHF Database Models - SQLAlchemy models for feedback storage.

This module defines the database schema for RLHF feedback collection:
- Feedback: Human feedback on model responses
- AnnotationTask: Tasks for human annotators
- Dataset: Exported datasets for training
- TrainingJob: Training job metadata and status

Uses SQLAlchemy ORM for database-agnostic storage.
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    Enum,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class FeedbackType(enum.Enum):
    """Type of feedback provided."""
    PREFERENCE = "preference"  # Pairwise preference (chosen vs rejected)
    SCORE = "score"  # Numeric score
    BINARY = "binary"  # Thumbs up/down, good/bad
    RANKING = "ranking"  # Ranking of multiple responses


class TaskStatus(enum.Enum):
    """Status of annotation task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class DatasetStatus(enum.Enum):
    """Status of dataset export."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(enum.Enum):
    """Status of training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Feedback(Base):
    """
    Human feedback on model responses.
    
    Stores all types of feedback: preferences, scores, binary labels, rankings.
    Used to create training datasets for RLHF algorithms.
    """
    __tablename__ = "rlhf_feedback"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Feedback metadata
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    annotator_id = Column(String(255), nullable=True)  # User/annotator identifier
    
    # Input (prompt)
    prompt = Column(Text, nullable=False)
    prompt_metadata = Column(JSON, nullable=True)  # Additional context
    
    # Response(s)
    response_chosen = Column(Text, nullable=True)  # For preference/binary
    response_rejected = Column(Text, nullable=True)  # For preference
    response_text = Column(Text, nullable=True)  # For score/binary
    
    # Feedback values
    score = Column(Float, nullable=True)  # Numeric score (e.g., 1-5)
    is_preferred = Column(Boolean, nullable=True)  # Binary preference
    ranking = Column(JSON, nullable=True)  # List of response rankings
    
    # Additional metadata
    confidence = Column(Float, nullable=True)  # Annotator confidence
    justification = Column(Text, nullable=True)  # Explanation
    tags = Column(JSON, nullable=True)  # Custom tags
    
    # Model information
    model_id = Column(String(255), nullable=True)
    model_version = Column(String(100), nullable=True)
    
    # Task association
    task_id = Column(Integer, ForeignKey("rlhf_tasks.id"), nullable=True)
    task = relationship("AnnotationTask", back_populates="feedback_items")
    
    # Dataset association
    dataset_id = Column(Integer, ForeignKey("rlhf_datasets.id"), nullable=True)
    dataset = relationship("Dataset", back_populates="feedback_items")
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_feedback_type", "feedback_type"),
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_annotator", "annotator_id"),
        Index("idx_feedback_task", "task_id"),
        Index("idx_feedback_dataset", "dataset_id"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "annotator_id": self.annotator_id,
            "prompt": self.prompt,
            "prompt_metadata": self.prompt_metadata,
            "response_chosen": self.response_chosen,
            "response_rejected": self.response_rejected,
            "response_text": self.response_text,
            "score": self.score,
            "is_preferred": self.is_preferred,
            "ranking": self.ranking,
            "confidence": self.confidence,
            "justification": self.justification,
            "tags": self.tags,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
        }


class AnnotationTask(Base):
    """
    Annotation task for human labelers.
    
    Groups prompts that need feedback annotation.
    Can be assigned to specific annotators with deadlines.
    """
    __tablename__ = "rlhf_tasks"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Task metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    
    # Assignment
    assigned_to = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    deadline = Column(DateTime, nullable=True)
    
    # Task configuration
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    instructions = Column(Text, nullable=True)
    num_prompts = Column(Integer, default=0, nullable=False)
    num_completed = Column(Integer, default=0, nullable=False)
    
    # Model information
    model_id = Column(String(255), nullable=True)
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)
    
    # Relationships
    feedback_items = relationship("Feedback", back_populates="task")
    
    # Indexes
    __table_args__ = (
        Index("idx_task_status", "status"),
        Index("idx_task_assigned", "assigned_to"),
        Index("idx_task_created", "created_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "feedback_type": self.feedback_type.value,
            "instructions": self.instructions,
            "num_prompts": self.num_prompts,
            "num_completed": self.num_completed,
            "model_id": self.model_id,
            "extra_metadata": self.extra_metadata,
        }


class Dataset(Base):
    """
    Exported dataset for training.
    
    Represents a collection of feedback compiled into a training dataset.
    Can be exported to various formats (Parquet, JSONL, HuggingFace).
    """
    __tablename__ = "rlhf_datasets"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dataset metadata
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.PENDING, nullable=False)
    
    # Content
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    num_samples = Column(Integer, default=0, nullable=False)
    
    # Export information
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    export_format = Column(String(50), nullable=True)  # parquet, jsonl, hf
    export_path = Column(String(500), nullable=True)  # File path or S3 URI
    
    # Statistics
    train_samples = Column(Integer, default=0)
    val_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)
    
    # Filtering criteria
    min_confidence = Column(Float, nullable=True)
    annotator_filter = Column(JSON, nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)
    
    # Relationships
    feedback_items = relationship("Feedback", back_populates="dataset")
    
    # Indexes
    __table_args__ = (
        Index("idx_dataset_status", "status"),
        Index("idx_dataset_created", "created_at"),
        Index("idx_dataset_name", "name"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "feedback_type": self.feedback_type.value,
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "export_format": self.export_format,
            "export_path": self.export_path,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "min_confidence": self.min_confidence,
            "annotator_filter": self.annotator_filter,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "extra_metadata": self.extra_metadata,
        }


class TrainingJob(Base):
    """
    RLHF training job metadata.
    
    Tracks training jobs initiated from feedback datasets.
    Links datasets to trained models.
    """
    __tablename__ = "rlhf_jobs"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Job metadata
    job_name = Column(String(255), nullable=False, unique=True)
    algorithm = Column(String(50), nullable=False)  # dpo, ppo, orpo, kto, etc.
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    
    # Model information
    base_model = Column(String(255), nullable=False)
    output_model = Column(String(500), nullable=True)
    reward_model = Column(String(255), nullable=True)
    
    # Dataset
    dataset_id = Column(Integer, ForeignKey("rlhf_datasets.id"), nullable=True)
    dataset = relationship("Dataset")
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Training metrics
    final_loss = Column(Float, nullable=True)
    total_steps = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Configuration
    config = Column(JSON, nullable=True)  # Complete RLHFConfig as JSON
    
    # Results
    checkpoint_path = Column(String(500), nullable=True)
    metrics = Column(JSON, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_type = Column(String(255), nullable=True)
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_job_status", "status"),
        Index("idx_job_algorithm", "algorithm"),
        Index("idx_job_created", "created_at"),
        Index("idx_job_dataset", "dataset_id"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_name": self.job_name,
            "algorithm": self.algorithm,
            "status": self.status.value,
            "base_model": self.base_model,
            "output_model": self.output_model,
            "reward_model": self.reward_model,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_loss": self.final_loss,
            "total_steps": self.total_steps,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "checkpoint_path": self.checkpoint_path,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "extra_metadata": self.extra_metadata,
        }
