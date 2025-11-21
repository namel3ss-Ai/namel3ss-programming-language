"""
RLHF (Reinforcement Learning from Human Feedback) training subsystem for Namel3ss.

This package provides production-grade RLHF capabilities including:
- Modern preference optimization algorithms (PPO, DPO, IPO, ORPO, KTO)
- Reward model training from human feedback
- Human feedback collection and annotation pipelines
- Integration with Hugging Face TRL, PEFT, and accelerate
- Distributed training support (DeepSpeed, FSDP)
- Experiment tracking (W&B, MLflow)
- Safety evaluation and monitoring

Architecture:
    
    N3 DSL → Parser → AST → Backend Codegen → FastAPI APIs
                                   ↓
    RLHFJobRunner → TRL Trainers → Checkpoints/Models
                          ↓
    Feedback DB → Datasets → Training Data

Key Components:
    - models: SQLAlchemy models for feedback storage
    - datasets: Preference dataset loaders and processors
    - trainers: RLHF algorithm implementations (PPO, DPO, etc.)
    - runners: Job orchestration and execution
    - api: FastAPI endpoints for feedback and job management
    - evaluation: Evaluation harness and safety filters
    - monitoring: Metrics and observability

Example N3 DSL:
    
    train rlhf "HelpfulAssistant" {
        base_model: "meta-llama/Meta-Llama-3-8B"
        algorithm: "dpo"
        dataset: "s3://my-bucket/preference-data"
        
        peft: {
            method: "qlora"
            r: 64
            alpha: 16
        }
        
        hyperparameters: {
            learning_rate: 1e-5
            batch_size: 64
            max_steps: 20000
            beta: 0.1  # DPO beta parameter
        }
        
        logging: {
            tracker: "wandb"
            project: "helpful-assistant"
        }
        
        output: {
            model_registry: "s3://my-bucket/models"
        }
    }

Usage:
    >>> from namel3ss.ml.rlhf import RLHFJobRunner, RLHFConfig
    >>> from namel3ss.ml.rlhf.trainers import DPOTrainer
    >>> 
    >>> # Configure RLHF training
    >>> config = RLHFConfig(
    ...     algorithm="dpo",
    ...     base_model="meta-llama/Meta-Llama-3-8B",
    ...     dataset_path="s3://my-bucket/preference-data",
    ...     output_dir="/models/helpful-assistant"
    ... )
    >>> 
    >>> # Run training
    >>> runner = RLHFJobRunner(config)
    >>> result = runner.run()
    >>> print(f"Model saved to: {result.model_path}")
"""

from .config import (
    RLHFConfig,
    RLHFAlgorithm,
    PEFTConfig,
    PEFTMethod,
    PPOConfig,
    DPOConfig,
    ORPOConfig,
    KTOConfig,
    LoggingConfig,
    SafetyConfig,
    ExperimentTracker,
)

from .errors import (
    RLHFError,
    RLHFConfigurationError,
    RLHFTrainingError,
    RLHFDatasetError,
    RLHFModelError,
    RLHFEvaluationError,
    RLHFStorageError,
)

from .runners import RLHFJobRunner, RLHFJobResult

from .datasets import (
    PreferenceDataset,
    FeedbackDataset,
    PreferenceSample,
    FeedbackSample,
    load_preference_dataset,
    load_feedback_dataset,
)

from .trainers import (
    BaseRLHFTrainer,
    PPOTrainer,
    DPOTrainer,
    ORPOTrainer,
    KTOTrainer,
    SFTTrainer,
    get_trainer_class,
)

from .storage import (
    StorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
    StorageManager,
    ArtifactMetadata,
    get_storage_manager,
)

from .database import (
    DatabaseManager,
    initialize_database,
    get_database,
    get_session,
)

from .models import (
    Feedback,
    AnnotationTask,
    Dataset,
    TrainingJob,
    FeedbackType,
    TaskStatus,
    DatasetStatus,
    JobStatus,
)

from .exporters import DatasetExporter

# API routers
from .api import (
    feedback_router,
    tasks_router,
    datasets_router,
    jobs_router,
)

__all__ = [
    # Configuration
    "RLHFConfig",
    "RLHFAlgorithm",
    "PEFTConfig",
    "PEFTMethod",
    "PPOConfig",
    "DPOConfig",
    "ORPOConfig",
    "KTOConfig",
    "LoggingConfig",
    "SafetyConfig",
    "ExperimentTracker",
    # Errors
    "RLHFError",
    "RLHFConfigurationError",
    "RLHFTrainingError",
    "RLHFDatasetError",
    "RLHFModelError",
    "RLHFEvaluationError",
    "RLHFStorageError",
    # Runners
    "RLHFJobRunner",
    "RLHFJobResult",
    # Datasets
    "PreferenceDataset",
    "FeedbackDataset",
    "PreferenceSample",
    "FeedbackSample",
    "load_preference_dataset",
    "load_feedback_dataset",
    # Trainers
    "BaseRLHFTrainer",
    "PPOTrainer",
    "DPOTrainer",
    "ORPOTrainer",
    "KTOTrainer",
    "SFTTrainer",
    "get_trainer_class",
    # Storage
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    "StorageManager",
    "ArtifactMetadata",
    "get_storage_manager",
    # Database
    "DatabaseManager",
    "initialize_database",
    "get_database",
    "get_session",
    # Models
    "Feedback",
    "AnnotationTask",
    "Dataset",
    "TrainingJob",
    "FeedbackType",
    "TaskStatus",
    "DatasetStatus",
    "JobStatus",
    # Exporters
    "DatasetExporter",
    # API routers
    "feedback_router",
    "tasks_router",
    "datasets_router",
    "jobs_router",
]

__version__ = "0.1.0"
