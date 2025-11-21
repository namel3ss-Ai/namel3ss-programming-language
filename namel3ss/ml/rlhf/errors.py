"""
Error types for RLHF training subsystem.

Provides domain-specific exceptions for different failure modes in RLHF
training pipelines, enabling targeted error handling and recovery.
"""

from typing import Any, Dict, Optional


class RLHFError(Exception):
    """Base exception for all RLHF-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RLHF error.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code (e.g., "RLHF001")
            context: Additional context for debugging
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}
    
    def format(self) -> str:
        """Format error with full context."""
        parts = []
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(self.message)
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        return self.format()


class RLHFConfigurationError(RLHFError):
    """
    Configuration validation errors.
    
    Raised when RLHF job configuration is invalid or incomplete.
    
    Error Codes:
        - RLHF001: Missing required configuration field
        - RLHF002: Invalid algorithm selection
        - RLHF003: Invalid hyperparameter value
        - RLHF004: Incompatible configuration options
        - RLHF005: Missing environment variable
    
    Example:
        >>> raise RLHFConfigurationError(
        ...     "Algorithm 'xyz' not supported",
        ...     code="RLHF002",
        ...     context={"algorithm": "xyz", "supported": ["ppo", "dpo"]}
        ... )
    """
    pass


class RLHFTrainingError(RLHFError):
    """
    Training execution errors.
    
    Raised when training fails during execution.
    
    Error Codes:
        - RLHF010: Training divergence (loss explosion, NaN values)
        - RLHF011: Out of memory (OOM)
        - RLHF012: Checkpoint save failure
        - RLHF013: Model loading failure
        - RLHF014: Distributed training synchronization failure
        - RLHF015: Early stopping triggered
    
    Example:
        >>> raise RLHFTrainingError(
        ...     "Training loss diverged",
        ...     code="RLHF010",
        ...     context={"step": 1523, "loss": float('inf')}
        ... )
    """
    pass


class RLHFDatasetError(RLHFError):
    """
    Dataset loading and processing errors.
    
    Raised when preference data cannot be loaded or is malformed.
    
    Error Codes:
        - RLHF020: Dataset not found
        - RLHF021: Invalid dataset format
        - RLHF022: Missing required columns
        - RLHF023: Insufficient data (< minimum samples)
        - RLHF024: Data corruption detected
        - RLHF025: Preference label mismatch
    
    Example:
        >>> raise RLHFDatasetError(
        ...     "Preference dataset missing 'chosen' column",
        ...     code="RLHF022",
        ...     context={"path": "s3://bucket/data", "columns": ["prompt", "rejected"]}
        ... )
    """
    pass


class RLHFModelError(RLHFError):
    """
    Model loading, initialization, or inference errors.
    
    Raised when base model or reward model cannot be loaded or used.
    
    Error Codes:
        - RLHF030: Model not found
        - RLHF031: Incompatible model architecture
        - RLHF032: Model loading timeout
        - RLHF033: Tokenizer mismatch
        - RLHF034: GPU memory exceeded
        - RLHF035: Reward model prediction failure
    
    Example:
        >>> raise RLHFModelError(
        ...     "Base model not found on Hugging Face Hub",
        ...     code="RLHF030",
        ...     context={"model_id": "meta-llama/does-not-exist"}
        ... )
    """
    pass


class RLHFEvaluationError(RLHFError):
    """
    Evaluation and safety filter errors.
    
    Raised when evaluation fails or safety checks cannot be performed.
    
    Error Codes:
        - RLHF040: Evaluation dataset unavailable
        - RLHF041: Metric computation failure
        - RLHF042: Safety filter failure
        - RLHF043: Benchmark not found
    
    Example:
        >>> raise RLHFEvaluationError(
        ...     "Safety filter could not load toxicity classifier",
        ...     code="RLHF042",
        ...     context={"filter": "toxicity", "model": "unitary/toxic-bert"}
        ... )
    """
    pass


class RLHFStorageError(RLHFError):
    """
    Storage and checkpoint management errors.
    
    Raised when checkpoints, models, or artifacts cannot be saved/loaded.
    
    Error Codes:
        - RLHF050: S3/GCS upload failure
        - RLHF051: Checkpoint directory not writable
        - RLHF052: Model registry unavailable
        - RLHF053: Artifact corruption
    
    Example:
        >>> raise RLHFStorageError(
        ...     "Failed to upload checkpoint to S3",
        ...     code="RLHF050",
        ...     context={"bucket": "my-bucket", "key": "checkpoints/step-1000"}
        ... )
    """
    pass


__all__ = [
    "RLHFError",
    "RLHFConfigurationError",
    "RLHFTrainingError",
    "RLHFDatasetError",
    "RLHFModelError",
    "RLHFEvaluationError",
    "RLHFStorageError",
]
