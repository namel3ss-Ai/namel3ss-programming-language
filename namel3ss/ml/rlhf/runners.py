"""
RLHF Job Runner - Orchestrates RLHF training jobs.

This module provides the main orchestration layer for RLHF training, managing:
- Configuration loading and validation
- Dataset preparation and loading
- Model and tokenizer initialization
- Trainer instantiation and execution
- Checkpoint management and artifact storage
- Result reporting and metrics collection

The RLHFJobRunner is the primary entry point for executing RLHF training jobs,
handling the complete lifecycle from setup to completion.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import RLHFConfig, RLHFAlgorithm
from .errors import (
    RLHFTrainingError,
    RLHFModelError,
    RLHFConfigurationError,
    RLHFStorageError,
)
from .storage import get_storage_manager, ArtifactMetadata

logger = logging.getLogger(__name__)


@dataclass
class RLHFJobResult:
    """
    Result of an RLHF training job.
    
    Contains all relevant metrics, paths, and metadata from the completed training run.
    """
    
    job_name: str
    algorithm: RLHFAlgorithm
    status: str  # "completed", "failed", "stopped"
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Model artifacts
    final_checkpoint_path: str
    best_checkpoint_path: Optional[str] = None
    output_dir: str = ""
    
    # Training metrics
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    total_steps: int = 0
    final_learning_rate: float = 0.0
    
    # Algorithm-specific metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation results
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    peak_gpu_memory_gb: Optional[float] = None
    total_gpu_hours: Optional[float] = None
    
    # Error information (if failed)
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "job_name": self.job_name,
            "algorithm": self.algorithm.value,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "final_checkpoint_path": self.final_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "output_dir": self.output_dir,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "total_steps": self.total_steps,
            "final_learning_rate": self.final_learning_rate,
            "metrics": self.metrics,
            "eval_metrics": self.eval_metrics,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            "total_gpu_hours": self.total_gpu_hours,
            "error_message": self.error_message,
            "error_type": self.error_type,
        }


class RLHFJobRunner:
    """
    Orchestrates RLHF training jobs from configuration to completion.
    
    This class manages the complete RLHF training lifecycle:
    1. Load and validate configuration
    2. Initialize model, tokenizer, and dataset
    3. Create appropriate trainer (PPO, DPO, etc.)
    4. Execute training with monitoring
    5. Save checkpoints and artifacts
    6. Return structured results
    
    Example:
        >>> config = RLHFConfig(
        ...     job_name="my_rlhf_job",
        ...     algorithm=RLHFAlgorithm.DPO,
        ...     base_model="meta-llama/Llama-2-7b-hf",
        ...     dataset_path="HuggingFaceH4/ultrafeedback_binarized",
        ...     output_dir="./outputs/dpo_llama2"
        ... )
        >>> runner = RLHFJobRunner(config)
        >>> result = runner.run()
        >>> print(f"Final loss: {result.final_loss}")
    """
    
    def __init__(self, config: RLHFConfig):
        """
        Initialize the RLHF job runner.
        
        Args:
            config: Complete RLHF configuration
            
        Raises:
            RLHFConfigurationError: If configuration is invalid
        """
        self.config = config
        self.start_time: Optional[datetime] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Any] = None  # Will be specific trainer type
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized RLHF job runner for job '{config.job_name}'")
        logger.info(f"Algorithm: {config.algorithm.value}")
        logger.info(f"Base model: {config.base_model}")
    
    def _validate_config(self) -> None:
        """
        Validate configuration before training.
        
        Raises:
            RLHFConfigurationError: If configuration is invalid
        """
        # Check output directory
        output_path = Path(self.config.output_dir)
        if not output_path.parent.exists():
            raise RLHFConfigurationError(
                f"Output directory parent does not exist: {output_path.parent}",
                code="RLHF001",
                context={"output_dir": self.config.output_dir}
            )
        
        # Validate algorithm-specific requirements
        if self.config.algorithm == RLHFAlgorithm.PPO:
            if not self.config.reward_model:
                raise RLHFConfigurationError(
                    "PPO algorithm requires reward_model to be specified",
                    code="RLHF003",
                    context={"algorithm": "PPO"}
                )
        
        # Check GPU availability if not using CPU
        if not self.config.metadata.get("force_cpu", False):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, will use CPU (training will be slow)")
    
    def _load_model_and_tokenizer(self) -> None:
        """
        Load base model and tokenizer.
        
        Raises:
            RLHFModelError: If model or tokenizer cannot be loaded
        """
        logger.info(f"Loading model: {self.config.base_model}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.config.bf16 else (
                    torch.float16 if self.config.fp16 else torch.float32
                ),
            }
            
            # Add quantization config if using QLoRA
            if self.config.peft and self.config.peft.quantization_bits:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.config.peft.quantization_bits == 4,
                    load_in_8bit=self.config.peft.quantization_bits == 8,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs
            )
            
            logger.info("Model and tokenizer loaded successfully")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU memory allocated: {memory_gb:.2f} GB")
        
        except Exception as e:
            raise RLHFModelError(
                f"Failed to load model or tokenizer: {str(e)}",
                code="RLHF030",
                context={
                    "base_model": self.config.base_model,
                    "error": str(e),
                }
            )
    
    def _create_trainer(self) -> Any:
        """
        Create appropriate trainer based on algorithm.
        
        Returns:
            Trainer instance (PPOTrainer, DPOTrainer, etc.)
            
        Raises:
            RLHFConfigurationError: If algorithm not supported
        """
        # Import trainers (will create these next)
        from .trainers import get_trainer_class
        
        trainer_class = get_trainer_class(self.config.algorithm)
        
        logger.info(f"Creating {self.config.algorithm.value} trainer")
        
        trainer = trainer_class(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def _save_checkpoint(self, checkpoint_path: str, is_final: bool = False) -> str:
        """
        Save model checkpoint with storage integration.
        
        Args:
            checkpoint_path: Path to save checkpoint (can be local, S3, etc.)
            is_final: Whether this is the final checkpoint
            
        Returns:
            Remote storage URI if uploaded, otherwise local path
            
        Raises:
            RLHFStorageError: If checkpoint cannot be saved
        """
        try:
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            
            # Save to temporary local directory first
            local_checkpoint_dir = Path(checkpoint_path)
            local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(local_checkpoint_dir)
            self.tokenizer.save_pretrained(local_checkpoint_dir)
            
            # Save config
            config_path = local_checkpoint_dir / "rlhf_config.json"
            import json
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info(f"Checkpoint saved locally")
            
            # Upload to remote storage if output_dir is a remote URI
            if self.config.output_dir.startswith(("s3://", "gs://", "azure://")):
                storage = get_storage_manager()
                
                # Create metadata
                metadata = ArtifactMetadata(
                    artifact_id=f"{self.config.job_name}_{'final' if is_final else 'checkpoint'}_{int(time.time())}",
                    artifact_type="checkpoint",
                    job_name=self.config.job_name,
                    algorithm=self.config.algorithm.value,
                    created_at=datetime.now(),
                    size_bytes=sum(f.stat().st_size for f in local_checkpoint_dir.rglob("*") if f.is_file()),
                    storage_path=checkpoint_path,
                    base_model=self.config.base_model,
                    peft_method=self.config.peft.method.value if self.config.peft else None,
                    tags={"is_final": str(is_final)},
                )
                
                # Upload
                remote_uri = storage.upload(
                    local_path=str(local_checkpoint_dir),
                    remote_uri=checkpoint_path,
                    metadata=metadata,
                )
                
                logger.info(f"Checkpoint uploaded to {remote_uri}")
                return remote_uri
            
            return str(local_checkpoint_dir)
        
        except Exception as e:
            raise RLHFStorageError(
                f"Failed to save checkpoint: {str(e)}",
                code="RLHF052",
                context={
                    "checkpoint_path": checkpoint_path,
                    "error": str(e),
                }
            )
    
    def run(self) -> RLHFJobResult:
        """
        Execute the RLHF training job.
        
        Returns:
            RLHFJobResult with training metrics and artifact paths
            
        Raises:
            RLHFTrainingError: If training fails
            RLHFModelError: If model loading fails
            RLHFConfigurationError: If configuration is invalid
        """
        self.start_time = datetime.now()
        
        try:
            logger.info(f"Starting RLHF job: {self.config.job_name}")
            
            # Load model and tokenizer
            self._load_model_and_tokenizer()
            
            # Create trainer
            self.trainer = self._create_trainer()
            
            # Run training
            logger.info("Starting training...")
            training_result = self.trainer.train()
            
            # Save final checkpoint
            final_checkpoint = os.path.join(
                self.config.output_dir,
                "final_checkpoint"
            )
            final_checkpoint_uri = self._save_checkpoint(final_checkpoint, is_final=True)
            
            # Push to hub if configured
            if self.config.push_to_hub and self.config.hub_model_id:
                logger.info(f"Pushing model to hub: {self.config.hub_model_id}")
                self.model.push_to_hub(self.config.hub_model_id)
                self.tokenizer.push_to_hub(self.config.hub_model_id)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            # Collect GPU metrics if available
            peak_memory_gb = None
            if torch.cuda.is_available():
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            
            # Create result
            result = RLHFJobResult(
                job_name=self.config.job_name,
                algorithm=self.config.algorithm,
                status="completed",
                start_time=self.start_time,
                end_time=end_time,
                duration_seconds=duration,
                final_checkpoint_path=final_checkpoint_uri,
                output_dir=self.config.output_dir,
                final_loss=training_result.get("final_loss"),
                best_loss=training_result.get("best_loss"),
                total_steps=training_result.get("total_steps", 0),
                final_learning_rate=training_result.get("final_lr", self.config.learning_rate),
                metrics=training_result.get("metrics", {}),
                eval_metrics=training_result.get("eval_metrics", {}),
                peak_gpu_memory_gb=peak_memory_gb,
            )
            
            logger.info(f"RLHF job completed successfully in {duration:.2f}s")
            logger.info(f"Final loss: {result.final_loss}")
            
            return result
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            logger.error(f"RLHF job failed: {str(e)}")
            
            # Create error result
            result = RLHFJobResult(
                job_name=self.config.job_name,
                algorithm=self.config.algorithm,
                status="failed",
                start_time=self.start_time,
                end_time=end_time,
                duration_seconds=duration,
                final_checkpoint_path="",
                output_dir=self.config.output_dir,
                error_message=str(e),
                error_type=type(e).__name__,
            )
            
            # Re-raise if it's already an RLHF error
            if isinstance(e, (RLHFTrainingError, RLHFModelError, RLHFConfigurationError)):
                raise
            
            # Wrap other exceptions
            raise RLHFTrainingError(
                f"Training job failed: {str(e)}",
                code="RLHF010",
                context={
                    "job_name": self.config.job_name,
                    "algorithm": self.config.algorithm.value,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            ) from e
