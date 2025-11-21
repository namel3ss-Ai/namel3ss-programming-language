"""
Configuration models for RLHF training jobs.

Provides strongly-typed configuration classes for all RLHF components:
- Base RLHF configuration
- Algorithm-specific parameters (PPO, DPO, ORPO, KTO)
- PEFT (LoRA/QLoRA) configuration
- Logging and monitoring
- Safety and evaluation

All configurations are validated at creation time to catch errors early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .errors import RLHFConfigurationError


class RLHFAlgorithm(str, Enum):
    """Supported RLHF algorithms."""
    PPO = "ppo"  # Proximal Policy Optimization
    DPO = "dpo"  # Direct Preference Optimization
    IPO = "ipo"  # Identity Preference Optimization
    ORPO = "orpo"  # Odds Ratio Preference Optimization
    KTO = "kto"  # Kahneman-Tversky Optimization
    REWARD_MODEL = "reward_model"  # Supervised reward model training
    SFT = "sft"  # Supervised Fine-Tuning (warmup)


class PEFTMethod(str, Enum):
    """Parameter-efficient fine-tuning methods."""
    LORA = "lora"  # Low-Rank Adaptation
    QLORA = "qlora"  # Quantized LoRA (4-bit/8-bit)
    IA3 = "ia3"  # Infused Adapter by Inhibiting and Amplifying Inner Activations
    ADALORA = "adalora"  # Adaptive LoRA
    FULL = "full"  # Full parameter fine-tuning


class ExperimentTracker(str, Enum):
    """Experiment tracking backends."""
    WANDB = "wandb"  # Weights & Biases
    MLFLOW = "mlflow"  # MLflow
    TENSORBOARD = "tensorboard"  # TensorBoard
    NONE = "none"  # No tracking


@dataclass
class PEFTConfig:
    """
    Parameter-efficient fine-tuning configuration.
    
    Enables memory-efficient training using LoRA, QLoRA, or other PEFT methods.
    
    Attributes:
        method: PEFT technique to use
        r: LoRA rank (typically 8-64)
        alpha: LoRA alpha scaling factor (typically 16-32)
        dropout: Dropout probability for LoRA layers
        target_modules: Module names to apply LoRA to (None = auto-detect)
        bias: Bias training strategy ("none", "all", "lora_only")
        task_type: Task type for PEFT ("CAUSAL_LM", "SEQ_CLS")
        quantization_bits: Quantization bits for QLoRA (4 or 8)
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
    
    Example:
        >>> config = PEFTConfig(
        ...     method=PEFTMethod.QLORA,
        ...     r=64,
        ...     alpha=16,
        ...     quantization_bits=4
        ... )
    """
    method: PEFTMethod = PEFTMethod.LORA
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    quantization_bits: Optional[int] = None  # 4 or 8 for QLoRA
    use_gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.r <= 0:
            raise RLHFConfigurationError(
                f"LoRA rank must be positive, got {self.r}",
                code="RLHF003",
                context={"r": self.r}
            )
        
        if self.alpha <= 0:
            raise RLHFConfigurationError(
                f"LoRA alpha must be positive, got {self.alpha}",
                code="RLHF003",
                context={"alpha": self.alpha}
            )
        
        if not 0 <= self.dropout < 1:
            raise RLHFConfigurationError(
                f"Dropout must be in [0, 1), got {self.dropout}",
                code="RLHF003",
                context={"dropout": self.dropout}
            )
        
        if self.quantization_bits is not None:
            if self.quantization_bits not in (4, 8):
                raise RLHFConfigurationError(
                    f"Quantization bits must be 4 or 8, got {self.quantization_bits}",
                    code="RLHF003",
                    context={"quantization_bits": self.quantization_bits}
                )
            if self.method != PEFTMethod.QLORA:
                raise RLHFConfigurationError(
                    "Quantization bits only supported for QLoRA",
                    code="RLHF004",
                    context={"method": self.method, "quantization_bits": self.quantization_bits}
                )


@dataclass
class PPOConfig:
    """
    PPO-specific hyperparameters.
    
    Proximal Policy Optimization parameters for RLHF training.
    
    Attributes:
        batch_size: Number of samples per update
        mini_batch_size: Mini-batch size for PPO updates
        ppo_epochs: Number of PPO epochs per batch
        gamma: Discount factor for rewards
        lam: GAE lambda for advantage estimation
        cliprange: PPO clipping range
        cliprange_value: Value function clipping range
        vf_coef: Value function loss coefficient
        init_kl_coef: Initial KL divergence coefficient
        target_kl: Target KL divergence threshold
        adap_kl_ctrl: Enable adaptive KL control
    
    Example:
        >>> config = PPOConfig(
        ...     batch_size=64,
        ...     mini_batch_size=16,
        ...     init_kl_coef=0.2,
        ...     target_kl=6.0
        ... )
    """
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True


@dataclass
class DPOConfig:
    """
    DPO-specific hyperparameters.
    
    Direct Preference Optimization parameters.
    
    Attributes:
        beta: DPO temperature parameter (controls sharpness of preference distribution)
        label_smoothing: Label smoothing factor
        loss_type: Loss function ("sigmoid", "hinge", "ipo")
        reference_free: Use reference-free DPO variant
    
    Example:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     loss_type="sigmoid"
        ... )
    """
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    reference_free: bool = False


@dataclass
class ORPOConfig:
    """
    ORPO-specific hyperparameters.
    
    Odds Ratio Preference Optimization parameters.
    
    Attributes:
        alpha: ORPO alpha parameter (weight for odds ratio term)
        beta: Temperature parameter
    
    Example:
        >>> config = ORPOConfig(alpha=1.0, beta=0.1)
    """
    alpha: float = 1.0
    beta: float = 0.1


@dataclass
class KTOConfig:
    """
    KTO-specific hyperparameters.
    
    Kahneman-Tversky Optimization parameters.
    
    Attributes:
        beta: KTO beta parameter
        desirable_weight: Weight for desirable samples
        undesirable_weight: Weight for undesirable samples
    
    Example:
        >>> config = KTOConfig(
        ...     beta=0.1,
        ...     desirable_weight=1.0,
        ...     undesirable_weight=1.0
        ... )
    """
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


@dataclass
class LoggingConfig:
    """
    Logging and experiment tracking configuration.
    
    Attributes:
        tracker: Experiment tracker backend
        project: Project name for grouping experiments
        run_name: Name for this specific run
        tags: Tags for organizing experiments
        log_interval: Steps between metric logging
        save_interval: Steps between checkpoint saves
        eval_interval: Steps between evaluations
        wandb_api_key: W&B API key (or use WANDB_API_KEY env var)
        mlflow_tracking_uri: MLflow tracking server URI
    
    Example:
        >>> config = LoggingConfig(
        ...     tracker=ExperimentTracker.WANDB,
        ...     project="helpful-assistant",
        ...     run_name="dpo-llama3-8b-v1"
        ... )
    """
    tracker: ExperimentTracker = ExperimentTracker.WANDB
    project: str = "namel3ss-rlhf"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    wandb_api_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None


@dataclass
class SafetyConfig:
    """
    Safety and evaluation configuration.
    
    Attributes:
        enable_safety_filter: Enable safety filtering during generation
        toxicity_threshold: Toxicity score threshold (0-1)
        safety_model: Model ID for safety classification
        enable_content_filter: Enable content policy filtering
        max_generation_length: Maximum generation length (safety limit)
        reject_unsafe_samples: Reject unsafe samples during training
    
    Example:
        >>> config = SafetyConfig(
        ...     enable_safety_filter=True,
        ...     toxicity_threshold=0.7,
        ...     safety_model="unitary/toxic-bert"
        ... )
    """
    enable_safety_filter: bool = False
    toxicity_threshold: float = 0.7
    safety_model: Optional[str] = None
    enable_content_filter: bool = False
    max_generation_length: int = 512
    reject_unsafe_samples: bool = False


@dataclass
class RLHFConfig:
    """
    Complete RLHF training configuration.
    
    Main configuration object that combines all RLHF settings.
    
    Attributes:
        job_name: Unique identifier for this training job
        algorithm: RLHF algorithm to use
        base_model: Hugging Face model ID or local path
        dataset_path: Path to preference dataset (S3, GCS, local, or HF dataset ID)
        reward_model: Reward model ID (for PPO) or None (for preference methods)
        output_dir: Directory for checkpoints and final model
        peft: PEFT configuration (None = full fine-tuning)
        ppo_config: PPO-specific parameters (if algorithm=PPO)
        dpo_config: DPO-specific parameters (if algorithm=DPO)
        orpo_config: ORPO-specific parameters (if algorithm=ORPO)
        kto_config: KTO-specific parameters (if algorithm=KTO)
        logging: Logging and tracking configuration
        safety: Safety and evaluation configuration
        learning_rate: Optimizer learning rate
        max_steps: Maximum training steps
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Learning rate warmup steps
        max_grad_norm: Gradient clipping norm
        seed: Random seed for reproducibility
        fp16: Enable mixed precision training (FP16)
        bf16: Enable bfloat16 training (better for modern GPUs)
        deepspeed_config: DeepSpeed configuration file path
        gradient_checkpointing: Enable gradient checkpointing
        dataloader_num_workers: DataLoader worker processes
        push_to_hub: Push final model to Hugging Face Hub
        hub_model_id: Model ID for Hub upload
        
    Example:
        >>> config = RLHFConfig(
        ...     job_name="helpful-assistant-v1",
        ...     algorithm=RLHFAlgorithm.DPO,
        ...     base_model="meta-llama/Meta-Llama-3-8B",
        ...     dataset_path="s3://my-bucket/preference-data",
        ...     output_dir="/models/helpful-assistant",
        ...     peft=PEFTConfig(method=PEFTMethod.QLORA, r=64),
        ...     dpo_config=DPOConfig(beta=0.1),
        ...     learning_rate=1e-5,
        ...     max_steps=20000
        ... )
    """
    # Required fields
    job_name: str
    algorithm: RLHFAlgorithm
    base_model: str
    dataset_path: str
    output_dir: str
    
    # Optional: Reward model (required for PPO)
    reward_model: Optional[str] = None
    
    # PEFT configuration
    peft: Optional[PEFTConfig] = None
    
    # Algorithm-specific configs
    ppo_config: Optional[PPOConfig] = None
    dpo_config: Optional[DPOConfig] = None
    orpo_config: Optional[ORPOConfig] = None
    kto_config: Optional[KTOConfig] = None
    
    # Logging and safety
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Precision and performance
    fp16: bool = False
    bf16: bool = True
    deepspeed_config: Optional[str] = None
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Model registry
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate algorithm-specific requirements
        if self.algorithm == RLHFAlgorithm.PPO:
            if self.reward_model is None:
                raise RLHFConfigurationError(
                    "PPO algorithm requires reward_model",
                    code="RLHF001",
                    context={"algorithm": self.algorithm}
                )
            if self.ppo_config is None:
                self.ppo_config = PPOConfig()
        
        elif self.algorithm == RLHFAlgorithm.DPO:
            if self.dpo_config is None:
                self.dpo_config = DPOConfig()
        
        elif self.algorithm == RLHFAlgorithm.ORPO:
            if self.orpo_config is None:
                self.orpo_config = ORPOConfig()
        
        elif self.algorithm == RLHFAlgorithm.KTO:
            if self.kto_config is None:
                self.kto_config = KTOConfig()
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise RLHFConfigurationError(
                f"Learning rate must be positive, got {self.learning_rate}",
                code="RLHF003",
                context={"learning_rate": self.learning_rate}
            )
        
        # Validate max_steps
        if self.max_steps <= 0:
            raise RLHFConfigurationError(
                f"max_steps must be positive, got {self.max_steps}",
                code="RLHF003",
                context={"max_steps": self.max_steps}
            )
        
        # Validate precision settings
        if self.fp16 and self.bf16:
            raise RLHFConfigurationError(
                "Cannot use both fp16 and bf16",
                code="RLHF004",
                context={"fp16": self.fp16, "bf16": self.bf16}
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            elif hasattr(value, "__dict__"):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


__all__ = [
    "RLHFAlgorithm",
    "PEFTMethod",
    "ExperimentTracker",
    "PEFTConfig",
    "PPOConfig",
    "DPOConfig",
    "ORPOConfig",
    "KTOConfig",
    "LoggingConfig",
    "SafetyConfig",
    "RLHFConfig",
]
