"""
RLHF (Reinforcement Learning from Human Feedback) AST nodes.

This module contains AST nodes for defining RLHF training jobs:
- RLHFJob: Complete RLHF training specification
- RLHFPEFTConfig: Parameter-efficient fine-tuning configuration
- RLHFAlgorithmConfig: Algorithm-specific hyperparameters
- RLHFComputeSpec: Distributed training resources
- RLHFLoggingConfig: Experiment tracking configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from ..base import Expression


# Type alias for config values (can be expressions or literals)
ConfigValue = Expression | Any


@dataclass
class RLHFPEFTConfig:
    """
    Parameter-efficient fine-tuning configuration.
    
    Supports LoRA, QLoRA, and other PEFT methods to reduce
    memory footprint and training time.
    
    Example DSL:
        peft: {
            method: "qlora",
            r: 64,
            lora_alpha: 16,
            lora_dropout: 0.1,
            target_modules: ["q_proj", "v_proj"]
        }
    """
    method: str  # lora, qlora, prefix_tuning, p_tuning, adapter
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    quantization: Optional[str] = None  # 4bit, 8bit
    options: Dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RLHFAlgorithmConfig:
    """
    Algorithm-specific hyperparameters for RLHF.
    
    Different algorithms (PPO, DPO, KTO, etc.) have different
    hyperparameters. This captures algorithm-specific settings.
    
    Example DSL (DPO):
        algorithm_config: {
            beta: 0.1,
            label_smoothing: 0.0,
            loss_type: "sigmoid"
        }
    
    Example DSL (PPO):
        algorithm_config: {
            init_kl_coef: 0.2,
            adap_kl_ctrl: true,
            target_kl: 6.0,
            gamma: 1.0,
            lam: 0.95
        }
    """
    algorithm: str  # ppo, dpo, ipo, orpo, kto, sft, reward
    
    # DPO/IPO parameters
    beta: Optional[float] = None
    label_smoothing: Optional[float] = None
    loss_type: Optional[str] = None  # sigmoid, hinge, ipo
    
    # PPO parameters
    init_kl_coef: Optional[float] = None
    adap_kl_ctrl: Optional[bool] = None
    target_kl: Optional[float] = None
    gamma: Optional[float] = None
    lam: Optional[float] = None
    cliprange: Optional[float] = None
    cliprange_value: Optional[float] = None
    vf_coef: Optional[float] = None
    
    # KTO parameters
    beta_kto: Optional[float] = None
    desirable_weight: Optional[float] = None
    undesirable_weight: Optional[float] = None
    
    # ORPO parameters
    beta_orpo: Optional[float] = None
    
    # Additional algorithm-specific options
    options: Dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RLHFComputeSpec:
    """
    Compute resources for distributed RLHF training.
    
    Specifies hardware, parallelism strategies, and optimization
    settings for large-scale training.
    
    Example DSL:
        compute: {
            backend: "aws_sagemaker",
            num_gpus: 4,
            gpu_type: "a100",
            strategy: "deepspeed_zero3",
            mixed_precision: "bf16",
            gradient_checkpointing: true
        }
    """
    backend: str = "local"  # local, aws_sagemaker, gcp_vertex, azure_ml
    num_gpus: int = 1
    gpu_type: Optional[str] = None  # a100, v100, h100
    num_nodes: int = 1
    
    # Distributed strategy
    strategy: Optional[str] = None  # ddp, fsdp, deepspeed_zero2, deepspeed_zero3
    
    # Optimization
    mixed_precision: Optional[str] = None  # fp16, bf16
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    
    # DeepSpeed config
    deepspeed_config: Optional[str] = None
    
    # Resource limits
    cpu_cores: Optional[int] = None
    memory_gb: Optional[int] = None
    
    # Additional options
    options: Dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RLHFLoggingConfig:
    """
    Experiment tracking and logging configuration.
    
    Integrates with Weights & Biases, MLflow, TensorBoard, etc.
    
    Example DSL:
        logging: {
            tracker: "wandb",
            project: "llama3_alignment",
            run_name: "dpo_helpful_v1",
            log_frequency: 100
        }
    """
    tracker: str = "wandb"  # wandb, mlflow, tensorboard, none
    project: Optional[str] = None
    run_name: Optional[str] = None
    entity: Optional[str] = None  # W&B entity/org
    tags: List[str] = field(default_factory=list)
    log_frequency: int = 100
    save_frequency: int = 500
    options: Dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RLHFSafetyConfig:
    """
    Safety filters and evaluation configuration.
    
    Specifies content moderation, toxicity detection, and
    safety guardrails for RLHF training.
    
    Example DSL:
        safety: {
            enable_content_filter: true,
            toxicity_threshold: 0.7,
            pii_detection: true,
            custom_filters: ["profanity", "bias"]
        }
    """
    enable_content_filter: bool = False
    toxicity_threshold: Optional[float] = None
    pii_detection: bool = False
    custom_filters: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    options: Dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RLHFJob:
    """
    Complete specification for an RLHF training job.
    
    Defines all aspects of RLHF training:
    - Base model and algorithm
    - Dataset and reward model
    - PEFT configuration
    - Hyperparameters
    - Compute resources
    - Logging and monitoring
    
    Example N3 DSL:
        train rlhf "helpful_assistant" {
            base_model: "meta-llama/Meta-Llama-3-8B"
            algorithm: "dpo"
            dataset: "s3://my-bucket/preference-data"
            
            peft: {
                method: "qlora"
                r: 64
                lora_alpha: 16
                target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
            
            algorithm_config: {
                beta: 0.1
                loss_type: "sigmoid"
            }
            
            hyperparameters: {
                learning_rate: 1e-5
                batch_size: 64
                max_steps: 20000
                warmup_ratio: 0.1
            }
            
            compute: {
                backend: "aws_sagemaker"
                num_gpus: 4
                gpu_type: "a100"
                strategy: "deepspeed_zero3"
                mixed_precision: "bf16"
            }
            
            logging: {
                tracker: "wandb"
                project: "llama3-alignment"
                run_name: "dpo_helpful_v1"
            }
            
            output: {
                registry: "s3://my-bucket/models"
                model_name: "helpful_assistant_v1"
            }
        }
    """
    name: str
    base_model: str
    algorithm: str  # ppo, dpo, ipo, orpo, kto, sft, reward
    dataset: str  # Path or URI to dataset
    
    # Optional reward model (for PPO)
    reward_model: Optional[str] = None
    
    # PEFT configuration
    peft: Optional[RLHFPEFTConfig] = None
    
    # Algorithm-specific config
    algorithm_config: Optional[RLHFAlgorithmConfig] = None
    
    # Training hyperparameters
    hyperparameters: Dict[str, ConfigValue] = field(default_factory=dict)
    
    # Compute resources
    compute: RLHFComputeSpec = field(default_factory=RLHFComputeSpec)
    
    # Logging and monitoring
    logging: RLHFLoggingConfig = field(default_factory=RLHFLoggingConfig)
    
    # Safety and evaluation
    safety: Optional[RLHFSafetyConfig] = None
    
    # Output configuration
    output_dir: Optional[str] = None
    output_registry: Optional[str] = None
    model_name: Optional[str] = None
    
    # Training control
    max_steps: Optional[int] = None
    num_epochs: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    
    # Optimization
    learning_rate: Optional[float] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    
    # Data processing
    max_prompt_length: Optional[int] = None
    max_response_length: Optional[int] = None
    train_split: Optional[float] = None
    val_split: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RLHFJob",
    "RLHFPEFTConfig",
    "RLHFAlgorithmConfig",
    "RLHFComputeSpec",
    "RLHFLoggingConfig",
    "RLHFSafetyConfig",
    "ConfigValue",
]
