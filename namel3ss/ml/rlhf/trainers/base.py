"""
Base RLHF Trainer - Abstract base class for all RLHF trainers.

Provides common functionality:
- Configuration management
- Model and tokenizer handling
- Dataset loading
- PEFT setup
- Logging and monitoring
- Checkpoint management
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ..config import RLHFConfig, PEFTMethod
from ..datasets import load_preference_dataset, load_feedback_dataset
from ..errors import RLHFConfigurationError, RLHFModelError

logger = logging.getLogger(__name__)


class BaseRLHFTrainer(ABC):
    """
    Abstract base class for RLHF trainers.
    
    Provides common infrastructure that all algorithm-specific trainers build upon.
    Subclasses must implement the train() method with their specific training logic.
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize base trainer.
        
        Args:
            config: RLHF configuration
            model: Pre-trained model
            tokenizer: Tokenizer
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = None
        
        # Setup PEFT if configured
        if self.config.peft:
            self._setup_peft()
    
    def _setup_peft(self) -> None:
        """
        Setup PEFT (LoRA/QLoRA) for efficient fine-tuning.
        
        Raises:
            RLHFModelError: If PEFT setup fails
        """
        peft_config = self.config.peft
        
        logger.info(f"Setting up PEFT: {peft_config.method.value}")
        
        try:
            # Prepare model for k-bit training if using quantization
            if peft_config.quantization_bits:
                logger.info(f"Preparing model for {peft_config.quantization_bits}-bit training")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=peft_config.gradient_checkpointing
                )
            
            # Create LoRA config
            if peft_config.method in [PEFTMethod.LORA, PEFTMethod.QLORA]:
                lora_config = LoraConfig(
                    r=peft_config.r,
                    lora_alpha=peft_config.alpha,
                    lora_dropout=peft_config.dropout,
                    target_modules=peft_config.target_modules or ["q_proj", "v_proj"],
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                self.model = get_peft_model(self.model, lora_config)
                
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                
                logger.info(f"PEFT setup complete")
                logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
            elif peft_config.method == PEFTMethod.FULL:
                logger.info("Using full fine-tuning (no PEFT)")
            
            else:
                raise RLHFModelError(
                    f"PEFT method not yet implemented: {peft_config.method.value}",
                    code="RLHF031",
                    context={"peft_method": peft_config.method.value}
                )
        
        except Exception as e:
            raise RLHFModelError(
                f"Failed to setup PEFT: {str(e)}",
                code="RLHF031",
                context={
                    "peft_method": peft_config.method.value,
                    "error": str(e),
                }
            )
    
    def _load_dataset(self) -> None:
        """
        Load dataset based on algorithm requirements.
        
        Subclasses can override to customize dataset loading.
        """
        if self.config.algorithm.value in ["ppo", "dpo", "ipo", "orpo"]:
            self.dataset = load_preference_dataset(
                path=self.config.dataset_path,
                split="train",
                algorithm=self.config.algorithm,
            )
        else:
            self.dataset = load_feedback_dataset(
                path=self.config.dataset_path,
                split="train",
                algorithm=self.config.algorithm,
            )
        
        logger.info(f"Loaded {len(self.dataset)} training samples")
    
    def _get_training_args(self) -> Dict[str, Any]:
        """
        Get training arguments common to all algorithms.
        
        Returns:
            Dictionary of training arguments
        """
        return {
            "output_dir": self.config.output_dir,
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_steps": self.config.warmup_steps,
            "max_grad_norm": self.config.max_grad_norm,
            "seed": self.config.seed,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "logging_steps": self.config.logging.log_interval,
            "save_steps": self.config.logging.save_interval,
            "eval_steps": self.config.logging.eval_interval,
        }
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute training.
        
        Must be implemented by subclasses.
        
        Returns:
            Dictionary with training results and metrics
        """
        pass
