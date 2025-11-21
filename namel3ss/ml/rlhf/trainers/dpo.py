"""
DPO Trainer - Direct Preference Optimization implementation.

Implements DPO and IPO algorithms using HuggingFace TRL library.
DPO is a simpler alternative to PPO that directly optimizes policy from preferences.
"""

import logging
from typing import Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import DPOTrainer as TRLDPOTrainer

from ..config import RLHFConfig
from ..errors import RLHFTrainingError
from .base import BaseRLHFTrainer

logger = logging.getLogger(__name__)


class DPOTrainer(BaseRLHFTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) and IPO algorithms.
    
    DPO directly optimizes a language model from preference data without
    requiring a separate reward model or reinforcement learning.
    
    IPO (Identity Preference Optimization) uses the same trainer but with
    a different loss function specified in config.
    
    Reference: https://arxiv.org/abs/2305.18290
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize DPO trainer."""
        super().__init__(config, model, tokenizer)
        
        logger.info(f"Initialized DPO trainer (algorithm: {config.algorithm.value})")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute DPO training.
        
        Returns:
            Dictionary with training results:
                - final_loss: Final training loss
                - best_loss: Best validation loss
                - total_steps: Total training steps
                - final_lr: Final learning rate
                - metrics: Algorithm-specific metrics
        """
        logger.info("Starting DPO training")
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get training arguments
            training_args_dict = self._get_training_args()
            training_args = TrainingArguments(**training_args_dict)
            
            # Get DPO-specific config
            dpo_config = self.config.dpo_config
            if not dpo_config:
                from ..config import DPOConfig
                dpo_config = DPOConfig()
            
            # Determine loss type (DPO vs IPO)
            loss_type = dpo_config.loss_type
            if self.config.algorithm.value == "ipo":
                loss_type = "ipo"
            
            # Create TRL DPO trainer
            logger.info(f"Creating DPO trainer with loss_type={loss_type}, beta={dpo_config.beta}")
            
            trainer = TRLDPOTrainer(
                model=self.model,
                ref_model=None,  # Will create reference model automatically
                args=training_args,
                beta=dpo_config.beta,
                train_dataset=self.dataset.get_hf_dataset(),
                tokenizer=self.tokenizer,
                max_length=512,  # TODO: Make configurable
                max_prompt_length=256,  # TODO: Make configurable
                loss_type=loss_type,
                label_smoothing=dpo_config.label_smoothing,
            )
            
            # Train
            logger.info("Beginning training loop")
            train_result = trainer.train()
            
            # Extract metrics
            final_loss = train_result.training_loss
            metrics = train_result.metrics
            
            logger.info(f"Training complete. Final loss: {final_loss:.4f}")
            
            return {
                "final_loss": final_loss,
                "best_loss": metrics.get("eval_loss"),
                "total_steps": metrics.get("total_flos", self.config.max_steps),
                "final_lr": self.config.learning_rate,
                "metrics": {
                    "train_loss": final_loss,
                    "dpo_beta": dpo_config.beta,
                    "loss_type": loss_type,
                    **metrics,
                },
            }
        
        except Exception as e:
            logger.error(f"DPO training failed: {str(e)}")
            raise RLHFTrainingError(
                f"DPO training failed: {str(e)}",
                code="RLHF010",
                context={
                    "algorithm": self.config.algorithm.value,
                    "error": str(e),
                }
            ) from e
