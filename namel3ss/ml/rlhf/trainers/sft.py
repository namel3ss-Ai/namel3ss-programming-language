"""
SFT Trainer - Supervised Fine-Tuning implementation.

Implements SFT using HuggingFace TRL library.
SFT is used for initial supervised fine-tuning before RLHF, or for training reward models.
"""

import logging
from typing import Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer as TRLSFTTrainer

from ..config import RLHFConfig
from ..errors import RLHFTrainingError
from .base import BaseRLHFTrainer

logger = logging.getLogger(__name__)


class SFTTrainer(BaseRLHFTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT).
    
    SFT is the first stage of RLHF, where a pre-trained model is fine-tuned
    on high-quality demonstration data. It can also be used to train reward models.
    
    This trainer wraps HuggingFace TRL's SFTTrainer with our configuration system.
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize SFT trainer."""
        super().__init__(config, model, tokenizer)
        
        logger.info("Initialized SFT trainer")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute SFT training.
        
        Returns:
            Dictionary with training results:
                - final_loss: Final training loss
                - best_loss: Best validation loss
                - total_steps: Total training steps
                - final_lr: Final learning rate
                - metrics: Training metrics
        """
        logger.info("Starting SFT training")
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get training arguments
            training_args_dict = self._get_training_args()
            training_args = TrainingArguments(**training_args_dict)
            
            # Create TRL SFT trainer
            logger.info("Creating SFT trainer")
            
            trainer = TRLSFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset.get_hf_dataset(),
                tokenizer=self.tokenizer,
                max_seq_length=512,  # TODO: Make configurable
                packing=False,  # TODO: Consider enabling packing for efficiency
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
                    **metrics,
                },
            }
        
        except Exception as e:
            logger.error(f"SFT training failed: {str(e)}")
            raise RLHFTrainingError(
                f"SFT training failed: {str(e)}",
                code="RLHF010",
                context={
                    "algorithm": "SFT",
                    "error": str(e),
                }
            ) from e
