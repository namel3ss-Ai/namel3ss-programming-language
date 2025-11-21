"""
ORPO Trainer - Odds Ratio Preference Optimization implementation.

Implements ORPO algorithm using HuggingFace TRL library.
ORPO combines SFT and preference learning in a single stage without a reference model.
"""

import logging
from typing import Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import ORPOTrainer as TRLORPOTrainer, ORPOConfig as TRLORPOConfig

from ..config import RLHFConfig
from ..errors import RLHFTrainingError
from .base import BaseRLHFTrainer

logger = logging.getLogger(__name__)


class ORPOTrainer(BaseRLHFTrainer):
    """
    Trainer for Odds Ratio Preference Optimization (ORPO).
    
    ORPO is a reference-model-free RLHF algorithm that combines supervised fine-tuning
    and preference learning in a single stage. It uses an odds ratio-based loss to
    optimize the model directly on preference data.
    
    Reference: https://arxiv.org/abs/2403.07691
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize ORPO trainer."""
        super().__init__(config, model, tokenizer)
        
        logger.info("Initialized ORPO trainer")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute ORPO training.
        
        Returns:
            Dictionary with training results:
                - final_loss: Final training loss
                - best_loss: Best validation loss
                - total_steps: Total training steps
                - final_lr: Final learning rate
                - metrics: Algorithm-specific metrics
        """
        logger.info("Starting ORPO training")
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get ORPO-specific config
            orpo_config = self.config.orpo_config
            if not orpo_config:
                from ..config import ORPOConfig
                orpo_config = ORPOConfig()
            
            # Create TRL ORPO config
            trl_orpo_config = TRLORPOConfig(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                max_steps=self.config.max_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_grad_norm=self.config.max_grad_norm,
                seed=self.config.seed,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                logging_steps=self.config.logging.log_interval,
                save_steps=self.config.logging.save_interval,
                eval_steps=self.config.logging.eval_interval,
                beta=orpo_config.beta,  # ORPO-specific parameter
            )
            
            # Create TRL ORPO trainer
            logger.info(f"Creating ORPO trainer with beta={orpo_config.beta}")
            
            trainer = TRLORPOTrainer(
                model=self.model,
                args=trl_orpo_config,
                train_dataset=self.dataset.get_hf_dataset(),
                tokenizer=self.tokenizer,
                max_length=512,  # TODO: Make configurable
                max_prompt_length=256,  # TODO: Make configurable
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
                    "orpo_beta": orpo_config.beta,
                    "orpo_alpha": orpo_config.alpha,
                    **metrics,
                },
            }
        
        except Exception as e:
            logger.error(f"ORPO training failed: {str(e)}")
            raise RLHFTrainingError(
                f"ORPO training failed: {str(e)}",
                code="RLHF010",
                context={
                    "algorithm": "ORPO",
                    "error": str(e),
                }
            ) from e
