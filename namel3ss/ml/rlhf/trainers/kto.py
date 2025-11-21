"""
KTO Trainer - Kahneman-Tversky Optimization implementation.

Implements KTO algorithm using HuggingFace TRL library.
KTO is inspired by prospect theory and works with binary feedback (desirable/undesirable).
"""

import logging
from typing import Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import KTOTrainer as TRLKTOTrainer, KTOConfig as TRLKTOConfig

from ..config import RLHFConfig
from ..errors import RLHFTrainingError
from .base import BaseRLHFTrainer

logger = logging.getLogger(__name__)


class KTOTrainer(BaseRLHFTrainer):
    """
    Trainer for Kahneman-Tversky Optimization (KTO).
    
    KTO is an RLHF algorithm inspired by Kahneman and Tversky's prospect theory.
    It optimizes models using binary feedback (desirable/undesirable) rather than
    pairwise preferences, making it more sample-efficient in some scenarios.
    
    Reference: https://arxiv.org/abs/2402.01306
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize KTO trainer."""
        super().__init__(config, model, tokenizer)
        
        logger.info("Initialized KTO trainer")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute KTO training.
        
        Returns:
            Dictionary with training results:
                - final_loss: Final training loss
                - best_loss: Best validation loss
                - total_steps: Total training steps
                - final_lr: Final learning rate
                - metrics: Algorithm-specific metrics
        """
        logger.info("Starting KTO training")
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get KTO-specific config
            kto_config = self.config.kto_config
            if not kto_config:
                from ..config import KTOConfig
                kto_config = KTOConfig()
            
            # Create TRL KTO config
            trl_kto_config = TRLKTOConfig(
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
                beta=kto_config.beta,
                desirable_weight=kto_config.desirable_weight,
                undesirable_weight=kto_config.undesirable_weight,
            )
            
            # Create TRL KTO trainer
            logger.info(
                f"Creating KTO trainer with beta={kto_config.beta}, "
                f"desirable_weight={kto_config.desirable_weight}"
            )
            
            trainer = TRLKTOTrainer(
                model=self.model,
                ref_model=None,  # Will create reference model automatically
                args=trl_kto_config,
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
                    "kto_beta": kto_config.beta,
                    "desirable_weight": kto_config.desirable_weight,
                    "undesirable_weight": kto_config.undesirable_weight,
                    **metrics,
                },
            }
        
        except Exception as e:
            logger.error(f"KTO training failed: {str(e)}")
            raise RLHFTrainingError(
                f"KTO training failed: {str(e)}",
                code="RLHF010",
                context={
                    "algorithm": "KTO",
                    "error": str(e),
                }
            ) from e
