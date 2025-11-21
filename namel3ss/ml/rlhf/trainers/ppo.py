"""
PPO Trainer - Proximal Policy Optimization implementation.

Implements PPO algorithm using HuggingFace TRL library.
PPO is the classic RLHF algorithm that uses a reward model to optimize policy.
"""

import logging
from typing import Dict, Any

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
)
from trl import PPOTrainer as TRLPPOTrainer, PPOConfig as TRLPPOConfig

from ..config import RLHFConfig
from ..errors import RLHFTrainingError, RLHFModelError
from .base import BaseRLHFTrainer

logger = logging.getLogger(__name__)


class PPOTrainer(BaseRLHFTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO).
    
    PPO optimizes a policy model using a separate reward model to score generations.
    This is the classic RLHF algorithm used by InstructGPT and ChatGPT.
    
    Reference: https://arxiv.org/abs/1707.06347
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize PPO trainer."""
        super().__init__(config, model, tokenizer)
        
        self.reward_model = None
        
        # Load reward model
        if config.reward_model:
            self._load_reward_model()
        else:
            raise RLHFModelError(
                "PPO requires a reward model",
                code="RLHF035",
                context={"algorithm": "PPO"}
            )
        
        logger.info("Initialized PPO trainer")
    
    def _load_reward_model(self) -> None:
        """
        Load reward model for scoring generations.
        
        Raises:
            RLHFModelError: If reward model cannot be loaded
        """
        logger.info(f"Loading reward model: {self.config.reward_model}")
        
        try:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model,
                num_labels=1,  # Regression head for reward
                trust_remote_code=True,
            )
            
            logger.info("Reward model loaded successfully")
        
        except Exception as e:
            raise RLHFModelError(
                f"Failed to load reward model: {str(e)}",
                code="RLHF030",
                context={
                    "reward_model": self.config.reward_model,
                    "error": str(e),
                }
            )
    
    def train(self) -> Dict[str, Any]:
        """
        Execute PPO training.
        
        Returns:
            Dictionary with training results:
                - final_loss: Final training loss
                - best_loss: Best validation loss
                - total_steps: Total training steps
                - final_lr: Final learning rate
                - metrics: Algorithm-specific metrics (KL divergence, rewards, etc.)
        """
        logger.info("Starting PPO training")
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get PPO-specific config
            ppo_config = self.config.ppo_config
            if not ppo_config:
                from ..config import PPOConfig
                ppo_config = PPOConfig()
            
            # Create TRL PPO config
            trl_ppo_config = TRLPPOConfig(
                model_name=self.config.base_model,
                learning_rate=self.config.learning_rate,
                batch_size=ppo_config.batch_size,
                mini_batch_size=ppo_config.mini_batch_size,
                ppo_epochs=ppo_config.ppo_epochs,
                gamma=ppo_config.gamma,
                lam=ppo_config.lam,
                cliprange=ppo_config.cliprange,
                vf_coef=ppo_config.vf_coef,
                init_kl_coef=ppo_config.init_kl_coef,
                target_kl=ppo_config.target_kl,
                adap_kl_ctrl=ppo_config.adap_kl_ctrl,
                seed=self.config.seed,
            )
            
            # Create TRL PPO trainer
            logger.info(f"Creating PPO trainer with batch_size={ppo_config.batch_size}")
            
            trainer = TRLPPOTrainer(
                config=trl_ppo_config,
                model=self.model,
                ref_model=None,  # Will create reference model automatically
                tokenizer=self.tokenizer,
                dataset=self.dataset.get_hf_dataset(),
            )
            
            # Training loop
            logger.info("Beginning PPO training loop")
            
            total_steps = 0
            final_loss = 0.0
            kl_divergences = []
            rewards = []
            
            for step, batch in enumerate(trainer.dataloader):
                if step >= self.config.max_steps:
                    break
                
                # Generate responses
                query_tensors = batch["input_ids"]
                response_tensors = trainer.generate(query_tensors)
                
                # Compute rewards using reward model
                batch_rewards = []
                for query, response in zip(query_tensors, response_tensors):
                    # Concatenate query and response
                    full_text = self.tokenizer.decode(
                        torch.cat([query, response]),
                        skip_special_tokens=True
                    )
                    
                    # Get reward from reward model
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    
                    with torch.no_grad():
                        reward = self.reward_model(**inputs).logits[0, 0].item()
                    
                    batch_rewards.append(reward)
                
                # PPO step
                stats = trainer.step(query_tensors, response_tensors, batch_rewards)
                
                # Collect metrics
                if "loss" in stats:
                    final_loss = stats["loss"]
                if "kl" in stats:
                    kl_divergences.append(stats["kl"])
                if "reward" in stats:
                    rewards.append(stats["reward"])
                
                total_steps += 1
                
                if step % self.config.logging.log_interval == 0:
                    logger.info(
                        f"Step {step}: loss={final_loss:.4f}, "
                        f"reward={batch_rewards[0]:.4f}, "
                        f"kl={stats.get('kl', 0):.4f}"
                    )
            
            avg_kl = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            
            logger.info(f"PPO training complete. Final loss: {final_loss:.4f}")
            logger.info(f"Average KL divergence: {avg_kl:.4f}")
            logger.info(f"Average reward: {avg_reward:.4f}")
            
            return {
                "final_loss": final_loss,
                "best_loss": final_loss,
                "total_steps": total_steps,
                "final_lr": self.config.learning_rate,
                "metrics": {
                    "avg_kl_divergence": avg_kl,
                    "avg_reward": avg_reward,
                    "ppo_epochs": ppo_config.ppo_epochs,
                    "cliprange": ppo_config.cliprange,
                },
            }
        
        except Exception as e:
            logger.error(f"PPO training failed: {str(e)}")
            raise RLHFTrainingError(
                f"PPO training failed: {str(e)}",
                code="RLHF010",
                context={
                    "algorithm": "PPO",
                    "error": str(e),
                }
            ) from e
