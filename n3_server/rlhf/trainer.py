"""
RLHF trainer using HuggingFace TRL library.

Implements PPO (Proximal Policy Optimization) for training language models
from human feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np

from .reward_model import RewardModel, RewardModelConfig, train_reward_model
from .dataset import prepare_feedback_dataset, prepare_ppo_dataset


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    max_length: int = 512
    max_prompt_length: int = 256
    train_split: float = 0.8
    normalize_scores: bool = True


@dataclass
class TrainingConfig:
    """Configuration for RLHF training."""
    
    # Model configuration
    base_model: str = "gpt2"  # Base LLM to train
    use_lora: bool = True  # Use LoRA for efficient fine-tuning
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0
    
    # PPO-specific
    init_kl_coef: float = 0.2  # KL divergence coefficient
    target_kl: float = 6.0  # Target KL divergence
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda
    cliprange: float = 0.2  # PPO clip range
    cliprange_value: float = 0.2  # Value function clip range
    vf_coef: float = 0.1  # Value function coefficient
    
    # Generation configuration
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Dataset configuration
    dataset: DatasetConfig = None
    
    # Training steps
    max_steps: int = 1000
    
    # Reward model training
    train_reward_model_first: bool = True
    reward_model_config: Optional[RewardModelConfig] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.reward_model_config is None:
            self.reward_model_config = RewardModelConfig(
                base_model=self.base_model,
                use_lora=self.use_lora,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                device=self.device,
            )


@dataclass
class TrainingResult:
    """Results from RLHF training."""
    
    model_path: str
    reward_model_path: Optional[str]
    feedback_count: int
    final_reward_mean: float
    final_reward_std: float
    training_steps: int
    reward_history: List[float]
    kl_history: List[float]
    loss_history: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> TrainingResult:
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class RLHFTrainer:
    """
    RLHF trainer using TRL's PPO implementation.
    
    Training pipeline:
    1. Train reward model on human feedback
    2. Initialize policy (base LLM with value head)
    3. Run PPO training loop:
       - Generate responses to prompts
       - Score responses with reward model
       - Update policy with PPO
    4. Save trained policy and metrics
    """
    
    def __init__(
        self,
        config: TrainingConfig,
    ):
        """
        Initialize RLHF trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.reward_model: Optional[RewardModel] = None
        self.policy_model = None
        self.ppo_trainer: Optional[PPOTrainer] = None
    
    def prepare_models(self) -> None:
        """Load and prepare tokenizer, reward model, and policy model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loaded tokenizer for {self.config.base_model}")
    
    def train_reward_model_from_feedback(
        self,
        feedbacks: List[Dict[str, Any]],
    ) -> RewardModel:
        """
        Train reward model from feedback data.
        
        Args:
            feedbacks: List of feedback dictionaries
        
        Returns:
            Trained reward model
        """
        print(f"Training reward model on {len(feedbacks)} feedback examples...")
        
        # Prepare datasets
        train_dataset, val_dataset = prepare_feedback_dataset(
            feedbacks,
            self.tokenizer,
            max_length=self.config.dataset.max_length,
            normalize_scores=self.config.dataset.normalize_scores,
            train_split=self.config.dataset.train_split,
        )
        
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Val: {len(val_dataset)} examples")
        
        # Train reward model
        reward_model = train_reward_model(
            train_dataset,
            val_dataset,
            self.config.reward_model_config,
            self.tokenizer,
        )
        
        print("Reward model training complete!")
        return reward_model
    
    def prepare_policy_model(self) -> AutoModelForCausalLMWithValueHead:
        """
        Prepare policy model with value head for PPO.
        
        Returns:
            Policy model ready for PPO training
        """
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
        
        # Apply LoRA if requested
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
            )
            model = get_peft_model(model, lora_config)
        
        # Add value head for PPO
        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model,
        )
        
        return policy_model
    
    def train(
        self,
        feedbacks: List[Dict[str, Any]],
        output_dir: Path,
        reward_model_path: Optional[Path] = None,
    ) -> TrainingResult:
        """
        Run complete RLHF training pipeline.
        
        Args:
            feedbacks: List of feedback dictionaries from database
            output_dir: Directory to save trained models
            reward_model_path: Pre-trained reward model path (optional)
        
        Returns:
            Training results with metrics and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare models
        self.prepare_models()
        
        # Step 2: Train or load reward model
        if reward_model_path and reward_model_path.exists():
            print(f"Loading pre-trained reward model from {reward_model_path}")
            self.reward_model = RewardModel.load_pretrained(
                reward_model_path,
                device=self.config.device,
            )
        elif self.config.train_reward_model_first:
            self.reward_model = self.train_reward_model_from_feedback(feedbacks)
            # Save reward model
            reward_model_save_path = output_dir / "reward_model"
            self.reward_model.save_pretrained(reward_model_save_path)
            print(f"Saved reward model to {reward_model_save_path}")
        else:
            raise ValueError("Must provide reward_model_path or set train_reward_model_first=True")
        
        # Step 3: Prepare policy model
        print("Preparing policy model...")
        self.policy_model = self.prepare_policy_model()
        
        # Step 4: Prepare PPO dataset (prompts only)
        ppo_dataset = prepare_ppo_dataset(
            feedbacks,
            self.tokenizer,
            max_prompt_length=self.config.dataset.max_prompt_length,
        )
        
        # Step 5: Configure PPO trainer
        ppo_config = PPOConfig(
            model_name=self.config.base_model,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            max_grad_norm=self.config.max_grad_norm,
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target_kl,
            gamma=self.config.gamma,
            lam=self.config.lam,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            tokenizer=self.tokenizer,
            dataset=ppo_dataset,
        )
        
        # Step 6: PPO training loop
        print(f"Starting PPO training for {self.config.max_steps} steps...")
        
        reward_history = []
        kl_history = []
        loss_history = []
        
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "do_sample": True,
        }
        
        for step, batch in enumerate(self.ppo_trainer.dataloader):
            if step >= self.config.max_steps:
                break
            
            # Get prompts
            prompt_tensors = batch["input_ids"]
            
            # Generate responses
            response_tensors = self.ppo_trainer.generate(
                prompt_tensors,
                return_prompt=False,
                **generation_kwargs,
            )
            
            # Combine prompts and responses
            full_tensors = [
                torch.cat([p, r])
                for p, r in zip(prompt_tensors, response_tensors)
            ]
            
            # Create attention masks
            attention_masks = [
                torch.ones_like(t)
                for t in full_tensors
            ]
            
            # Pad to same length
            max_len = max(t.size(0) for t in full_tensors)
            padded_tensors = []
            padded_masks = []
            
            for t, m in zip(full_tensors, attention_masks):
                pad_len = max_len - t.size(0)
                if pad_len > 0:
                    padded_t = torch.cat([
                        t,
                        torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=t.dtype)
                    ])
                    padded_m = torch.cat([
                        m,
                        torch.zeros(pad_len, dtype=m.dtype)
                    ])
                else:
                    padded_t = t
                    padded_m = m
                
                padded_tensors.append(padded_t)
                padded_masks.append(padded_m)
            
            # Stack into batch
            input_ids_batch = torch.stack(padded_tensors).to(self.config.device)
            attention_mask_batch = torch.stack(padded_masks).to(self.config.device)
            
            # Get rewards from reward model
            with torch.no_grad():
                rewards = self.reward_model(input_ids_batch, attention_mask_batch)
            
            # Convert to list for PPO trainer
            rewards_list = [r.unsqueeze(0) for r in rewards]
            
            # Run PPO step
            stats = self.ppo_trainer.step(
                prompt_tensors,
                response_tensors,
                rewards_list,
            )
            
            # Track metrics
            if "ppo/returns/mean" in stats:
                reward_history.append(stats["ppo/returns/mean"])
            if "ppo/policy/approxkl" in stats:
                kl_history.append(stats["ppo/policy/approxkl"])
            if "ppo/loss/total" in stats:
                loss_history.append(stats["ppo/loss/total"])
            
            # Log progress
            if step % 10 == 0:
                print(f"Step {step}/{self.config.max_steps}")
                if reward_history:
                    print(f"  Reward: {reward_history[-1]:.4f}")
                if kl_history:
                    print(f"  KL: {kl_history[-1]:.4f}")
                if loss_history:
                    print(f"  Loss: {loss_history[-1]:.4f}")
        
        # Step 7: Save trained policy
        policy_save_path = output_dir / "policy"
        self.policy_model.save_pretrained(policy_save_path)
        self.tokenizer.save_pretrained(policy_save_path)
        print(f"Saved trained policy to {policy_save_path}")
        
        # Step 8: Compile results
        result = TrainingResult(
            model_path=str(policy_save_path),
            reward_model_path=str(output_dir / "reward_model") if self.config.train_reward_model_first else None,
            feedback_count=len(feedbacks),
            final_reward_mean=float(np.mean(reward_history)) if reward_history else 0.0,
            final_reward_std=float(np.std(reward_history)) if reward_history else 0.0,
            training_steps=step + 1,
            reward_history=reward_history,
            kl_history=kl_history,
            loss_history=loss_history,
        )
        
        result.save(output_dir / "training_results.json")
        print("Training complete!")
        
        return result
    
    def estimate_training(
        self,
        feedbacks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Estimate training without actually training.
        
        Args:
            feedbacks: List of feedback dictionaries
        
        Returns:
            Estimation with metrics and parameters
        """
        scores = [f["score"] for f in feedbacks]
        
        return {
            "feedback_count": len(feedbacks),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "estimated_steps": self.config.max_steps,
            "estimated_time_minutes": (
                self.config.max_steps * self.config.batch_size * 2  # rough estimate
            ) / 60,
            "model_config": {
                "base_model": self.config.base_model,
                "use_lora": self.config.use_lora,
                "lora_r": self.config.lora_r,
                "learning_rate": self.config.learning_rate,
            },
        }
