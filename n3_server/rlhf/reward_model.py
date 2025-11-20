"""
Reward model for RLHF training.

Trains a model to predict human preference scores from prompt-response pairs.
This reward model is used during PPO training to score generated responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""
    
    # Model configuration
    base_model: str = "gpt2"  # Base model name from HuggingFace
    use_lora: bool = True  # Use LoRA for efficient fine-tuning
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.1  # LoRA dropout
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Data configuration
    max_length: int = 512
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModelHead(nn.Module):
    """
    Reward model head that predicts a scalar reward.
    
    Takes the final hidden state from the base model and predicts
    a single reward value.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            rewards: [batch_size] scalar rewards
        """
        # Use the last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply dropout and linear layer
        x = self.dropout(last_hidden)
        rewards = self.linear(x).squeeze(-1)  # [batch_size]
        
        return rewards


class RewardModel(nn.Module):
    """
    Complete reward model.
    
    Consists of:
    1. Base transformer model (e.g., GPT-2)
    2. Reward head that predicts scalar rewards
    
    Can be trained with LoRA for efficiency.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: RewardModelConfig,
    ):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Apply LoRA if requested
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
            )
            self.base_model = get_peft_model(base_model, lora_config)
        
        # Reward head
        hidden_size = base_model.config.hidden_size
        self.reward_head = RewardModelHead(hidden_size)
        
        self.to(config.device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            rewards: [batch_size] predicted rewards
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Predict rewards
        rewards = self.reward_head(hidden_states)
        
        return rewards
    
    def train_reward_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the reward model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            Training history with losses and metrics
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        criterion = nn.MSELoss()
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.train()
            train_losses = []
            train_maes = []
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                scores = batch["score"].to(self.config.device)
                
                # Forward pass
                predictions = self.forward(input_ids, attention_mask)
                loss = criterion(predictions, scores)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_losses.append(loss.item())
                mae = torch.abs(predictions - scores).mean().item()
                train_maes.append(mae)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            history["train_loss"].append(avg_train_loss)
            history["train_mae"].append(avg_train_mae)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_losses = []
                val_maes = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.config.device)
                        attention_mask = batch["attention_mask"].to(self.config.device)
                        scores = batch["score"].to(self.config.device)
                        
                        predictions = self.forward(input_ids, attention_mask)
                        loss = criterion(predictions, scores)
                        
                        val_losses.append(loss.item())
                        mae = torch.abs(predictions - scores).mean().item()
                        val_maes.append(mae)
                
                avg_val_loss = np.mean(val_losses)
                avg_val_mae = np.mean(val_maes)
                history["val_loss"].append(avg_val_loss)
                history["val_mae"].append(avg_val_mae)
                
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        
        return history
    
    def save_pretrained(self, path: Path) -> None:
        """Save reward model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save base model
        if self.config.use_lora:
            # Save LoRA adapters
            self.base_model.save_pretrained(path / "base_model")
        else:
            # Save full model
            self.base_model.save_pretrained(path / "base_model")
        
        # Save reward head
        torch.save(
            self.reward_head.state_dict(),
            path / "reward_head.pt",
        )
        
        # Save config
        import json
        with open(path / "config.json", "w") as f:
            json.dump(
                {
                    "base_model": self.config.base_model,
                    "use_lora": self.config.use_lora,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "hidden_size": self.base_model.config.hidden_size,
                },
                f,
                indent=2,
            )
    
    @classmethod
    def load_pretrained(
        cls,
        path: Path,
        device: str = "cpu",
    ) -> RewardModel:
        """Load reward model from disk."""
        import json
        
        # Load config
        with open(path / "config.json") as f:
            saved_config = json.load(f)
        
        # Create config
        config = RewardModelConfig(
            base_model=saved_config["base_model"],
            use_lora=saved_config["use_lora"],
            lora_r=saved_config["lora_r"],
            lora_alpha=saved_config["lora_alpha"],
            lora_dropout=saved_config["lora_dropout"],
            device=device,
        )
        
        # Load base model
        if saved_config["use_lora"]:
            from peft import AutoPeftModel
            base_model = AutoPeftModel.from_pretrained(path / "base_model")
        else:
            base_model = AutoModel.from_pretrained(path / "base_model")
        
        # Create reward model
        model = cls(base_model, config)
        
        # Load reward head
        model.reward_head.load_state_dict(
            torch.load(path / "reward_head.pt", map_location=device)
        )
        
        return model


def train_reward_model(
    train_dataset,
    val_dataset,
    config: RewardModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> RewardModel:
    """
    Train a reward model from scratch.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Reward model configuration
        tokenizer: Tokenizer for base model
    
    Returns:
        Trained reward model
    """
    # Load base model
    base_model = AutoModel.from_pretrained(config.base_model)
    
    # Create reward model
    reward_model = RewardModel(base_model, config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    ) if val_dataset else None
    
    # Train
    reward_model.train_reward_model(train_loader, val_loader)
    
    return reward_model
