"""
Feedback dataset preparation for RLHF training.

Converts database feedback into training datasets for reward model and PPO training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class FeedbackExample:
    """Single feedback example for training."""
    prompt: str
    response: str
    score: float
    run_id: str
    notes: Optional[str] = None


class FeedbackDataset(Dataset):
    """
    PyTorch dataset for feedback data.
    
    Tokenizes prompts and responses for reward model or PPO training.
    Handles batching and padding automatically.
    """
    
    def __init__(
        self,
        examples: List[FeedbackExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        normalize_scores: bool = True,
    ):
        """
        Initialize feedback dataset.
        
        Args:
            examples: List of feedback examples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            normalize_scores: Normalize scores to [-1, 1] range
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Normalize scores if requested
        if normalize_scores and examples:
            scores = [ex.score for ex in examples]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for ex in self.examples:
                    # Normalize to [-1, 1]
                    ex.score = 2 * ((ex.score - min_score) / score_range) - 1
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Tokenize prompt and response
        text = f"{example.prompt}\n\n{example.response}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "score": torch.tensor(example.score, dtype=torch.float32),
            "prompt": example.prompt,
            "response": example.response,
        }


class PPODataset(Dataset):
    """
    Dataset for PPO training.
    
    Provides prompts for the policy to generate responses,
    which are then scored by the reward model.
    """
    
    def __init__(
        self,
        examples: List[FeedbackExample],
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 256,
    ):
        """
        Initialize PPO dataset.
        
        Args:
            examples: List of feedback examples (we only use prompts)
            tokenizer: HuggingFace tokenizer
            max_prompt_length: Maximum prompt length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single prompt."""
        example = self.examples[idx]
        
        encoding = self.tokenizer(
            example.prompt,
            max_length=self.max_prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "prompt": example.prompt,
        }


def prepare_feedback_dataset(
    feedbacks: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    normalize_scores: bool = True,
    train_split: float = 0.8,
) -> tuple[FeedbackDataset, FeedbackDataset]:
    """
    Prepare train and validation datasets from feedback.
    
    Args:
        feedbacks: List of feedback dictionaries from database
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        normalize_scores: Normalize scores
        train_split: Fraction of data for training
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Convert to FeedbackExample objects
    examples = [
        FeedbackExample(
            prompt=f["prompt"],
            response=f["response"],
            score=f["score"],
            run_id=f["run_id"],
            notes=f.get("notes"),
        )
        for f in feedbacks
    ]
    
    # Shuffle and split
    import random
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Create datasets
    train_dataset = FeedbackDataset(
        train_examples,
        tokenizer,
        max_length,
        normalize_scores,
    )
    
    val_dataset = FeedbackDataset(
        val_examples,
        tokenizer,
        max_length,
        normalize_scores,
    )
    
    return train_dataset, val_dataset


def prepare_ppo_dataset(
    feedbacks: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int = 256,
) -> PPODataset:
    """
    Prepare dataset for PPO training.
    
    Args:
        feedbacks: List of feedback dictionaries
        tokenizer: HuggingFace tokenizer
        max_prompt_length: Maximum prompt length
    
    Returns:
        PPO dataset with prompts
    """
    examples = [
        FeedbackExample(
            prompt=f["prompt"],
            response=f["response"],
            score=f["score"],
            run_id=f["run_id"],
            notes=f.get("notes"),
        )
        for f in feedbacks
    ]
    
    return PPODataset(examples, tokenizer, max_prompt_length)


def save_dataset(
    dataset: FeedbackDataset,
    path: Path,
) -> None:
    """
    Save dataset to disk.
    
    Args:
        dataset: Feedback dataset
        path: Path to save to
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "examples": [
            {
                "prompt": ex.prompt,
                "response": ex.response,
                "score": ex.score,
                "run_id": ex.run_id,
                "notes": ex.notes,
            }
            for ex in dataset.examples
        ],
        "max_length": dataset.max_length,
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(
    path: Path,
    tokenizer: PreTrainedTokenizer,
) -> FeedbackDataset:
    """
    Load dataset from disk.
    
    Args:
        path: Path to load from
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Loaded feedback dataset
    """
    with open(path) as f:
        data = json.load(f)
    
    examples = [
        FeedbackExample(**ex)
        for ex in data["examples"]
    ]
    
    return FeedbackDataset(
        examples,
        tokenizer,
        max_length=data.get("max_length", 512),
        normalize_scores=False,  # Already normalized when saved
    )
