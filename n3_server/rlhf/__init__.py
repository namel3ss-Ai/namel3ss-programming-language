"""RLHF training pipeline using HuggingFace TRL library."""

from .trainer import RLHFTrainer, TrainingConfig, TrainingResult, DatasetConfig
from .reward_model import RewardModel, RewardModelConfig, train_reward_model
from .dataset import FeedbackDataset, prepare_feedback_dataset, prepare_ppo_dataset

__all__ = [
    "RLHFTrainer",
    "TrainingConfig",
    "TrainingResult",
    "DatasetConfig",
    "RewardModel",
    "RewardModelConfig",
    "train_reward_model",
    "FeedbackDataset",
    "prepare_feedback_dataset",
    "prepare_ppo_dataset",
]
