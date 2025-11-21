"""
RLHF Trainers - Algorithm-specific trainer implementations.

This package provides trainer classes for different RLHF algorithms:
- PPOTrainer: Proximal Policy Optimization
- DPOTrainer: Direct Preference Optimization
- IPOTrainer: Identity Preference Optimization
- ORPOTrainer: Odds Ratio Preference Optimization
- KTOTrainer: Kahneman-Tversky Optimization
- RewardModelTrainer: Reward model training
- SFTTrainer: Supervised Fine-Tuning

All trainers wrap HuggingFace TRL library with our configuration system.
"""

from typing import Type, Dict
from ..config import RLHFAlgorithm
from ..errors import RLHFConfigurationError

# Import trainers
from .base import BaseRLHFTrainer
from .ppo import PPOTrainer
from .dpo import DPOTrainer
from .orpo import ORPOTrainer
from .kto import KTOTrainer
from .sft import SFTTrainer

__all__ = [
    "BaseRLHFTrainer",
    "PPOTrainer",
    "DPOTrainer",
    "ORPOTrainer",
    "KTOTrainer",
    "SFTTrainer",
    "get_trainer_class",
]


# Trainer registry
_TRAINER_REGISTRY: Dict[RLHFAlgorithm, Type[BaseRLHFTrainer]] = {
    RLHFAlgorithm.PPO: PPOTrainer,
    RLHFAlgorithm.DPO: DPOTrainer,
    RLHFAlgorithm.IPO: DPOTrainer,  # IPO uses DPO trainer with different loss
    RLHFAlgorithm.ORPO: ORPOTrainer,
    RLHFAlgorithm.KTO: KTOTrainer,
    RLHFAlgorithm.SFT: SFTTrainer,
    RLHFAlgorithm.REWARD_MODEL: SFTTrainer,  # Reward model uses SFT with classification head
}


def get_trainer_class(algorithm: RLHFAlgorithm) -> Type[BaseRLHFTrainer]:
    """
    Get trainer class for specified algorithm.
    
    Args:
        algorithm: RLHF algorithm
        
    Returns:
        Trainer class
        
    Raises:
        RLHFConfigurationError: If algorithm not supported
    """
    if algorithm not in _TRAINER_REGISTRY:
        raise RLHFConfigurationError(
            f"No trainer available for algorithm: {algorithm.value}",
            code="RLHF002",
            context={"algorithm": algorithm.value}
        )
    
    return _TRAINER_REGISTRY[algorithm]
