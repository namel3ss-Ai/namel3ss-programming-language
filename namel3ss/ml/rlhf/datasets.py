"""
RLHF Dataset Loaders - Handle preference and feedback datasets.

This module provides dataset loading and preprocessing for RLHF training:
- PreferenceDataset: Loads pairwise preference data (chosen/rejected)
- FeedbackDataset: Loads human feedback scores and rankings
- Dataset validation and normalization
- Train/validation splitting
- Integration with HuggingFace Datasets

Supports multiple formats:
- HuggingFace Hub datasets
- Local Parquet files
- JSON/JSONL files
- Custom data loaders
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import datasets
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset

from .config import RLHFAlgorithm
from .errors import RLHFDatasetError

logger = logging.getLogger(__name__)


@dataclass
class PreferenceSample:
    """
    A single preference data sample.
    
    Contains a prompt and two responses (chosen/rejected) for pairwise preference learning.
    Used by DPO, IPO, ORPO algorithms.
    """
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeedbackSample:
    """
    A single feedback data sample.
    
    Contains a prompt, response, and feedback score for reward modeling or KTO.
    """
    prompt: str
    response: str
    score: Optional[float] = None  # For reward modeling
    is_desirable: Optional[bool] = None  # For KTO
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PreferenceDataset(TorchDataset):
    """
    Dataset for pairwise preference learning (DPO, IPO, ORPO).
    
    Loads and preprocesses preference data where each sample contains:
    - A prompt
    - A chosen (preferred) response
    - A rejected (non-preferred) response
    
    Supports loading from:
    - HuggingFace Hub (e.g., "HuggingFaceH4/ultrafeedback_binarized")
    - Local Parquet files
    - JSON/JSONL files
    
    Example:
        >>> dataset = PreferenceDataset(
        ...     path="HuggingFaceH4/ultrafeedback_binarized",
        ...     split="train",
        ...     prompt_col="prompt",
        ...     chosen_col="chosen",
        ...     rejected_col="rejected"
        ... )
        >>> sample = dataset[0]
        >>> print(sample.prompt)
    """
    
    def __init__(
        self,
        path: str,
        split: str = "train",
        prompt_col: str = "prompt",
        chosen_col: str = "chosen",
        rejected_col: str = "rejected",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize preference dataset.
        
        Args:
            path: Dataset path (HuggingFace Hub ID or local path)
            split: Dataset split to load ("train", "validation", "test")
            prompt_col: Column name for prompts
            chosen_col: Column name for chosen responses
            rejected_col: Column name for rejected responses
            max_samples: Maximum number of samples to load (None = all)
            seed: Random seed for shuffling
            
        Raises:
            RLHFDatasetError: If dataset cannot be loaded or is invalid
        """
        self.path = path
        self.split = split
        self.prompt_col = prompt_col
        self.chosen_col = chosen_col
        self.rejected_col = rejected_col
        
        logger.info(f"Loading preference dataset from {path}, split={split}")
        
        try:
            # Load dataset
            self.dataset = self._load_dataset()
            
            # Validate columns
            self._validate_columns()
            
            # Shuffle and limit samples
            self.dataset = self.dataset.shuffle(seed=seed)
            if max_samples:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
            logger.info(f"Loaded {len(self.dataset)} preference samples")
        
        except Exception as e:
            raise RLHFDatasetError(
                f"Failed to load preference dataset: {str(e)}",
                code="RLHF020",
                context={
                    "path": path,
                    "split": split,
                    "error": str(e),
                }
            )
    
    def _load_dataset(self) -> Dataset:
        """
        Load dataset from HuggingFace Hub or local path.
        
        Returns:
            Loaded HuggingFace Dataset
        """
        path_obj = Path(self.path)
        
        # Check if local file
        if path_obj.exists():
            if path_obj.suffix == ".parquet":
                return load_dataset("parquet", data_files=str(path_obj), split=self.split)
            elif path_obj.suffix in [".json", ".jsonl"]:
                return load_dataset("json", data_files=str(path_obj), split=self.split)
            else:
                raise RLHFDatasetError(
                    f"Unsupported file format: {path_obj.suffix}",
                    code="RLHF021",
                    context={"path": self.path, "format": path_obj.suffix}
                )
        
        # Load from HuggingFace Hub
        return load_dataset(self.path, split=self.split)
    
    def _validate_columns(self) -> None:
        """
        Validate that required columns exist.
        
        Raises:
            RLHFDatasetError: If required columns are missing
        """
        required_cols = [self.prompt_col, self.chosen_col, self.rejected_col]
        available_cols = self.dataset.column_names
        
        missing_cols = [col for col in required_cols if col not in available_cols]
        
        if missing_cols:
            raise RLHFDatasetError(
                f"Dataset missing required columns: {missing_cols}",
                code="RLHF022",
                context={
                    "path": self.path,
                    "missing_columns": missing_cols,
                    "available_columns": available_cols,
                }
            )
        
        # Check for empty samples
        if len(self.dataset) == 0:
            raise RLHFDatasetError(
                "Dataset is empty",
                code="RLHF023",
                context={"path": self.path, "split": self.split}
            )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> PreferenceSample:
        """
        Get a single preference sample.
        
        Args:
            idx: Sample index
            
        Returns:
            PreferenceSample with prompt, chosen, and rejected responses
        """
        sample = self.dataset[idx]
        
        # Extract text from possible nested structures
        prompt = self._extract_text(sample[self.prompt_col])
        chosen = self._extract_text(sample[self.chosen_col])
        rejected = self._extract_text(sample[self.rejected_col])
        
        # Collect metadata
        metadata = {k: v for k, v in sample.items() 
                   if k not in [self.prompt_col, self.chosen_col, self.rejected_col]}
        
        return PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata
        )
    
    def _extract_text(self, value: Union[str, List, Dict]) -> str:
        """
        Extract text from various data structures.
        
        Args:
            value: Value to extract text from
            
        Returns:
            Extracted text string
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            # Assume list of messages, concatenate
            if value and isinstance(value[0], dict) and "content" in value[0]:
                return "\n".join([msg["content"] for msg in value])
            return "\n".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Try common keys
            for key in ["content", "text", "message"]:
                if key in value:
                    return str(value[key])
            # Fallback to first value
            return str(next(iter(value.values())))
        else:
            return str(value)
    
    def get_hf_dataset(self) -> Dataset:
        """
        Get underlying HuggingFace Dataset.
        
        Returns:
            HuggingFace Dataset object
        """
        return self.dataset
    
    def train_test_split(
        self,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction of data for test set
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        
        train_ds = PreferenceDataset.__new__(PreferenceDataset)
        train_ds.__dict__.update(self.__dict__)
        train_ds.dataset = split_dataset["train"]
        
        test_ds = PreferenceDataset.__new__(PreferenceDataset)
        test_ds.__dict__.update(self.__dict__)
        test_ds.dataset = split_dataset["test"]
        
        return train_ds, test_ds


class FeedbackDataset(TorchDataset):
    """
    Dataset for feedback-based learning (reward modeling, KTO).
    
    Loads and preprocesses feedback data where each sample contains:
    - A prompt
    - A response
    - A feedback score (for reward modeling) or desirability label (for KTO)
    
    Example:
        >>> dataset = FeedbackDataset(
        ...     path="my_feedback_data.parquet",
        ...     split="train",
        ...     prompt_col="prompt",
        ...     response_col="response",
        ...     score_col="score"
        ... )
        >>> sample = dataset[0]
        >>> print(f"Score: {sample.score}")
    """
    
    def __init__(
        self,
        path: str,
        split: str = "train",
        prompt_col: str = "prompt",
        response_col: str = "response",
        score_col: Optional[str] = "score",
        desirable_col: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize feedback dataset.
        
        Args:
            path: Dataset path (HuggingFace Hub ID or local path)
            split: Dataset split to load
            prompt_col: Column name for prompts
            response_col: Column name for responses
            score_col: Column name for feedback scores (for reward modeling)
            desirable_col: Column name for desirability labels (for KTO)
            max_samples: Maximum number of samples to load
            seed: Random seed
            
        Raises:
            RLHFDatasetError: If dataset cannot be loaded or is invalid
        """
        self.path = path
        self.split = split
        self.prompt_col = prompt_col
        self.response_col = response_col
        self.score_col = score_col
        self.desirable_col = desirable_col
        
        logger.info(f"Loading feedback dataset from {path}, split={split}")
        
        try:
            # Load dataset
            self.dataset = self._load_dataset()
            
            # Validate columns
            self._validate_columns()
            
            # Shuffle and limit
            self.dataset = self.dataset.shuffle(seed=seed)
            if max_samples:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
            logger.info(f"Loaded {len(self.dataset)} feedback samples")
        
        except Exception as e:
            raise RLHFDatasetError(
                f"Failed to load feedback dataset: {str(e)}",
                code="RLHF020",
                context={"path": path, "split": split, "error": str(e)}
            )
    
    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace Hub or local path."""
        path_obj = Path(self.path)
        
        if path_obj.exists():
            if path_obj.suffix == ".parquet":
                return load_dataset("parquet", data_files=str(path_obj), split=self.split)
            elif path_obj.suffix in [".json", ".jsonl"]:
                return load_dataset("json", data_files=str(path_obj), split=self.split)
            else:
                raise RLHFDatasetError(
                    f"Unsupported file format: {path_obj.suffix}",
                    code="RLHF021",
                    context={"path": self.path, "format": path_obj.suffix}
                )
        
        return load_dataset(self.path, split=self.split)
    
    def _validate_columns(self) -> None:
        """Validate that required columns exist."""
        required_cols = [self.prompt_col, self.response_col]
        if self.score_col:
            required_cols.append(self.score_col)
        if self.desirable_col:
            required_cols.append(self.desirable_col)
        
        available_cols = self.dataset.column_names
        missing_cols = [col for col in required_cols if col not in available_cols]
        
        if missing_cols:
            raise RLHFDatasetError(
                f"Dataset missing required columns: {missing_cols}",
                code="RLHF022",
                context={
                    "path": self.path,
                    "missing_columns": missing_cols,
                    "available_columns": available_cols,
                }
            )
        
        if len(self.dataset) == 0:
            raise RLHFDatasetError(
                "Dataset is empty",
                code="RLHF023",
                context={"path": self.path, "split": self.split}
            )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> FeedbackSample:
        """Get a single feedback sample."""
        sample = self.dataset[idx]
        
        prompt = str(sample[self.prompt_col])
        response = str(sample[self.response_col])
        score = float(sample[self.score_col]) if self.score_col and self.score_col in sample else None
        is_desirable = bool(sample[self.desirable_col]) if self.desirable_col and self.desirable_col in sample else None
        
        metadata = {k: v for k, v in sample.items() 
                   if k not in [self.prompt_col, self.response_col, self.score_col, self.desirable_col]}
        
        return FeedbackSample(
            prompt=prompt,
            response=response,
            score=score,
            is_desirable=is_desirable,
            metadata=metadata
        )
    
    def get_hf_dataset(self) -> Dataset:
        """Get underlying HuggingFace Dataset."""
        return self.dataset


def load_preference_dataset(
    path: str,
    split: str = "train",
    algorithm: Optional[RLHFAlgorithm] = None,
    **kwargs
) -> PreferenceDataset:
    """
    Load a preference dataset with automatic column detection.
    
    Args:
        path: Dataset path
        split: Dataset split
        algorithm: RLHF algorithm (for algorithm-specific defaults)
        **kwargs: Additional arguments for PreferenceDataset
        
    Returns:
        PreferenceDataset instance
    """
    # Try common column name patterns
    if "prompt_col" not in kwargs:
        common_prompt_cols = ["prompt", "instruction", "question", "input"]
        kwargs["prompt_col"] = "prompt"  # Default
    
    if "chosen_col" not in kwargs:
        kwargs["chosen_col"] = "chosen"
    
    if "rejected_col" not in kwargs:
        kwargs["rejected_col"] = "rejected"
    
    return PreferenceDataset(path=path, split=split, **kwargs)


def load_feedback_dataset(
    path: str,
    split: str = "train",
    algorithm: Optional[RLHFAlgorithm] = None,
    **kwargs
) -> FeedbackDataset:
    """
    Load a feedback dataset with automatic column detection.
    
    Args:
        path: Dataset path
        split: Dataset split
        algorithm: RLHF algorithm (for algorithm-specific defaults)
        **kwargs: Additional arguments for FeedbackDataset
        
    Returns:
        FeedbackDataset instance
    """
    if "prompt_col" not in kwargs:
        kwargs["prompt_col"] = "prompt"
    
    if "response_col" not in kwargs:
        kwargs["response_col"] = "response"
    
    # Set score/desirable column based on algorithm
    if algorithm == RLHFAlgorithm.KTO:
        if "desirable_col" not in kwargs:
            kwargs["desirable_col"] = "label"
    else:
        if "score_col" not in kwargs:
            kwargs["score_col"] = "score"
    
    return FeedbackDataset(path=path, split=split, **kwargs)
