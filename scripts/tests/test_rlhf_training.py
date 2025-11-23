"""
Tests for RLHF training pipeline.

Tests dataset preparation, reward model training, and PPO training.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from n3_server.rlhf import (
    FeedbackDataset,
    prepare_feedback_dataset,
    prepare_ppo_dataset,
    RewardModel,
    RewardModelConfig,
    train_reward_model,
    RLHFTrainer,
    TrainingConfig,
    DatasetConfig,
)
from n3_server.rlhf.dataset import FeedbackExample


# Sample feedback data
SAMPLE_FEEDBACKS = [
    {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "score": 0.9,
        "run_id": "run_1",
        "notes": "Accurate answer",
    },
    {
        "prompt": "What is 2 + 2?",
        "response": "2 + 2 equals 4.",
        "score": 1.0,
        "run_id": "run_2",
        "notes": "Correct math",
    },
    {
        "prompt": "Explain quantum physics",
        "response": "Quantum physics is hard.",
        "score": 0.3,
        "run_id": "run_3",
        "notes": "Too vague",
    },
    {
        "prompt": "What is the speed of light?",
        "response": "The speed of light is approximately 299,792,458 meters per second.",
        "score": 0.95,
        "run_id": "run_4",
        "notes": "Accurate and precise",
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
        "score": 1.0,
        "run_id": "run_5",
        "notes": "Perfect answer",
    },
    {
        "prompt": "What is the largest planet?",
        "response": "Jupiter is the largest planet in our solar system.",
        "score": 0.9,
        "run_id": "run_6",
        "notes": "Correct and clear",
    },
    {
        "prompt": "How do airplanes fly?",
        "response": "They have wings.",
        "score": 0.2,
        "run_id": "run_7",
        "notes": "Incomplete explanation",
    },
    {
        "prompt": "What is DNA?",
        "response": "DNA is the molecule that carries genetic information.",
        "score": 0.8,
        "run_id": "run_8",
        "notes": "Good basic explanation",
    },
    {
        "prompt": "Name three programming languages",
        "response": "Python, JavaScript, and Java are programming languages.",
        "score": 1.0,
        "run_id": "run_9",
        "notes": "Complete answer",
    },
    {
        "prompt": "What causes rain?",
        "response": "Rain occurs when water vapor condenses in clouds.",
        "score": 0.7,
        "run_id": "run_10",
        "notes": "Good but could be more detailed",
    },
]


@pytest.fixture
def tokenizer():
    """Load GPT-2 tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_feedbacks():
    """Get sample feedback data."""
    return SAMPLE_FEEDBACKS.copy()


class TestFeedbackDataset:
    """Test feedback dataset preparation."""
    
    def test_dataset_creation(self, tokenizer, sample_feedbacks):
        """Test creating feedback dataset."""
        examples = [FeedbackExample(**f) for f in sample_feedbacks]
        dataset = FeedbackDataset(examples, tokenizer, max_length=128)
        
        assert len(dataset) == len(sample_feedbacks)
        
        # Check first example
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "score" in item
        assert "prompt" in item
        assert "response" in item
        
        # Check shapes
        assert item["input_ids"].shape == (128,)
        assert item["attention_mask"].shape == (128,)
        assert isinstance(item["score"], torch.Tensor)
    
    def test_score_normalization(self, tokenizer, sample_feedbacks):
        """Test score normalization to [-1, 1] range."""
        examples = [FeedbackExample(**f) for f in sample_feedbacks]
        dataset = FeedbackDataset(
            examples,
            tokenizer,
            max_length=128,
            normalize_scores=True,
        )
        
        # Check scores are normalized
        scores = [dataset[i]["score"].item() for i in range(len(dataset))]
        assert min(scores) >= -1.0
        assert max(scores) <= 1.0
    
    def test_no_normalization(self, tokenizer, sample_feedbacks):
        """Test dataset without score normalization."""
        examples = [FeedbackExample(**f) for f in sample_feedbacks]
        dataset = FeedbackDataset(
            examples,
            tokenizer,
            max_length=128,
            normalize_scores=False,
        )
        
        # Check scores are unchanged
        for i, feedback in enumerate(sample_feedbacks):
            assert abs(dataset[i]["score"].item() - feedback["score"]) < 1e-6


class TestDatasetPreparation:
    """Test dataset preparation utilities."""
    
    def test_prepare_feedback_dataset(self, tokenizer, sample_feedbacks):
        """Test preparing train and validation datasets."""
        train_dataset, val_dataset = prepare_feedback_dataset(
            sample_feedbacks,
            tokenizer,
            max_length=128,
            train_split=0.8,
        )
        
        # Check split sizes
        total_size = len(sample_feedbacks)
        expected_train_size = int(total_size * 0.8)
        expected_val_size = total_size - expected_train_size
        
        assert len(train_dataset) == expected_train_size
        assert len(val_dataset) == expected_val_size
    
    def test_prepare_ppo_dataset(self, tokenizer, sample_feedbacks):
        """Test preparing PPO dataset (prompts only)."""
        ppo_dataset = prepare_ppo_dataset(
            sample_feedbacks,
            tokenizer,
            max_prompt_length=64,
        )
        
        assert len(ppo_dataset) == len(sample_feedbacks)
        
        # Check first item
        item = ppo_dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "prompt" in item
        assert item["input_ids"].shape == (64,)


class TestRewardModel:
    """Test reward model training and inference."""
    
    @pytest.fixture
    def reward_model_config(self):
        """Create reward model config for testing."""
        return RewardModelConfig(
            base_model="gpt2",
            use_lora=True,
            batch_size=2,
            num_epochs=1,  # Quick test
            device="cpu",
        )
    
    def test_reward_model_creation(self, reward_model_config):
        """Test creating reward model."""
        from transformers import AutoModel
        
        base_model = AutoModel.from_pretrained("gpt2")
        reward_model = RewardModel(base_model, reward_model_config)
        
        assert reward_model is not None
        assert hasattr(reward_model, "base_model")
        assert hasattr(reward_model, "reward_head")
    
    def test_reward_model_forward(self, reward_model_config):
        """Test reward model forward pass."""
        from transformers import AutoModel
        
        base_model = AutoModel.from_pretrained("gpt2")
        reward_model = RewardModel(base_model, reward_model_config)
        
        # Create dummy input
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        rewards = reward_model(input_ids, attention_mask)
        
        # Check output shape
        assert rewards.shape == (batch_size,)
    
    def test_reward_model_training(
        self,
        tokenizer,
        sample_feedbacks,
        reward_model_config,
        tmp_path,
    ):
        """Test training reward model."""
        # Prepare datasets
        train_dataset, val_dataset = prepare_feedback_dataset(
            sample_feedbacks,
            tokenizer,
            max_length=128,
            train_split=0.8,
        )
        
        # Train reward model (1 epoch for speed)
        reward_model = train_reward_model(
            train_dataset,
            val_dataset,
            reward_model_config,
            tokenizer,
        )
        
        assert reward_model is not None
        
        # Test inference
        item = train_dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)
        
        with torch.no_grad():
            reward = reward_model(input_ids, attention_mask)
        
        assert reward.shape == (1,)
    
    def test_reward_model_save_load(
        self,
        tokenizer,
        sample_feedbacks,
        reward_model_config,
        tmp_path,
    ):
        """Test saving and loading reward model."""
        # Train reward model
        train_dataset, val_dataset = prepare_feedback_dataset(
            sample_feedbacks,
            tokenizer,
            max_length=128,
            train_split=0.8,
        )
        
        reward_model = train_reward_model(
            train_dataset,
            val_dataset,
            reward_model_config,
            tokenizer,
        )
        
        # Save model
        save_path = tmp_path / "reward_model"
        reward_model.save_pretrained(save_path)
        
        assert (save_path / "base_model").exists()
        assert (save_path / "reward_head.pt").exists()
        assert (save_path / "config.json").exists()
        
        # Load model
        loaded_model = RewardModel.load_pretrained(save_path, device="cpu")
        
        assert loaded_model is not None
        
        # Test inference with loaded model
        item = train_dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)
        
        with torch.no_grad():
            original_reward = reward_model(input_ids, attention_mask)
            loaded_reward = loaded_model(input_ids, attention_mask)
        
        # Should produce similar results
        assert torch.allclose(original_reward, loaded_reward, atol=1e-3)


class TestRLHFTrainer:
    """Test RLHF trainer."""
    
    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        return TrainingConfig(
            base_model="gpt2",
            use_lora=True,
            learning_rate=1e-5,
            batch_size=2,
            max_steps=5,  # Very short for testing
            train_reward_model_first=True,
            dataset=DatasetConfig(
                max_length=128,
                max_prompt_length=64,
            ),
            reward_model_config=RewardModelConfig(
                base_model="gpt2",
                use_lora=True,
                batch_size=2,
                num_epochs=1,
                device="cpu",
            ),
            device="cpu",
        )
    
    def test_trainer_creation(self, training_config):
        """Test creating RLHF trainer."""
        trainer = RLHFTrainer(training_config)
        
        assert trainer is not None
        assert trainer.config == training_config
    
    def test_estimate_training(self, training_config, sample_feedbacks):
        """Test training estimation."""
        trainer = RLHFTrainer(training_config)
        
        estimate = trainer.estimate_training(sample_feedbacks)
        
        assert "feedback_count" in estimate
        assert "score_mean" in estimate
        assert "score_std" in estimate
        assert "estimated_steps" in estimate
        assert "estimated_time_minutes" in estimate
        assert "model_config" in estimate
        
        assert estimate["feedback_count"] == len(sample_feedbacks)
        assert estimate["estimated_steps"] == training_config.max_steps
    
    def test_prepare_models(self, training_config):
        """Test preparing models."""
        trainer = RLHFTrainer(training_config)
        trainer.prepare_models()
        
        assert trainer.tokenizer is not None
        assert trainer.tokenizer.pad_token is not None
    
    def test_train_reward_model_from_feedback(
        self,
        training_config,
        sample_feedbacks,
    ):
        """Test training reward model from feedback."""
        trainer = RLHFTrainer(training_config)
        trainer.prepare_models()
        
        reward_model = trainer.train_reward_model_from_feedback(sample_feedbacks)
        
        assert reward_model is not None
        assert isinstance(reward_model, RewardModel)
    
    def test_prepare_policy_model(self, training_config):
        """Test preparing policy model."""
        trainer = RLHFTrainer(training_config)
        trainer.prepare_models()
        
        policy_model = trainer.prepare_policy_model()
        
        assert policy_model is not None
        # Should have value head for PPO
        assert hasattr(policy_model, "v_head")
    
    @pytest.mark.slow
    def test_full_training(
        self,
        training_config,
        sample_feedbacks,
        tmp_path,
    ):
        """Test full RLHF training pipeline."""
        trainer = RLHFTrainer(training_config)
        
        output_dir = tmp_path / "trained_model"
        
        # Run training
        result = trainer.train(
            feedbacks=sample_feedbacks,
            output_dir=output_dir,
        )
        
        # Check result
        assert result is not None
        assert result.feedback_count == len(sample_feedbacks)
        assert result.training_steps > 0
        assert result.model_path is not None
        assert Path(result.model_path).exists()
        
        # Check saved files
        assert (output_dir / "policy").exists()
        assert (output_dir / "reward_model").exists()
        assert (output_dir / "training_results.json").exists()


class TestIntegration:
    """Integration tests for RLHF system."""
    
    def test_end_to_end_pipeline(
        self,
        tokenizer,
        sample_feedbacks,
        tmp_path,
    ):
        """Test complete pipeline from feedback to trained policy."""
        # Create config
        config = TrainingConfig(
            base_model="gpt2",
            use_lora=True,
            max_steps=5,
            batch_size=2,
            train_reward_model_first=True,
            reward_model_config=RewardModelConfig(
                base_model="gpt2",
                use_lora=True,
                batch_size=2,
                num_epochs=1,
                device="cpu",
            ),
            device="cpu",
        )
        
        # Create trainer
        trainer = RLHFTrainer(config)
        
        # Run training
        output_dir = tmp_path / "rlhf_test"
        result = trainer.train(
            feedbacks=sample_feedbacks,
            output_dir=output_dir,
        )
        
        # Verify outputs
        assert result.feedback_count == len(sample_feedbacks)
        assert result.training_steps == config.max_steps
        assert len(result.reward_history) > 0
        
        # Verify saved models
        policy_path = Path(result.model_path)
        assert policy_path.exists()
        assert (policy_path / "config.json").exists()
        
        # Verify reward model
        if result.reward_model_path:
            reward_model_path = Path(result.reward_model_path)
            assert reward_model_path.exists()
    
    def test_training_with_preloaded_reward_model(
        self,
        tokenizer,
        sample_feedbacks,
        tmp_path,
    ):
        """Test training with pre-trained reward model."""
        # First, train a reward model
        config = RewardModelConfig(
            base_model="gpt2",
            use_lora=True,
            batch_size=2,
            num_epochs=1,
            device="cpu",
        )
        
        train_dataset, val_dataset = prepare_feedback_dataset(
            sample_feedbacks,
            tokenizer,
            max_length=128,
            train_split=0.8,
        )
        
        reward_model = train_reward_model(
            train_dataset,
            val_dataset,
            config,
            tokenizer,
        )
        
        # Save reward model
        reward_model_path = tmp_path / "pretrained_reward_model"
        reward_model.save_pretrained(reward_model_path)
        
        # Now train policy with pre-trained reward model
        training_config = TrainingConfig(
            base_model="gpt2",
            use_lora=True,
            max_steps=5,
            batch_size=2,
            train_reward_model_first=False,
            device="cpu",
        )
        
        trainer = RLHFTrainer(training_config)
        
        output_dir = tmp_path / "policy_with_pretrained_reward"
        result = trainer.train(
            feedbacks=sample_feedbacks,
            output_dir=output_dir,
            reward_model_path=reward_model_path,
        )
        
        # Should not have trained new reward model
        assert result.reward_model_path is None
        assert result.model_path is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
