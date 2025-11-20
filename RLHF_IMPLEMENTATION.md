# RLHF Training Implementation

Complete implementation of Reinforcement Learning from Human Feedback (RLHF) using HuggingFace TRL library with PPO (Proximal Policy Optimization).

## Overview

This implementation enables N3 agents to learn from human feedback and continuously improve their responses through policy optimization.

### Architecture

```
Feedback Collection → Reward Model Training → PPO Training → Trained Policy
       ↓                      ↓                     ↓              ↓
   Database           Score Predictions       Policy Updates    Model Checkpoint
```

### Key Components

1. **Dataset Preparation** (`n3_server/rlhf/dataset.py`)
   - `FeedbackDataset`: PyTorch dataset for reward model training
   - `PPODataset`: Dataset for PPO training (prompts only)
   - Score normalization and train/val splitting

2. **Reward Model** (`n3_server/rlhf/reward_model.py`)
   - Transformer base model + reward prediction head
   - LoRA fine-tuning support for efficiency
   - MSE loss training on human feedback scores

3. **PPO Trainer** (`n3_server/rlhf/trainer.py`)
   - Complete RLHF pipeline using TRL library
   - Policy model with value head
   - KL divergence constraint to prevent over-optimization

4. **API Integration** (`n3_server/api/policies.py`)
   - POST `/api/policies/train_policy/{agent_id}`: Train policy with RLHF
   - POST `/api/policies/feedback/{agent_id}`: Submit feedback
   - GET `/api/policies/{agent_id}`: List trained policies

## Installation

Dependencies are already in `requirements.txt`:

```bash
# RLHF dependencies
trl>=0.7.0              # TRL library for PPO/RLHF
transformers>=4.35.0    # HuggingFace transformers
torch>=2.1.0            # PyTorch
peft>=0.7.0             # Parameter-Efficient Fine-Tuning (LoRA)
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Feedback

Submit human feedback for agent responses:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/policies/feedback/my_agent",
    json={
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "score": 0.9,  # Score between 0 (bad) and 1 (excellent)
        "runId": "execution_123",
        "notes": "Accurate and concise answer"
    }
)
```

**Minimum 10 feedback samples required** for training.

### 2. Estimate Training (Dry Run)

Get training estimates without actually training:

```python
response = requests.post(
    "http://localhost:8000/api/policies/train_policy/my_agent",
    json={
        "dryRun": True,
        "maxSteps": 1000,
        "learningRate": 1e-5
    }
)

estimate = response.json()
print(f"Feedback count: {estimate['feedbackCount']}")
print(f"Estimated time: {estimate['estimatedTimeMinutes']} minutes")
print(f"Score mean: {estimate['scoreMean']}")
```

### 3. Train Policy

Run actual RLHF training:

```python
response = requests.post(
    "http://localhost:8000/api/policies/train_policy/my_agent",
    json={
        "dryRun": False,
        "maxSteps": 1000,
        "learningRate": 1e-5
    }
)

result = response.json()
print(f"Policy ID: {result['policyId']}")
print(f"Version: {result['version']}")
print(f"Model path: {result['modelPath']}")
print(f"Training steps: {result['trainingSteps']}")
print(f"Final reward: {result['rewardMean']} ± {result['rewardStd']}")
```

### 4. List Trained Policies

Get all trained policies for an agent:

```python
response = requests.get("http://localhost:8000/api/policies/my_agent")
policies = response.json()

for policy in policies:
    print(f"{policy['version']}: {policy['metrics']['rewardMean']:.3f}")
```

## Training Configuration

### Default Configuration

```python
from n3_server.rlhf import TrainingConfig

config = TrainingConfig(
    # Model settings
    base_model="gpt2",           # Base LLM to train
    use_lora=True,               # Use LoRA for efficient fine-tuning
    lora_r=8,                    # LoRA rank
    lora_alpha=32,               # LoRA alpha
    lora_dropout=0.1,            # LoRA dropout
    
    # Training hyperparameters
    learning_rate=1e-5,          # Learning rate
    batch_size=8,                # Training batch size
    mini_batch_size=4,           # PPO mini-batch size
    ppo_epochs=4,                # PPO update epochs per batch
    max_steps=1000,              # Total training steps
    
    # PPO-specific
    init_kl_coef=0.2,            # KL divergence coefficient
    target_kl=6.0,               # Target KL divergence
    gamma=1.0,                   # Discount factor
    lam=0.95,                    # GAE lambda
    cliprange=0.2,               # PPO clip range
    vf_coef=0.1,                 # Value function coefficient
    
    # Generation settings
    max_new_tokens=128,          # Max tokens to generate
    temperature=1.0,             # Sampling temperature
    top_k=50,                    # Top-K sampling
    top_p=0.95,                  # Nucleus sampling
)
```

### Recommended Settings by Use Case

#### Quick Prototyping
```python
config = TrainingConfig(
    base_model="gpt2",
    max_steps=100,
    batch_size=4,
    learning_rate=1e-4,
)
```

#### Production Training
```python
config = TrainingConfig(
    base_model="gpt2-medium",  # or gpt2-large
    max_steps=5000,
    batch_size=16,
    learning_rate=1e-5,
    use_lora=True,
    lora_r=16,
)
```

#### Memory-Constrained
```python
config = TrainingConfig(
    base_model="gpt2",
    use_lora=True,
    lora_r=4,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
)
```

## Training Pipeline Details

### Step 1: Reward Model Training

The reward model learns to predict human preference scores:

```python
from n3_server.rlhf import RewardModelConfig, train_reward_model

config = RewardModelConfig(
    base_model="gpt2",
    use_lora=True,
    batch_size=8,
    num_epochs=3,
    learning_rate=1e-5,
)

reward_model = train_reward_model(
    train_dataset,
    val_dataset,
    config,
    tokenizer,
)

# Save reward model
reward_model.save_pretrained("models/my_agent/reward_model")
```

### Step 2: PPO Training

The policy is trained using PPO to maximize the reward model's predictions:

```python
from n3_server.rlhf import RLHFTrainer

trainer = RLHFTrainer(training_config)

result = trainer.train(
    feedbacks=feedback_list,
    output_dir="models/my_agent/v1",
)

print(f"Final reward: {result.final_reward_mean}")
print(f"Training steps: {result.training_steps}")
```

### Step 3: Policy Deployment

Load the trained policy for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/my_agent/v1/policy")
tokenizer = AutoTokenizer.from_pretrained("models/my_agent/v1/policy")

# Generate response
prompt = "What is quantum computing?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance Optimization

### GPU Training

Enable CUDA for faster training:

```python
config = TrainingConfig(
    device="cuda",  # Use GPU
    batch_size=32,  # Larger batch size with GPU
    mixed_precision="fp16",  # Use mixed precision
)
```

### LoRA Fine-Tuning

LoRA reduces memory usage by 3-10x:

```python
config = TrainingConfig(
    use_lora=True,
    lora_r=8,        # Lower rank = less memory, faster training
    lora_alpha=32,   # Alpha/r ratio controls adaptation strength
    lora_dropout=0.1,
)
```

### Memory Usage Estimates

| Model | Full Fine-Tuning | LoRA (r=8) |
|-------|------------------|------------|
| GPT-2 (124M) | ~8GB | ~2GB |
| GPT-2 Medium (355M) | ~16GB | ~4GB |
| GPT-2 Large (774M) | ~32GB | ~8GB |

## Monitoring Training

### Training Metrics

The training process tracks:

- **Reward**: Average reward from reward model
- **KL Divergence**: Distance from initial policy (prevents over-optimization)
- **Loss**: PPO objective loss
- **Value Error**: Value function prediction error

### Visualization

Training results include full history:

```python
import matplotlib.pyplot as plt

# Load results
from n3_server.rlhf import TrainingResult

result = TrainingResult.load("models/my_agent/v1/training_results.json")

# Plot reward over time
plt.plot(result.reward_history)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.show()
```

## Advanced Usage

### Custom Base Models

Use different base models:

```python
config = TrainingConfig(
    base_model="facebook/opt-125m",  # OPT model
    # or
    base_model="EleutherAI/gpt-neo-125M",  # GPT-Neo
    # or
    base_model="meta-llama/Llama-2-7b-hf",  # Llama 2
)
```

### Pre-trained Reward Model

Reuse reward model across training runs:

```python
trainer = RLHFTrainer(config)

result = trainer.train(
    feedbacks=feedbacks,
    output_dir="models/my_agent/v2",
    reward_model_path="models/my_agent/v1/reward_model",  # Reuse
)
```

### Custom Reward Functions

Implement custom reward computation:

```python
from n3_server.rlhf import RewardModel

class CustomRewardModel(RewardModel):
    def forward(self, input_ids, attention_mask):
        # Custom reward logic
        base_rewards = super().forward(input_ids, attention_mask)
        
        # Apply custom transformations
        rewards = base_rewards * 2.0 - 1.0  # Scale to [-1, 1]
        
        return rewards
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Solution**: Reduce batch size or use LoRA

```python
config = TrainingConfig(
    batch_size=4,
    mini_batch_size=2,
    use_lora=True,
    lora_r=4,
)
```

#### 2. Training Instability

**Solution**: Lower learning rate or increase KL coefficient

```python
config = TrainingConfig(
    learning_rate=1e-6,  # Lower learning rate
    init_kl_coef=0.5,    # Stronger KL constraint
)
```

#### 3. Low Reward

**Solution**: Check feedback quality and increase training steps

```python
# Review feedback scores
response = requests.get("http://localhost:8000/api/policies/my_agent")
policies = response.json()

# Ensure feedback is high-quality
# - Scores should vary (not all 0.5)
# - Good examples should have scores > 0.7
# - Bad examples should have scores < 0.3

config = TrainingConfig(
    max_steps=2000,  # More training steps
)
```

#### 4. Reward Model Overfitting

**Solution**: Use more feedback or regularization

```python
config = RewardModelConfig(
    num_epochs=2,  # Fewer epochs
    lora_dropout=0.2,  # Higher dropout
)
```

## Testing

Run tests with pytest:

```bash
# Unit tests for RLHF components
pytest test_rlhf_training.py -v

# API integration tests
pytest test_rlhf_api.py -v

# Run only fast tests (skip slow training tests)
pytest test_rlhf_training.py -v -m "not slow"
```

## API Reference

### Training Configuration

```python
@dataclass
class TrainingConfig:
    base_model: str = "gpt2"
    use_lora: bool = True
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_steps: int = 1000
    # ... see full config in trainer.py
```

### Training Result

```python
@dataclass
class TrainingResult:
    model_path: str
    reward_model_path: Optional[str]
    feedback_count: int
    final_reward_mean: float
    final_reward_std: float
    training_steps: int
    reward_history: List[float]
    kl_history: List[float]
    loss_history: List[float]
```

### API Endpoints

#### POST /api/policies/feedback/{agent_id}

Submit feedback for an agent.

**Request Body**:
```json
{
  "prompt": "string",
  "response": "string",
  "score": 0.0-1.0,
  "runId": "string",
  "notes": "string (optional)"
}
```

**Response**:
```json
{
  "id": "feedback_id",
  "agentId": "agent_id",
  "score": 0.9,
  "createdAt": "2024-01-01T00:00:00Z"
}
```

#### POST /api/policies/train_policy/{agent_id}

Train a new policy using RLHF.

**Request Body**:
```json
{
  "dryRun": false,
  "maxSteps": 1000,
  "learningRate": 1e-5
}
```

**Response (Training)**:
```json
{
  "status": "trained",
  "policyId": "policy_id",
  "version": "v15",
  "modelPath": "models/agent_id/v15/policy",
  "rewardModelPath": "models/agent_id/v15/reward_model",
  "feedbackCount": 15,
  "trainingSteps": 1000,
  "rewardMean": 0.82,
  "rewardStd": 0.15
}
```

**Response (Dry Run)**:
```json
{
  "status": "dry_run",
  "feedbackCount": 15,
  "estimatedSteps": 1000,
  "estimatedTimeMinutes": 45,
  "scoreMean": 0.75,
  "scoreStd": 0.18,
  "modelConfig": {
    "base_model": "gpt2",
    "use_lora": true,
    "learning_rate": 1e-5
  }
}
```

#### GET /api/policies/{agent_id}

List all trained policies for an agent.

**Response**:
```json
[
  {
    "id": "policy_id",
    "agentId": "agent_id",
    "version": "v15",
    "createdAt": "2024-01-01T00:00:00Z",
    "metrics": {
      "feedbackCount": 15,
      "rewardMean": 0.82,
      "rewardStd": 0.15
    }
  }
]
```

## Best Practices

### 1. Feedback Collection

- **Quality over Quantity**: 50 high-quality feedback samples beats 500 low-quality ones
- **Diverse Prompts**: Collect feedback across different prompt types
- **Score Distribution**: Aim for varied scores (not all 0.5)
  - Excellent responses: 0.8-1.0
  - Good responses: 0.6-0.8
  - Mediocre responses: 0.4-0.6
  - Poor responses: 0.0-0.4

### 2. Incremental Training

Train incrementally as feedback accumulates:

```python
# Initial training with 15 samples
# Version v15

# Add 10 more samples → retrain
# Version v25

# Add 20 more samples → retrain
# Version v45
```

### 3. A/B Testing

Compare policy versions:

```python
# Deploy v15 to 50% of users
# Deploy v25 to 50% of users
# Collect feedback on both
# Train v35 on combined feedback
```

### 4. Monitoring

Track metrics over versions:

```python
versions = ["v15", "v25", "v35"]
rewards = [0.75, 0.82, 0.88]  # Increasing = good!
stds = [0.20, 0.15, 0.12]     # Decreasing = more consistent
```

## Examples

See test files for complete examples:

- `test_rlhf_training.py`: Unit tests for training components
- `test_rlhf_api.py`: Integration tests for API endpoints

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [RLHF Blog Post](https://huggingface.co/blog/rlhf)

## License

Same as main N3 project license.
