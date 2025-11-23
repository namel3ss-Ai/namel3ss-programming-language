# RLHF Training Quick Reference

## Quick Start

### 1. Collect Feedback (Minimum 10 samples)

```bash
curl -X POST http://localhost:8000/api/policies/feedback/my_agent \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "response": "Paris is the capital of France.",
    "score": 0.9,
    "runId": "run_123",
    "notes": "Accurate answer"
  }'
```

### 2. Estimate Training (Dry Run)

```bash
curl -X POST http://localhost:8000/api/policies/train_policy/my_agent \
  -H "Content-Type: application/json" \
  -d '{
    "dryRun": true,
    "maxSteps": 1000,
    "learningRate": 1e-5
  }'
```

### 3. Train Policy

```bash
curl -X POST http://localhost:8000/api/policies/train_policy/my_agent \
  -H "Content-Type: application/json" \
  -d '{
    "dryRun": false,
    "maxSteps": 1000,
    "learningRate": 1e-5
  }'
```

### 4. List Policies

```bash
curl http://localhost:8000/api/policies/my_agent
```

## Python API

### Train with Custom Configuration

```python
from n3_server.rlhf import RLHFTrainer, TrainingConfig
from pathlib import Path

# Create configuration
config = TrainingConfig(
    base_model="gpt2",
    use_lora=True,
    learning_rate=1e-5,
    max_steps=1000,
    batch_size=8,
)

# Initialize trainer
trainer = RLHFTrainer(config)

# Prepare feedback data
feedbacks = [
    {
        "prompt": "Question 1",
        "response": "Answer 1",
        "score": 0.9,
        "run_id": "run_1",
    },
    # ... more feedback (min 10)
]

# Train policy
result = trainer.train(
    feedbacks=feedbacks,
    output_dir=Path("models/my_agent/v1"),
)

print(f"Training complete!")
print(f"Model saved to: {result.model_path}")
print(f"Final reward: {result.final_reward_mean:.3f} ± {result.final_reward_std:.3f}")
print(f"Training steps: {result.training_steps}")
```

### Use Trained Policy

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained policy
model = AutoModelForCausalLM.from_pretrained("models/my_agent/v1/policy")
tokenizer = AutoTokenizer.from_pretrained("models/my_agent/v1/policy")

# Generate response
prompt = "What is quantum computing?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Configuration Presets

### Quick Test (5 minutes)
```python
config = TrainingConfig(
    base_model="gpt2",
    max_steps=50,
    batch_size=4,
    learning_rate=1e-4,
)
```

### Production (2-4 hours)
```python
config = TrainingConfig(
    base_model="gpt2-medium",
    max_steps=5000,
    batch_size=16,
    learning_rate=1e-5,
    use_lora=True,
    lora_r=16,
)
```

### Low Memory (8GB GPU)
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

### High Performance (40GB GPU)
```python
config = TrainingConfig(
    base_model="gpt2-large",
    use_lora=False,  # Full fine-tuning
    batch_size=32,
    learning_rate=1e-5,
    device="cuda",
)
```

## Feedback Scoring Guidelines

| Score | Quality | Example |
|-------|---------|---------|
| 0.9-1.0 | Excellent | Perfect answer, comprehensive, accurate |
| 0.7-0.9 | Good | Correct, clear, minor improvements possible |
| 0.5-0.7 | Mediocre | Acceptable but vague or incomplete |
| 0.3-0.5 | Poor | Partially incorrect or very incomplete |
| 0.0-0.3 | Very Poor | Wrong, misleading, or unhelpful |

## Training Metrics

### Reward
- **Meaning**: Average score from reward model
- **Good**: Increasing over time
- **Target**: > 0.7 after training

### KL Divergence
- **Meaning**: Distance from initial policy
- **Good**: < 10.0 (stays close to base model)
- **Bad**: > 20.0 (over-optimized, may produce nonsense)

### Loss
- **Meaning**: PPO objective loss
- **Good**: Decreasing over time
- **Target**: Converges to stable value

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
config.batch_size = 4
config.mini_batch_size = 2

# Use LoRA
config.use_lora = True
config.lora_r = 4
```

### Training Unstable
```python
# Lower learning rate
config.learning_rate = 1e-6

# Increase KL constraint
config.init_kl_coef = 0.5
config.target_kl = 3.0
```

### Low Reward
```python
# More training steps
config.max_steps = 2000

# Check feedback quality
# - Ensure scores vary (not all 0.5)
# - Good examples should have scores > 0.7
# - Bad examples should have scores < 0.3
```

### Reward Model Overfitting
```python
# Use more feedback data (> 50 samples)
# Or reduce reward model epochs
config.reward_model_config.num_epochs = 2
config.reward_model_config.lora_dropout = 0.2
```

## File Structure

```
models/
  my_agent/
    v1/
      policy/                    # Trained policy model
        config.json
        pytorch_model.bin
        tokenizer_config.json
      reward_model/              # Trained reward model
        base_model/
          adapter_config.json
          adapter_model.bin
        reward_head.pt
        config.json
      training_results.json      # Training metrics
```

## API Responses

### Training Success
```json
{
  "status": "trained",
  "policyId": "abc123",
  "version": "v15",
  "modelPath": "models/my_agent/v15/policy",
  "rewardModelPath": "models/my_agent/v15/reward_model",
  "feedbackCount": 15,
  "trainingSteps": 1000,
  "rewardMean": 0.82,
  "rewardStd": 0.15
}
```

### Dry Run
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

### Error: Insufficient Feedback
```json
{
  "detail": "Insufficient feedback: 5 samples (minimum 10 required)"
}
```

## Testing

```bash
# Run all tests
pytest test_rlhf_training.py -v

# Run specific test
pytest test_rlhf_training.py::TestRLHFTrainer::test_estimate_training -v

# Skip slow tests (full training)
pytest test_rlhf_training.py -v -m "not slow"

# Test API endpoints
pytest test_rlhf_api.py -v
```

## Monitoring Commands

### Check Training Progress
```bash
# View training logs
tail -f models/my_agent/v1/training.log

# Load and analyze results
python -c "
from n3_server.rlhf import TrainingResult
result = TrainingResult.load('models/my_agent/v1/training_results.json')
print(f'Reward: {result.final_reward_mean:.3f} ± {result.final_reward_std:.3f}')
print(f'Steps: {result.training_steps}')
"
```

### Compare Versions
```bash
curl http://localhost:8000/api/policies/my_agent | jq '.[] | {version, reward: .metrics.rewardMean}'
```

## Best Practices

1. **Start Small**: Test with `max_steps=50` first
2. **Collect Quality Feedback**: 50 good samples > 500 mediocre ones
3. **Monitor KL**: Keep KL divergence < 10.0
4. **Incremental Training**: Retrain as feedback grows (v15 → v25 → v35)
5. **A/B Test**: Compare policy versions in production
6. **Use LoRA**: Saves 3-10x memory with minimal quality loss

## Environment Variables

```bash
# Use GPU
export CUDA_VISIBLE_DEVICES=0

# Set models directory
export MODELS_DIR=/path/to/models

# Enable debug logging
export LOG_LEVEL=DEBUG
```

## Dependencies

```bash
pip install trl>=0.7.0 transformers>=4.35.0 torch>=2.1.0 peft>=0.7.0
```

## Support

- Full documentation: `RLHF_IMPLEMENTATION.md`
- Test examples: `test_rlhf_training.py`
- API tests: `test_rlhf_api.py`

## License

Same as N3 project license.
