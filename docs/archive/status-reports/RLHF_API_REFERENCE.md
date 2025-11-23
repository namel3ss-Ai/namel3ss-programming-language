# RLHF API Quick Reference

## Basic Usage

### 1. DPO Training (Simplest)

```python
from namel3ss.ml.rlhf import RLHFConfig, RLHFAlgorithm, RLHFJobRunner

config = RLHFConfig(
    job_name="my_dpo_job",
    algorithm=RLHFAlgorithm.DPO,
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="HuggingFaceH4/ultrafeedback_binarized",
    output_dir="./outputs/dpo",
)

runner = RLHFJobRunner(config)
result = runner.run()
print(f"Loss: {result.final_loss}")
```

### 2. DPO with LoRA (Efficient)

```python
from namel3ss.ml.rlhf import (
    RLHFConfig, RLHFAlgorithm, PEFTConfig, PEFTMethod, RLHFJobRunner
)

config = RLHFConfig(
    job_name="dpo_lora",
    algorithm=RLHFAlgorithm.DPO,
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="HuggingFaceH4/ultrafeedback_binarized",
    output_dir="./outputs/dpo_lora",
    peft=PEFTConfig(
        method=PEFTMethod.LORA,
        r=64,
        alpha=16,
    ),
)

result = RLHFJobRunner(config).run()
```

### 3. DPO with QLoRA (Memory Efficient)

```python
config = RLHFConfig(
    job_name="dpo_qlora",
    algorithm=RLHFAlgorithm.DPO,
    base_model="meta-llama/Llama-2-13b-hf",  # Larger model!
    dataset_path="HuggingFaceH4/ultrafeedback_binarized",
    output_dir="./outputs/dpo_qlora",
    peft=PEFTConfig(
        method=PEFTMethod.QLORA,
        r=64,
        alpha=16,
        quantization_bits=4,  # 4-bit quantization
    ),
    bf16=True,
)

result = RLHFJobRunner(config).run()
```

### 4. PPO Training (with Reward Model)

```python
config = RLHFConfig(
    job_name="ppo_training",
    algorithm=RLHFAlgorithm.PPO,
    base_model="meta-llama/Llama-2-7b-hf",
    reward_model="my-org/llama2-reward-model",  # Required for PPO
    dataset_path="my_prompts_dataset",
    output_dir="./outputs/ppo",
)

result = RLHFJobRunner(config).run()
```

### 5. ORPO Training (Reference-Free)

```python
config = RLHFConfig(
    job_name="orpo_training",
    algorithm=RLHFAlgorithm.ORPO,
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="HuggingFaceH4/ultrafeedback_binarized",
    output_dir="./outputs/orpo",
)

result = RLHFJobRunner(config).run()
```

### 6. KTO Training (Binary Feedback)

```python
config = RLHFConfig(
    job_name="kto_training",
    algorithm=RLHFAlgorithm.KTO,
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="my_feedback_dataset",  # Must have desirable/undesirable labels
    output_dir="./outputs/kto",
)

result = RLHFJobRunner(config).run()
```

## Advanced Configuration

### Algorithm-Specific Parameters

```python
from namel3ss.ml.rlhf import DPOConfig, PPOConfig, ORPOConfig, KTOConfig

# DPO configuration
dpo_config = DPOConfig(
    beta=0.1,                    # DPO beta parameter
    label_smoothing=0.0,         # Label smoothing
    loss_type="sigmoid",         # "sigmoid", "hinge", or "ipo"
)

# PPO configuration
ppo_config = PPOConfig(
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,
    gamma=1.0,
    lam=0.95,
    cliprange=0.2,
    init_kl_coef=0.2,
    target_kl=6.0,
)

# ORPO configuration
orpo_config = ORPOConfig(
    alpha=1.0,
    beta=0.1,
)

# KTO configuration
kto_config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

# Use in main config
config = RLHFConfig(
    # ... other params ...
    dpo_config=dpo_config,  # or ppo_config, orpo_config, kto_config
)
```

### Experiment Tracking

```python
from namel3ss.ml.rlhf import LoggingConfig, ExperimentTracker

logging_config = LoggingConfig(
    tracker=ExperimentTracker.WANDB,  # or MLFLOW, TENSORBOARD
    project="my-rlhf-project",
    run_name="dpo-llama2-7b",
    tags=["dpo", "llama2", "helpful"],
    log_interval=10,
    save_interval=500,
    eval_interval=100,
    wandb_api_key="your-wandb-key",  # or set WANDB_API_KEY env var
)

config = RLHFConfig(
    # ... other params ...
    logging=logging_config,
)
```

### Safety Configuration

```python
from namel3ss.ml.rlhf import SafetyConfig

safety_config = SafetyConfig(
    enable_safety_filter=True,
    toxicity_threshold=0.7,
    safety_model="unitary/toxic-bert",
    enable_content_filter=True,
    max_generation_length=512,
    reject_unsafe_samples=True,
)

config = RLHFConfig(
    # ... other params ...
    safety=safety_config,
)
```

### Hub Publishing

```python
config = RLHFConfig(
    # ... other params ...
    push_to_hub=True,
    hub_model_id="your-username/llama2-7b-dpo-helpful",
)
```

## Dataset Loading

### Load Preference Dataset Manually

```python
from namel3ss.ml.rlhf import PreferenceDataset

dataset = PreferenceDataset(
    path="HuggingFaceH4/ultrafeedback_binarized",
    split="train",
    prompt_col="prompt",
    chosen_col="chosen",
    rejected_col="rejected",
    max_samples=10000,  # Limit dataset size
)

# Access samples
sample = dataset[0]
print(sample.prompt)
print(sample.chosen)
print(sample.rejected)

# Train/test split
train_ds, test_ds = dataset.train_test_split(test_size=0.1)
```

### Load Feedback Dataset Manually

```python
from namel3ss.ml.rlhf import FeedbackDataset

dataset = FeedbackDataset(
    path="my_feedback_data.parquet",
    split="train",
    prompt_col="prompt",
    response_col="response",
    score_col="score",  # For reward modeling
    # OR
    desirable_col="label",  # For KTO (True/False)
)

sample = dataset[0]
print(sample.score)  # or sample.is_desirable
```

## Error Handling

```python
from namel3ss.ml.rlhf import (
    RLHFConfigurationError,
    RLHFTrainingError,
    RLHFDatasetError,
    RLHFModelError,
)

try:
    runner = RLHFJobRunner(config)
    result = runner.run()
except RLHFConfigurationError as e:
    print(f"Configuration error: {e.format()}")
    print(f"Error code: {e.code}")
    print(f"Context: {e.context}")
except RLHFTrainingError as e:
    print(f"Training failed: {e.format()}")
except RLHFDatasetError as e:
    print(f"Dataset loading failed: {e.format()}")
except RLHFModelError as e:
    print(f"Model loading failed: {e.format()}")
```

## Trainer-Specific Usage

```python
from namel3ss.ml.rlhf import get_trainer_class, RLHFAlgorithm

# Get specific trainer class
DPOTrainer = get_trainer_class(RLHFAlgorithm.DPO)
PPOTrainer = get_trainer_class(RLHFAlgorithm.PPO)
ORPOTrainer = get_trainer_class(RLHFAlgorithm.ORPO)

# Use directly (advanced)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

trainer = DPOTrainer(config=config, model=model, tokenizer=tokenizer)
result = trainer.train()
```

## Result Inspection

```python
result = runner.run()

# Check status
print(f"Status: {result.status}")  # "completed", "failed", "stopped"

# Training metrics
print(f"Final loss: {result.final_loss}")
print(f"Best loss: {result.best_loss}")
print(f"Total steps: {result.total_steps}")

# Paths
print(f"Checkpoint: {result.final_checkpoint_path}")
print(f"Output dir: {result.output_dir}")

# Resource usage
print(f"Duration: {result.duration_seconds}s")
print(f"Peak GPU memory: {result.peak_gpu_memory_gb} GB")

# Algorithm-specific metrics
print(f"Metrics: {result.metrics}")

# Serialize
result_dict = result.to_dict()
```

## Environment Variables

```bash
# Experiment tracking
export WANDB_API_KEY="your-wandb-key"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Storage
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export GCS_PROJECT_ID="your-gcs-project"

# Model cache
export HF_HOME="/path/to/cache"
export TRANSFORMERS_CACHE="/path/to/cache"

# Distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
```

## DeepSpeed Configuration

```python
config = RLHFConfig(
    # ... other params ...
    deepspeed_config="./configs/deepspeed_config.json",
)
```

Example `deepspeed_config.json`:
```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

## Common Patterns

### Small Model, Fast Iteration
```python
config = RLHFConfig(
    algorithm=RLHFAlgorithm.DPO,
    base_model="gpt2",  # Small model
    dataset_path="HuggingFaceH4/ultrafeedback_binarized",
    output_dir="./outputs/test",
    max_steps=100,  # Quick test
    bf16=True,
)
```

### Large Model, Production Training
```python
config = RLHFConfig(
    algorithm=RLHFAlgorithm.DPO,
    base_model="meta-llama/Llama-2-70b-hf",
    dataset_path="s3://my-bucket/preference-data",
    output_dir="s3://my-bucket/models/llama2-70b-dpo",
    peft=PEFTConfig(method=PEFTMethod.QLORA, r=64, quantization_bits=4),
    max_steps=50000,
    gradient_accumulation_steps=16,
    deepspeed_config="./configs/deepspeed_zero3.json",
    push_to_hub=True,
    hub_model_id="my-org/llama2-70b-dpo-helpful",
)
```

### Multi-GPU Training
```bash
# Use accelerate
accelerate launch train.py

# Or torchrun
torchrun --nproc_per_node=8 train.py
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""Complete RLHF training script."""

from namel3ss.ml.rlhf import (
    RLHFConfig,
    RLHFAlgorithm,
    PEFTConfig,
    PEFTMethod,
    DPOConfig,
    LoggingConfig,
    ExperimentTracker,
    RLHFJobRunner,
)

def main():
    config = RLHFConfig(
        job_name="production_dpo_llama2",
        algorithm=RLHFAlgorithm.DPO,
        base_model="meta-llama/Llama-2-7b-hf",
        dataset_path="HuggingFaceH4/ultrafeedback_binarized",
        output_dir="./outputs/dpo_llama2",
        peft=PEFTConfig(
            method=PEFTMethod.LORA,
            r=64,
            alpha=16,
            dropout=0.1,
        ),
        dpo_config=DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
        ),
        learning_rate=5e-5,
        max_steps=10000,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        logging=LoggingConfig(
            tracker=ExperimentTracker.WANDB,
            project="rlhf-production",
            run_name="dpo-llama2-7b",
        ),
        push_to_hub=True,
        hub_model_id="my-org/llama2-7b-dpo",
    )
    
    runner = RLHFJobRunner(config)
    result = runner.run()
    
    print(f"✅ Training complete!")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Model: {result.final_checkpoint_path}")

if __name__ == "__main__":
    main()
```
