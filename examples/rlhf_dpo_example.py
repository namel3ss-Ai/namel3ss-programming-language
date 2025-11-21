"""
Example RLHF Training Script - Demonstrates the complete RLHF API.

This script shows how to use the Namel3ss RLHF subsystem to train a model
using Direct Preference Optimization (DPO) with LoRA efficient fine-tuning.

Usage:
    python examples/rlhf_dpo_example.py
"""

import logging
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Run DPO training example."""
    
    logger.info("=" * 80)
    logger.info("Namel3ss RLHF Training Example - DPO with LoRA")
    logger.info("=" * 80)
    
    # Configure PEFT (LoRA)
    peft_config = PEFTConfig(
        method=PEFTMethod.LORA,
        r=64,
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    # Configure DPO algorithm
    dpo_config = DPOConfig(
        beta=0.1,
        label_smoothing=0.0,
        loss_type="sigmoid",
    )
    
    # Configure logging
    logging_config = LoggingConfig(
        tracker=ExperimentTracker.WANDB,
        project="namel3ss-rlhf-example",
        run_name="dpo-llama2-7b",
        log_interval=10,
        save_interval=500,
        eval_interval=100,
    )
    
    # Create main RLHF configuration
    config = RLHFConfig(
        job_name="dpo_llama2_helpful_assistant",
        algorithm=RLHFAlgorithm.DPO,
        base_model="meta-llama/Llama-2-7b-hf",
        dataset_path="HuggingFaceH4/ultrafeedback_binarized",
        output_dir="./outputs/dpo_llama2",
        
        # PEFT configuration
        peft=peft_config,
        
        # Algorithm-specific configuration
        dpo_config=dpo_config,
        
        # Training hyperparameters
        learning_rate=5e-5,
        max_steps=10000,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_grad_norm=1.0,
        
        # Compute configuration
        bf16=True,  # Use bfloat16 for efficient training
        gradient_checkpointing=True,
        
        # Logging configuration
        logging=logging_config,
        
        # Model registry
        push_to_hub=True,
        hub_model_id="your-username/llama2-7b-dpo-helpful",
        
        # Metadata
        metadata={
            "description": "Llama 2 7B fine-tuned with DPO for helpful assistant",
            "dataset": "ultrafeedback_binarized",
            "author": "Namel3ss Team",
        }
    )
    
    logger.info("Configuration:")
    logger.info(f"  Job Name: {config.job_name}")
    logger.info(f"  Algorithm: {config.algorithm.value}")
    logger.info(f"  Base Model: {config.base_model}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  PEFT: {config.peft.method.value} (r={config.peft.r})")
    logger.info(f"  DPO Beta: {dpo_config.beta}")
    logger.info("")
    
    # Create and run training job
    logger.info("Initializing RLHF job runner...")
    runner = RLHFJobRunner(config)
    
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    try:
        result = runner.run()
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("")
        logger.info("Results:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Duration: {result.duration_seconds:.2f} seconds")
        logger.info(f"  Final Loss: {result.final_loss:.4f}")
        logger.info(f"  Total Steps: {result.total_steps}")
        logger.info(f"  Checkpoint: {result.final_checkpoint_path}")
        
        if result.peak_gpu_memory_gb:
            logger.info(f"  Peak GPU Memory: {result.peak_gpu_memory_gb:.2f} GB")
        
        logger.info("")
        logger.info("Algorithm Metrics:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80)
        
        return result
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
