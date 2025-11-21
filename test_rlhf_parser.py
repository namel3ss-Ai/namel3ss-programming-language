"""Test RLHF job parsing and integration."""

from namel3ss.parser import Parser
from namel3ss.ast import RLHFJob


def test_parse_basic_rlhf_job():
    """Test basic RLHF job parsing."""
    source = '''
app "RLHF Trainer".

train rlhf "helpful_assistant":
    base_model: "meta-llama/Meta-Llama-3-8B"
    algorithm: "dpo"
    dataset: "s3://bucket/preference-data"
    
page "Home" at "/":
    show text "ok"
'''
    
    app = Parser(source).parse_app()
    
    assert len(app.rlhf_jobs) == 1
    job = app.rlhf_jobs[0]
    assert isinstance(job, RLHFJob)
    assert job.name == "helpful_assistant"
    assert job.base_model == "meta-llama/Meta-Llama-3-8B"
    assert job.algorithm == "dpo"
    assert job.dataset == "s3://bucket/preference-data"


def test_parse_rlhf_job_with_peft():
    """Test RLHF job with PEFT configuration."""
    source = '''
app "RLHF Trainer".

train rlhf "aligned_model":
    base_model: "mistralai/Mistral-7B-v0.1"
    algorithm: "dpo"
    dataset: "anthropic/hh-rlhf"
    
    peft:
        method: "qlora"
        r: 64
        lora_alpha: 16
        target_modules: ["q_proj", "v_proj", "k_proj"]
        quantization: "nf4"
    
page "Home" at "/":
    show text "ok"
'''
    
    app = Parser(source).parse_app()
    
    assert len(app.rlhf_jobs) == 1
    job = app.rlhf_jobs[0]
    assert job.name == "aligned_model"
    assert job.peft is not None
    assert job.peft.method == "qlora"
    assert job.peft.r == 64
    assert job.peft.lora_alpha == 16
    assert job.peft.target_modules == ["q_proj", "v_proj", "k_proj"]
    assert job.peft.quantization == "nf4"


def test_parse_rlhf_job_with_all_configs():
    """Test RLHF job with complete configuration."""
    source = '''
app "RLHF Trainer".

train rlhf "production_model":
    base_model: "meta-llama/Meta-Llama-3-8B"
    algorithm: "dpo"
    dataset: "s3://bucket/preference-data"
    reward_model: "reward-model-v1"
    
    peft:
        method: "lora"
        r: 32
        lora_alpha: 16
    
    algorithm_config:
        beta: 0.1
        loss_type: "sigmoid"
        label_smoothing: 0.0
    
    hyperparameters:
        learning_rate: 0.00001
        batch_size: 64
        warmup_ratio: 0.1
    
    compute:
        backend: "vertex-ai"
        num_gpus: 4
        gpu_type: "a100"
        strategy: "deepspeed_zero3"
        mixed_precision: "bf16"
        gradient_checkpointing: true
    
    logging:
        tracker: "wandb"
        project: "llama3-alignment"
        run_name: "dpo-experiment-1"
        log_frequency: 10
    
    safety:
        enable_content_filter: true
        toxicity_threshold: 0.8
        pii_detection: true
    
    max_steps: 20000
    eval_steps: 500
    save_steps: 1000
    
page "Home" at "/":
    show text "ok"
'''
    
    app = Parser(source).parse_app()
    
    assert len(app.rlhf_jobs) == 1
    job = app.rlhf_jobs[0]
    
    # Basic fields
    assert job.name == "production_model"
    assert job.base_model == "meta-llama/Meta-Llama-3-8B"
    assert job.algorithm == "dpo"
    assert job.dataset == "s3://bucket/preference-data"
    assert job.reward_model == "reward-model-v1"
    
    # PEFT config
    assert job.peft is not None
    assert job.peft.method == "lora"
    assert job.peft.r == 32
    
    # Algorithm config
    assert job.algorithm_config is not None
    assert job.algorithm_config.beta == 0.1
    assert job.algorithm_config.loss_type == "sigmoid"
    
    # Hyperparameters
    assert job.hyperparameters["learning_rate"] == 0.00001
    assert job.hyperparameters["batch_size"] == 64
    
    # Compute
    assert job.compute.backend == "vertex-ai"
    assert job.compute.num_gpus == 4
    assert job.compute.gpu_type == "a100"
    assert job.compute.strategy == "deepspeed_zero3"
    assert job.compute.mixed_precision == "bf16"
    assert job.compute.gradient_checkpointing is True
    
    # Logging
    assert job.logging.tracker == "wandb"
    assert job.logging.project == "llama3-alignment"
    assert job.logging.run_name == "dpo-experiment-1"
    
    # Safety
    assert job.safety is not None
    assert job.safety.enable_content_filter is True
    assert job.safety.toxicity_threshold == 0.8
    
    # Training params
    assert job.max_steps == 20000
    assert job.eval_steps == 500
    assert job.save_steps == 1000


def test_parse_multiple_rlhf_algorithms():
    """Test parsing different RLHF algorithms."""
    algorithms = ["ppo", "dpo", "ipo", "orpo", "kto", "sft", "reward"]
    
    for algo in algorithms:
        source = f'''
app "Test".

train rlhf "job_{algo}":
    base_model: "llama-3-8b"
    algorithm: "{algo}"
    dataset: "data.json"

page "Home" at "/":
    show text "ok"
'''
        app = Parser(source).parse_app()
        assert len(app.rlhf_jobs) == 1
        assert app.rlhf_jobs[0].algorithm == algo


if __name__ == "__main__":
    test_parse_basic_rlhf_job()
    print("✓ Basic RLHF job parsing works")
    
    test_parse_rlhf_job_with_peft()
    print("✓ RLHF job with PEFT works")
    
    test_parse_rlhf_job_with_all_configs()
    print("✓ Complete RLHF job configuration works")
    
    test_parse_multiple_rlhf_algorithms()
    print("✓ All RLHF algorithms parse correctly")
    
    print("\n✅ All RLHF parser tests passed!")
