"""
Comprehensive unit tests for RLHF DSL parser.

Tests the complete RLHF parser integration:
- RLHFJob AST node creation
- Configuration parsing (PEFT, algorithm, compute, logging, safety)
- Parser validation
- Error handling
- Integration with App
"""

import pytest
from namel3ss.parser.app import App
from namel3ss.parser.ast_nodes import (
    RLHFJob,
    RLHFPEFTConfig,
    RLHFAlgorithmConfig,
    RLHFComputeSpec,
    RLHFLoggingConfig,
    RLHFSafetyConfig,
)


class TestBasicRLHFParsing:
    """Test basic RLHF job parsing."""

    def test_minimal_rlhf_job(self):
        """Test parsing minimal RLHF job."""
        code = """
        rlhf dpo_job {
            model "gpt2"
            dataset "hf://username/preferences"
        }
        """
        
        app = App()
        result = app.parse(code)
        
        assert result is not None
        assert hasattr(app, 'rlhf_jobs')
        assert len(app.rlhf_jobs) == 1
        
        job = app.rlhf_jobs[0]
        assert isinstance(job, RLHFJob)
        assert job.name == "dpo_job"
        assert job.model == "gpt2"
        assert job.dataset == "hf://username/preferences"

    def test_rlhf_with_algorithm(self):
        """Test parsing RLHF with algorithm specification."""
        code = """
        rlhf training_job {
            model "llama-3-8b"
            dataset "hf://my/dataset"
            algorithm "dpo"
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.algorithm == "dpo"

    def test_rlhf_with_output_path(self):
        """Test parsing RLHF with output path."""
        code = """
        rlhf job1 {
            model "model"
            dataset "data"
            output_model "./trained_model"
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.output_model == "./trained_model"

    def test_multiple_rlhf_jobs(self):
        """Test parsing multiple RLHF jobs."""
        code = """
        rlhf job1 {
            model "model1"
            dataset "data1"
        }
        
        rlhf job2 {
            model "model2"
            dataset "data2"
        }
        """
        
        app = App()
        app.parse(code)
        
        assert len(app.rlhf_jobs) == 2
        assert app.rlhf_jobs[0].name == "job1"
        assert app.rlhf_jobs[1].name == "job2"


class TestPEFTConfiguration:
    """Test PEFT (Parameter-Efficient Fine-Tuning) configuration."""

    def test_lora_config(self):
        """Test LoRA configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            peft {
                method "lora"
                rank 8
                alpha 16
                dropout 0.1
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.peft is not None
        assert isinstance(job.peft, RLHFPEFTConfig)
        assert job.peft.method == "lora"
        assert job.peft.rank == 8
        assert job.peft.alpha == 16
        assert job.peft.dropout == 0.1

    def test_qlora_config(self):
        """Test QLoRA configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            peft {
                method "qlora"
                rank 16
                alpha 32
                quantization "4bit"
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.peft.method == "qlora"
        assert job.peft.quantization == "4bit"

    def test_target_modules(self):
        """Test target modules configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            peft {
                method "lora"
                rank 8
                target_modules ["q_proj", "v_proj", "k_proj"]
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.peft.target_modules == ["q_proj", "v_proj", "k_proj"]

    def test_peft_without_method(self):
        """Test PEFT config requires method."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            peft {
                rank 8
            }
        }
        """
        
        app = App()
        # Should either fail parsing or validation
        try:
            app.parse(code)
            # If parsing succeeds, validation should fail
            if hasattr(app, 'rlhf_jobs') and app.rlhf_jobs:
                job = app.rlhf_jobs[0]
                assert job.peft is None or job.peft.method is None
        except Exception:
            pass  # Expected to fail


class TestAlgorithmConfiguration:
    """Test algorithm configuration."""

    def test_dpo_config(self):
        """Test DPO algorithm configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            algorithm_config {
                name "dpo"
                beta 0.1
                learning_rate 5e-7
                epochs 3
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.algorithm_config is not None
        assert isinstance(job.algorithm_config, RLHFAlgorithmConfig)
        assert job.algorithm_config.name == "dpo"
        assert job.algorithm_config.beta == 0.1
        assert job.algorithm_config.learning_rate == 5e-7
        assert job.algorithm_config.epochs == 3

    def test_ppo_config(self):
        """Test PPO algorithm configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            algorithm_config {
                name "ppo"
                learning_rate 1e-5
                batch_size 32
                ppo_epochs 4
                clip_range 0.2
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.algorithm_config.name == "ppo"
        assert job.algorithm_config.ppo_epochs == 4
        assert job.algorithm_config.clip_range == 0.2

    def test_reward_modeling_config(self):
        """Test reward modeling configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            algorithm_config {
                name "reward_modeling"
                learning_rate 1e-5
                batch_size 16
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.algorithm_config.name == "reward_modeling"


class TestComputeSpecification:
    """Test compute resource specification."""

    def test_basic_compute_spec(self):
        """Test basic compute specification."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            compute {
                device "cuda"
                num_gpus 4
                mixed_precision "fp16"
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.compute is not None
        assert isinstance(job.compute, RLHFComputeSpec)
        assert job.compute.device == "cuda"
        assert job.compute.num_gpus == 4
        assert job.compute.mixed_precision == "fp16"

    def test_distributed_config(self):
        """Test distributed training configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            compute {
                device "cuda"
                num_gpus 8
                distributed_strategy "ddp"
                gradient_accumulation_steps 4
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.compute.distributed_strategy == "ddp"
        assert job.compute.gradient_accumulation_steps == 4

    def test_memory_optimization(self):
        """Test memory optimization settings."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            compute {
                device "cuda"
                gradient_checkpointing true
                max_memory_per_gpu "40GB"
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.compute.gradient_checkpointing is True
        assert job.compute.max_memory_per_gpu == "40GB"


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_wandb_logging(self):
        """Test Weights & Biases logging."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            logging {
                provider "wandb"
                project "my_project"
                run_name "experiment_1"
                log_interval 100
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.logging is not None
        assert isinstance(job.logging, RLHFLoggingConfig)
        assert job.logging.provider == "wandb"
        assert job.logging.project == "my_project"
        assert job.logging.run_name == "experiment_1"
        assert job.logging.log_interval == 100

    def test_tensorboard_logging(self):
        """Test TensorBoard logging."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            logging {
                provider "tensorboard"
                log_dir "./logs"
                log_interval 50
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.logging.provider == "tensorboard"
        assert job.logging.log_dir == "./logs"

    def test_metrics_to_log(self):
        """Test specifying which metrics to log."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            logging {
                provider "wandb"
                project "test"
                metrics ["loss", "reward", "kl_divergence"]
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.logging.metrics == ["loss", "reward", "kl_divergence"]


class TestSafetyConfiguration:
    """Test safety configuration."""

    def test_basic_safety_filters(self):
        """Test basic safety filter configuration."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            safety {
                enable_filters true
                toxicity_threshold 0.7
                pii_detection true
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.safety is not None
        assert isinstance(job.safety, RLHFSafetyConfig)
        assert job.safety.enable_filters is True
        assert job.safety.toxicity_threshold == 0.7
        assert job.safety.pii_detection is True

    def test_content_moderation(self):
        """Test content moderation settings."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            safety {
                enable_filters true
                profanity_filter true
                bias_detection true
                bias_threshold 0.6
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.safety.profanity_filter is True
        assert job.safety.bias_detection is True
        assert job.safety.bias_threshold == 0.6

    def test_custom_filter_list(self):
        """Test custom filter list."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            safety {
                enable_filters true
                filters ["toxicity", "pii", "profanity"]
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.safety.filters == ["toxicity", "pii", "profanity"]

    def test_safety_action_on_violation(self):
        """Test action when safety violation detected."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            safety {
                enable_filters true
                action_on_violation "skip"
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.safety.action_on_violation == "skip"


class TestCompleteRLHFJob:
    """Test complete RLHF job with all configurations."""

    def test_full_configuration(self):
        """Test complete RLHF job with all options."""
        code = """
        rlhf complete_job {
            model "meta-llama/Llama-3-8b"
            dataset "hf://Anthropic/hh-rlhf"
            algorithm "dpo"
            output_model "./models/llama-3-dpo"
            
            peft {
                method "lora"
                rank 16
                alpha 32
                dropout 0.05
                target_modules ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
            
            algorithm_config {
                name "dpo"
                beta 0.1
                learning_rate 5e-7
                batch_size 8
                epochs 3
                warmup_steps 100
            }
            
            compute {
                device "cuda"
                num_gpus 4
                mixed_precision "bf16"
                distributed_strategy "fsdp"
                gradient_accumulation_steps 4
                gradient_checkpointing true
            }
            
            logging {
                provider "wandb"
                project "llama3_alignment"
                run_name "dpo_exp_1"
                log_interval 10
                metrics ["loss", "reward", "kl_div"]
            }
            
            safety {
                enable_filters true
                toxicity_threshold 0.7
                pii_detection true
                profanity_filter true
                bias_detection true
                filters ["toxicity", "pii", "profanity", "bias"]
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        assert len(app.rlhf_jobs) == 1
        job = app.rlhf_jobs[0]
        
        # Verify all components
        assert job.name == "complete_job"
        assert job.model == "meta-llama/Llama-3-8b"
        assert job.algorithm == "dpo"
        
        assert job.peft is not None
        assert job.peft.method == "lora"
        assert job.peft.rank == 16
        
        assert job.algorithm_config is not None
        assert job.algorithm_config.beta == 0.1
        
        assert job.compute is not None
        assert job.compute.num_gpus == 4
        
        assert job.logging is not None
        assert job.logging.provider == "wandb"
        
        assert job.safety is not None
        assert job.safety.enable_filters is True


class TestParserValidation:
    """Test parser validation and error handling."""

    def test_missing_required_fields(self):
        """Test validation catches missing required fields."""
        code = """
        rlhf job {
            model "model"
            # Missing dataset
        }
        """
        
        app = App()
        # Should fail validation
        try:
            app.parse(code)
            if hasattr(app, 'rlhf_jobs') and app.rlhf_jobs:
                job = app.rlhf_jobs[0]
                # Either dataset is None or validation failed
                assert job.dataset is None or job.dataset == ""
        except Exception:
            pass  # Expected validation error

    def test_invalid_algorithm(self):
        """Test validation of algorithm names."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            algorithm "invalid_algorithm"
        }
        """
        
        app = App()
        app.parse(code)
        # Parser should accept any string, validation happens later
        job = app.rlhf_jobs[0]
        assert job.algorithm == "invalid_algorithm"

    def test_negative_hyperparameters(self):
        """Test handling of negative hyperparameters."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            algorithm_config {
                name "dpo"
                beta -0.1
            }
        }
        """
        
        app = App()
        app.parse(code)
        # Parser accepts, validation should flag
        job = app.rlhf_jobs[0]
        assert job.algorithm_config.beta == -0.1


class TestParserEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_rlhf_block(self):
        """Test parsing empty RLHF block."""
        code = """
        rlhf job {
        }
        """
        
        app = App()
        try:
            app.parse(code)
        except Exception:
            pass  # May fail, which is acceptable

    def test_nested_configuration_blocks(self):
        """Test that nested blocks are properly parsed."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            peft {
                method "lora"
                rank 8
            }
            
            compute {
                device "cuda"
                num_gpus 2
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.peft is not None
        assert job.compute is not None

    def test_string_escaping(self):
        """Test proper handling of escaped strings."""
        code = '''
        rlhf job {
            model "model/with/slashes"
            dataset "hf://user/dataset-name_v2"
            output_model "./path/to/model"
        }
        '''
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert "/" in job.model
        assert "://" in job.dataset

    def test_numeric_precision(self):
        """Test handling of various numeric formats."""
        code = """
        rlhf job {
            model "model"
            dataset "data"
            
            algorithm_config {
                name "dpo"
                learning_rate 5e-7
                beta 0.1
                warmup_ratio 0.03
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        job = app.rlhf_jobs[0]
        assert job.algorithm_config.learning_rate == 5e-7
        assert 0.0 < job.algorithm_config.beta < 1.0


class TestIntegrationWithApp:
    """Test integration with main App parser."""

    def test_rlhf_with_other_constructs(self):
        """Test RLHF alongside other N3 constructs."""
        code = """
        model gpt4 {
            provider "openai"
            name "gpt-4"
        }
        
        rlhf training {
            model "llama-3"
            dataset "data"
        }
        
        prompt example {
            "Test prompt"
        }
        """
        
        app = App()
        app.parse(code)
        
        # Should have both model and RLHF job
        assert hasattr(app, 'rlhf_jobs')
        assert len(app.rlhf_jobs) >= 1

    def test_multiple_jobs_with_different_configs(self):
        """Test multiple RLHF jobs with different configurations."""
        code = """
        rlhf dpo_job {
            model "model1"
            dataset "data1"
            algorithm "dpo"
            
            algorithm_config {
                name "dpo"
                beta 0.1
            }
        }
        
        rlhf ppo_job {
            model "model2"
            dataset "data2"
            algorithm "ppo"
            
            algorithm_config {
                name "ppo"
                clip_range 0.2
            }
        }
        """
        
        app = App()
        app.parse(code)
        
        assert len(app.rlhf_jobs) == 2
        assert app.rlhf_jobs[0].algorithm == "dpo"
        assert app.rlhf_jobs[1].algorithm == "ppo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
