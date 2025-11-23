"""
Quick Test - Verify RLHF imports and configuration.

This script validates that all RLHF components can be imported
and basic configurations can be created without errors.
"""

def test_imports():
    """Test that all RLHF components can be imported."""
    print("Testing RLHF imports...")
    
    # Configuration
    from namel3ss.ml.rlhf import (
        RLHFConfig,
        RLHFAlgorithm,
        PEFTConfig,
        PEFTMethod,
        PPOConfig,
        DPOConfig,
        ORPOConfig,
        KTOConfig,
        LoggingConfig,
        SafetyConfig,
        ExperimentTracker,
    )
    print("✓ Configuration imports successful")
    
    # Errors
    from namel3ss.ml.rlhf import (
        RLHFError,
        RLHFConfigurationError,
        RLHFTrainingError,
        RLHFDatasetError,
        RLHFModelError,
        RLHFEvaluationError,
        RLHFStorageError,
    )
    print("✓ Error imports successful")
    
    # Runners
    from namel3ss.ml.rlhf import RLHFJobRunner, RLHFJobResult
    print("✓ Runner imports successful")
    
    # Datasets
    from namel3ss.ml.rlhf import (
        PreferenceDataset,
        FeedbackDataset,
        PreferenceSample,
        FeedbackSample,
        load_preference_dataset,
        load_feedback_dataset,
    )
    print("✓ Dataset imports successful")
    
    # Trainers
    from namel3ss.ml.rlhf import (
        BaseRLHFTrainer,
        PPOTrainer,
        DPOTrainer,
        ORPOTrainer,
        KTOTrainer,
        SFTTrainer,
        get_trainer_class,
    )
    print("✓ Trainer imports successful")
    
    return True


def test_configuration():
    """Test that configurations can be created."""
    print("\nTesting RLHF configuration creation...")
    
    from namel3ss.ml.rlhf import (
        RLHFConfig,
        RLHFAlgorithm,
        PEFTConfig,
        PEFTMethod,
        DPOConfig,
    )
    
    # Create PEFT config
    peft_config = PEFTConfig(
        method=PEFTMethod.LORA,
        r=64,
        alpha=16,
    )
    print("✓ PEFT config created")
    
    # Create DPO config
    dpo_config = DPOConfig(beta=0.1)
    print("✓ DPO config created")
    
    # Create main RLHF config
    config = RLHFConfig(
        job_name="test_job",
        algorithm=RLHFAlgorithm.DPO,
        base_model="test-model",
        dataset_path="test-dataset",
        output_dir="/tmp/test",
        peft=peft_config,
        dpo_config=dpo_config,
    )
    print("✓ RLHF config created")
    
    # Test serialization
    config_dict = config.to_dict()
    print(f"✓ Config serialized ({len(config_dict)} keys)")
    
    return True


def test_error_creation():
    """Test that errors can be created and formatted."""
    print("\nTesting RLHF error handling...")
    
    from namel3ss.ml.rlhf import (
        RLHFConfigurationError,
        RLHFTrainingError,
        RLHFModelError,
    )
    
    # Create configuration error
    try:
        raise RLHFConfigurationError(
            "Test configuration error",
            code="RLHF001",
            context={"test": "value"}
        )
    except RLHFConfigurationError as e:
        assert e.code == "RLHF001"
        assert "test" in e.context
        formatted = e.format()
        assert "RLHF001" in formatted
        print("✓ Configuration error works")
    
    # Create training error
    try:
        raise RLHFTrainingError(
            "Test training error",
            code="RLHF010",
            context={"step": 100}
        )
    except RLHFTrainingError as e:
        assert e.code == "RLHF010"
        print("✓ Training error works")
    
    return True


def test_trainer_registry():
    """Test that trainer registry works."""
    print("\nTesting trainer registry...")
    
    from namel3ss.ml.rlhf import RLHFAlgorithm, get_trainer_class
    from namel3ss.ml.rlhf import PPOTrainer, DPOTrainer, ORPOTrainer, KTOTrainer, SFTTrainer
    
    # Test getting trainers
    assert get_trainer_class(RLHFAlgorithm.PPO) == PPOTrainer
    print("✓ PPO trainer resolved")
    
    assert get_trainer_class(RLHFAlgorithm.DPO) == DPOTrainer
    print("✓ DPO trainer resolved")
    
    assert get_trainer_class(RLHFAlgorithm.ORPO) == ORPOTrainer
    print("✓ ORPO trainer resolved")
    
    assert get_trainer_class(RLHFAlgorithm.KTO) == KTOTrainer
    print("✓ KTO trainer resolved")
    
    assert get_trainer_class(RLHFAlgorithm.SFT) == SFTTrainer
    print("✓ SFT trainer resolved")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Namel3ss RLHF Quick Test Suite")
    print("=" * 60)
    
    try:
        test_imports()
        test_configuration()
        test_error_creation()
        test_trainer_registry()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nRLHF subsystem is properly configured and ready to use!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements-rlhf.txt")
        print("  2. Run example: python examples/rlhf_dpo_example.py")
        print("  3. Check docs: docs/RLHF_TRAINING.md")
        
        return True
    
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
