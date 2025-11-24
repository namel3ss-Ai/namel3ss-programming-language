"""Unit tests for training and tuning AST nodes."""

from namel3ss.ast import TrainingJob, TuningJob, HyperparamSpec, EarlyStoppingSpec


def test_training_job_ast_fields():
    """Test that TrainingJob AST has all new fields."""
    job = TrainingJob(
        name="test_job",
        model="test_model",
        dataset="test_data",
        objective="accuracy",
        target="label",
        features=["feat1", "feat2"],
        framework="sklearn",
        split={"train": 0.7, "validation": 0.15, "test": 0.15},
        validation_split=0.2,
        early_stopping=EarlyStoppingSpec(
            metric="val_accuracy",
            patience=5,
            min_delta=0.001,
            mode="max"
        ),
        hyperparameters={"n_estimators": 100}
    )
    
    assert job.name == "test_job"
    assert job.target == "label"
    assert job.features == ["feat1", "feat2"]
    assert job.framework == "sklearn"
    assert job.split == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert job.validation_split == 0.2
    assert job.early_stopping.metric == "val_accuracy"
    assert job.early_stopping.patience == 5


def test_tuning_job_ast_fields():
    """Test that TuningJob AST has all necessary fields."""
    job = TuningJob(
        name="test_tuning",
        training_job="base_job",
        search_space={
            "n_estimators": HyperparamSpec(
                type="int",
                min=50,
                max=200,
                step=50
            ),
            "learning_rate": HyperparamSpec(
                type="float",
                min=0.001,
                max=0.1,
                log=True
            )
        },
        strategy="random",
        max_trials=20,
        objective_metric="f1"
    )
    
    assert job.name == "test_tuning"
    assert job.training_job == "base_job"
    assert job.strategy == "random"
    assert job.max_trials == 20
    assert "n_estimators" in job.search_space
    assert job.search_space["n_estimators"].type == "int"
    assert job.search_space["learning_rate"].log is True


def test_hyperparameter_spec():
    """Test HyperparamSpec initialization."""
    spec_int = HyperparamSpec(type="int", min=1, max=100, step=10)
    assert spec_int.type == "int"
    assert spec_int.min == 1
    assert spec_int.max == 100
    assert spec_int.step == 10
    
    spec_float = HyperparamSpec(type="float", min=0.01, max=1.0, log=True)
    assert spec_float.type == "float"
    assert spec_float.log is True
    
    spec_categorical = HyperparamSpec(type="categorical", values=["a", "b", "c"])
    assert spec_categorical.type == "categorical"
    assert len(spec_categorical.values) == 3


def test_early_stopping_spec():
    """Test EarlyStoppingSpec initialization."""
    spec = EarlyStoppingSpec(
        metric="accuracy",
        patience=10,
        min_delta=0.005,
        mode="max"
    )
    
    assert spec.metric == "accuracy"
    assert spec.patience == 10
    assert spec.min_delta == 0.005
    assert spec.mode == "max"
