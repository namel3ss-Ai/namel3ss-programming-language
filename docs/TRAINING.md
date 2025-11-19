# Training and Tuning in Namel3ss (N3)

Namel3ss provides first-class support for machine learning training and hyperparameter tuning workflows, fully integrated with the DSL, codegen, runtime, and model registry.

## Overview

Training and tuning pipelines in N3 allow you to:
- Define training jobs with dataset references, target/feature specifications, and hyperparameters
- Execute training with pluggable backends (sklearn, PyTorch, TensorFlow)
- Perform hyperparameter search with grid, random, or bayesian strategies
- Persist trained models to the model registry with versioning and metrics
- Expose training/tuning endpoints via automatically generated FastAPI routes

## Training Jobs

### Basic Syntax

```n3
training "job_name":
    model: model_reference
    dataset: dataset_reference
    target: target_column
    features:
        - feature1
        - feature2
        - feature3
    framework: sklearn
    objective: maximize_accuracy
    hyperparameters:
        param1: value1
        param2: value2
    split:
        train: 0.7
        validation: 0.15
        test: 0.15
    compute:
        backend: local
        resources:
            memory: 4GB
    metrics:
        - accuracy
        - f1
        - precision
```

### Required Fields

- **name**: Unique identifier for the training job
- **model**: Reference to a model definition
- **dataset**: Reference to a dataset or frame
- **objective**: Training objective (e.g., `maximize_accuracy`, `minimize_loss`)

### Optional Fields

- **target**: Target column name (if not inferred from model)
- **features**: List of feature column names (if not using all columns)
- **framework**: Training framework (`sklearn`, `pytorch`, `tensorflow`, defaults to `sklearn`)
- **hyperparameters**: Dictionary of model hyperparameters
- **split**: Train/validation/test split ratios (defaults to 0.7/0.15/0.15)
- **validation_split**: Simple validation split (alternative to full split configuration)
- **early_stopping**: Early stopping configuration (see below)
- **compute**: Compute backend and resource configuration
- **metrics**: List of metrics to track
- **output_registry**: Custom model registry key
- **description**: Human-readable description
- **metadata**: Additional metadata

### Early Stopping

```n3
training "with_early_stopping":
    model: my_model
    dataset: my_data
    target: label
    objective: accuracy
    early_stopping:
        metric: val_accuracy
        patience: 5
        min_delta: 0.001
        mode: max
```

## Tuning Jobs

### Basic Syntax

```n3
tuning "tuning_job_name":
    training_job: base_training_job
    strategy: random
    max_trials: 20
    parallel_trials: 2
    objective_metric: accuracy
    search_space:
        hyperparameter1:
            type: float
            min: 0.001
            max: 0.1
            log: true
        hyperparameter2:
            type: int
            min: 10
            max: 100
            step: 10
        hyperparameter3:
            type: categorical
            values:
                - option1
                - option2
                - option3
    early_stopping:
        metric: val_accuracy
        patience: 3
        min_delta: 0.01
        mode: max
```

### Required Fields

- **name**: Unique identifier for the tuning job
- **training_job**: Reference to base training job
- **search_space**: Hyperparameter search space specification

### Optional Fields

- **strategy**: Search strategy (`grid`, `random`, defaults to `grid`)
- **max_trials**: Maximum number of trials to run (default: 10)
- **parallel_trials**: Number of trials to run in parallel (default: 1)
- **objective_metric**: Metric to optimize (default: `accuracy`)
- **early_stopping**: Early stopping for tuning process
- **metadata**: Additional metadata

### Search Space Types

#### Numeric Ranges

```n3
learning_rate:
    type: float
    min: 0.0001
    max: 0.1
    log: true  # Log-uniform sampling
```

#### Integer Ranges

```n3
n_estimators:
    type: int
    min: 50
    max: 200
    step: 50
```

#### Categorical Values

```n3
activation:
    type: categorical
    values:
        - relu
        - tanh
        - sigmoid
```

## Complete Example

```n3
app:
    name: customer_churn_prediction

# Dataset definition
dataset "customer_data":
    source: postgres
    table: customers
    schema:
        - customer_id: int
        - tenure: int
        - monthly_spend: float
        - support_calls: int
        - contract_type: string
        - churned: bool

# Model definition
model "churn_classifier":
    type: classifier
    engine: sklearn

# Base training job
training "baseline_churn_model":
    model: churn_classifier
    dataset: customer_data
    target: churned
    features:
        - tenure
        - monthly_spend
        - support_calls
    framework: sklearn
    objective: maximize_accuracy
    hyperparameters:
        n_estimators: 100
        max_depth: 10
        min_samples_split: 2
    split:
        train: 0.7
        validation: 0.15
        test: 0.15
    early_stopping:
        metric: val_accuracy
        patience: 10
        min_delta: 0.001
        mode: max
    compute:
        backend: local
    metrics:
        - accuracy
        - precision
        - recall
        - f1
        - auc
    description: "Baseline random forest model for churn prediction"

# Hyperparameter tuning
tuning "optimized_churn_model":
    training_job: baseline_churn_model
    strategy: random
    max_trials: 50
    parallel_trials: 4
    objective_metric: auc
    search_space:
        n_estimators:
            type: int
            min: 50
            max: 300
            step: 50
        max_depth:
            type: int
            min: 5
            max: 30
        min_samples_split:
            type: int
            min: 2
            max: 20
        learning_rate:
            type: float
            min: 0.001
            max: 0.3
            log: true
    early_stopping:
        metric: auc
        patience: 5
        min_delta: 0.005
        mode: max
    metadata:
        experiment: churn_optimization_v1
        owner: data_science_team
```

## Runtime API

### Training Execution

Execute a training job programmatically:

```python
from generated.runtime import run_training_job

result = await run_training_job(
    name="baseline_churn_model",
    payload={},
    overrides={
        "hyperparameters": {
            "n_estimators": 150
        }
    }
)

print(f"Training status: {result['status']}")
print(f"Model: {result['model']}")
print(f"Metrics: {result['metrics']}")
print(f"Registry key: {result.get('registry_key')}")
```

### Tuning Execution

Execute a tuning job:

```python
from generated.runtime import run_tuning_job

result = await run_tuning_job(
    name="optimized_churn_model",
    payload={},
    overrides={}
)

print(f"Total trials: {result['total_trials']}")
print(f"Best hyperparameters: {result['best_hyperparameters']}")
print(f"Best metrics: {result['best_metrics']}")
print(f"Best model registry key: {result.get('best_model_registry_key')}")
```

## HTTP API

The N3 compiler automatically generates FastAPI endpoints for all training and tuning jobs.

### Training Endpoints

#### List All Training Jobs
```bash
GET /api/training/jobs
```

Response:
```json
["baseline_churn_model", "image_classifier_training", ...]
```

#### Get Training Job Spec
```bash
GET /api/training/jobs/{job_name}
```

Response:
```json
{
    "name": "baseline_churn_model",
    "model": "churn_classifier",
    "dataset": "customer_data",
    "target": "churned",
    "features": ["tenure", "monthly_spend", "support_calls"],
    ...
}
```

#### Execute Training Job
```bash
POST /api/training/jobs/{job_name}/run
Content-Type: application/json

{
    "payload": {},
    "overrides": {
        "hyperparameters": {
            "n_estimators": 150
        }
    }
}
```

Response:
```json
{
    "status": "ok",
    "job": "baseline_churn_model",
    "model": "churn_classifier",
    "dataset": "customer_data",
    "metrics": {
        "accuracy": 0.8567,
        "f1": 0.8234,
        "auc": 0.9012
    },
    "registry_key": "churn_classifier_baseline_churn_model_v1678901234",
    ...
}
```

#### Get Training Metrics
```bash
GET /api/training/jobs/{job_name}/metrics
```

Response:
```json
{
    "job": "baseline_churn_model",
    "status": "ok",
    "metrics": {
        "accuracy": 0.8567,
        "f1": 0.8234
    },
    "timestamp": 1678901234.567
}
```

#### Get Training History
```bash
GET /api/training/jobs/{job_name}/history
```

Response:
```json
[
    {
        "status": "ok",
        "ts": 1678901234.567,
        "duration_ms": 12345.678,
        "metrics": {...},
        ...
    },
    ...
]
```

### Tuning Endpoints

#### List All Tuning Jobs
```bash
GET /api/training/tuning/jobs
```

#### Get Tuning Job Spec
```bash
GET /api/training/tuning/jobs/{job_name}
```

#### Execute Tuning Job
```bash
POST /api/training/tuning/jobs/{job_name}/run
Content-Type: application/json

{
    "payload": {},
    "overrides": {}
}
```

Response:
```json
{
    "status": "ok",
    "job": "optimized_churn_model",
    "training_job": "baseline_churn_model",
    "strategy": "random",
    "total_trials": 50,
    "best_trial": {...},
    "best_hyperparameters": {
        "n_estimators": 200,
        "max_depth": 15,
        ...
    },
    "best_metrics": {
        "auc": 0.9234,
        "accuracy": 0.8890
    },
    "best_model_registry_key": "churn_classifier_baseline_churn_model_v1678901456",
    ...
}
```

#### Get Trial Results
```bash
GET /api/training/tuning/jobs/{job_name}/trials
```

Response:
```json
[
    {
        "trial_index": 0,
        "status": "ok",
        "hyperparameters": {...},
        "metrics": {...}
    },
    ...
]
```

#### Get Best Trial
```bash
GET /api/training/tuning/jobs/{job_name}/best
```

## Training Backends

N3 supports pluggable training backends:

- **local**: Deterministic local backend for development/testing
- **sklearn**: Production scikit-learn training with real model persistence
- **pytorch**: PyTorch training (placeholder for future implementation)
- **tensorflow**: TensorFlow training (placeholder for future implementation)
- **ray**: Distributed training with Ray (requires ray package)

### Sklearn Backend

The sklearn backend provides production-grade training:

- Real dataset loading and preprocessing
- Train/validation/test splitting
- Model training with configurable hyperparameters
- Metric evaluation (accuracy, precision, recall, F1, etc.)
- Model persistence to registry with versioning

Supported sklearn models:
- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression
- LinearRegression
- DecisionTreeClassifier
- SVC

## Model Registry Integration

Trained models are automatically persisted to the model registry with:

- **Version**: Timestamp-based version identifier
- **Metrics**: All evaluated metrics from training
- **Metadata**: Training job info, dataset, hyperparameters, timestamps
- **Artifacts**: Serialized model object, feature names, target name

Access trained models:

```python
from generated.runtime import MODEL_REGISTRY

# Get trained model
model_key = "churn_classifier_baseline_churn_model_v1678901234"
model_entry = MODEL_REGISTRY[model_key]

print(f"Framework: {model_entry['framework']}")
print(f"Metrics: {model_entry['metrics']}")
print(f"Training job: {model_entry['metadata']['training_job']}")

# Get model object for inference
model_object = model_entry['metadata']['model_object']
predictions = model_object.predict(new_data)
```

## Configuration

Training and tuning can be configured via environment variables:

```bash
# Training backend
export NAMEL3SS_TRAINING_BACKEND=sklearn

# Model registry location
export NAMEL3SS_MODEL_REGISTRY=/path/to/registry.json
export NAMEL3SS_MODEL_ROOT=/path/to/models/
```

## Best Practices

1. **Always specify target and features explicitly** for clarity and validation
2. **Use appropriate split ratios** (70/15/15 is a good default)
3. **Start with small max_trials** for tuning jobs to test the pipeline
4. **Use log-uniform sampling** for learning rate and similar exponential-scale parameters
5. **Enable early stopping** to avoid unnecessary computation
6. **Track multiple metrics** but optimize for one primary metric
7. **Store training metadata** for experiment tracking and reproducibility
8. **Use grid search** for small search spaces, **random search** for larger spaces
9. **Version your datasets** and reference specific versions in training jobs
10. **Monitor training job history** to track model performance over time

## Limitations

- PyTorch and TensorFlow backends are placeholders (not yet implemented)
- Parallel trial execution runs sequentially (async parallel execution coming soon)
- Bayesian optimization strategy not yet implemented
- Custom metric definitions must be done in custom code
- Distributed training requires Ray backend configuration

## Troubleshooting

### Training Job Fails with "dataset_load_failed"

Ensure the dataset is properly defined and accessible:
```n3
dataset "my_data":
    source: csv
    path: "/absolute/path/to/data.csv"
    schema:
        - field1: type1
        - field2: type2
```

### Target/Feature Field Not Found

Verify field names match dataset schema exactly:
```n3
training "my_job":
    target: label  # Must exist in dataset schema
    features:
        - feature_a  # Must exist in dataset schema
```

### Sklearn Backend Unavailable

Install required dependencies:
```bash
pip install scikit-learn pandas numpy
```

### Registry Key Missing

Check if model was persisted successfully:
```python
result = await run_training_job("my_job")
if "registry_key" in result:
    print(f"Model saved as: {result['registry_key']}")
else:
    print(f"Warning: {result.get('registry_warning')}")
```

## See Also

- [Model Registry Guide](llm-provider-guide.md)
- [Dataset Guide](EVAL_SUITES.md)
- [RAG Guide](RAG_GUIDE.md)
