"""Model parsing tests for the Namel3ss parser."""

from namel3ss.ast import Model
from namel3ss.parser import Parser


def test_parse_model_declaration() -> None:
    source = (
        'app "Models".\n'
        '\n'
        'model "sales_forecast" using regression engine sklearn:\n'
        '  from dataset sales_features\n'
        '  target: revenue\n'
        '  features: week, spend\n'
        '  schedule: daily\n'
        '  options:\n'
        '    alpha: 0.1\n'
        '    epochs: 200\n'
        '  registry:\n'
        '    version: v1\n'
        '    accuracy: 0.92\n'
        '    promoted: true\n'
    )

    app = Parser(source).parse()
    assert len(app.models) == 1
    model = app.models[0]
    assert isinstance(model, Model)
    assert model.model_type == 'regression'
    assert model.engine == 'sklearn'
    assert model.training.source_type == 'dataset'
    assert model.training.source == 'sales_features'
    assert model.training.target == 'revenue'
    assert model.training.features == ['week', 'spend']
    assert model.training.schedule == 'daily'
    assert model.options['alpha'] == 0.1
    assert model.options['epochs'] == 200
    assert model.registry.version == 'v1'
    assert model.registry.accuracy == 0.92
    assert model.registry.metadata['promoted'] is True


def test_parse_model_with_extended_metadata() -> None:
    source = (
        'app "AdvancedModels".\n'
        '\n'
        'model "churn" using classifier engine pytorch:\n'
        '  framework: pytorch\n'
        '  objective: binary\n'
        '  from dataset churn_features\n'
        '  target: churn_flag\n'
        '  features: age, spend\n'
        '  datasets:\n'
        '    train:\n'
        '      name: churn_train\n'
        '      filter: age > 18\n'
        '    validation: churn_validation\n'
        '  transform "scale_inputs":\n'
        '    type: expression\n'
        '    inputs: age, spend\n'
        '    output: features_scaled\n'
        '    expression: (age + spend) / 2\n'
        '  hyperparameters:\n'
        '    learning_rate:\n'
        '      value: 0.01\n'
        '      tunable: true\n'
        '  feature "features_scaled":\n'
        '    role: feature\n'
        '    required: true\n'
        '  monitoring:\n'
        '    schedule: daily\n'
        '    metrics:\n'
        '      accuracy:\n'
        '        value: 0.9\n'
        '        threshold: 0.8\n'
        '  serving:\n'
        '    endpoints: https://api.example.com/predict\n'
        '  deployments:\n'
        '    production:\n'
        '      environment: prod\n'
        '  tags: churn, production\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )

    app = Parser(source).parse()
    model = app.models[0]

    assert model.training.framework == 'pytorch'
    assert model.training.objective == 'binary'
    assert model.training.datasets and model.training.datasets[0].name == 'churn_train'
    assert model.training.datasets[0].filters and model.training.datasets[0].filters[0] is not None
    assert model.training.transforms and model.training.transforms[0].name == 'scale_inputs'
    assert model.training.hyperparameters and model.training.hyperparameters[0].tunable is True
    assert model.features_spec and model.features_spec[0].name == 'features_scaled'
    assert model.monitoring is not None and model.monitoring.schedule == 'daily'
    assert model.monitoring.metrics[0].name == 'accuracy'
    assert model.serving is not None and model.serving.endpoints[0] == 'https://api.example.com/predict'
    assert model.deployments and model.deployments[0].environment == 'prod'
    assert 'churn' in model.tags
