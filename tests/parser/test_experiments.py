"""Tests for experiment parsing in Namel3ss."""

from namel3ss.ast import Experiment, ExperimentMetric, ExperimentVariant
from namel3ss.parser import Parser


def test_parse_experiment_with_variants_and_metrics() -> None:
    source = (
        'app "Lab".\n'
        '\n'
        'experiment "sentiment_test":\n'
        '  description: "A/B test for sentiment flows"\n'
        '  variants:\n'
        '    v1 uses model sentiment_v1\n'
        '    v2 uses chain summarize_chain:\n'
        '      temperature: 0.3\n'
        '      max_tokens: 256\n'
        '  metrics:\n'
        '    accuracy from dataset reviews goal 0.90\n'
        '    latency_ms goal 250\n'
        '  metadata:\n'
        '    owner: "research"\n'
    )

    app = Parser(source).parse()

    assert len(app.experiments) == 1
    experiment = app.experiments[0]
    assert isinstance(experiment, Experiment)
    assert experiment.name == "sentiment_test"
    assert experiment.description == "A/B test for sentiment flows"
    assert experiment.metadata["owner"] == "research"

    assert len(experiment.variants) == 2
    variant_a, variant_b = experiment.variants
    assert isinstance(variant_a, ExperimentVariant)
    assert variant_a.target_type == "model"
    assert variant_a.target_name == "sentiment_v1"
    assert isinstance(variant_b, ExperimentVariant)
    assert variant_b.target_type == "chain"
    assert variant_b.config["temperature"] == 0.3
    assert variant_b.config["max_tokens"] == 256

    assert len(experiment.metrics) == 2
    metric_accuracy, metric_latency = experiment.metrics
    assert isinstance(metric_accuracy, ExperimentMetric)
    assert metric_accuracy.source_kind == "dataset"
    assert metric_accuracy.source_name == "reviews"
    assert metric_accuracy.goal == "0.90"
    assert metric_latency.name == "latency_ms"
    assert metric_latency.goal == "250"
