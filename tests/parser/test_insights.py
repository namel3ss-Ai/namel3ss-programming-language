"""Insight parsing tests for the Namel3ss parser."""

from namel3ss.ast import (
    CallExpression,
    InsightAssignment,
    InsightEmit,
    InsightMetric,
    InsightNarrative,
    InsightSelect,
    InsightThreshold,
    Literal,
    ShowChart,
)
from namel3ss.parser import Parser


def test_parse_insight_logic_and_outputs() -> None:
    source = (
        'app "Insights".\n'
        '\n'
        'insight "high_revenue_regions" from dataset sales_summary:\n'
        '  logic:\n'
        '    threshold = 100000\n'
        '    regions = rows\n'
        '    select regions where total_revenue > threshold limit 5 order by total_revenue desc\n'
        '    emit narrative "Top regions are outperforming"\n'
        '  metrics:\n'
        '    total_revenue:\n'
        '      label: "Total Revenue"\n'
        '      value: sum("total_revenue")\n'
        '      baseline: avg("total_revenue")\n'
        '      unit: "USD"\n'
        '  thresholds:\n'
        '    revenue_warning:\n'
        '      metric: total_revenue\n'
        '      operator: ">"\n'
        '      value: 90000\n'
        '      level: warning\n'
        '  narratives:\n'
        '    summary:\n'
        '      template: "Top region is {regions[0].name}"\n'
        '  expose:\n'
        '    top_region: regions[0].name\n'
    )

    app = Parser(source).parse_app()
    assert len(app.insights) == 1
    insight = app.insights[0]
    assert insight.name == 'high_revenue_regions'
    assert len(insight.logic) == 4
    assert isinstance(insight.logic[0], InsightAssignment)
    assert isinstance(insight.logic[1], InsightAssignment)
    select_step = insight.logic[2]
    assert isinstance(select_step, InsightSelect)
    assert select_step.limit == 5
    assert select_step.order_by is not None
    assert select_step.order_by[0].startswith('total_revenue')
    assert isinstance(insight.logic[3], InsightEmit)

    assert len(insight.metrics) == 1
    metric = insight.metrics[0]
    assert isinstance(metric, InsightMetric)
    assert metric.name == 'total_revenue'
    assert metric.label == 'Total Revenue'
    assert isinstance(metric.value, CallExpression)
    assert metric.unit == 'USD'

    assert len(insight.thresholds) == 1
    threshold = insight.thresholds[0]
    assert isinstance(threshold, InsightThreshold)
    assert threshold.name == 'revenue_warning'
    assert threshold.metric == 'total_revenue'
    assert threshold.level == 'warning'

    assert len(insight.narratives) == 1
    narrative = insight.narratives[0]
    assert isinstance(narrative, InsightNarrative)
    assert narrative.template.startswith('Top region')

    expose_expr = insight.expose_as['top_region']
    assert isinstance(expose_expr, Literal)
    assert expose_expr.value == 'regions[0].name'
    assert 'revenue_warning' in insight.alert_thresholds



def test_parse_insight_with_compute_and_emit_narrative() -> None:
    source = (
        'app "Insights".\n'
        '\n'
        'insight "revenue_growth" from dataset monthly_sales:\n'
        '  compute:\n'
        '    current = avg(revenue)\n'
        '    previous = avg(revenue)\n'
        '  emit narrative:\n'
        '    "Revenue changed {delta_percent}% since last month."\n'
    )

    app = Parser(source).parse_app()
    assert len(app.insights) == 1
    insight = app.insights[0]
    assert insight.name == 'revenue_growth'
    assert insight.source_dataset == 'monthly_sales'
    assert len(insight.logic) == 3
    assert isinstance(insight.logic[0], InsightAssignment)
    assert isinstance(insight.logic[-1], InsightEmit)
    assert insight.logic[-1].kind == 'narrative'
    assert 'Revenue changed' in insight.logic[-1].content



def test_parse_chart_with_inline_insight_reference() -> None:
    source = (
        'app "Charts".\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show chart "Revenue" from dataset monthly_sales:\n'
        '    x: month\n'
        '    y: revenue\n'
        '    insight: "revenue_growth"\n'
    )

    app = Parser(source).parse_app()
    assert len(app.pages) == 1
    page = app.pages[0]
    chart = next(stmt for stmt in page.statements if isinstance(stmt, ShowChart))
    assert chart.insight == 'revenue_growth'
