"""Frame parsing tests for the Namel3ss parser."""

from namel3ss.ast import FrameColumnConstraint
from namel3ss.parser import Parser


def test_parse_frame_with_columns_relationships_and_samples() -> None:
    source = (
        'app "FrameApp".\n'
        '\n'
        'frame "OrdersFrame" from dataset orders:\n'
        '  description: "Normalized orders view"\n'
        '  tags: finance, ops\n'
        '  column order_id uuid required:\n'
        '    description: "Primary id"\n'
        '    tags: primary, id\n'
        '  column total_amount decimal:\n'
        '    nullable: false\n'
        '    default: 0\n'
        '    expression: subtotal + tax\n'
        '    validations:\n'
        '      non_negative:\n'
        '        expression: total_amount >= 0\n'
        '        message: "Totals must be >= 0"\n'
        '  index "by_customer" on customer_id\n'
        '  relationship "customer" to frame CustomersFrame:\n'
        '    local_key: customer_id\n'
        '    remote_key: id\n'
        '  constraint "positive_total":\n'
        '    expression: total_amount >= 0\n'
        '    severity: warn\n'
        '  access:\n'
        '    public: false\n'
        '    roles: admin, finance\n'
        '    cache_seconds: 30\n'
        '  sample:\n'
        '    order_id: "ord_1"\n'
        '    total_amount: 42\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )

    app = Parser(source).parse()
    assert len(app.frames) == 1
    frame = app.frames[0]
    assert frame.name == 'OrdersFrame'
    assert frame.source == 'orders'
    assert frame.description == 'Normalized orders view'
    assert frame.tags == ['finance', 'ops']
    assert len(frame.columns) == 2

    id_column = frame.columns[0]
    assert id_column.name == 'order_id'
    assert id_column.dtype == 'uuid'
    assert id_column.nullable is False
    assert id_column.tags == ['primary', 'id']

    amount_column = frame.columns[1]
    assert amount_column.default == 0
    assert amount_column.expression is not None
    assert amount_column.nullable is False
    assert amount_column.validations
    assert isinstance(amount_column.validations[0], FrameColumnConstraint)
    assert amount_column.validations[0].message == 'Totals must be >= 0'

    assert frame.indexes[0].columns == ['customer_id']
    assert frame.relationships[0].target_frame == 'CustomersFrame'
    assert frame.constraints[0].severity == 'warn'
    assert frame.access is not None
    assert frame.access.roles == ['admin', 'finance']
    assert frame.access.cache_seconds == 30
    assert frame.examples[0]['order_id'] == 'ord_1'


def test_parse_frame_with_inline_options_and_defaults() -> None:
    source = (
        'app "OptionsApp".\n'
        '\n'
        'frame "MetricsFrame" from dataset metrics:\n'
        '  column metric_name string:\n'
        '    default: "visitors"\n'
        '  with option caching.ttl 30\n'
        '  with option caching.strategy "memory"\n'
        '  options:\n'
        '    format: parquet\n'
        '\n'
        'page "Metrics" at "/metrics":\n'
        '  show text "metrics"\n'
    )

    app = Parser(source).parse()
    frame = app.frames[0]
    assert frame.options['caching']['ttl'] == 30
    assert frame.options['caching']['strategy'] == 'memory'
    assert frame.options['format'] == 'parquet'
    assert frame.columns[0].default == 'visitors'
