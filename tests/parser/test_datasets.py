"""Dataset parsing tests for the Namel3ss parser."""

from namel3ss.ast import (
    BinaryOp,
    CallExpression,
    CachePolicy,
    ComputedColumnOp,
    ContextValue,
    FilterOp,
    JoinOp,
    OrderByOp,
    RefreshPolicy,
    WindowFrame,
    WindowOp,
)
from namel3ss.parser import Parser


def test_parse_context_references_in_options() -> None:
    source = (
        'app "CtxApp".\n'
        '\n'
        'dataset "secure" from rest SECURE_ENDPOINT endpoint "https://api.example.com/data":\n'
        '  with option headers.Authorization env:API_TOKEN\n'
        '  with option params.user_id ctx:user.id\n'
        '  filter by: ctx:user.id == 42\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Welcome {ctx:user.name}!"\n'
    )

    app = Parser(source).parse_app()

    dataset = app.datasets[0]
    headers = dataset.connector.options['headers']
    params = dataset.connector.options['params']

    auth_value = headers['Authorization']
    assert isinstance(auth_value, ContextValue)
    assert auth_value.scope == 'env'
    assert auth_value.path == ['API_TOKEN']

    user_param = params['user_id']
    assert isinstance(user_param, ContextValue)
    assert user_param.scope == 'ctx'
    assert user_param.path == ['user', 'id']

    filter_op = dataset.operations[0]
    assert isinstance(filter_op, FilterOp)
    assert isinstance(filter_op.condition.left, ContextValue)
    assert filter_op.condition.left.scope == 'ctx'
    assert filter_op.condition.left.path == ['user', 'id']


def test_parse_dataset_computed_column_and_filter_expression() -> None:
    source = (
        'app "Data".\n'
        '\n'
        'dataset "orders_enriched" from table orders:\n'
        '  add column total_with_tax = (subtotal + tax)\n'
        '  filter by: total_with_tax > 100 and status == "PAID"\n'
        '\n'
        'page "Dummy" at "/d":\n'
        '  show text "ok"\n'
    )
    app = Parser(source).parse_app()
    dataset = next(d for d in app.datasets if d.name == 'orders_enriched')

    computed = next(op for op in dataset.operations if isinstance(op, ComputedColumnOp))
    assert computed.name == 'total_with_tax'
    assert isinstance(computed.expression, BinaryOp)
    assert computed.expression.op == '+'

    filter_op = next(op for op in dataset.operations if isinstance(op, FilterOp))
    assert isinstance(filter_op.condition, BinaryOp)
    assert filter_op.condition.op == 'and'

    left = filter_op.condition.left
    assert isinstance(left, BinaryOp)
    assert left.op == '>'

    right = filter_op.condition.right
    assert isinstance(right, BinaryOp)
    assert right.op == '=='



def test_parse_dataset_with_connectors_and_runtime_policies() -> None:
    source = (
        'app "DataConnectors".\n'
        '\n'
        'dataset "sales" from sql "MAIN_DB" table "sales":\n'
        '  reactive: true\n'
        '  auto refresh every 30 seconds\n'
        '  cache:\n'
        '    strategy: memory\n'
        '    ttl_seconds: 120\n'
        '  pagination:\n'
        '    page_size: 50\n'
        '  filter by: status == "OPEN"\n'
        '  add column amount_usd = convert_to_usd(amount)\n'
        '  add column rolling_7d = avg(amount) over last 7 days\n'
        '  group by: region\n'
        '  join dataset regions on sales.region_id == regions.id\n'
        '  order by: region\n'
    )

    app = Parser(source).parse_app()
    dataset = app.datasets[0]

    assert dataset.source_type == 'sql'
    assert dataset.connector is not None
    assert dataset.connector.connector_type == 'sql'
    assert dataset.connector.connector_name == 'MAIN_DB'
    assert dataset.connector.options.get('table') == 'sales'
    assert dataset.reactive is True
    assert isinstance(dataset.refresh_policy, RefreshPolicy)
    assert dataset.refresh_policy.interval_seconds == 30
    assert isinstance(dataset.cache_policy, CachePolicy)
    assert dataset.cache_policy.strategy == 'memory'
    assert dataset.cache_policy.ttl_seconds == 120
    assert dataset.pagination is not None
    assert dataset.pagination.page_size == 50

    operations = dataset.operations
    assert any(isinstance(op, FilterOp) for op in operations)

    computed = next(op for op in operations if isinstance(op, ComputedColumnOp))
    assert computed.name == 'amount_usd'
    assert isinstance(computed.expression, CallExpression)

    window = next(op for op in operations if isinstance(op, WindowOp))
    assert window.name == 'rolling_7d'
    assert window.function == 'avg'
    assert isinstance(window.frame, WindowFrame)
    assert window.frame.interval_value == 7
    assert window.frame.interval_unit == 'days'

    join = next(op for op in operations if isinstance(op, JoinOp))
    assert join.target_type == 'dataset'
    assert join.target_name == 'regions'

    order = next(op for op in operations if isinstance(op, OrderByOp))
    assert 'region' in order.columns



def test_parse_dataset_with_inline_connector_options() -> None:
    source = (
        'app "EnvConnect".\n'
        '\n'
        'dataset "orders" from rest "ORDERS_API" endpoint "https://api.example.com/orders":\n'
        '  with option headers.Authorization env:NAMEL3SS_ORDERS_TOKEN\n'
        '  with option params.api_key ${NAMEL3SS_ORDERS_TOKEN}\n'
        '  with option timeout 15\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show table "Orders" from dataset orders\n'
    )

    app = Parser(source).parse_app()
    dataset = app.datasets[0]

    assert dataset.connector is not None
    options = dataset.connector.options
    auth_value = options['headers']['Authorization']
    assert isinstance(auth_value, ContextValue)
    assert auth_value.scope == 'env'
    assert auth_value.path == ['NAMEL3SS_ORDERS_TOKEN']
    assert options['params']['api_key'] == '${NAMEL3SS_ORDERS_TOKEN}'
    assert options['timeout'] == 15


def test_parse_dataset_with_schema_transforms_and_metadata() -> None:
    source = (
        'app "Advanced".\n'
        '\n'
        'dataset "orders_enriched" from table orders:\n'
        '  transform "normalize_total":\n'
        '    type: expression\n'
        '    inputs: subtotal, tax\n'
        '    output: total_norm\n'
        '    expression: (subtotal + tax) / 100\n'
        '  schema:\n'
        '    column subtotal:\n'
        '      dtype: float\n'
        '      nullable: false\n'
        '      tags: numeric\n'
        '    column total_norm:\n'
        '      dtype: float\n'
        '  feature "total_norm":\n'
        '    role: feature\n'
        '    dtype: float\n'
        '    description: "Normalized total"\n'
        '  target "is_vip":\n'
        '    kind: classification\n'
        '    expression: vip_flag\n'
        '    positive_class: "Y"\n'
        '  quality "no_missing_total":\n'
        '    condition: count_nulls(total_norm) == 0\n'
        '    severity: warn\n'
        '  profile:\n'
        '    row_count: 1000\n'
        '    stats:\n'
        '      freshness: "hourly"\n'
        '  metadata:\n'
        '    owner: "data_science"\n'
        '  lineage:\n'
        '    upstream: raw_orders\n'
        '  tags: finance, orders\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )

    app = Parser(source).parse_app()
    dataset = app.datasets[0]

    assert dataset.transforms and dataset.transforms[0].name == 'normalize_total'
    assert dataset.transforms[0].output == 'total_norm'
    assert dataset.schema and dataset.schema[0].name == 'subtotal'
    assert dataset.schema[0].dtype == 'float'
    assert dataset.schema[0].nullable is False
    assert 'numeric' in dataset.schema[0].tags
    assert dataset.features and dataset.features[0].name == 'total_norm'
    assert dataset.targets and dataset.targets[0].positive_class == 'Y'
    assert dataset.quality_checks and dataset.quality_checks[0].severity == 'warn'
    assert dataset.profile is not None and dataset.profile.row_count == 1000
    assert dataset.metadata.get('owner') == 'data_science'
    assert dataset.lineage.get('upstream') == 'raw_orders'
    assert dataset.tags == ['finance', 'orders']
