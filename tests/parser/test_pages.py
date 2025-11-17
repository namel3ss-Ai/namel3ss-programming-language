"""Page and statement parsing tests for the Namel3ss parser."""

from namel3ss.ast import (
    Action,
    AskConnectorOperation,
    AttributeRef,
    BinaryOp,
    CallPythonOperation,
    ForLoop,
    IfBlock,
    LayoutMeta,
    Literal,
    NameRef,
    PredictStatement,
    RunChainOperation,
    RunPromptOperation,
    RefreshPolicy,
    ShowChart,
    ShowForm,
    ShowTable,
    ShowText,
    ToastOperation,
    VariableAssignment,
)
from namel3ss.parser import Parser


def test_parse_basic_app() -> None:
    source = (
        'app "Demo" connects to postgres "PRIMARY".\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Welcome"\n'
        '  show table "Users" from table users\n'
    )
    app = Parser(source).parse()

    assert app.name == "Demo"
    assert app.database == "PRIMARY"
    assert len(app.pages) == 1

    page = app.pages[0]
    assert page.route == "/"
    assert any(isinstance(stmt, ShowText) for stmt in page.statements)
    assert any(isinstance(stmt, ShowTable) for stmt in page.statements)


def test_parse_form_with_submit_actions() -> None:
    source = (
        'app "Forms".\n'
        '\n'
        'page "Contact" at "/contact":\n'
        '  show form "Contact Us":\n'
        '    fields: name, email\n'
        '    on submit:\n'
        '      show toast "Thank you"\n'
        '  action "Notify":\n'
        '    when user clicks "Send":\n'
        '      show toast "Notified"\n'
    )
    app = Parser(source).parse()

    form = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, ShowForm))
    assert form.fields[0].name == "name"
    assert isinstance(form.on_submit_ops[0], ToastOperation)

    action = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, Action))
    assert action.trigger == 'user clicks "Send"'
    assert isinstance(action.operations[0], ToastOperation)


def test_parse_if_block() -> None:
    """If block should parse into a BinaryOp with attribute comparison."""

    source = (
        'app "IfTest".\n'
        '\n'
        'page "Admin" at "/admin":\n'
        '  if user.role == "admin":\n'
        '    show text "Welcome, admin!"\n'
        '    show text "You have full access"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    assert len(page.statements) == 1

    if_block = page.statements[0]
    assert isinstance(if_block, IfBlock)
    assert isinstance(if_block.condition, BinaryOp)
    assert if_block.condition.op == '=='
    assert isinstance(if_block.condition.left, AttributeRef)
    assert if_block.condition.left.base == 'user'
    assert if_block.condition.left.attr == 'role'
    assert isinstance(if_block.condition.right, Literal)
    assert if_block.condition.right.value == 'admin'
    assert len(if_block.body) == 2
    assert isinstance(if_block.body[0], ShowText)
    assert if_block.body[0].text == "Welcome, admin!"
    assert isinstance(if_block.body[1], ShowText)
    assert if_block.body[1].text == "You have full access"
    assert if_block.else_body is None


def test_parse_if_else_block() -> None:
    """If/else blocks should include parsed else_body statements."""

    source = (
        'app "IfElseTest".\n'
        '\n'
        'page "Access" at "/access":\n'
        '  if user.role == "admin":\n'
        '    show text "Welcome, admin!"\n'
        '  else:\n'
        '    show text "Access denied."\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    if_block = page.statements[0]

    assert isinstance(if_block, IfBlock)
    assert isinstance(if_block.condition, BinaryOp)
    assert if_block.condition.op == '=='
    assert len(if_block.body) == 1
    assert if_block.body[0].text == "Welcome, admin!"

    assert if_block.else_body is not None
    assert len(if_block.else_body) == 1
    assert isinstance(if_block.else_body[0], ShowText)
    assert if_block.else_body[0].text == "Access denied."


def test_parse_if_with_multiple_statements() -> None:
    """If blocks can hold multiple statements and numeric comparisons."""

    source = (
        'app "ComplexIf".\n'
        '\n'
        'page "Dashboard" at "/dash":\n'
        '  if count > 0:\n'
        '    show text "Data available"\n'
        '    show table "Results" from table results\n'
        '  else:\n'
        '    show text "No data found"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    if_block = page.statements[0]

    assert isinstance(if_block, IfBlock)
    assert isinstance(if_block.condition, BinaryOp)
    assert if_block.condition.op == '>'
    assert isinstance(if_block.condition.left, NameRef)
    assert if_block.condition.left.name == 'count'
    assert isinstance(if_block.condition.right, Literal)
    assert if_block.condition.right.value == 0
    assert len(if_block.body) == 2
    assert isinstance(if_block.body[0], ShowText)
    assert isinstance(if_block.body[1], ShowTable)


def test_parse_for_loop_dataset() -> None:
    """For loops over datasets should yield ForLoop nodes with dataset source."""

    source = (
        'app "ForTest".\n'
        '\n'
        'dataset "latest_orders" from table orders:\n'
        '  filter by: status == "NEW"\n'
        '\n'
        'page "Orders" at "/orders":\n'
        '  for order in dataset latest_orders:\n'
        '    show text "{order.id} – {order.total}"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    assert len(page.statements) == 1

    for_loop = page.statements[0]
    assert isinstance(for_loop, ForLoop)
    assert for_loop.loop_var == "order"
    assert for_loop.source_kind == "dataset"
    assert for_loop.source_name == "latest_orders"
    assert len(for_loop.body) == 1
    assert isinstance(for_loop.body[0], ShowText)
    assert for_loop.body[0].text == "{order.id} – {order.total}"


def test_parse_for_loop_table() -> None:
    """For loops can iterate over tables and include multiple statements."""

    source = (
        'app "ForTableTest".\n'
        '\n'
        'page "Users" at "/users":\n'
        '  for row in table orders:\n'
        '    show text "{row.id} – {row.status}"\n'
        '    show text "Total: {row.total}"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    for_loop = page.statements[0]

    assert isinstance(for_loop, ForLoop)
    assert for_loop.loop_var == "row"
    assert for_loop.source_kind == "table"
    assert for_loop.source_name == "orders"
    assert len(for_loop.body) == 2
    assert isinstance(for_loop.body[0], ShowText)
    assert isinstance(for_loop.body[1], ShowText)


def test_parse_show_table_from_frame() -> None:
    source = (
        'app "FrameTest".\n'
        '\n'
        'frame "TopUsers" from dataset active_users:\n'
        '  column id string\n'
        '\n'
        'page "Dash" at "/dash":\n'
        '  show table "Top" from frame TopUsers\n'
    )

    app = Parser(source).parse()

    page = app.pages[0]
    table = next(stmt for stmt in page.statements if isinstance(stmt, ShowTable))
    assert table.source_type == "frame"
    assert table.source == "TopUsers"


def test_parse_for_loop_frame() -> None:
    source = (
        'app "FrameLoop".\n'
        '\n'
        'frame "Recent" from dataset recent_events:\n'
        '  column event_id string\n'
        '\n'
        'page "Events" at "/events":\n'
        '  for event in frame Recent:\n'
        '    show text "{event.event_id}"\n'
    )

    app = Parser(source).parse()

    page = app.pages[0]
    loop = next(stmt for stmt in page.statements if isinstance(stmt, ForLoop))
    assert loop.source_kind == "frame"
    assert loop.source_name == "Recent"


def test_parse_ai_operations_in_form() -> None:
    source = (
        'app "AIApp".\n'
        '\n'
        'page "Predict" at "/predict":\n'
        '  show form "Inference":\n'
        '    fields: text\n'
        '    on submit:\n'
        '      call python "models.py" method "predict" with:\n'
        '        input = form.text\n'
        '        user_id = ctx:user.id\n'
        '      ask connector openai with:\n'
        '        prompt = form.text\n'
        '        temperature = 0.2\n'
        '      run chain summarize_chain with:\n'
        '        text = form.text\n'
    )

    app = Parser(source).parse()

    form = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, ShowForm))
    operations = form.on_submit_ops
    assert len(operations) == 3

    call_op = operations[0]
    assert isinstance(call_op, CallPythonOperation)
    assert call_op.module == "models.py"
    assert call_op.method == "predict"
    assert set(call_op.arguments.keys()) == {"input", "user_id"}

    connector_op = operations[1]
    assert isinstance(connector_op, AskConnectorOperation)
    assert connector_op.connector_name == "openai"
    assert set(connector_op.arguments.keys()) == {"prompt", "temperature"}

    chain_op = operations[2]
    assert isinstance(chain_op, RunChainOperation)
    assert chain_op.chain_name == "summarize_chain"
    assert set(chain_op.inputs.keys()) == {"text"}


def test_parse_run_prompt_operation() -> None:
    source = (
        'app "AIApp".\n'
        '\n'
        'page "Home" at "/":\n'
        '  action "Submit":\n'
        '    when form.submit:\n'
        '      run prompt summarize_ticket with:\n'
        '        ticket = form.ticket\n'
    )

    app = Parser(source).parse()

    action = next(stmt for stmt in app.pages[0].statements if isinstance(stmt, Action))
    operation = action.operations[0]
    assert isinstance(operation, RunPromptOperation)
    assert operation.prompt_name == "summarize_ticket"
    assert "ticket" in operation.arguments


def test_parse_nested_if_in_for() -> None:
    """Nested control flow should retain expression structure inside loops."""

    source = (
        'app "NestedTest".\n'
        '\n'
        'page "Report" at "/report":\n'
        '  for item in table items:\n'
        '    if item.status == "active":\n'
        '      show text "{item.name} is active"\n'
        '    else:\n'
        '      show text "{item.name} is inactive"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    for_loop = page.statements[0]

    assert isinstance(for_loop, ForLoop)
    assert len(for_loop.body) == 1

    if_block = for_loop.body[0]
    assert isinstance(if_block, IfBlock)
    assert isinstance(if_block.condition, BinaryOp)
    assert if_block.condition.op == '=='
    assert isinstance(if_block.condition.left, AttributeRef)
    assert if_block.condition.left.base == 'item'
    assert if_block.condition.left.attr == 'status'
    assert isinstance(if_block.condition.right, Literal)
    assert if_block.condition.right.value == 'active'
    assert len(if_block.body) == 1
    assert if_block.else_body is not None
    assert len(if_block.else_body) == 1


def test_parse_mixed_statements() -> None:
    """Pages can mix direct statements with control flow."""

    source = (
        'app "MixedTest".\n'
        '\n'
        'page "Mixed" at "/mixed":\n'
        '  show text "Header"\n'
        '  if user.logged_in:\n'
        '    show text "Welcome back!"\n'
        '  show table "Data" from table data\n'
        '  for row in table items:\n'
        '    show text "{row.name}"\n'
    )
    app = Parser(source).parse()

    page = app.pages[0]
    assert len(page.statements) == 4

    assert isinstance(page.statements[0], ShowText)
    assert isinstance(page.statements[1], IfBlock)
    cond = page.statements[1].condition
    assert isinstance(cond, AttributeRef)
    assert cond.base == 'user'
    assert cond.attr == 'logged_in'
    assert isinstance(page.statements[2], ShowTable)
    assert isinstance(page.statements[3], ForLoop)


def test_parse_page_reactive_and_predict_statement() -> None:
    """Pages can define refresh policies, layout, and inline predictions."""

    source = (
        'app "Realtime".\n'
        '\n'
        'page "Live Metrics" at "/live":\n'
        '  reactive: true\n'
        '  auto refresh every 5 seconds\n'
        '  layout:\n'
        '    template: grid\n'
        '  show chart "Active Users" from dataset active_users:\n'
        '    type: line\n'
        '    layout:\n'
        '      width: 2\n'
        '      variant: card\n'
        '  predict using model "churn" with dataset active_users into variable churn_prediction\n'
        '    params:\n'
        '      horizon: 7\n'
    )

    app = Parser(source).parse()
    page = app.pages[0]

    assert page.reactive is True
    assert isinstance(page.refresh_policy, RefreshPolicy)
    assert page.refresh_policy.interval_seconds == 5
    assert page.layout.get('template') == 'grid'

    chart = next(stmt for stmt in page.statements if isinstance(stmt, ShowChart))
    assert chart.chart_type == 'line'
    assert isinstance(chart.layout, LayoutMeta)
    assert chart.layout.width == 2
    assert chart.layout.variant == 'card'

    predict = next(stmt for stmt in page.statements if isinstance(stmt, PredictStatement))
    assert predict.model_name == 'churn'
    assert predict.input_kind == 'dataset'
    assert predict.input_ref == 'active_users'
    assert predict.assign.kind == 'variable'
    assert predict.assign.name == 'churn_prediction'
    assert predict.parameters['horizon'] == 7


def test_parse_page_reactive_header_short_form() -> None:
    source = (
        'app "Realtime".\n'
        '\n'
        'page "Live Dashboard" reactive:\n'
        '  show text "Hi"\n'
    )

    app = Parser(source).parse()
    page = app.pages[0]

    assert page.route == '/live-dashboard'
    assert page.reactive is True


def test_parse_page_reactive_kind_with_explicit_route() -> None:
    source = (
        'app "Realtime".\n'
        '\n'
        'page "Ops" at "/ops" kind reactive:\n'
        '  show text "Hi"\n'
    )

    app = Parser(source).parse()
    page = app.pages[0]

    assert page.route == '/ops'
    assert page.reactive is True


def test_parse_variable_assignment() -> None:
    source = (
        'app "Vars".\n'
        '\n'
        'page "Config" at "/cfg":\n'
        '  set tax_rate = 0.18\n'
        '  set title = "Dashboard"\n'
    )
    app = Parser(source).parse()
    page = app.pages[0]
    assignments = [stmt for stmt in page.statements if isinstance(stmt, VariableAssignment)]
    assert len(assignments) == 2

    tax_rate = assignments[0]
    assert tax_rate.name == 'tax_rate'
    assert isinstance(tax_rate.value, Literal)
    assert tax_rate.value.value == 0.18

    title = assignments[1]
    assert title.name == 'title'
    assert isinstance(title.value, Literal)
    assert title.value.value == 'Dashboard'


def test_parse_chart_layout_metadata() -> None:
    source = (
        'app "Visual".\n'
        '\n'
        'dataset "sales_summary" from table sales:\n'
        '  filter by: status == "OPEN"\n'
        '\n'
        'page "Dashboard" at "/dash":\n'
        '  show chart "Revenue" from dataset sales_summary:\n'
        '    x: month\n'
        '    y: revenue\n'
        '    layout:\n'
        '      width: 2\n'
        '      height: 1\n'
        '      variant: "card"\n'
        '      align: "center"\n'
        '      emphasis: "primary"\n'
    )

    app = Parser(source).parse()
    page = app.pages[0]
    chart = next(stmt for stmt in page.statements if isinstance(stmt, ShowChart))

    assert chart.layout is not None
    assert isinstance(chart.layout, LayoutMeta)
    assert chart.layout.width == 2
    assert chart.layout.height == 1
    assert chart.layout.variant == 'card'
    assert chart.layout.align == 'center'
    assert chart.layout.emphasis == 'primary'
