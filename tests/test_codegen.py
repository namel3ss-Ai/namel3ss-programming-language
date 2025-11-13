"""Tests for code generation helpers."""

import asyncio
import html
import json
import re
from pathlib import Path

from namel3ss.ast import LayoutMeta, RefreshPolicy, ShowChart
from namel3ss.codegen.backend import generate_backend
from namel3ss.codegen.frontend import build_chart_config, generate_site
from namel3ss.parser import Parser
from tests.backend_test_utils import load_backend_module as _load_backend_module


def _build_sample_app():
    source = (
        'app "Sample".\n'
        '\n'
        'dataset "monthly_sales" from table sales:\n'
        '  group by: month\n'
        '  sum: revenue as total_revenue\n'
        '  add column revenue_plus_tax = (revenue + tax)\n'
        '  filter by: revenue_plus_tax > 100\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show text "Hello"\n'
        '  show table "Users" from table users\n'
        '  show chart "Sales" from dataset monthly_sales\n'
        '  action "Refresh":\n'
        '    when user clicks "Refresh":\n'
        '      show toast "Refreshing"\n'
    )
    return Parser(source).parse()


def test_generate_site_creates_expected_files(tmp_path: Path) -> None:
    app = _build_sample_app()

    generate_site(app, tmp_path)

    assert (tmp_path / 'index.html').exists()
    assert (tmp_path / 'scripts.js').exists()
    assert (tmp_path / 'styles.css').exists()
    assert (tmp_path / 'index.html').read_text(encoding='utf-8').startswith('<!DOCTYPE html>')


def test_generate_site_includes_realtime_runtime(tmp_path: Path) -> None:
    app = _build_sample_app()
    app.pages[0].route = '/dashboard'
    app.pages[0].reactive = True
    app.pages[0].refresh_policy = RefreshPolicy(interval_seconds=12)

    generate_site(app, tmp_path, enable_realtime=True)

    scripts_js = (tmp_path / 'scripts.js').read_text(encoding='utf-8')
    assert 'global.N3Realtime' in scripts_js
    assert 'realtime.connectPage' in scripts_js

    dashboard_html = (tmp_path / 'dashboard.html').read_text(encoding='utf-8')
    assert "window.N3Realtime.connectPage('dashboard'" in dashboard_html


def test_generate_site_adds_ai_dashboards(tmp_path: Path) -> None:
    source = (
        'app "AI".\n'
        '\n'
        'model "baseline" using python:\n'
        '  registry:\n'
        '    version: "v1"\n'
        '\n'
        'experiment "compare":\n'
        '  variants:\n'
        '    local uses model baseline\n'
        '    registry uses model image_classifier\n'
        '  metrics:\n'
        '    accuracy goal 0.9\n'
    )

    app = Parser(source).parse()

    generate_site(app, tmp_path)

    models_html = (tmp_path / 'models.html').read_text(encoding='utf-8')
    experiments_html = (tmp_path / 'experiments.html').read_text(encoding='utf-8')

    assert 'Registered Models' in models_html
    assert 'baseline' in models_html
    assert '/api/pages/model/metrics' in models_html

    assert 'Experiments' in experiments_html
    assert '/api/experiments/' in experiments_html
    assert 'compare' in experiments_html


def test_generate_backend_creates_main_module(tmp_path: Path) -> None:
    app = _build_sample_app()

    generate_backend(app, tmp_path)

    main_path = tmp_path / 'main.py'
    init_path = tmp_path / '__init__.py'
    database_path = tmp_path / 'database.py'
    generated_dir = tmp_path / 'generated'
    schemas_path = generated_dir / 'schemas' / '__init__.py'
    runtime_path = generated_dir / 'runtime.py'
    pages_router_path = generated_dir / 'routers' / 'pages.py'
    helpers_path = generated_dir / 'helpers' / '__init__.py'
    assert main_path.exists()
    assert init_path.exists()
    assert database_path.exists()
    assert schemas_path.exists()
    assert runtime_path.exists()
    assert pages_router_path.exists()
    assert helpers_path.exists()
    custom_api_path = tmp_path / 'custom' / 'routes' / 'custom_api.py'
    assert custom_api_path.exists()

    main_content = main_path.read_text(encoding='utf-8')
    assert 'FastAPI' in main_content
    assert 'include_generated_routers(app)' in main_content
    assert '@app.get("/api/health")' in main_content
    assert 'predict_model = runtime.predict_model' in main_content
    # Ensure computed column serialized
    runtime_content = runtime_path.read_text(encoding='utf-8')
    assert 'computed_column' in runtime_content
    assert 'revenue_plus_tax' in runtime_content
    assert 'compile_dataset_to_sql' in runtime_content
    assert 'DATASETS' in runtime_content
    pages_router_content = pages_router_path.read_text(encoding='utf-8')
    assert 'response_model=ChartResponse' in pages_router_content
    assert '/api/pages/' in pages_router_content

    database_content = database_path.read_text(encoding='utf-8')
    assert 'DATABASE_URL_ENV' in database_content
    assert 'create_async_engine' in database_content

    schemas_content = schemas_path.read_text(encoding='utf-8')
    assert 'class TableResponse' in schemas_content
    assert 'pydantic' in schemas_content
    assert 'insights: Dict[str, Any]' in schemas_content
    assert 'class InsightResponse' in schemas_content
    assert 'class PredictionResponse' in schemas_content
    assert 'class ExperimentResult' in schemas_content


def test_backend_emits_model_registry_stub(tmp_path: Path) -> None:
    app = _build_sample_app()

    generate_backend(app, tmp_path)

    runtime_content = (tmp_path / 'generated' / 'runtime.py').read_text(encoding='utf-8')
    assert 'MODEL_REGISTRY' in runtime_content
    assert 'churn_classifier' in runtime_content
    assert 'image_classifier' in runtime_content
    assert 'def predict(' in runtime_content
    assert 'AI_CONNECTORS' in runtime_content
    assert 'AI_TEMPLATES' in runtime_content
    assert 'AI_CHAINS' in runtime_content
    assert 'AI_EXPERIMENTS' in runtime_content
    assert 'call_python_model(' in runtime_content
    assert 'call_llm_connector(' in runtime_content
    assert 'run_chain(' in runtime_content
    assert 'evaluate_experiment(' in runtime_content
    models_router_content = (tmp_path / 'generated' / 'routers' / 'models.py').read_text(encoding='utf-8')
    assert '@router.post("/api/models/{model_name}/predict"' in models_router_content
    assert '@router.post("/api/experiments/{slug}"' not in models_router_content
    experiments_router_content = (tmp_path / 'generated' / 'routers' / 'experiments.py').read_text(encoding='utf-8')
    assert '@router.get("/api/experiments/{slug}"' in experiments_router_content


def test_codegen_wires_chart_insight_reference(tmp_path: Path) -> None:
    source = (
        'app "Dash".\n'
        '\n'
        'dataset "monthly_sales" from table monthly_sales:\n'
        '  cache:\n'
        '    strategy: none\n'
        '\n'
        'insight "revenue_growth" from dataset monthly_sales:\n'
        '  logic:\n'
        '    emit narrative "Placeholder"\n'
        '\n'
    'page "Dashboard" at "/dashboard":\n'
    '  show chart "Revenue" from dataset monthly_sales\n'
        '    x: month\n'
        '    y: revenue\n'
        '    insight: "revenue_growth"\n'
    )

    app = Parser(source).parse()

    backend_dir = tmp_path / 'backend'
    frontend_dir = tmp_path / 'site'
    generate_backend(app, backend_dir)
    generate_site(app, frontend_dir)

    runtime_content = (backend_dir / 'generated' / 'runtime.py').read_text(encoding='utf-8')
    assert "'insight': 'revenue_growth'" in runtime_content
    pages_router_content = (backend_dir / 'generated' / 'routers' / 'pages.py').read_text(encoding='utf-8')
    assert "insight='revenue_growth'" in pages_router_content

    page_html = (frontend_dir / 'dashboard.html').read_text(encoding='utf-8')
    assert 'data-n3-insight="revenue_growth"' in page_html

    scripts_js = (frontend_dir / 'scripts.js').read_text(encoding='utf-8')
    assert 'config.insight = def.insight' in scripts_js
    assert 'data-n3-insight-ref' in scripts_js


def test_predict_stub_returns_deterministic_payload(tmp_path: Path, monkeypatch) -> None:
    app = _build_sample_app()
    backend_dir = tmp_path / 'backend_ml'
    generate_backend(app, backend_dir)

    with _load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        payload = {"feature_a": 1.0, "feature_b": 2.0}
        result = module.predict('image_classifier', payload)
        assert result['model'] == 'image_classifier'
        assert result['framework'] == 'pytorch'
        assert result['status'] == 'error'
        error_message = result['output']['error']
        assert isinstance(error_message, str)
        assert 'image model driver' in error_message
    api_result = asyncio.run(module.predict_model('image_classifier', payload))
    assert api_result['output']['status'] == 'error'
    assert api_result['output']['error'] == error_message


def test_backend_ai_helpers_return_deterministic_payloads(tmp_path: Path, monkeypatch) -> None:
    source = (
        'app "AIApp".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-stub"\n'
        '\n'
        'define template "summary":\n'
        '  prompt = "Summary: {input}"\n'
        '\n'
        'define chain "summarize_chain":\n'
        '  input -> template summary -> connector openai\n'
        '\n'
        'experiment "ai_eval":\n'
        '  variants:\n'
        '    chain_flow uses chain summarize_chain\n'
        '    model_flow uses model image_classifier\n'
        '  metrics:\n'
        '    quality goal 0.9\n'
        '    latency_ms goal 250\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Hello"\n'
    )

    app = Parser(source).parse()
    backend_dir = tmp_path / 'backend_ai'
    generate_backend(app, backend_dir)

    model_file = backend_dir / 'local_model.py'
    model_file.write_text(
        'def predict(input, user_id=None):\n'
        '    return {"echo": input, "user": user_id}\n',
        encoding='utf-8',
    )

    with _load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        python_result = module.call_python_model(str(model_file), 'predict', {'input': 'hello', 'user_id': 'abc'})
        assert python_result['status'] == 'ok'
        assert python_result['result']['echo'] == 'hello'
        assert python_result['result']['user'] == 'abc'

        monkeypatch.delenv('NAMEL3SS_ALLOW_STUBS', raising=False)
        fallback_result = module.call_python_model('missing.module', 'predict', {'input': 'x'})
        assert fallback_result['status'] == 'error'
        assert 'missing.module' in fallback_result['error']

        monkeypatch.setenv('NAMEL3SS_ALLOW_STUBS', '1')
        stub_result = module.call_python_model('missing.module', 'predict', {'input': 'x'})
        assert stub_result['status'] == 'stub'
        assert stub_result['result'] == 'stub_prediction'

        runtime_module = module.runtime
        connector_spec = runtime_module.AI_CONNECTORS['openai']
        connector_spec.setdefault('config', {})
        connector_spec['config'].update(
            {
                'endpoint': 'https://example.com/v1/chat',
                'api_key_env': 'OPENAI_API_KEY',
                'mode': 'chat',
            }
        )

        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

        def _fake_post(url, data, headers, timeout):
            assert headers['Authorization'].startswith('Bearer')
            payload = {
                'choices': [
                    {
                        'message': {
                            'content': 'LLM says hi',
                        }
                    }
                ],
                'usage': {'total_tokens': 12},
            }
            raw_text = json.dumps(payload)
            return 200, raw_text, payload

        if hasattr(module, '_http_post_json'):
            monkeypatch.setattr(module, '_http_post_json', _fake_post)
        monkeypatch.setattr(runtime_module, '_http_post_json', _fake_post)

        connector_result = module.call_llm_connector('openai', {'prompt': 'Hi there'})
        assert connector_result['status'] == 'ok'
        assert connector_result['result']['text'] == 'LLM says hi'
        assert connector_result['result']['json']['usage'] == {'total_tokens': 12}
        assert connector_result['provider'] == 'openai'
        assert connector_result['metadata']['http_status'] == 200

        chain_result = module.run_chain('summarize_chain', {'input': 'Important data'})
        assert chain_result['status'] in {'ok', 'partial'}
        assert chain_result['result'] is not None
        assert len(chain_result['steps']) == 2

    experiment_result = module.evaluate_experiment('ai_eval', {'input': {'text': 'Feedback'}})
    assert experiment_result['experiment'] == 'ai_eval'
    assert len(experiment_result['variants']) == 2
    variants_by_type = {entry['target_type']: entry for entry in experiment_result['variants']}
    assert set(variants_by_type.keys()) >= {'chain', 'model'}

    chain_variant = variants_by_type['chain']
    assert chain_variant['status'] in {'ok', 'partial'}
    assert chain_variant['result'] is not None

    model_variant = variants_by_type['model']
    assert model_variant['status'] == 'error'
    reported_error = model_variant.get('error') or ''
    if not reported_error and isinstance(model_variant.get('result'), dict):
        reported_error = model_variant['result'].get('output', {}).get('error', '')
    assert 'image model driver' in reported_error

    assert experiment_result['metrics'] is not None
    assert experiment_result['winner'] in {'chain_flow', 'model_flow', None}

def test_frontend_interpolates_text_variables(tmp_path: Path) -> None:
    source = (
        'app "TaxApp".\n'
        'set tax_rate = 0.2\n'
        'page "Home" at "/home":\n'
        '  show text "Tax is {tax_rate}"\n'
    )
    app = Parser(source).parse()

    generate_site(app, tmp_path)

    page_path = tmp_path / 'home.html'
    assert page_path.exists()
    html_content = page_path.read_text(encoding='utf-8')

    assert 'data-n3-text-template="Tax is {tax_rate}"' in html_content
    assert 'window.N3_VARS = vars;' in html_content
    assert '/api/pages/home' in html_content
    assert 'tpl.replace(/\\{([a-zA-Z_][a-zA-Z0-9_]*)\\}/g' in html_content


def test_frontend_renders_insight_widgets(tmp_path: Path) -> None:
    source = (
        'app "InsightUI".\n'
        '\n'
        'dataset "sales" from table sales:\n'
        '  cache:\n'
        '    strategy: none\n'
        '\n'
        'insight "summary" from dataset sales:\n'
        '  logic:\n'
        '    emit narrative "Summary ready"\n'
        '\n'
        'page "Home" at "/home":\n'
        '  show text "Hello"\n'
    )

    app = Parser(source).parse()
    generate_site(app, tmp_path)

    page_html = (tmp_path / 'home.html').read_text(encoding='utf-8')
    scripts_js = (tmp_path / 'scripts.js').read_text(encoding='utf-8')

    assert 'class="n3-insights"' in page_html
    assert 'data-n3-insight="metrics"' in page_html
    assert 'renderInsight' in scripts_js
    assert 'populateMetrics' in scripts_js
    assert 'populateNarratives' in scripts_js


def test_generate_backend_emits_insight_routes_and_registries(tmp_path: Path) -> None:
    source = (
        'app "Insightful".\n'
        '\n'
        'dataset "sales_summary" from table sales:\n'
        '  cache:\n'
        '    strategy: memory\n'
        '\n'
        'insight "top_regions" from dataset sales_summary:\n'
        '  logic:\n'
        '    emit narrative "Top regions outperform"\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show table "Summary" from dataset sales_summary\n'
    )
    app = Parser(source).parse()

    generate_backend(app, tmp_path)

    runtime_content = (tmp_path / 'generated' / 'runtime.py').read_text(encoding='utf-8')
    assert 'CONNECTORS: Dict[str, Dict[str, Any]]' in runtime_content
    assert 'INSIGHTS: Dict[str, Dict[str, Any]]' in runtime_content
    assert 'evaluate_insights_for_dataset' in runtime_content
    assert 'EMBED_INSIGHTS: bool = False' in runtime_content
    assert 'httpx.AsyncClient' in runtime_content

    insights_router_content = (tmp_path / 'generated' / 'routers' / 'insights.py').read_text(encoding='utf-8')
    assert '@router.get("/api/insights/{slug}"' in insights_router_content
    assert 'response_model=InsightResponse' in insights_router_content
    assert 'return evaluate_insight' in insights_router_content

    schemas_content = (tmp_path / 'generated' / 'schemas' / '__init__.py').read_text(encoding='utf-8')
    assert 'class InsightResponse' in schemas_content
    assert 'result: Dict[str, Any]' in schemas_content
    assert 'insight: Optional[str] = None' in schemas_content


def test_cli_train_and_deploy_commands(tmp_path: Path, capsys) -> None:
    source_path = tmp_path / 'ml_app.n3'
    source_path.write_text('app "MLApp".\n')

    from namel3ss import cli

    cli.main(['train', str(source_path), '--model', 'image_classifier'])
    train_output = capsys.readouterr().out
    assert 'Starting training pipeline for image_classifier' in train_output
    assert '[train:image_classifier]' in train_output
    assert 'Artifacts written to models/image_classifier.pt' in train_output
    train_status_line = next(
        (line for line in reversed(train_output.strip().splitlines()) if line.strip()),
        '{}',
    )
    train_status = json.loads(train_status_line)
    assert train_status.get('status') == 'ok'
    assert train_status.get('model') == 'image_classifier'

    cli.main(['deploy', str(source_path), '--model', 'image_classifier'])
    deploy_output = capsys.readouterr().out
    assert 'Publishing image_classifier artifact' in deploy_output
    assert 'Deployment completed: https://models.example.com/image_classifier/predict' in deploy_output
    deploy_status_line = next(
        (line for line in reversed(deploy_output.strip().splitlines()) if line.strip()),
        '{}',
    )
    deploy_status = json.loads(deploy_status_line)
    assert deploy_status.get('status') == 'ok'
    assert deploy_status.get('model') == 'image_classifier'
    assert deploy_status.get('endpoint') == 'https://models.example.com/image_classifier/predict'

    def test_generated_backend_runs_insight_runtime(tmp_path: Path, monkeypatch) -> None:
        source = (
            'app "Metrics".\n'
            '\n'
            'dataset "sales" from table sales:\n'
            '  cache:\n'
            '    strategy: none\n'
            '\n'
            'insight "sales_kpi" from dataset sales:\n'
            '  logic:\n'
            '    regions = rows\n'
            '    emit narrative "Sales updated"\n'
            '  metrics:\n'
            '    total_revenue:\n'
            '      label: "Total Revenue"\n'
            '      value: sum("total")\n'
            '      baseline: avg("total")\n'
            '      unit: "USD"\n'
            '  thresholds:\n'
            '    revenue_goal:\n'
            '      metric: total_revenue\n'
            '      operator: ">="\n'
            '      value: 200\n'
            '      level: critical\n'
            '  narratives:\n'
            '    summary:\n'
            '      template: "Total revenue {metrics.total_revenue.formatted}"\n'
            '  expose:\n'
            '    first_region: regions[0].region\n'
        )

        app = Parser(source).parse()
        backend_dir = tmp_path / 'backend_insight'
        generate_backend(app, backend_dir)

        with _load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
            spec = module.INSIGHTS['sales_kpi']
            rows = [
                {'region': 'North', 'total': 250},
                {'region': 'South', 'total': 150},
            ]
            base_context = module.build_context(None)
            result = module._run_insight(spec, rows, base_context)

            metrics = result['metrics']
            assert metrics and metrics[0]['name'] == 'total_revenue'
            assert metrics[0]['value'] == 400
            assert metrics[0]['baseline'] == 200
            assert metrics[0]['trend'] == 'up'
            assert metrics[0]['formatted'].startswith('$')

            alerts = result['alerts_list']
            assert alerts and alerts[0]['triggered'] is True
            assert alerts[0]['level'] == 'critical'

            narratives = result['narratives']
            assert narratives and 'Total revenue' in narratives[0]['text']

            variables = result['variables']
            assert variables.get('first_region') == 'North'

def test_generate_backend_with_embed_insights_true(tmp_path: Path) -> None:
    source = (
        'app "Embed".\n'
        '\n'
        'dataset "metrics" from table metrics:\n'
        '  group by: category\n'
        '\n'
        'insight "trend" from dataset metrics:\n'
        '  logic:\n'
        '    emit narrative "Trends are steady"\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show chart "Metrics" from dataset metrics\n'
    )
    app = Parser(source).parse()

    generate_backend(app, tmp_path, embed_insights=True)

    runtime_content = (tmp_path / 'generated' / 'runtime.py').read_text(encoding='utf-8')
    assert 'EMBED_INSIGHTS: bool = True' in runtime_content
    assert 'positive_label' in runtime_content  # prediction helper present
    pages_router_content = (tmp_path / 'generated' / 'routers' / 'pages.py').read_text(encoding='utf-8')
    assert 'insight_results: Dict[str, Any] = {}' in pages_router_content
    assert 'insights=insight_results' in pages_router_content
    assert "if runtime.EMBED_INSIGHTS and dataset.get('name')" in pages_router_content


def test_generate_backend_includes_context_runtime_helpers(tmp_path: Path) -> None:
    source = (
        'app "EnvApp".\n'
        '\n'
        'dataset "orders" from rest "ORDERS_API" endpoint "https://api.example.com/orders":\n'
        '  filter by: status == "OPEN"\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show table "Orders" from dataset orders\n'
    )
    app = Parser(source).parse()

    generate_backend(app, tmp_path)

    runtime_content = (tmp_path / 'generated' / 'runtime.py').read_text(encoding='utf-8')

    assert 'CONTEXT_MARKER_KEY = "__context__"' in runtime_content
    assert 'class ContextRegistry' in runtime_content
    assert 'def _resolve_context_path' in runtime_content
    assert 'def _render_template_value' in runtime_content


def test_generated_backend_resolves_env_placeholders(tmp_path: Path, monkeypatch) -> None:
    source = (
        'app "RuntimeEnv".\n'
        '\n'
        'dataset "orders" from rest "ORDERS_API" endpoint "https://api.example.com/orders":\n'
        '  with option headers.Authorization env:NAMEL3SS_ORDERS_TOKEN\n'
        '  with option params.api_key ${NAMEL3SS_ORDERS_TOKEN}\n'
        '  with option params.user_id ctx:user.id\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  show text "Welcome {ctx:user.name}! Token {env:NAMEL3SS_ORDERS_TOKEN}"\n'
        '  show table "Orders" from dataset orders\n'
    )
    app = Parser(source).parse()

    backend_dir = tmp_path / 'backend'
    generate_backend(app, backend_dir)

    monkeypatch.setenv('NAMEL3SS_ORDERS_TOKEN', 'super-secret')

    with _load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime_module = module.runtime
        dataset_key = next(iter(runtime_module.DATASETS))
        dataset_spec = runtime_module.DATASETS[dataset_key]

        auth_marker = dataset_spec['connector']['options']['headers']['Authorization']
        assert auth_marker['__context__']['scope'] == 'env'
        assert auth_marker['__context__']['path'] == ['NAMEL3SS_ORDERS_TOKEN']

        user_marker = dataset_spec['connector']['options']['params']['user_id']
        assert user_marker['__context__']['scope'] == 'ctx'
        assert user_marker['__context__']['path'] == ['user', 'id']

        runtime_module.CONTEXT.set_global({'user': {'name': 'Ava', 'id': 'u-42'}})
        context = runtime_module.build_context(dataset_spec.get('context_page'))
        connector = runtime_module._resolve_connector(dataset_spec, context)

        assert connector['options']['headers']['Authorization'] == 'super-secret'
        assert connector['options']['params']['api_key'] == 'super-secret'
        assert connector['options']['params']['user_id'] == 'u-42'

        page_func = getattr(runtime_module, 'page_dashboard_0')
        result = asyncio.run(page_func())
        assert any(comp['type'] == 'table' for comp in result['components'])
        text_components = [comp for comp in result['components'] if comp['type'] == 'text']
        assert text_components
        assert text_components[0]['text'] == 'Welcome Ava! Token super-secret'


def test_generated_backend_handles_loop_control_flow(tmp_path: Path, monkeypatch) -> None:
    source = (
        'app "ControlFlow".\n'
        '\n'
        'dataset "orders" from table orders:\n'
        '  filter by: status == "OPEN"\n'
        '\n'
        'page "Dashboard" at "/":\n'
        '  set seen = 0\n'
        '  for order in dataset orders:\n'
        '    if order.value == 20:\n'
        '      continue\n'
        '    set seen = seen + 1\n'
        '    show text "Order {vars:order.id} seen {vars:seen}"\n'
        '    if seen == 2:\n'
        '      break\n'
        '  while seen < 3:\n'
        '    set seen = seen + 1\n'
        '  show text "Final total {vars:seen}"\n'
    )

    app = Parser(source).parse()
    backend_dir = tmp_path / 'backend_loops'
    generate_backend(app, backend_dir)

    with _load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime_module = module.runtime
        page_func = getattr(runtime_module, 'page_dashboard_0')
        result = asyncio.run(page_func())

        components = result['components']
        texts = [comp['text'] for comp in components if comp['type'] == 'text']

        assert texts == ['Order 1 seen 1', 'Order 3 seen 2', 'Final total 3']
        assert all(comp['type'] == 'text' for comp in components)


def test_frontend_generates_widget_bootstrap_with_layout(tmp_path: Path) -> None:
    source = (
        'app "Viz".\n'
        '\n'
        'dataset "sales_summary" from table sales:\n'
        '  filter by: status == "OPEN"\n'
        '\n'
        'dataset "users" from table users:\n'
        '  filter by: active == true\n'
        '\n'
        'page "Dashboard" at "/dashboard":\n'
        '  show chart "Revenue" from dataset sales_summary:\n'
        '    layout:\n'
        '      variant: "card"\n'
        '      width: 2\n'
        '      align: "center"\n'
        '  show table "Users" from dataset users:\n'
        '    layout:\n'
        '      variant: "dense"\n'
        '      emphasis: "secondary"\n'
    )

    app = Parser(source).parse()
    generate_site(app, tmp_path)

    page_html = (tmp_path / 'dashboard.html').read_text(encoding='utf-8')
    scripts_js = (tmp_path / 'scripts.js').read_text(encoding='utf-8')

    assert 'N3Widgets.bootstrap' in page_html
    assert 'renderChart' in scripts_js
    assert 'renderTable' in scripts_js

    layout_match = re.search(r'data-n3-layout="([^\"]+)"', page_html)
    assert layout_match is not None
    layout_payload = json.loads(html.unescape(layout_match.group(1)))
    assert layout_payload['variant'] == 'card'
    assert layout_payload['width'] == 2


def test_build_chart_config_handles_empty_dataset() -> None:
    chart = ShowChart(
        heading='Sales Trend',
        title='Trend',
        source_type='dataset',
        source='sales',
        chart_type='line',
        layout=LayoutMeta(variant='card', emphasis='primary'),
    )

    config = build_chart_config(chart, {}, theme='dark')

    assert config['type'] == 'line'
    assert config['data']['labels'] == []
    assert config['data']['datasets'] == []
    title_opts = config['options']['plugins']['title']
    assert title_opts['display'] is True
    assert title_opts['font']['weight'] == '600'
    assert title_opts['color'] == '#f5f5f5'
    legend_opts = config['options']['plugins']['legend']
    assert legend_opts['display'] is False
