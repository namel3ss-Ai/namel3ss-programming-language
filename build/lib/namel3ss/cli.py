"""
Command line interface for Namel3ss.

This module exposes a CLI with subcommands for building and running
Namel3ss applications. It provides:

- build: Generate static frontend and/or backend scaffold
- run: Start a development server with hot reload

The CLI is the primary entrypoint for working with .n3 files.
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ast import App, Chain, Experiment
from .parser import Parser, N3SyntaxError
from .codegen import generate_backend, generate_site
from .ml import get_default_model_registry
from .utils import iter_dependency_reports

ENV_ALIAS_MAP = {
    "production": ".env.prod",
    "prod": ".env.prod",
    "development": ".env.dev",
    "dev": ".env.dev",
    "local": ".env.local",
    "locally": ".env.local",
    "test": ".env.test",
}

DEFAULT_MODEL_REGISTRY = get_default_model_registry()

_RUN_ENV_PREPOSITIONS = {"using", "in", "on", "with"}


def _find_first_n3_file() -> Optional[Path]:
    candidates = sorted(Path.cwd().glob('*.n3'))
    return candidates[0] if candidates else None


def _resolve_env_reference(token: str) -> Optional[str]:
    normalized = token.strip().strip('"\'')
    if not normalized:
        return None
    lower = normalized.lower()
    if lower in ENV_ALIAS_MAP:
        return ENV_ALIAS_MAP[lower]
    if normalized.startswith('.env'):
        return normalized
    return None


def _normalize_run_command_args(argv: List[str]) -> List[str]:
    if not argv or argv[0] != 'run':
        return argv

    new_args: List[str] = ['run']
    env_specs: List[str] = []
    tokens = argv[1:]
    idx = 0
    file_seen = False

    while idx < len(tokens):
        token = tokens[idx]
        lower = token.lower()

        if lower in _RUN_ENV_PREPOSITIONS and idx + 1 < len(tokens):
            candidate = tokens[idx + 1]
            resolved = _resolve_env_reference(candidate)
            if resolved is not None:
                env_specs.append(resolved)
                idx += 2
                continue

        resolved_single = _resolve_env_reference(token)
        if resolved_single is not None:
            env_specs.append(resolved_single)
            idx += 1
            continue

        if token.endswith('.n3') and not file_seen:
            new_args.append(token)
            file_seen = True
            idx += 1
            continue

        new_args.append(token)
        idx += 1

    for env_entry in env_specs:
        new_args.extend(['--env', env_entry])

    return new_args


def _apply_env_overrides(overrides: Optional[List[str]]) -> None:
    """Set environment variables provided as KEY=VALUE strings."""

    if not overrides:
        return

    for entry in overrides:
        if not entry:
            continue

        if '=' in entry:
            key, value = entry.split('=', 1)
            os.environ[key] = value
            continue

        _load_env_file(entry)


def _load_env_file(path_str: str) -> None:
    env_path = Path(path_str)
    if not env_path.exists():
        print(f"Error: environment file not found: {env_path}", file=sys.stderr)
        sys.exit(1)

    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
    except Exception as exc:  # pragma: no cover - IO error handling
        print(f"Error reading environment file {env_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _load_cli_app(source_path: Path) -> App:
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    source = source_path.read_text(encoding='utf-8')
    try:
        return Parser(source).parse()
    except N3SyntaxError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def _slugify_model_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return normalized.lower() or "model"


def _resolve_model_spec(app: App, model_name: str) -> Dict[str, Any]:
    for model in app.models:
        if model.name == model_name:
            registry_info = model.registry
            spec = {
                "type": model.model_type,
                "framework": model.engine or model.model_type,
                "version": registry_info.version or "v1",
                "metrics": dict(registry_info.metrics or {}),
                "metadata": dict(registry_info.metadata or {}),
            }
            return spec

    fallback = DEFAULT_MODEL_REGISTRY.get(model_name)
    if fallback:
        return dict(fallback)

    return {
        "type": "custom",
        "framework": "unknown",
        "version": "v1",
        "metrics": {},
        "metadata": {},
    }


def _backend_summary_lines(app: App) -> List[str]:
    connectors = {
        str(ds.connector.connector_name or ds.source or ds.name)
        for ds in app.datasets
        if ds.connector is not None
    }
    connectors.update(conn.name for conn in app.connectors)
    dataset_count = len(app.datasets)
    insight_count = len(app.insights)
    model_count = len(app.models)
    template_count = len(app.templates)
    chain_count = len(app.chains)
    experiment_count = len(app.experiments)

    return [
        f"✓ {dataset_count} {_plural('dataset', dataset_count)} available",
        f"✓ {len(connectors)} {_plural('connector', len(connectors))} registered",
        f"✓ {insight_count} {_plural('insight', insight_count)} routed",
        f"✓ {model_count} {_plural('model', model_count)} declared",
        f"✓ {template_count} {_plural('template', template_count)} cached",
        f"✓ {chain_count} {_plural('chain', chain_count)} composed",
        f"✓ {experiment_count} {_plural('experiment', experiment_count)} queued",
    ]


def _find_chain(app: App, name: str) -> Optional[Chain]:
    for chain in app.chains:
        if chain.name == name:
            return chain
    return None


def _find_experiment(app: App, name: str) -> Optional[Experiment]:
    for experiment in app.experiments:
        if experiment.name == name:
            return experiment
    return None


def _simulate_chain_run(chain: Chain, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a stub ``PredictionResponse`` for CLI previews."""

    request_payload = dict(payload or {})
    steps = []
    for index, step in enumerate(chain.steps, start=1):
        steps.append({
            "index": index,
            "kind": step.kind,
            "target": step.target,
            "options": dict(step.options),
        })

    metadata: Dict[str, Any] = {
        "chain": chain.name,
        "input_key": chain.input_key,
        "steps": steps,
    }

    version = None
    framework = None
    if isinstance(chain.metadata, dict):
        version = chain.metadata.get("version")
        framework = chain.metadata.get("framework") or chain.metadata.get("engine")
        if chain.metadata:
            metadata["chain_metadata"] = dict(chain.metadata)

    return {
        "model": chain.name,
        "version": version,
        "framework": framework or "chain",
        "input": request_payload,
        "output": {
            "status": "stub",
            "result": f"{chain.name}_output",
        },
        "explanations": {},
        "metadata": metadata,
    }


def _simulate_experiment(
    experiment: Experiment,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a stub ``ExperimentResult`` for CLI previews."""

    request_payload = dict(payload or {})
    variants = []
    for index, variant in enumerate(experiment.variants, start=1):
        variants.append({
            "name": variant.name or f"variant_{index}",
            "target_type": variant.target_type,
            "target_name": variant.target_name,
            "score": round(0.68 + 0.045 * index, 3),
            "result": {
                "status": "stub",
                "summary": f"Simulated outcome for {variant.target_type}:{variant.target_name}",
            },
            "config": dict(variant.config),
        })

    metrics = []
    for index, metric in enumerate(experiment.metrics, start=1):
        metrics.append({
            "name": metric.name or f"metric_{index}",
            "value": round(0.74 + 0.03 * index, 3),
            "goal": metric.goal,
            "source_kind": metric.source_kind,
            "source_name": metric.source_name,
            "metadata": dict(metric.metadata),
        })

    leaderboard = sorted(
        (variant for variant in variants if variant.get("score") is not None),
        key=lambda item: item["score"],
        reverse=True,
    )
    winner = leaderboard[0]["name"] if leaderboard else None

    metadata = dict(experiment.metadata)
    if experiment.description:
        metadata.setdefault("description", experiment.description)

    return {
        "experiment": experiment.name,
        "slug": _slugify_model_name(experiment.name),
        "status": "stub",
        "variants": variants,
        "metrics": metrics,
        "leaderboard": [entry["name"] for entry in leaderboard],
        "winner": winner,
        "inputs": request_payload,
        "metadata": metadata,
    }


def _print_prediction_response_text(response: Dict[str, Any]) -> None:
    """Pretty-print a stub prediction response for humans."""

    framework = response.get("framework") or "n/a"
    version = response.get("version") or "n/a"
    print(f"Model: {response.get('model')} (framework: {framework}, version: {version})")
    if response.get("input"):
        print("Input payload:")
        print(json.dumps(response["input"], indent=2))
    print("Output:")
    print(json.dumps(response.get("output", {}), indent=2))
    metadata = response.get("metadata", {})
    if metadata:
        print("Metadata:")
        print(json.dumps(metadata, indent=2))


def _print_experiment_result_text(result: Dict[str, Any]) -> None:
    """Pretty-print a stub experiment result for humans."""

    description = result.get("metadata", {}).get("description")
    print(f"Experiment: {result.get('experiment')} (slug: {result.get('slug')})")
    print(f"Status: {result.get('status')}")
    if description:
        print(f"Description: {description}")
    print(f"Winner: {result.get('winner') or 'n/a'}")

    variants = result.get("variants") or []
    if variants:
        print("Variants:")
        for entry in variants:
            target = f"{entry.get('target_type')}:{entry.get('target_name') or 'default'}"
            score = entry.get("score")
            score_display = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            print(
                f"  - {entry.get('name')} ({target}) score={score_display}"
            )

    metrics = result.get("metrics") or []
    if metrics:
        print("Metrics:")
        for metric in metrics:
            source_kind = metric.get("source_kind") or "source"
            source_name = metric.get("source_name") or "n/a"
            value = metric.get("value")
            value_display = f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
            goal_display = metric.get("goal") or "n/a"
            print(
                f"  - {metric.get('name')}: {value_display} "
                f"goal={goal_display} "
                f"source={source_kind}:{source_name}"
            )

    if result.get("inputs"):
        print("Inputs:")
        print(json.dumps(result["inputs"], indent=2))


def prepare_backend(
    source_path: Path,
    backend_dir: str,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
) -> App:
    """Parse N3 file and generate backend scaffold.
    
    Parameters
    ----------
    source_path : Path
        Path to the .n3 source file
    backend_dir : str
        Directory where backend scaffold will be generated
    embed_insights : bool, optional
        Whether to embed evaluated insight payloads into dataset endpoints
        generated for tables and charts. Default is ``False``.
        
    Returns
    -------
    App
        The parsed application AST
        
    Raises
    ------
    N3SyntaxError
        If the source file contains syntax errors
    FileNotFoundError
        If the source file does not exist
    """
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")
    
    source = source_path.read_text(encoding='utf-8')
    app: App = Parser(source).parse()
    
    # Generate backend scaffold
    generate_backend(
        app,
        backend_dir,
        embed_insights=embed_insights,
        enable_realtime=enable_realtime,
    )
    
    return app


def check_uvicorn_available() -> bool:
    """Check if uvicorn is installed and importable.
    
    Returns
    -------
    bool
        True if uvicorn is available, False otherwise
    """
    try:
        import uvicorn  # type: ignore
        return True
    except ImportError:
        return False


def run_dev_server(
    source_path: Path,
    backend_dir: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
) -> None:
    """Run a development server for a Namel3ss app.
    
    This function:
    1. Parses the .n3 file
    2. Generates a backend scaffold
    3. Starts a uvicorn dev server with hot reload
    
    Parameters
    ----------
    source_path : Path
        Path to the .n3 source file
    backend_dir : Optional[str]
        Directory for backend scaffold. If None, uses a temp directory
    host : str
        Host to bind the server to
    port : int
        Port to bind the server to
    reload : bool
        Whether to enable hot reload
    embed_insights : bool, optional
        Forwarded to backend generation to control insight embedding.
    """
    # Check uvicorn availability first
    if not check_uvicorn_available():
        print(
            "Error: uvicorn is not installed.\n"
            "Please install it with: pip install uvicorn[standard]",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Import uvicorn only after checking it's available
    import uvicorn  # type: ignore
    
    # Use temp directory if none specified
    if backend_dir is None:
        backend_dir = os.path.join(tempfile.gettempdir(), '.namel3ss_dev_backend')
    
    try:
        # Prepare backend
        app = prepare_backend(
            source_path,
            backend_dir,
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
        )
        
        print(f"✓ Parsed: {app.name}")
        print(f"✓ Backend generated in: {backend_dir}")
        for line in _backend_summary_lines(app):
            print(line)
        if enable_realtime:
            print("✓ Realtime streaming enabled (ws://{host}:{port}/ws/pages/<slug>)".format(host=host, port=port))
        print(f"\nNamel3ss dev server running at http://{host}:{port}")
        print("Press CTRL+C to stop\n")
        
        # Change to backend directory so imports work
        original_dir = os.getcwd()
        os.chdir(backend_dir)
        
        try:
            # Start uvicorn dev server
            uvicorn.run(
                "main:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    except N3SyntaxError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    """Handle the 'build' subcommand."""
    source_path = Path(args.file)
    
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)
    
    source = source_path.read_text(encoding='utf-8')
    
    try:
        app: App = Parser(source).parse()
    except N3SyntaxError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    
    if args.print_ast:
        # Pretty print AST as JSON
        def default(o):
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        print(json.dumps(app, default=default, indent=2))
        return
    
    # Generate static site
    realtime_flag = getattr(args, 'realtime', False)

    if not args.backend_only:
        out_dir = args.out
        generate_site(app, out_dir, enable_realtime=realtime_flag)
        print(f"✓ Static site generated in {out_dir}")
    
    # Generate backend if requested
    if args.build_backend or args.backend_only:
        backend_dir = args.backend_out
        embed_flag = getattr(args, 'embed_insights', False)
        _apply_env_overrides(getattr(args, 'env', []))
        generate_backend(
            app,
            backend_dir,
            embed_insights=embed_flag,
            enable_realtime=realtime_flag,
        )
        print(f"✓ Backend scaffold generated in {backend_dir}")
        for line in _backend_summary_lines(app):
            print(line)


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    namespace = vars(args)
    target = namespace.get('target')
    _apply_env_overrides(namespace.get('env', []))
    force_dev = namespace.get('dev', False)
    chain_mode = False
    if not force_dev and target:
        if target.endswith('.n3'):
            chain_mode = False
        else:
            potential_path = Path(target)
            chain_mode = not potential_path.exists()
    elif not target:
        chain_mode = False

    if chain_mode:
        chain_name = target or ''
        source_arg = namespace.get('file')
        if source_arg is None:
            default_file = _find_first_n3_file()
            if default_file is None:
                print("Error: no .n3 file found to resolve chain execution.", file=sys.stderr)
                sys.exit(1)
            source_arg = str(default_file)
        source_path = Path(source_arg)
        app = _load_cli_app(source_path)
        chain = _find_chain(app, chain_name)
        if chain is None:
            available = ", ".join(sorted(c.name for c in app.chains)) or "none"
            print(
                f"Error: chain '{chain_name}' not found. Available chains: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        result = _simulate_chain_run(chain)
        if namespace.get('json', False):
            print(json.dumps(result, indent=2))
        else:
            _print_prediction_response_text(result)
        return

    source_path_arg = None
    if target and (target.endswith('.n3') or Path(target).exists()):
        source_path_arg = target
    if source_path_arg is None:
        fallback = namespace.get('file')
        if fallback:
            source_path_arg = fallback
        else:
            default_file = _find_first_n3_file()
            if default_file is None:
                print("Error: no .n3 file found in the current directory.", file=sys.stderr)
                sys.exit(1)
            source_path_arg = str(default_file)

    source_path = Path(source_path_arg)
    embed_flag = namespace.get('embed_insights', False)
    realtime_flag = namespace.get('realtime', False)
    run_dev_server(
        source_path,
        backend_dir=namespace.get('backend_out'),
        host=namespace.get('host', '127.0.0.1'),
        port=namespace.get('port', 8000),
        reload=not namespace.get('no_reload', False),
        embed_insights=embed_flag,
        enable_realtime=realtime_flag,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' subcommand for experiments."""

    experiment_name = args.experiment
    source_arg = args.file
    if source_arg is None:
        default_file = _find_first_n3_file()
        if default_file is None:
            print("Error: no .n3 file found to evaluate experiments.", file=sys.stderr)
            sys.exit(1)
        source_arg = str(default_file)

    app = _load_cli_app(Path(source_arg))
    experiment = _find_experiment(app, experiment_name)
    if experiment is None:
        available = ", ".join(sorted(exp.name for exp in app.experiments)) or "none"
        print(
            f"Error: experiment '{experiment_name}' not found. Available experiments: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = _simulate_experiment(experiment)
    if getattr(args, 'format', 'json') == 'text':
        _print_experiment_result_text(result)
        return

    print(json.dumps(result, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' subcommand using optional user hooks."""

    source_path = Path(args.file)
    app = _load_cli_app(source_path)
    model_name = args.model
    spec = _resolve_model_spec(app, model_name)
    framework = spec.get("framework", "unknown")
    metadata = spec.get("metadata", {})
    trainer_hook: Optional[str] = metadata.get("trainer")
    if trainer_hook:
        try:
            module_path, _, attr = trainer_hook.partition(":")
            if module_path and attr:
                module = importlib.import_module(module_path)
                trainer = getattr(module, attr, None)
                if callable(trainer):
                    trainer(model_name, spec, args)
                    print(f"Training hook completed for '{model_name}'.")
                    return
        except Exception as exc:
            print(f"Warning: trainer hook failed ({exc}). Falling back to stub.")

    print(f"Training model '{model_name}' (framework: {framework})")
    loss_schedule = metadata.get("loss_schedule") or [0.12, 0.10, 0.09, 0.08, 0.07]
    total_epochs = len(loss_schedule)
    for index, loss in enumerate(loss_schedule, start=1):
        print(f"Epoch {index}/{total_epochs} - loss={float(loss):.2f} (stub)")
    print("Metrics saved to registry.")


def cmd_deploy(args: argparse.Namespace) -> None:
    """Handle the 'deploy' subcommand invoking optional user hooks."""

    source_path = Path(args.file)
    app = _load_cli_app(source_path)
    model_name = args.model
    spec = _resolve_model_spec(app, model_name)
    slug = _slugify_model_name(model_name)
    version = spec.get("version", "v1")
    metadata = spec.get("metadata", {})
    deployer_hook: Optional[str] = metadata.get("deployer")
    if deployer_hook:
        try:
            module_path, _, attr = deployer_hook.partition(":")
            if module_path and attr:
                module = importlib.import_module(module_path)
                deployer = getattr(module, attr, None)
                if callable(deployer):
                    endpoint = deployer(model_name, spec, args)
                    if endpoint:
                        print(f"Model '{model_name}' deployed at {endpoint}")
                    else:
                        print(f"Model '{model_name}' deployment hook executed.")
                    return
        except Exception as exc:
            print(f"Warning: deployer hook failed ({exc}). Falling back to stub.")

    print(f"Model '{model_name}' deployed at /api/models/{slug}/predict")
    print(f"Framework: {spec.get('framework', 'unknown')} | version: {version} (stub deploy)")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report the availability of core and optional dependencies."""

    reports = iter_dependency_reports()
    core_missing = False

    for report in reports:
        prefix = "Optional" if report.optional else "Core"
        status = "available" if not report.missing else "missing"
        symbol = "✓" if status == "available" else "✗"
        print(f"{symbol} {prefix} {report.title}: {status}")
        if report.missing:
            missing_modules = ", ".join(report.missing)
            print(f"    Missing modules: {missing_modules}")
            if report.advice:
                print(f"    → {report.advice}")
            if not report.optional:
                core_missing = True

    if core_missing:
        raise SystemExit(1)


def main(argv: Optional[list] = None) -> None:
    """Main CLI entrypoint with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Namel3ss (N3) language compiler – build full‑stack apps in plain English",
        prog="namel3ss"
    )
    
    # Check if old-style invocation (no subcommand, just a file)
    # For backward compatibility
    if argv is None:
        argv = sys.argv[1:]
    
    # If first arg looks like a file and not a known subcommand, use legacy mode
    if (len(argv) > 0 and 
        not argv[0].startswith('-') and 
        argv[0] not in ['build', 'run', 'help'] and
        (argv[0].endswith('.n3') or Path(argv[0]).exists())):
        # Legacy mode: treat as build command
        print("Note: Using legacy invocation. Consider using 'namel3ss build' instead.", file=sys.stderr)
        argv = ['build'] + argv
    
    if argv and argv[0] == 'run':
        argv = _normalize_run_command_args(argv)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build subcommand
    build_parser = subparsers.add_parser(
        'build',
        help='Generate static site and/or backend scaffold'
    )
    build_parser.add_argument('file', help='Path to the .n3 source file')
    build_parser.add_argument(
        '--out', '-o', default='build', help='Output directory for static files'
    )
    build_parser.add_argument(
        '--print-ast', action='store_true', help='Print the parsed AST and exit'
    )
    build_parser.add_argument(
        '--build-backend', action='store_true', help='Also generate FastAPI backend scaffold'
    )
    build_parser.add_argument(
        '--realtime', action='store_true', help='Enable realtime websocket scaffolding'
    )
    build_parser.add_argument(
        '--backend-only', action='store_true', help='Only generate backend, skip static site'
    )
    build_parser.add_argument(
        '--backend-out', default='backend_build', help='Output directory for backend scaffold'
    )
    build_parser.add_argument(
        '--embed-insights', action='store_true', help='Embed insight results directly in dataset responses'
    )
    build_parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set environment variable for backend generation (may be provided multiple times)'
    )
    build_parser.set_defaults(func=cmd_build)
    
    # Run subcommand
    run_parser = subparsers.add_parser(
        'run',
        help='Execute an AI chain or launch the development server'
    )
    run_parser.add_argument(
        'target',
        nargs='?',
        help='Chain name to execute or path to the .n3 source file'
    )
    run_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Explicit .n3 source file to use when running a chain'
    )
    run_parser.add_argument(
        '--dev',
        action='store_true',
        help='Force dev server mode even if the target looks like a chain'
    )
    run_parser.add_argument(
        '--backend-out',
        default=None,
        help='Output directory for backend scaffold (default: temp directory)'
    )
    run_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind server to (default: 127.0.0.1)'
    )
    run_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind server to (default: 8000)'
    )
    run_parser.add_argument(
        '--no-reload',
        action='store_true',
        help='Disable hot reload'
    )
    run_parser.add_argument(
        '--embed-insights',
        action='store_true',
        help='Embed insight results directly in dataset responses'
    )
    run_parser.add_argument(
        '--realtime',
        action='store_true',
        help='Enable realtime websocket scaffolding'
    )
    run_parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set environment variable before launching the dev server (may be provided multiple times)'
    )
    run_parser.add_argument(
        '--json',
        action='store_true',
        help='Emit structured JSON when executing a chain stub'
    )
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser(
        'eval',
        help='Evaluate experiment variants using deterministic stubs'
    )
    eval_parser.add_argument('experiment', help='Name of the experiment to evaluate')
    eval_parser.add_argument(
        '-f', '--file',
        dest='file',
        help='Path to the .n3 source file (defaults to first .n3 in current directory)'
    )
    eval_parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format for experiment results'
    )
    eval_parser.set_defaults(func=cmd_eval)

    train_parser = subparsers.add_parser(
        'train',
        help='Simulate model training for registered ML/DL assets'
    )
    train_parser.add_argument('file', help='Path to the .n3 source file')
    train_parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to train (must exist in the DSL or model registry)'
    )
    train_parser.set_defaults(func=cmd_train)

    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Simulate deployment of a model prediction endpoint'
    )
    deploy_parser.add_argument('file', help='Path to the .n3 source file')
    deploy_parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to deploy (must exist in the DSL or model registry)'
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    doctor_parser = subparsers.add_parser(
        'doctor',
        help='Diagnose installed dependencies and optional extras'
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    
    args = parser.parse_args(argv)
    
    # If no command specified, print help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    args.func(args)


if __name__ == '__main__':  # pragma: no cover
    main()
