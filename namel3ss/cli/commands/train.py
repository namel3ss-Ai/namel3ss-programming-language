"""
Training command implementation.

This module handles the 'train' subcommand for inspecting and executing
training jobs defined in N3 applications.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..errors import handle_cli_exception
from ..loading import load_json_argument, load_n3_app, load_runtime_module
from ..utils import find_first_source_file


def _format_error_detail(exc: BaseException) -> str:
    """Format exception as concise error detail string."""
    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def cmd_train(args: argparse.Namespace) -> None:
    """
    Handle the 'train' subcommand for training job management.
    
    Provides commands to:
    - List available training jobs
    - List available training backends  
    - View training job history
    - Resolve training plans
    - Execute training jobs
    
    Args:
        args: Parsed command-line arguments containing:
            - file: Path to .ai source file
            - list: List available training jobs (flag)
            - backends: List available training backends (flag)
            - job: Training job name to execute or inspect (optional)
            - history: Show training history for job (flag)
            - history_limit: Number of history entries (default: 5)
            - plan: Resolve and display training plan without executing (flag)
            - payload: Inline JSON payload for training
            - payload_file: File path containing JSON payload
            - overrides: Inline JSON overrides
            - overrides_file: File path containing JSON overrides
            - json: Output in JSON format (flag)
    
    Raises:
        SystemExit: On any error during training operations
    
    Examples:
        List jobs:
        >>> args = argparse.Namespace(file='app.ai', list=True)
        >>> cmd_train(args)  # doctest: +SKIP
        {
          "status": "ok",
          "jobs": ["fine_tune_gpt", "train_classifier"]
        }
        
        Execute job:
        >>> args = argparse.Namespace(  # doctest: +SKIP
        ...     file='app.ai',
        ...     job='fine_tune_gpt',
        ...     payload='{"examples": [...]}',
        ...     json=True
        ... )
        >>> cmd_train(args)
        {
          "status": "ok",
          "job": "fine_tune_gpt",
          "job_id": "ft-abc123"
        }
    """
    try:
        source_path = Path(args.file)
        app = load_n3_app(source_path)
        cache_key = str(source_path.resolve())
        runtime = load_runtime_module(app, cache_key)
        
        def _emit(payload: Dict[str, Any]) -> None:
            """Emit result payload in requested format."""
            if getattr(args, "json", False):
                print(json.dumps(payload))
            else:
                print(json.dumps(payload, indent=2))
        
        def _error(code: str, detail: str) -> None:
            """Emit error payload and exit."""
            _emit({"status": "error", "error": code, "detail": detail})
            sys.exit(1)
        
        # Check if runtime supports training
        list_jobs = getattr(runtime, "list_training_jobs", None)
        if not callable(list_jobs):
            _error(
                "training_not_supported",
                "This app does not declare any training jobs."
            )
        
        # Load arguments
        try:
            payload = load_json_argument(
                getattr(args, "payload", None),
                getattr(args, "payload_file", None),
                "payload"
            )
            overrides = load_json_argument(
                getattr(args, "overrides", None),
                getattr(args, "overrides_file", None),
                "overrides"
            )
        except ValueError as exc:
            _error("invalid_training_arguments", str(exc))
        
        # Get available jobs
        jobs = sorted(str(name) for name in (list_jobs() or []))
        available_backends_fn = getattr(runtime, "available_training_backends", None)
        history_fn = getattr(runtime, "training_job_history", None)
        resolve_plan = getattr(runtime, "resolve_training_job_plan", None)
        run_job = getattr(runtime, "run_training_job", None)
        
        # Handle --backends flag
        if args.backends:
            backends = available_backends_fn() if callable(available_backends_fn) else []
            _emit({"status": "ok", "backends": backends})
            return
        
        # Handle --list flag
        if args.list:
            _emit({"status": "ok", "jobs": jobs})
            return
        
        # Ensure at least one job exists
        if not jobs:
            _error(
                "training_job_not_found",
                "No training jobs are defined in this app."
            )
        
        # Ensure job name is specified
        job_name = getattr(args, "job", None)
        if not job_name:
            _error(
                "training_job_required",
                "Provide --job to select a training job or use --list to see available options."
            )
        
        # Validate job exists
        if jobs and job_name not in jobs:
            _error(
                "training_job_not_found",
                f"Training job '{job_name}' was not found. Available jobs: {', '.join(jobs) or 'none'}."
            )
        
        # Handle conflicting flags
        if args.history and args.plan:
            _error(
                "invalid_training_arguments",
                "--plan cannot be combined with --history."
            )
        
        # Handle --history flag
        if args.history:
            if not callable(history_fn):
                _error(
                    "training_history_unavailable",
                    "Training history is not available in this runtime."
                )
            limit = getattr(args, "history_limit", 5)
            limit = limit if isinstance(limit, int) and limit > 0 else 5
            history = history_fn(job_name, limit=limit)
            _emit({
                "status": "ok",
                "job": job_name,
                "history": history,
                "limit": limit
            })
            return
        
        # Handle --plan flag
        if args.plan:
            if not callable(resolve_plan):
                _error(
                    "training_runtime_missing",
                    "Training plan helpers are unavailable; regenerate the backend."
                )
            try:
                plan = resolve_plan(job_name, payload, overrides)
            except ValueError as exc:
                _error("training_job_not_found", str(exc))
            except Exception as exc:
                _error("training_plan_failed", _format_error_detail(exc))
            
            _emit({"status": "ok", "job": job_name, "plan": plan})
            return
        
        # Execute training job
        if not callable(run_job):
            _error(
                "training_runtime_missing",
                "Training execution helpers are unavailable; regenerate the backend."
            )
        
        try:
            result = run_job(job_name, payload, overrides)
        except Exception as exc:
            _error("training_backend_failed", _format_error_detail(exc))
        
        if not isinstance(result, dict):
            _error(
                "training_backend_invalid",
                "Training runtime returned an invalid response."
            )
        
        result.setdefault("job", job_name)
        _emit(result)
    
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))
