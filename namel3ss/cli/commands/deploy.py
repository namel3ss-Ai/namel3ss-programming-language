"""
Deploy command implementation.

This module handles the 'deploy' subcommand for deploying models
using configured deployer hooks.
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Optional

from ..errors import CLIRuntimeError, handle_cli_exception
from ..loading import load_n3_app
from ..utils import resolve_model_spec, slugify_model_name


def cmd_deploy(args: argparse.Namespace) -> None:
    """
    Handle the 'deploy' subcommand invoking optional user hooks.
    
    Deploys a model by invoking a configured deployer hook from the model's
    metadata. The deployer hook is a Python callable that handles the
    deployment logic and returns status information.
    
    Args:
        args: Parsed command-line arguments containing:
            - file: Path to .ai source file
            - model: Name of model to deploy
            
    Raises:
        SystemExit: On any error during deployment
    
    Examples:
        >>> args = argparse.Namespace(  # doctest: +SKIP
        ...     file='app.ai',
        ...     model='my_classifier'
        ... )
        >>> cmd_deploy(args)
        Model 'my_classifier' deployed at https://api.example.com/v1/my_classifier
        {
          "status": "ok",
          "model": "my_classifier",
          "version": "v1",
          "endpoint": "https://api.example.com/v1/my_classifier"
        }
    """
    try:
        source_path = Path(args.file)
        app = load_n3_app(source_path)
        model_name = args.model
        spec = resolve_model_spec(app, model_name)
        slug = slugify_model_name(model_name)
        version = spec.get("version", "v1")
        metadata = spec.get("metadata", {})
        
        # Check for deployer hook
        deployer_hook: Optional[str] = metadata.get("deployer")
        if not deployer_hook:
            result = {
                "status": "error",
                "error": "deployer_not_configured",
                "detail": f"No deployer hook configured for model '{model_name}'.",
            }
            print(json.dumps(result))
            return
        
        # Parse and validate hook format
        module_path, _, attr = deployer_hook.partition(":")
        if not module_path or not attr:
            result = {
                "status": "error",
                "error": "deployer_invalid_hook",
                "detail": f"Deployer hook '{deployer_hook}' is not importable.",
            }
            print(json.dumps(result))
            return
        
        # Import and execute deployer
        try:
            module = importlib.import_module(module_path)
            deployer = getattr(module, attr)
            if not callable(deployer):
                raise TypeError(f"Deployer '{deployer_hook}' is not callable")
            output = deployer(model_name, spec, args)
        except Exception as exc:
            result = {
                "status": "error",
                "error": "deployer_failed",
                "detail": str(exc),
            }
            print(json.dumps(result))
            return
        
        # Format output
        if isinstance(output, dict):
            result = {
                "status": output.get("status", "ok"),
                **{k: v for k, v in output.items() if k != "status"}
            }
        elif isinstance(output, str) and output:
            result = {"status": "ok", "endpoint": output}
        else:
            result = {"status": "ok", "detail": "Deployment hook executed."}
        
        result.setdefault("model", model_name)
        result.setdefault("version", version)
        result.setdefault("endpoint", result.get("endpoint"))
        
        # Print success message if endpoint available
        endpoint = result.get("endpoint")
        if endpoint:
            print(f"Model '{model_name}' deployed at {endpoint}")
        
        print(json.dumps(result))
    
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))
