"""
Local model deployment commands for Namel3ss CLI.

This module provides commands for managing local model deployments
using vLLM, Ollama, and LocalAI.
"""

import argparse
import asyncio
import json
import logging
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...ast.ai import AIModel
from ...providers.local import VLLMProvider, OllamaProvider, LocalAIProvider
from ...providers.factory import create_provider_from_spec
from ..errors import CLIRuntimeError, handle_cli_exception
from ..loading import load_n3_app
from ..output import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def cmd_deploy_local(args: argparse.Namespace) -> None:
    """
    Deploy a local model using vLLM, Ollama, or LocalAI.
    
    Args:
        args: Parsed command-line arguments
    """
    try:
        if args.local_command == 'start':
            _cmd_local_start(args)
        elif args.local_command == 'stop':
            _cmd_local_stop(args)
        elif args.local_command == 'status':
            _cmd_local_status(args)
        elif args.local_command == 'logs':
            _cmd_local_logs(args)
        elif args.local_command == 'scale':
            _cmd_local_scale(args)
        elif args.local_command == 'list':
            _cmd_local_list(args)
        else:
            print_error(f"Unknown local deployment command: {args.local_command}")
            sys.exit(1)
            
    except Exception as exc:
        handle_cli_exception(exc, verbose=getattr(args, "verbose", False))


def _cmd_local_start(args: argparse.Namespace) -> None:
    """Start a local model deployment."""
    print_info(f"Starting local model deployment: {args.model}")
    
    # Load N3 application
    source_path = Path(args.file) if args.file else Path.cwd() / "app.n3"
    app = load_n3_app(source_path)
    
    # Find the specified model
    target_model: Optional[AIModel] = None
    for model in app.ai_models:
        if model.name == args.model:
            target_model = model
            break
    
    if not target_model:
        print_error(f"Model '{args.model}' not found in {source_path}")
        sys.exit(1)
    
    if not target_model.is_local:
        print_error(f"Model '{args.model}' is not configured for local deployment")
        print_info("Local deployment requires providers: vllm, ollama, local_ai, or lm_studio")
        sys.exit(1)
    
    # Create deployment configuration
    deployment_config = _build_deployment_config(target_model, args)
    
    # Start the deployment
    asyncio.run(_start_deployment(target_model, deployment_config))


def _cmd_local_stop(args: argparse.Namespace) -> None:
    """Stop a local model deployment."""
    print_info(f"Stopping local model deployment: {args.model}")
    
    # Find running deployment
    deployment_info = _find_running_deployment(args.model)
    if not deployment_info:
        print_warning(f"No running deployment found for model: {args.model}")
        return
    
    # Stop the deployment
    asyncio.run(_stop_deployment(args.model, deployment_info))
    print_success(f"Stopped deployment: {args.model}")


def _cmd_local_status(args: argparse.Namespace) -> None:
    """Check status of local model deployments."""
    if args.model:
        # Status for specific model
        deployment_info = _find_running_deployment(args.model)
        if deployment_info:
            status = asyncio.run(_get_deployment_status(args.model, deployment_info))
            print(json.dumps(status, indent=2))
        else:
            print_info(f"Model '{args.model}' is not deployed")
    else:
        # Status for all deployments
        deployments = _list_all_deployments()
        print(json.dumps(deployments, indent=2))


def _cmd_local_logs(args: argparse.Namespace) -> None:
    """Show logs for a local model deployment."""
    deployment_info = _find_running_deployment(args.model)
    if not deployment_info:
        print_error(f"No running deployment found for model: {args.model}")
        sys.exit(1)
    
    _show_deployment_logs(args.model, deployment_info, args.follow)


def _cmd_local_scale(args: argparse.Namespace) -> None:
    """Scale a local model deployment."""
    print_info(f"Scaling model '{args.model}' to {args.replicas} replicas")
    
    # Find running deployment
    deployment_info = _find_running_deployment(args.model)
    if not deployment_info:
        print_error(f"No running deployment found for model: {args.model}")
        sys.exit(1)
    
    # Scale the deployment
    asyncio.run(_scale_deployment(args.model, deployment_info, args.replicas))
    print_success(f"Scaled deployment '{args.model}' to {args.replicas} replicas")


def _cmd_local_list(args: argparse.Namespace) -> None:
    """List all local model deployments."""
    deployments = _list_all_deployments()
    
    if not deployments:
        print_info("No local model deployments found")
        return
    
    # Print deployment table
    print(f"{'MODEL':<20} {'PROVIDER':<10} {'STATUS':<10} {'URL':<30} {'REPLICAS':<8}")
    print("-" * 78)
    
    for deployment in deployments:
        print(f"{deployment['model']:<20} "
              f"{deployment['provider']:<10} "
              f"{deployment['status']:<10} "
              f"{deployment.get('url', 'N/A'):<30} "
              f"{deployment.get('replicas', 1):<8}")


async def _start_deployment(model: AIModel, config: Dict[str, Any]) -> None:
    """Start a local model deployment."""
    try:
        # Create provider instance
        provider = create_provider_from_spec(
            name=f"local_{model.name}",
            provider_type=model.provider,
            model=model.model_name,
            config={**model.config, **config}
        )
        
        # Start the deployment
        if hasattr(provider, 'start_server') and config.get('auto_start_server', True):
            if isinstance(provider, VLLMProvider):
                await provider._ensure_server_running()
                print_success(f"vLLM deployment started for model: {model.name}")
                print_info(f"Server URL: {provider.base_url}")
                
            elif isinstance(provider, OllamaProvider):
                await provider._ensure_ready()
                print_success(f"Ollama deployment started for model: {model.name}")
                print_info(f"Server URL: {provider.base_url}")
                
            elif isinstance(provider, LocalAIProvider):
                await provider._ensure_server_running()
                print_success(f"LocalAI deployment started for model: {model.name}")
                print_info(f"Server URL: {provider.base_url}")
        
        # Save deployment info
        _save_deployment_info(model.name, {
            'model': model.name,
            'provider': model.provider,
            'model_name': model.model_name,
            'status': 'running',
            'url': getattr(provider, 'base_url', None),
            'config': config,
            'started_at': time.time(),
        })
        
    except Exception as e:
        print_error(f"Failed to start deployment: {e}")
        raise


async def _stop_deployment(model_name: str, deployment_info: Dict[str, Any]) -> None:
    """Stop a local model deployment."""
    try:
        provider_type = deployment_info['provider']
        config = deployment_info['config']
        
        # Create provider instance to stop
        provider = create_provider_from_spec(
            name=f"local_{model_name}",
            provider_type=provider_type,
            model=deployment_info['model_name'],
            config=config
        )
        
        # Stop the deployment
        if hasattr(provider, 'close'):
            await provider.close()
        
        # Remove deployment info
        _remove_deployment_info(model_name)
        
    except Exception as e:
        print_error(f"Failed to stop deployment: {e}")
        raise


async def _get_deployment_status(model_name: str, deployment_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get status of a deployment."""
    try:
        provider_type = deployment_info['provider']
        config = deployment_info['config']
        
        # Create provider instance to check health
        provider = create_provider_from_spec(
            name=f"local_{model_name}",
            provider_type=provider_type,
            model=deployment_info['model_name'],
            config=config
        )
        
        if hasattr(provider, 'health_check'):
            health = await provider.health_check()
            return {**deployment_info, 'health': health}
        else:
            return {**deployment_info, 'health': {'status': 'unknown'}}
            
    except Exception as e:
        return {**deployment_info, 'health': {'status': 'error', 'error': str(e)}}


async def _scale_deployment(model_name: str, deployment_info: Dict[str, Any], replicas: int) -> None:
    """Scale a deployment (placeholder for future implementation)."""
    # For now, local deployments don't support horizontal scaling
    # This would be implemented with Docker Compose or Kubernetes
    print_warning("Local deployment scaling not yet implemented")
    print_info("Future versions will support scaling with Docker Compose/Kubernetes")


def _build_deployment_config(model: AIModel, args: argparse.Namespace) -> Dict[str, Any]:
    """Build deployment configuration from model and CLI args."""
    config = model.config.copy()
    deployment_config = model.deployment_config.copy()
    
    # Override with CLI arguments
    if hasattr(args, 'host') and args.host:
        config['host'] = args.host
    if hasattr(args, 'port') and args.port:
        config['port'] = args.port
    if hasattr(args, 'gpu') and args.gpu is not None:
        deployment_config['gpu_memory_utilization'] = args.gpu
    
    config['deployment_config'] = deployment_config
    return config


def _find_running_deployment(model_name: str) -> Optional[Dict[str, Any]]:
    """Find running deployment by model name."""
    deployments_file = _get_deployments_file()
    if not deployments_file.exists():
        return None
    
    try:
        with deployments_file.open('r') as f:
            deployments = json.load(f)
        return deployments.get(model_name)
    except (json.JSONDecodeError, IOError):
        return None


def _list_all_deployments() -> List[Dict[str, Any]]:
    """List all running deployments."""
    deployments_file = _get_deployments_file()
    if not deployments_file.exists():
        return []
    
    try:
        with deployments_file.open('r') as f:
            deployments = json.load(f)
        return list(deployments.values())
    except (json.JSONDecodeError, IOError):
        return []


def _save_deployment_info(model_name: str, info: Dict[str, Any]) -> None:
    """Save deployment information."""
    deployments_file = _get_deployments_file()
    deployments_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing deployments
    deployments = {}
    if deployments_file.exists():
        try:
            with deployments_file.open('r') as f:
                deployments = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Add/update deployment
    deployments[model_name] = info
    
    # Save deployments
    with deployments_file.open('w') as f:
        json.dump(deployments, f, indent=2)


def _remove_deployment_info(model_name: str) -> None:
    """Remove deployment information."""
    deployments_file = _get_deployments_file()
    if not deployments_file.exists():
        return
    
    try:
        with deployments_file.open('r') as f:
            deployments = json.load(f)
        
        deployments.pop(model_name, None)
        
        with deployments_file.open('w') as f:
            json.dump(deployments, f, indent=2)
    except (json.JSONDecodeError, IOError):
        pass


def _get_deployments_file() -> Path:
    """Get path to deployments tracking file."""
    return Path.home() / ".namel3ss" / "deployments.json"


def _show_deployment_logs(model_name: str, deployment_info: Dict[str, Any], follow: bool = False) -> None:
    """Show logs for a deployment."""
    print_info(f"Showing logs for deployment: {model_name}")
    print_warning("Log viewing not yet implemented for local deployments")
    print_info("Future versions will support log aggregation and viewing")


def add_local_deploy_command(parser: argparse.ArgumentParser) -> None:
    """Add local deployment subcommands to the CLI parser."""
    local_parser = parser.add_parser(
        'local',
        help='Manage local model deployments'
    )
    
    # Add subcommands
    local_subparsers = local_parser.add_subparsers(
        dest='local_command',
        help='Local deployment operations'
    )
    
    # Start command
    start_parser = local_subparsers.add_parser(
        'start',
        help='Start a local model deployment'
    )
    start_parser.add_argument(
        'model',
        help='Name of the model to deploy'
    )
    start_parser.add_argument(
        '--file', '-f',
        help='Path to .n3 source file (default: ./app.n3)'
    )
    start_parser.add_argument(
        '--host',
        help='Host to bind the server to (default: model config)'
    )
    start_parser.add_argument(
        '--port', 
        type=int,
        help='Port to bind the server to (default: model config)'
    )
    start_parser.add_argument(
        '--gpu',
        type=float,
        help='GPU memory utilization (0.0-1.0, vLLM only)'
    )
    
    # Stop command
    stop_parser = local_subparsers.add_parser(
        'stop',
        help='Stop a local model deployment'
    )
    stop_parser.add_argument(
        'model',
        help='Name of the model deployment to stop'
    )
    
    # Status command
    status_parser = local_subparsers.add_parser(
        'status',
        help='Check status of local model deployments'
    )
    status_parser.add_argument(
        'model',
        nargs='?',
        help='Name of specific model to check (optional)'
    )
    
    # Logs command
    logs_parser = local_subparsers.add_parser(
        'logs',
        help='Show logs for a local model deployment'
    )
    logs_parser.add_argument(
        'model',
        help='Name of the model deployment'
    )
    logs_parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Follow log output'
    )
    
    # Scale command
    scale_parser = local_subparsers.add_parser(
        'scale',
        help='Scale a local model deployment'
    )
    scale_parser.add_argument(
        'model',
        help='Name of the model deployment to scale'
    )
    scale_parser.add_argument(
        'replicas',
        type=int,
        help='Number of replicas'
    )
    
    # List command
    local_subparsers.add_parser(
        'list',
        help='List all local model deployments'
    )
    
    # Set the command handler
    local_parser.set_defaults(func=cmd_deploy_local)