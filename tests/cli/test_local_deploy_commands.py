"""Tests for CLI local deployment commands."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import asyncio
from click.testing import CliRunner

from namel3ss.cli.commands.local_deploy import (
    _cmd_local_start, _cmd_local_stop, _cmd_local_list, 
    _cmd_local_status, _cmd_local_scale, cmd_deploy_local
)


class TestLocalDeploymentCLI:
    """Test cases for local deployment CLI commands."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def mock_provider_factory(self):
        """Mock provider factory."""
        with patch('namel3ss.cli.commands.local_deploy.get_provider_class') as mock_factory:
            yield mock_factory
    
    def test_start_deployment_success(self, cli_runner, mock_provider_factory):
        """Test successful deployment start."""
        # Mock provider class and instance
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.start_deployment = AsyncMock()
        mock_provider.get_deployment_info = Mock(return_value={
            'status': 'running',
            'url': 'http://localhost:8001',
            'model': 'microsoft/DialoGPT-medium',
            'pid': 12345
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        # Test command execution
        result = cli_runner.invoke(start_deployment, [
            '--provider', 'vllm',
            '--model', 'microsoft/DialoGPT-medium',
            '--host', '127.0.0.1',
            '--port', '8001'
        ])
        
        assert result.exit_code == 0
        assert 'Deployment started successfully' in result.output
        assert 'Status: running' in result.output
        assert 'URL: http://localhost:8001' in result.output
        
        # Verify provider was called correctly
        mock_provider_class.assert_called_once()
        mock_provider.start_deployment.assert_called_once()
    
    def test_start_deployment_failure(self, cli_runner, mock_provider_factory):
        """Test deployment start failure."""
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.start_deployment = AsyncMock(side_effect=Exception("Deployment failed"))
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        result = cli_runner.invoke(start_deployment, [
            '--provider', 'vllm',
            '--model', 'test-model'
        ])
        
        assert result.exit_code == 1
        assert 'Failed to start deployment' in result.output
        assert 'Deployment failed' in result.output
    
    def test_start_deployment_with_config_file(self, cli_runner, mock_provider_factory):
        """Test deployment start with configuration file."""
        # Create temporary config file
        import tempfile
        import json
        
        config_data = {
            'provider': 'ollama',
            'model': 'llama3:8b',
            'config': {
                'temperature': 0.8,
                'host': '127.0.0.1',
                'port': 11434
            },
            'deployment_config': {
                'num_gpu': 1,
                'num_thread': 8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name
        
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.start_deployment = AsyncMock()
        mock_provider.get_deployment_info = Mock(return_value={
            'status': 'running',
            'model': 'llama3:8b'
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        try:
            result = cli_runner.invoke(start_deployment, [
                '--config-file', config_file_path
            ])
            
            assert result.exit_code == 0
            assert 'Deployment started successfully' in result.output
        finally:
            import os
            os.unlink(config_file_path)
    
    def test_stop_deployment_success(self, cli_runner, mock_provider_factory):
        """Test successful deployment stop."""
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.stop_deployment = AsyncMock()
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        result = cli_runner.invoke(stop_deployment, [
            '--provider', 'vllm',
            '--model', 'test-model'
        ])
        
        assert result.exit_code == 0
        assert 'Deployment stopped successfully' in result.output
        mock_provider.stop_deployment.assert_called_once()
    
    def test_list_deployments(self, cli_runner, mock_provider_factory):
        """Test listing deployments."""
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.list_deployments = Mock(return_value=[
            {
                'name': 'vllm-model1',
                'provider': 'vllm',
                'model': 'microsoft/DialoGPT-medium',
                'status': 'running',
                'url': 'http://localhost:8001',
                'pid': 12345
            },
            {
                'name': 'ollama-model1',
                'provider': 'ollama', 
                'model': 'llama3:8b',
                'status': 'running',
                'url': 'http://localhost:11434',
                'pid': 12346
            }
        ])
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        result = cli_runner.invoke(list_deployments)
        
        assert result.exit_code == 0
        assert 'Active Deployments' in result.output
        assert 'vllm-model1' in result.output
        assert 'ollama-model1' in result.output
        assert 'microsoft/DialoGPT-medium' in result.output
        assert 'llama3:8b' in result.output
    
    def test_deployment_status(self, cli_runner, mock_provider_factory):
        """Test getting deployment status."""
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.get_deployment_status = Mock(return_value={
            'status': 'running',
            'health': 'healthy',
            'uptime': '2 hours',
            'memory_usage': '2.1GB',
            'gpu_usage': '85%',
            'requests_served': 156,
            'avg_response_time': '245ms'
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        result = cli_runner.invoke(deployment_status, [
            '--provider', 'vllm',
            '--model', 'test-model'
        ])
        
        assert result.exit_code == 0
        assert 'Deployment Status' in result.output
        assert 'Status: running' in result.output
        assert 'Health: healthy' in result.output
        assert 'Memory Usage: 2.1GB' in result.output
        assert 'GPU Usage: 85%' in result.output
    
    def test_scale_deployment(self, cli_runner, mock_provider_factory):
        """Test scaling deployment."""
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.scale_deployment = AsyncMock()
        mock_provider.get_deployment_info = Mock(return_value={
            'status': 'running',
            'replicas': 3,
            'load_balancer': 'http://localhost:8001'
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        result = cli_runner.invoke(scale_deployment, [
            '--provider', 'vllm',
            '--model', 'test-model',
            '--replicas', '3'
        ])
        
        assert result.exit_code == 0
        assert 'Deployment scaled successfully' in result.output
        assert 'Replicas: 3' in result.output
        mock_provider.scale_deployment.assert_called_once_with(replicas=3)
    
    def test_deploy_group_success(self, cli_runner, mock_provider_factory):
        """Test deploying multiple models as a group."""
        # Create temporary group configuration
        import tempfile
        import json
        
        group_config = {
            'group_name': 'production_models',
            'models': [
                {
                    'name': 'chat_model',
                    'provider': 'vllm',
                    'model': 'microsoft/DialoGPT-medium',
                    'config': {'temperature': 0.7, 'port': 8001}
                },
                {
                    'name': 'completion_model',
                    'provider': 'ollama',
                    'model': 'llama3:8b',
                    'config': {'temperature': 0.8, 'port': 11434}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(group_config, f)
            group_file_path = f.name
        
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider.start_deployment = AsyncMock()
        mock_provider.get_deployment_info = Mock(return_value={
            'status': 'running',
            'model': 'test-model'
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        try:
            result = cli_runner.invoke(deploy_group, [
                '--config-file', group_file_path
            ])
            
            assert result.exit_code == 0
            assert 'Group deployment completed' in result.output
            assert 'Successfully deployed: 2' in result.output
            
            # Verify both models were deployed
            assert mock_provider.start_deployment.call_count == 2
        finally:
            import os
            os.unlink(group_file_path)
    
    def test_deploy_group_partial_failure(self, cli_runner, mock_provider_factory):
        """Test group deployment with some failures."""
        # Create temporary group configuration
        import tempfile
        import json
        
        group_config = {
            'group_name': 'test_group',
            'models': [
                {
                    'name': 'model1',
                    'provider': 'vllm',
                    'model': 'test-model1'
                },
                {
                    'name': 'model2', 
                    'provider': 'ollama',
                    'model': 'test-model2'
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(group_config, f)
            group_file_path = f.name
        
        mock_provider_class = Mock()
        mock_provider = Mock()
        # First deployment succeeds, second fails
        mock_provider.start_deployment = AsyncMock(side_effect=[
            None,  # Success
            Exception("Deployment failed")  # Failure
        ])
        mock_provider.get_deployment_info = Mock(return_value={
            'status': 'running',
            'model': 'test-model'
        })
        mock_provider_class.return_value = mock_provider
        mock_provider_factory.return_value = mock_provider_class
        
        try:
            result = cli_runner.invoke(deploy_group, [
                '--config-file', group_file_path
            ])
            
            assert result.exit_code == 0  # Partial success
            assert 'Group deployment completed' in result.output
            assert 'Successfully deployed: 1' in result.output
            assert 'Failed deployments: 1' in result.output
        finally:
            import os
            os.unlink(group_file_path)


class TestCLIConfigurationValidation:
    """Test cases for CLI configuration validation."""
    
    def test_validate_port_range(self):
        """Test port range validation."""
        from namel3ss.cli.commands.local_deploy import _validate_port
        
        # Valid ports
        assert _validate_port(8080) == 8080
        assert _validate_port(11434) == 11434
        assert _validate_port(65535) == 65535
        
        # Invalid ports
        with pytest.raises(ValueError):
            _validate_port(0)
        
        with pytest.raises(ValueError):
            _validate_port(65536)
        
        with pytest.raises(ValueError):
            _validate_port(-1)
    
    def test_validate_provider_name(self):
        """Test provider name validation."""
        from namel3ss.cli.commands.local_deploy import _validate_provider
        
        # Valid providers
        valid_providers = ['vllm', 'ollama', 'local_ai']
        for provider in valid_providers:
            assert _validate_provider(provider) == provider
        
        # Invalid provider
        with pytest.raises(ValueError):
            _validate_provider('invalid_provider')
    
    def test_parse_deployment_config(self):
        """Test deployment configuration parsing.""" 
        from namel3ss.cli.commands.local_deploy import _parse_deployment_config
        
        # Valid JSON config
        config_str = '{"gpu_memory_utilization": 0.9, "tensor_parallel_size": 2}'
        config = _parse_deployment_config(config_str)
        assert config['gpu_memory_utilization'] == 0.9
        assert config['tensor_parallel_size'] == 2
        
        # Invalid JSON
        with pytest.raises(ValueError):
            _parse_deployment_config('{"invalid": json}')
    
    def test_load_config_file(self):
        """Test configuration file loading."""
        from namel3ss.cli.commands.local_deploy import _load_config_file
        import tempfile
        import json
        
        config_data = {
            'provider': 'vllm',
            'model': 'test-model',
            'config': {'temperature': 0.7}
        }
        
        # Test valid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name
        
        try:
            loaded_config = _load_config_file(config_file_path)
            assert loaded_config == config_data
        finally:
            import os
            os.unlink(config_file_path)
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            _load_config_file('/non/existent/file.json')


class TestCLIAsyncHelpers:
    """Test cases for CLI async helpers."""
    
    def test_run_async_command(self):
        """Test async command execution wrapper."""
        from namel3ss.cli.commands.local_deploy import _run_async
        
        async def mock_async_func(arg1, arg2):
            return f"result: {arg1} + {arg2}"
        
        result = _run_async(mock_async_func("hello", "world"))
        assert result == "result: hello + world"
    
    def test_run_async_command_with_exception(self):
        """Test async command execution with exception."""
        from namel3ss.cli.commands.local_deploy import _run_async
        
        async def mock_failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            _run_async(mock_failing_func())


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_cli_help_messages(self, cli_runner=None):
        """Test CLI help messages."""
        if cli_runner is None:
            cli_runner = CliRunner()
        
        # Test main command help
        result = cli_runner.invoke(start_deployment, ['--help'])
        assert result.exit_code == 0
        assert 'Start a local model deployment' in result.output
        
        result = cli_runner.invoke(stop_deployment, ['--help'])
        assert result.exit_code == 0
        assert 'Stop a local model deployment' in result.output
        
        result = cli_runner.invoke(list_deployments, ['--help'])
        assert result.exit_code == 0
        assert 'List all active deployments' in result.output
    
    def test_cli_required_arguments(self):
        """Test CLI required argument validation."""
        cli_runner = CliRunner()
        
        # Start deployment requires provider and model
        result = cli_runner.invoke(start_deployment, [])
        assert result.exit_code != 0
        
        # Stop deployment requires provider and model
        result = cli_runner.invoke(stop_deployment, [])
        assert result.exit_code != 0
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        cli_runner = CliRunner()
        
        # Test invalid provider
        with patch('namel3ss.cli.commands.local_deploy.get_provider_class', side_effect=Exception("Provider not found")):
            result = cli_runner.invoke(start_deployment, [
                '--provider', 'invalid',
                '--model', 'test'
            ])
            assert result.exit_code == 1
            assert 'Provider not found' in result.output


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])