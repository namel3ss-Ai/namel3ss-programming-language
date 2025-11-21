"""
N3 Runtime Integration Module.

This module bridges the N3 DSL configuration with the Python runtime,
allowing N3 declarations to control application behavior dynamically.

Usage:
    from n3_integration import load_n3_config, get_dataset_config
    
    config = load_n3_config("app.n3")
    dataset = get_dataset_config(config, "{{ dataset_name }}")
"""

from pathlib import Path
from typing import Any, Optional


class N3Config:
    """
    N3 configuration container.
    
    Holds parsed N3 configuration for runtime access.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize N3 config.
        
        Args:
            config_path: Path to .n3 file
        """
        self.config_path = config_path
        self.datasets = {}
        self.apis = {}
        self.backends = {}
        self.deployments = {}
        self._raw_config = None
    
    def get_dataset(self, name: str) -> Optional[dict]:
        """Get dataset configuration by name."""
        return self.datasets.get(name)
    
    def get_api(self, name: str) -> Optional[dict]:
        """Get API configuration by name."""
        return self.apis.get(name)
    
    def get_backend(self, name: str) -> Optional[dict]:
        """Get backend configuration by name."""
        return self.backends.get(name)
    
    def get_deployment(self, name: str) -> Optional[dict]:
        """Get deployment configuration by name."""
        return self.deployments.get(name)


def load_n3_config(config_path: str | Path) -> N3Config:
    """
    Load and parse N3 configuration file.
    
    This is a placeholder that demonstrates the integration point.
    In production, this would use the actual N3 parser/compiler.
    
    Args:
        config_path: Path to .n3 configuration file
        
    Returns:
        Parsed N3 configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"N3 config file not found: {config_path}")
    
    # TODO: Integrate with actual N3 parser
    # For now, return a stub config
    config = N3Config(config_path)
    
    # In production, this would:
    # 1. Parse the .n3 file using the N3 compiler
    # 2. Extract dataset, api, backend configurations
    # 3. Validate the configuration
    # 4. Populate the config object
    
    return config


def apply_n3_validation(item: Any, dataset_config: dict) -> tuple[bool, list[str]]:
    """
    Apply N3 validation rules to an item.
    
    Args:
        item: Item to validate
        dataset_config: Dataset configuration from N3
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Extract validation rules from config
    validation_rules = dataset_config.get("validation", {})
    
    # TODO: Implement N3 validation rule execution
    # This would evaluate the rules defined in the N3 file
    # against the item data
    
    return len(errors) == 0, errors


def apply_n3_hooks(event: str, item: Any, dataset_config: dict) -> Any:
    """
    Apply N3 hooks for a given event.
    
    Args:
        event: Event name (e.g., "before_create", "after_update")
        item: Item being processed
        dataset_config: Dataset configuration from N3
        
    Returns:
        Modified item after hooks
    """
    hooks = dataset_config.get("hooks", {})
    event_hooks = [h for h in hooks if h.get("trigger") == event]
    
    # TODO: Execute N3 hooks
    # This would run the hook logic defined in the N3 file
    
    return item


def get_n3_filters(dataset_config: dict) -> list[str]:
    """
    Extract available filters from N3 config.
    
    Args:
        dataset_config: Dataset configuration from N3
        
    Returns:
        List of filterable field names
    """
    api_config = dataset_config.get("api", {})
    filters = api_config.get("filters", {})
    return list(filters.keys())


def get_n3_pagination_config(dataset_config: dict) -> dict:
    """
    Extract pagination configuration from N3 config.
    
    Args:
        dataset_config: Dataset configuration from N3
        
    Returns:
        Pagination configuration dict
    """
    api_config = dataset_config.get("api", {})
    pagination = api_config.get("pagination", {})
    
    return {
        "default_page_size": pagination.get("default_page_size", 20),
        "max_page_size": pagination.get("max_page_size", 100),
    }


# Extension point: N3 runtime integration
class N3Runtime:
    """
    N3 runtime integration for dynamic behavior.
    
    This class provides the bridge between N3 declarations and
    Python runtime behavior, enabling:
    - Dynamic validation based on N3 rules
    - Hook execution at lifecycle events
    - Configuration-driven API behavior
    """
    
    def __init__(self, config: N3Config):
        """
        Initialize N3 runtime.
        
        Args:
            config: Loaded N3 configuration
        """
        self.config = config
    
    def validate_item(self, dataset_name: str, item: Any) -> tuple[bool, list[str]]:
        """
        Validate item against N3 rules.
        
        Args:
            dataset_name: Name of the dataset
            item: Item to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        dataset_config = self.config.get_dataset(dataset_name)
        if not dataset_config:
            return True, []
        
        return apply_n3_validation(item, dataset_config)
    
    def execute_hook(self, dataset_name: str, event: str, item: Any) -> Any:
        """
        Execute N3 hooks for an event.
        
        Args:
            dataset_name: Name of the dataset
            event: Event name
            item: Item being processed
            
        Returns:
            Modified item
        """
        dataset_config = self.config.get_dataset(dataset_name)
        if not dataset_config:
            return item
        
        return apply_n3_hooks(event, item, dataset_config)
    
    def get_filters(self, dataset_name: str) -> list[str]:
        """Get available filters for dataset."""
        dataset_config = self.config.get_dataset(dataset_name)
        if not dataset_config:
            return []
        
        return get_n3_filters(dataset_config)
    
    def get_pagination_config(self, dataset_name: str) -> dict:
        """Get pagination configuration for dataset."""
        dataset_config = self.config.get_dataset(dataset_name)
        if not dataset_config:
            return {"default_page_size": 20, "max_page_size": 100}
        
        return get_n3_pagination_config(dataset_config)


# Global runtime instance (initialized at app startup)
_runtime: Optional[N3Runtime] = None


def init_n3_runtime(config_path: str | Path) -> N3Runtime:
    """
    Initialize global N3 runtime.
    
    Args:
        config_path: Path to N3 config file
        
    Returns:
        N3 runtime instance
    """
    global _runtime
    config = load_n3_config(config_path)
    _runtime = N3Runtime(config)
    return _runtime


def get_n3_runtime() -> Optional[N3Runtime]:
    """Get global N3 runtime instance."""
    return _runtime
