"""LLM configuration standards for the Namel3ss standard library."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class LLMConfigField(Enum):
    """Standard LLM configuration fields with provider-neutral semantics."""
    
    # Core generation parameters
    TEMPERATURE = "temperature"
    """Controls sampling entropy. Range: [0.0, 2.0]. Lower = more deterministic."""
    
    MAX_TOKENS = "max_tokens"
    """Maximum tokens to generate. Range: [1, model_limit]. Controls response length."""
    
    TOP_P = "top_p"
    """Nucleus sampling parameter. Range: [0.0, 1.0]. Controls diversity."""
    
    TOP_K = "top_k" 
    """Top-k sampling parameter. Range: [1, vocab_size]. Limits token candidates."""
    
    FREQUENCY_PENALTY = "frequency_penalty"
    """Frequency penalty. Range: [-2.0, 2.0]. Reduces repetition."""
    
    PRESENCE_PENALTY = "presence_penalty"
    """Presence penalty. Range: [-2.0, 2.0]. Encourages topic diversity."""
    
    # Control parameters
    STOP_SEQUENCES = "stop_sequences"
    """List of strings where generation should stop."""
    
    SEED = "seed"
    """Random seed for reproducible generation. Range: [0, 2^63-1]."""
    
    STREAM = "stream"
    """Whether to stream responses. Boolean."""
    
    # System configuration
    SYSTEM_PROMPT = "system_prompt"
    """System message/instructions for the model."""
    
    # Provider selection
    PROVIDER = "provider"
    """LLM provider name (e.g., 'openai', 'anthropic', 'azure')."""
    
    MODEL = "model"
    """Model identifier (e.g., 'gpt-4', 'claude-3-opus')."""


@dataclass(frozen=True)
class LLMConfigSpec:
    """Specification for an LLM configuration field."""
    
    field: LLMConfigField
    description: str
    field_type: str  # 'float', 'int', 'str', 'bool', 'list'
    required: bool = False
    default_value: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[str]] = None
    
    def validate_value(self, value: Any) -> bool:
        """Check if a value is valid for this field."""
        if value is None:
            return not self.required
        
        # Type validation
        if self.field_type == 'float':
            if not isinstance(value, (int, float)):
                return False
            value = float(value)
        elif self.field_type == 'int':
            if not isinstance(value, int):
                return False
        elif self.field_type == 'str':
            if not isinstance(value, str):
                return False
        elif self.field_type == 'bool':
            if not isinstance(value, bool):
                return False
        elif self.field_type == 'list':
            if not isinstance(value, list):
                return False
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        
        # Enum validation
        if self.valid_values is not None and str(value) not in self.valid_values:
            return False
        
        return True


# Standard LLM configuration field specifications
STANDARD_LLM_FIELDS: Dict[LLMConfigField, LLMConfigSpec] = {
    LLMConfigField.TEMPERATURE: LLMConfigSpec(
        field=LLMConfigField.TEMPERATURE,
        description="Controls sampling entropy. Range: [0.0, 2.0]. Lower = more deterministic.",
        field_type='float',
        default_value=0.7,
        min_value=0.0,
        max_value=2.0
    ),
    
    LLMConfigField.MAX_TOKENS: LLMConfigSpec(
        field=LLMConfigField.MAX_TOKENS,
        description="Maximum tokens to generate. Range: [1, model_limit]. Controls response length.",
        field_type='int',
        default_value=1024,
        min_value=1,
        max_value=200000  # Reasonable upper bound
    ),
    
    LLMConfigField.TOP_P: LLMConfigSpec(
        field=LLMConfigField.TOP_P,
        description="Nucleus sampling parameter. Range: [0.0, 1.0]. Controls diversity.",
        field_type='float',
        default_value=0.9,
        min_value=0.0,
        max_value=1.0
    ),
    
    LLMConfigField.TOP_K: LLMConfigSpec(
        field=LLMConfigField.TOP_K,
        description="Top-k sampling parameter. Range: [1, vocab_size]. Limits token candidates.",
        field_type='int',
        default_value=50,
        min_value=1,
        max_value=100000  # Reasonable upper bound
    ),
    
    LLMConfigField.FREQUENCY_PENALTY: LLMConfigSpec(
        field=LLMConfigField.FREQUENCY_PENALTY,
        description="Frequency penalty. Range: [-2.0, 2.0]. Reduces repetition.",
        field_type='float',
        default_value=0.0,
        min_value=-2.0,
        max_value=2.0
    ),
    
    LLMConfigField.PRESENCE_PENALTY: LLMConfigSpec(
        field=LLMConfigField.PRESENCE_PENALTY,
        description="Presence penalty. Range: [-2.0, 2.0]. Encourages topic diversity.",
        field_type='float',
        default_value=0.0,
        min_value=-2.0,
        max_value=2.0
    ),
    
    LLMConfigField.STOP_SEQUENCES: LLMConfigSpec(
        field=LLMConfigField.STOP_SEQUENCES,
        description="List of strings where generation should stop.",
        field_type='list',
        default_value=[]
    ),
    
    LLMConfigField.SEED: LLMConfigSpec(
        field=LLMConfigField.SEED,
        description="Random seed for reproducible generation. Range: [0, 2^63-1].",
        field_type='int',
        min_value=0,
        max_value=9223372036854775807  # 2^63-1
    ),
    
    LLMConfigField.STREAM: LLMConfigSpec(
        field=LLMConfigField.STREAM,
        description="Whether to stream responses. Boolean.",
        field_type='bool',
        default_value=True
    ),
    
    LLMConfigField.SYSTEM_PROMPT: LLMConfigSpec(
        field=LLMConfigField.SYSTEM_PROMPT,
        description="System message/instructions for the model.",
        field_type='str'
    ),
    
    LLMConfigField.PROVIDER: LLMConfigSpec(
        field=LLMConfigField.PROVIDER,
        description="LLM provider name (e.g., 'openai', 'anthropic', 'azure').",
        field_type='str',
        required=True,
        valid_values=['openai', 'anthropic', 'azure', 'vertex', 'ollama', 'local']
    ),
    
    LLMConfigField.MODEL: LLMConfigSpec(
        field=LLMConfigField.MODEL,
        description="Model identifier (e.g., 'gpt-4', 'claude-3-opus').",
        field_type='str',
        required=True
    )
}


def get_llm_config_spec(field: Union[str, LLMConfigField]) -> LLMConfigSpec:
    """
    Get the specification for an LLM configuration field.
    
    Args:
        field: Configuration field name or enum value
        
    Returns:
        LLM configuration field specification
        
    Raises:
        ValueError: If field is not recognized
    """
    if isinstance(field, str):
        try:
            field = LLMConfigField(field)
        except ValueError:
            valid_fields = [f.value for f in LLMConfigField]
            raise ValueError(
                f"Unknown LLM config field '{field}'. "
                f"Valid fields: {', '.join(valid_fields)}"
            )
    
    return STANDARD_LLM_FIELDS[field]


def list_llm_config_fields() -> List[str]:
    """List all available LLM configuration field names."""
    return [field.value for field in LLMConfigField]


def get_field_description(field: Union[str, LLMConfigField]) -> str:
    """Get human-readable description of an LLM configuration field."""
    spec = get_llm_config_spec(field)
    return spec.description


def get_default_value(field: Union[str, LLMConfigField]) -> Any:
    """Get default value for an LLM configuration field."""
    spec = get_llm_config_spec(field)
    return spec.default_value


def get_standard_llm_config() -> Dict[str, Any]:
    """Get a standard LLM configuration with recommended defaults."""
    config = {}
    for field, spec in STANDARD_LLM_FIELDS.items():
        if spec.default_value is not None:
            config[field.value] = spec.default_value
    return config