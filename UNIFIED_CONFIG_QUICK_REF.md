# Unified Config Parsing - Quick Reference

## For Parser Development

### Adding a New Declaration Type

```python
def parse_mydecl_declaration(self):
    """Parse mydecl declaration with unified config pattern."""
    from namel3ss.ast import MyDecl
    
    # 1. Parse header
    token = self.expect(TokenType.MYDECL)
    name = self.expect(TokenType.STRING).value
    
    self.declare_symbol(f"mydecl:{name}", token.line)
    
    # 2. Parse block
    self.expect(TokenType.LBRACE)
    
    # Define special handlers for complex fields (optional)
    special_handlers = {
        "schema": lambda: self.parse_schema_definition(),
        "complex_field": lambda: self.parse_custom_structure(),
    }
    
    config = self._parse_config_block(
        allow_any_keyword=True,
        special_handlers=special_handlers
    )
    
    self.expect(TokenType.RBRACE)
    self.skip_newlines()
    
    # 3. Transform legacy fields if needed
    if 'old_name' in config:
        config['new_name'] = config.pop('old_name')
    
    # 4. Build with unified pattern
    return build_dataclass_with_config(
        MyDecl,
        config,
        declared_name=name,
        path=self.path,
        line=token.line,
        column=token.column,
        name=name,  # Explicit fields override config
    )
```

### Adding Field Aliases

```python
# In config_filter.py

MYDECL_ALIASES = {
    "short_name": "full_field_name",
    "llm": "llm_name",  # Common pattern
    "system": "system_prompt",  # Common pattern
}

# Register in ALIAS_REGISTRY
ALIAS_REGISTRY = {
    # ... existing entries
    "MyDecl": MYDECL_ALIASES,
}

# Export
__all__ = [
    # ... existing exports
    "MYDECL_ALIASES",
]
```

### Adding Validation Rules

```python
# In config_validator.py

# 1. Add disallowed fields (if any)
DISALLOWED_FIELDS: Dict[str, Set[str]] = {
    # ... existing entries
    "MyDecl": {
        "temperature",  # Example: model param not allowed
        "max_tokens",
    },
}

# 2. Add type-specific validation function (if needed)
def validate_mydecl_specific(
    config: Dict[str, Any],
    path: str = "",
    line: int = 0,
    column: int = 0
) -> Dict[str, Any]:
    """MyDecl-specific validation logic."""
    if "field_a" in config and "field_b" in config:
        raise create_syntax_error(
            "Cannot use both 'field_a' and 'field_b'",
            path=path,
            line=line,
            column=column,
        )
    return config

# 3. Integrate in validate_config_for_declaration()
def validate_config_for_declaration(...):
    # ... existing code
    
    class_name = dataclass_type.__name__
    if class_name == "MyDecl":
        config = validate_mydecl_specific(config, path, line, column)
    
    return config
```

## For AST Development

### Dataclass Design Guidelines

```python
@dataclass
class MyDecl:
    """My declaration type.
    
    Config Sink Strategy:
    - Use 'config: Dict[str, Any]' for unknown fields in most types
    - Use 'metadata: Dict[str, Any]' for LLM-related types
    - Use 'parameters: Dict[str, Any]' for Prompt (special case)
    
    The filtering system will automatically route unknown fields to
    the appropriate sink based on _has_config_sink() logic.
    """
    # Required fields first
    name: str
    
    # Optional fields with defaults
    description: Optional[str] = None
    
    # Config sink (choose ONE based on type)
    config: Dict[str, Any] = field(default_factory=dict)  # Most common
    # OR
    # metadata: Dict[str, Any] = field(default_factory=dict)  # For LLM types
    # OR
    # parameters: Dict[str, Any] = field(default_factory=dict)  # For Prompt only
```

## For Testing

### Test Pattern for New Declarations

```python
class TestMyDeclParsing:
    """Test MyDecl declaration parsing."""
    
    def test_basic_mydecl(self):
        """Test basic MyDecl with required fields only."""
        source = """
mydecl "test" {
    required_field: "value"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        mydecl = module.body[0]
        assert isinstance(mydecl, MyDecl)
        assert mydecl.name == "test"
        assert mydecl.required_field == "value"
    
    def test_mydecl_with_aliases(self):
        """Test that aliases work correctly."""
        source = """
mydecl "test" {
    short_name: "value"  # Uses alias
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        mydecl = module.body[0]
        assert mydecl.full_field_name == "value"  # Alias resolved
    
    def test_mydecl_with_unknown_fields(self):
        """Test that unknown fields go to config sink."""
        source = """
mydecl "test" {
    required_field: "value"
    custom_field: 42
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        mydecl = module.body[0]
        assert mydecl.config["custom_field"] == 42
    
    def test_mydecl_validation_error(self):
        """Test that invalid configs are caught."""
        source = """
mydecl "test" {
    temperature: 0.7  # Disallowed field
}
"""
        parser = N3Parser(source, path="test.n3")
        
        with pytest.raises(N3SyntaxError) as exc_info:
            parser.parse()
        
        assert "temperature" in str(exc_info.value).lower()
```

## Common Patterns

### Pattern: Conditional Aliasing

```python
# Problem: field "llm" should alias to "model", but only if "model" not present
# Solution: In filter_config_for_dataclass()

if class_name == "MyDecl" and "model" in config and "llm" in aliases:
    aliases = {k: v for k, v in aliases.items() if k != "llm"}
```

### Pattern: Legacy Field Mapping

```python
# Problem: DSL field name different from dataclass field name
# Solution: In parse_mydecl_declaration()

if 'input' in config:
    config['input_fields'] = config.pop('input')  # Rename before building
```

### Pattern: Special Field Handlers

```python
# Problem: Some fields need custom parsing logic
# Solution: Use special_handlers in _parse_config_block()

special_handlers = {
    "schema": lambda: self.parse_schema_definition(),  # Custom parser
    "args": lambda: self.parse_value(),  # Generic parser
}

config = self._parse_config_block(
    allow_any_keyword=True,
    special_handlers=special_handlers
)
```

### Pattern: Config Sink Selection

```python
# In config_filter.py _has_config_sink()

# Special case for specific type
if class_name == "MySpecialDecl" and "special_sink" in field_names:
    return True, "special_sink"

# Default priority: config > metadata
if "config" in field_names:
    return True, "config"
elif "metadata" in field_names:
    return True, "metadata"
else:
    return False, None
```

## Debugging Tips

### Enable Verbose Logging

```python
# In parse_mydecl_declaration()
print(f"DEBUG: Raw config: {config}")
print(f"DEBUG: After filtering: {constructor_kwargs}")
print(f"DEBUG: Leftover: {leftover}")
```

### Check Dataclass Fields

```python
from dataclasses import fields
from namel3ss.ast import MyDecl

for f in fields(MyDecl):
    print(f"{f.name}: {f.type}")
```

### Test Filtering Directly

```python
from namel3ss.lang.parser.config_filter import filter_config_for_dataclass
from namel3ss.ast import MyDecl

config = {"field_a": "value", "unknown": 42}
kwargs, leftover = filter_config_for_dataclass(config, MyDecl)
print(f"Constructor kwargs: {kwargs}")
print(f"Leftover config: {leftover}")
```

### Test Validation Directly

```python
from namel3ss.lang.parser.config_validator import validate_config_for_declaration
from namel3ss.ast import MyDecl

config = {"field_a": "value"}
validated = validate_config_for_declaration(MyDecl, config)
print(f"Validated config: {validated}")
```

## Error Messages

### Good Error Message Example

```python
raise create_syntax_error(
    f"Cannot use '{field_a}' and '{field_b}' together in {decl_type}. "
    f"Use '{field_a}' for modern syntax or remove both and use '{field_c}'.",
    path=path,
    line=line,
    column=column,
)
```

**Why Good:**
- Explains WHAT is wrong
- Explains WHY it's wrong
- Provides actionable fix

### Bad Error Message Example

```python
raise create_syntax_error("Invalid configuration", path=path, line=line)
```

**Why Bad:**
- Doesn't say what's invalid
- No context
- No suggested fix

## Quick Decision Tree

### "Should I add an alias?"

```
Does the user's natural field name differ from the dataclass field?
├─ Yes → Add to MYDECL_ALIASES
└─ No → Use field name as-is
```

### "Should I add validation?"

```
Can the user write invalid but parseable config?
├─ Yes → Add validation
│   ├─ Type-agnostic? → Add to validate_field_restrictions()
│   └─ Type-specific? → Add new validate_mydecl_*() function
└─ No → Skip validation (dataclass will catch it)
```

### "Should I use special_handlers?"

```
Does the field need non-standard parsing?
├─ Yes → Add to special_handlers dict
│   ├─ Schema-like? → Use parse_schema_definition()
│   ├─ Block-like? → Use parse_block()
│   └─ Custom? → Write custom parser
└─ No → Use default parse_value()
```

### "Where do unknown fields go?"

```
What type is this declaration?
├─ Prompt? → parameters (special case)
├─ LLM/Agent? → metadata (if present) or config
└─ Other? → config (if present) or error
```

## Files to Edit

| Task | Files |
|------|-------|
| Add new declaration parser | `namel3ss/lang/parser/declarations.py` |
| Add field aliases | `namel3ss/lang/parser/config_filter.py` |
| Add validation rules | `namel3ss/lang/parser/config_validator.py` |
| Update AST | `namel3ss/ast/*.py` |
| Add tests | `tests/parser/test_*.py` |

## Related Documentation

- Full implementation details: `UNIFIED_CONFIG_PARSING_COMPLETE.md`
- Parser architecture: `PARSER_REFACTOR_DESIGN.md`
- AST structure: `namel3ss/ast/README.md` (if exists)
