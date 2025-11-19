"""Pattern matching engine with structural unification."""

from __future__ import annotations

from typing import Any, Dict, Optional

from namel3ss.ast.expressions import (
    ConstructorPattern,
    DictPattern,
    ListPattern,
    LiteralPattern,
    Pattern,
    TuplePattern,
    VarPattern,
    WildcardPattern,
)

__all__ = [
    "match_pattern",
    "PatternMatchError",
]


class PatternMatchError(Exception):
    """Raised when pattern matching fails."""
    pass


def match_pattern(
    pattern: Pattern,
    value: Any,
    bindings: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Attempt to match pattern against value.
    
    Args:
        pattern: Pattern to match
        value: Value to match against
        bindings: Existing variable bindings (optional)
    
    Returns:
        Updated bindings dict on success, None on failure
    """
    if bindings is None:
        bindings = {}
    else:
        bindings = dict(bindings)  # Copy to avoid mutation
    
    result = _match_pattern_impl(pattern, value, bindings)
    return result if result is not None else None


def _match_pattern_impl(
    pattern: Pattern,
    value: Any,
    bindings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Internal pattern matching implementation."""
    
    # Wildcard matches anything
    if isinstance(pattern, WildcardPattern):
        return bindings
    
    # Literal pattern: exact equality
    if isinstance(pattern, LiteralPattern):
        if pattern.value == value:
            return bindings
        return None
    
    # Variable pattern: bind to any value
    if isinstance(pattern, VarPattern):
        name = pattern.name
        
        # If variable already bound, check consistency
        if name in bindings:
            if bindings[name] == value:
                return bindings
            return None
        
        # Bind variable
        bindings[name] = value
        return bindings
    
    # List pattern
    if isinstance(pattern, ListPattern):
        if not isinstance(value, (list, tuple)):
            return None
        
        value_list = list(value)
        
        # Handle rest pattern
        if pattern.rest_var:
            # Must have at least as many elements as non-rest patterns
            if len(value_list) < len(pattern.elements):
                return None
            
            # Match prefix elements
            for i, elem_pattern in enumerate(pattern.elements):
                result = _match_pattern_impl(elem_pattern, value_list[i], bindings)
                if result is None:
                    return None
                bindings = result
            
            # Bind rest to remaining elements
            rest_values = value_list[len(pattern.elements):]
            bindings[pattern.rest_var] = rest_values
            return bindings
        else:
            # Exact length match required
            if len(value_list) != len(pattern.elements):
                return None
            
            # Match each element
            for elem_pattern, elem_value in zip(pattern.elements, value_list):
                result = _match_pattern_impl(elem_pattern, elem_value, bindings)
                if result is None:
                    return None
                bindings = result
            
            return bindings
    
    # Dict pattern
    if isinstance(pattern, DictPattern):
        if not isinstance(value, dict):
            return None
        
        value_dict = dict(value)
        
        # Match each key-pattern pair
        for key, key_pattern in pattern.pairs:
            if key not in value_dict:
                return None
            
            result = _match_pattern_impl(key_pattern, value_dict[key], bindings)
            if result is None:
                return None
            bindings = result
        
        # Handle rest pattern
        if pattern.rest_var:
            matched_keys = {key for key, _ in pattern.pairs}
            rest_dict = {k: v for k, v in value_dict.items() if k not in matched_keys}
            bindings[pattern.rest_var] = rest_dict
        
        return bindings
    
    # Tuple pattern
    if isinstance(pattern, TuplePattern):
        if not isinstance(value, (tuple, list)):
            return None
        
        value_tuple = tuple(value)
        
        # Exact length match required
        if len(value_tuple) != len(pattern.elements):
            return None
        
        # Match each element
        for elem_pattern, elem_value in zip(pattern.elements, value_tuple):
            result = _match_pattern_impl(elem_pattern, elem_value, bindings)
            if result is None:
                return None
            bindings = result
        
        return bindings
    
    # Constructor pattern (symbolic terms)
    if isinstance(pattern, ConstructorPattern):
        # Value must be a dict with 'type' field matching constructor name
        # Or a custom object with matching attributes
        
        if isinstance(value, dict):
            # Dict-based symbolic term
            if value.get('_type') != pattern.name and value.get('type') != pattern.name:
                return None
            
            # Match arguments
            args = value.get('args', value.get('_args', []))
            if not isinstance(args, (list, tuple)):
                return None
            
            args_list = list(args)
            if len(args_list) != len(pattern.args):
                return None
            
            for arg_pattern, arg_value in zip(pattern.args, args_list):
                result = _match_pattern_impl(arg_pattern, arg_value, bindings)
                if result is None:
                    return None
                bindings = result
            
            return bindings
        
        # Object-based symbolic term
        if hasattr(value, '__class__'):
            class_name = value.__class__.__name__
            if class_name != pattern.name:
                return None
            
            # Try to extract args from object
            # Common patterns: .args attribute, or positional fields
            if hasattr(value, 'args'):
                args = value.args
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes, dict)):
                args = list(value)
            else:
                return None
            
            if len(args) != len(pattern.args):
                return None
            
            for arg_pattern, arg_value in zip(pattern.args, args):
                result = _match_pattern_impl(arg_pattern, arg_value, bindings)
                if result is None:
                    return None
                bindings = result
            
            return bindings
        
        return None
    
    # Unknown pattern type
    return None


def extract_pattern_vars(pattern: Pattern) -> set[str]:
    """Extract all variable names from a pattern."""
    vars_set: set[str] = set()
    
    if isinstance(pattern, VarPattern):
        vars_set.add(pattern.name)
    elif isinstance(pattern, ListPattern):
        for elem in pattern.elements:
            vars_set.update(extract_pattern_vars(elem))
        if pattern.rest_var:
            vars_set.add(pattern.rest_var)
    elif isinstance(pattern, DictPattern):
        for _, val_pattern in pattern.pairs:
            vars_set.update(extract_pattern_vars(val_pattern))
        if pattern.rest_var:
            vars_set.add(pattern.rest_var)
    elif isinstance(pattern, TuplePattern):
        for elem in pattern.elements:
            vars_set.update(extract_pattern_vars(elem))
    elif isinstance(pattern, ConstructorPattern):
        for arg in pattern.args:
            vars_set.update(extract_pattern_vars(arg))
    
    return vars_set
