"""Built-in functions for the symbolic expression evaluator."""

from __future__ import annotations

import functools
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = [
    "BUILTIN_FUNCTIONS",
    "get_builtin",
]


def _map_fn(func: Callable, seq: Iterable) -> List[Any]:
    """Map function over sequence."""
    return [func(item) for item in seq]


def _filter_fn(pred: Callable, seq: Iterable) -> List[Any]:
    """Filter sequence by predicate."""
    return [item for item in seq if pred(item)]


def _reduce_fn(func: Callable, seq: Iterable, *args) -> Any:
    """Reduce sequence with function."""
    if args:
        return functools.reduce(func, seq, args[0])
    return functools.reduce(func, seq)


def _fold_right(func: Callable, seq: Iterable, init: Any) -> Any:
    """Right fold over sequence."""
    result = init
    for item in reversed(list(seq)):
        result = func(item, result)
    return result


def _zip_fn(*seqs) -> List[tuple]:
    """Zip sequences together."""
    return list(zip(*seqs))


def _enumerate_fn(seq: Iterable, start: int = 0) -> List[tuple]:
    """Enumerate sequence with indices."""
    return list(enumerate(seq, start))


# List operations
def _head(xs: List) -> Any:
    """Return first element of list."""
    if not xs:
        raise IndexError("head of empty list")
    return xs[0]


def _tail(xs: List) -> List:
    """Return list without first element."""
    if not xs:
        return []
    return xs[1:]


def _init(xs: List) -> List:
    """Return list without last element."""
    if not xs:
        return []
    return xs[:-1]


def _last(xs: List) -> Any:
    """Return last element of list."""
    if not xs:
        raise IndexError("last of empty list")
    return xs[-1]


def _cons(x: Any, xs: List) -> List:
    """Prepend element to list."""
    return [x] + list(xs)


def _append(xs: List, ys: List) -> List:
    """Concatenate two lists."""
    return list(xs) + list(ys)


def _concat(lists: List[List]) -> List:
    """Concatenate list of lists."""
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def _reverse(xs: List) -> List:
    """Reverse list."""
    return list(reversed(xs))


def _sort(xs: List, key: Optional[Callable] = None, reverse: bool = False) -> List:
    """Sort list."""
    return sorted(xs, key=key, reverse=reverse)


def _length(xs: Iterable) -> int:
    """Return length of sequence."""
    if hasattr(xs, '__len__'):
        return len(xs)
    return sum(1 for _ in xs)


def _nth(xs: List, n: int) -> Any:
    """Return nth element of list (0-indexed)."""
    if 0 <= n < len(xs):
        return xs[n]
    raise IndexError(f"index {n} out of range for list of length {len(xs)}")


def _take(n: int, xs: List) -> List:
    """Take first n elements."""
    return list(xs)[:n]


def _drop(n: int, xs: List) -> List:
    """Drop first n elements."""
    return list(xs)[n:]


def _slice(xs: List, start: Optional[int] = None, end: Optional[int] = None, step: Optional[int] = None) -> List:
    """Slice list."""
    return list(xs)[start:end:step]


# Numeric operations
def _sum(xs: Iterable) -> Any:
    """Sum of sequence."""
    return sum(xs)


def _product(xs: Iterable) -> Any:
    """Product of sequence."""
    return functools.reduce(operator.mul, xs, 1)


def _min_fn(*args) -> Any:
    """Minimum value."""
    if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
        return min(args[0])
    return min(args)


def _max_fn(*args) -> Any:
    """Maximum value."""
    if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
        return max(args[0])
    return max(args)


def _abs_fn(x: Any) -> Any:
    """Absolute value."""
    return abs(x)


def _round_fn(x: float, ndigits: int = 0) -> Any:
    """Round number."""
    return round(x, ndigits)


def _floor(x: float) -> int:
    """Floor function."""
    import math
    return math.floor(x)


def _ceil(x: float) -> int:
    """Ceiling function."""
    import math
    return math.ceil(x)


def _range_fn(*args) -> List[int]:
    """Generate range of integers."""
    return list(range(*args))


# String operations
def _concat_str(*strs) -> str:
    """Concatenate strings."""
    return ''.join(str(s) for s in strs)


def _split(s: str, sep: Optional[str] = None, maxsplit: int = -1) -> List[str]:
    """Split string."""
    return s.split(sep, maxsplit)


def _join(sep: str, xs: Iterable) -> str:
    """Join sequence with separator."""
    return sep.join(str(x) for x in xs)


def _lower(s: str) -> str:
    """Convert to lowercase."""
    return s.lower()


def _upper(s: str) -> str:
    """Convert to uppercase."""
    return s.upper()


def _strip(s: str, chars: Optional[str] = None) -> str:
    """Strip whitespace or characters."""
    return s.strip(chars)


def _replace(s: str, old: str, new: str, count: int = -1) -> str:
    """Replace substring."""
    return s.replace(old, new, count)


def _startswith(s: str, prefix: str) -> bool:
    """Check if string starts with prefix."""
    return s.startswith(prefix)


def _endswith(s: str, suffix: str) -> bool:
    """Check if string ends with suffix."""
    return s.endswith(suffix)


# Dict operations
def _keys(d: Dict) -> List:
    """Get dict keys."""
    return list(d.keys())


def _values(d: Dict) -> List:
    """Get dict values."""
    return list(d.values())


def _items(d: Dict) -> List[tuple]:
    """Get dict items."""
    return list(d.items())


def _get(d: Dict, key: Any, default: Any = None) -> Any:
    """Get dict value with default."""
    return d.get(key, default)


def _has_key(d: Dict, key: Any) -> bool:
    """Check if dict has key."""
    return key in d


def _merge(*dicts) -> Dict:
    """Merge dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


# Predicates and logic
def _all_fn(xs: Iterable) -> bool:
    """Check if all elements are truthy."""
    return all(xs)


def _any_fn(xs: Iterable) -> bool:
    """Check if any element is truthy."""
    return any(xs)


def _not_fn(x: Any) -> bool:
    """Logical not."""
    return not x


def _is_empty(xs: Any) -> bool:
    """Check if sequence is empty."""
    if hasattr(xs, '__len__'):
        return len(xs) == 0
    return not any(True for _ in xs)


def _is_none(x: Any) -> bool:
    """Check if value is None."""
    return x is None


def _is_list(x: Any) -> bool:
    """Check if value is a list."""
    return isinstance(x, list)


def _is_dict(x: Any) -> bool:
    """Check if value is a dict."""
    return isinstance(x, dict)


def _is_string(x: Any) -> bool:
    """Check if value is a string."""
    return isinstance(x, str)


def _is_number(x: Any) -> bool:
    """Check if value is a number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_bool(x: Any) -> bool:
    """Check if value is a boolean."""
    return isinstance(x, bool)


# Utilities
def _identity(x: Any) -> Any:
    """Identity function."""
    return x


def _const(x: Any) -> Callable:
    """Constant function."""
    return lambda *args, **kwargs: x


def _compose(*funcs) -> Callable:
    """Compose functions right-to-left."""
    def composed(x):
        result = x
        for func in reversed(funcs):
            result = func(result)
        return result
    return composed


def _pipe(*funcs) -> Callable:
    """Compose functions left-to-right."""
    def piped(x):
        result = x
        for func in funcs:
            result = func(result)
        return result
    return piped


def _assert_fn(condition: bool, message: str = "Assertion failed") -> None:
    """Assert condition is true."""
    if not condition:
        raise AssertionError(message)


# Type conversions
def _int(x: Any) -> int:
    """Convert to integer."""
    return int(x)


def _float(x: Any) -> float:
    """Convert to float."""
    return float(x)


def _str(x: Any) -> str:
    """Convert to string."""
    return str(x)


def _bool(x: Any) -> bool:
    """Convert to boolean."""
    return bool(x)


def _list(x: Any) -> List:
    """Convert to list."""
    return list(x)


def _dict(x: Any) -> Dict:
    """Convert to dict."""
    return dict(x)


# Built-in functions registry
BUILTIN_FUNCTIONS: Dict[str, Callable] = {
    # Higher-order functions
    'map': _map_fn,
    'filter': _filter_fn,
    'reduce': _reduce_fn,
    'fold': _fold_right,
    'zip': _zip_fn,
    'enumerate': _enumerate_fn,
    
    # List operations
    'head': _head,
    'tail': _tail,
    'init': _init,
    'last': _last,
    'cons': _cons,
    'append': _append,
    'concat': _concat,
    'reverse': _reverse,
    'sort': _sort,
    'length': _length,
    'len': _length,
    'nth': _nth,
    'take': _take,
    'drop': _drop,
    'slice': _slice,
    
    # Numeric
    'sum': _sum,
    'product': _product,
    'min': _min_fn,
    'max': _max_fn,
    'abs': _abs_fn,
    'round': _round_fn,
    'floor': _floor,
    'ceil': _ceil,
    'range': _range_fn,
    
    # String operations
    'concat_str': _concat_str,
    'split': _split,
    'join': _join,
    'lower': _lower,
    'upper': _upper,
    'strip': _strip,
    'replace': _replace,
    'startswith': _startswith,
    'endswith': _endswith,
    
    # Dict operations
    'keys': _keys,
    'values': _values,
    'items': _items,
    'get': _get,
    'has_key': _has_key,
    'merge': _merge,
    
    # Predicates
    'all': _all_fn,
    'any': _any_fn,
    'not': _not_fn,
    'is_empty': _is_empty,
    'is_none': _is_none,
    'is_list': _is_list,
    'is_dict': _is_dict,
    'is_string': _is_string,
    'is_str': _is_string,  # Alias
    'is_number': _is_number,
    'is_int': lambda x: isinstance(x, int) and not isinstance(x, bool),
    'is_float': lambda x: isinstance(x, float),
    'is_bool': _is_bool,
    
    # Utilities
    'identity': _identity,
    'const': _const,
    'compose': _compose,
    'pipe': _pipe,
    'assert': _assert_fn,
    
    # Type conversions
    'int': _int,
    'float': _float,
    'str': _str,
    'bool': _bool,
    'list': _list,
    'dict': _dict,
}


def get_builtin(name: str) -> Optional[Callable]:
    """Get built-in function by name."""
    return BUILTIN_FUNCTIONS.get(name)
