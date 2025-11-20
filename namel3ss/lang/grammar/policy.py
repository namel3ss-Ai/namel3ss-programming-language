"""Policy parsing."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast.policy import PolicyDefinition


class PolicyParserMixin:
    """Mixin providing policy parsing."""

    def _parse_policy(self, line: _Line) -> None:
        """
        Parse a policy definition block.
        
        Grammar:
            policy <name> {
                block_categories: ["self-harm", "hate", "sexual_minors"]
                allow_categories: ["educational"]
                alert_only_categories: ["profanity"]
                redact_pii: true
                max_tokens: 512
                fallback_message: "I can't help with that."
                log_level: "full"
            }
        """
        match = _POLICY_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "policy declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract policy fields
        block_categories = properties.get('block_categories', [])
        if isinstance(block_categories, str):
            # Parse string representation of list
            block_categories = self._parse_string_list(block_categories)
        
        allow_categories = properties.get('allow_categories', [])
        if isinstance(allow_categories, str):
            allow_categories = self._parse_string_list(allow_categories)
        
        alert_only_categories = properties.get('alert_only_categories', [])
        if isinstance(alert_only_categories, str):
            alert_only_categories = self._parse_string_list(alert_only_categories)
        
        redact_pii = properties.get('redact_pii', False)
        if isinstance(redact_pii, str):
            redact_pii = redact_pii.lower() in ('true', 'yes', '1')
        
        max_tokens = properties.get('max_tokens')
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        
        fallback_message = properties.get('fallback_message')
        if fallback_message and isinstance(fallback_message, str):
            # Remove quotes if present
            if fallback_message.startswith('"') and fallback_message.endswith('"'):
                fallback_message = fallback_message[1:-1]
            elif fallback_message.startswith("'") and fallback_message.endswith("'"):
                fallback_message = fallback_message[1:-1]
        
        log_level = properties.get('log_level', 'full')
        if isinstance(log_level, str):
            log_level = log_level.strip('"').strip("'")
        
        # Build config from remaining properties
        config = {
            k: v for k, v in properties.items()
            if k not in {
                'block_categories', 'allow_categories', 'alert_only_categories',
                'redact_pii', 'max_tokens', 'fallback_message', 'log_level'
            }
        }
        
        policy = PolicyDefinition(
            name=name,
            block_categories=block_categories,
            allow_categories=allow_categories,
            alert_only_categories=alert_only_categories,
            redact_pii=redact_pii,
            max_tokens=max_tokens,
            fallback_message=fallback_message,
            log_level=log_level,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.policies.append(policy)

    def _parse_kv_block_braces(self, parent_indent: int) -> dict[str, any]:
        """KV block parser - moved to utility_parsers.py mixin."""
        pass

__all__ = ['PolicyParserMixin']
