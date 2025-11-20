"""Response parsing for LLM API responses."""

from textwrap import dedent

RESPONSE_PARSER = dedent(
    '''
def _extract_llm_text(
    provider: str,
    payload: Any,
    config: Dict[str, Any],
) -> str:
    response_path = config.get("response_path") or config.get("result_path")
    if response_path:
        value = _traverse_attribute_path(payload, _as_path_segments(response_path))
        if value is not None:
            return str(value)
    provider_key = provider.lower()
    if isinstance(payload, dict):
        if provider_key == "openai":
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            segments: List[str] = []
                            for entry in content:
                                if isinstance(entry, dict) and "text" in entry:
                                    segments.append(str(entry["text"]))
                                else:
                                    segments.append(str(entry))
                            joined = "\\n".join(segment for segment in segments if segment)
                            if joined:
                                return joined
                        if content is not None:
                            return str(content)
                    if "text" in first_choice:
                        return str(first_choice["text"])
        if provider_key == "anthropic":
            content = payload.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first["text"])
                    if "value" in first:
                        return str(first["value"])
            if isinstance(content, str):
                return content
        if "result" in payload and isinstance(payload["result"], str):
            return payload["result"]
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _extract_memory_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
        return names
    return []


_SCHEMA_TYPE_MAPPING = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
}
'''
).strip()

__all__ = ['RESPONSE_PARSER']
