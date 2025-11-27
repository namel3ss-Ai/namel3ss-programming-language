from __future__ import annotations

import pytest

from namel3ss.lang.parser import N3SyntaxError
from namel3ss.lang.grammar import parse_module


def test_comment_metadata_is_captured() -> None:
    source = """
# 💬 Module header
app "Demo".

# ⚠️ Page description
"""

    module = parse_module(source, path="comments.ai")

    assert len(module.comments) == 2
    header = module.comments[0]
    assert header.text == "Module header"
    assert header.emoji == "💬"
    assert header.line == 2

    warning = module.comments[1]
    assert warning.text == "Page description"
    assert warning.emoji == "⚠"
    assert warning.line == 5

    app = module.body[0]
    assert getattr(app, "name", "") == "Demo"


@pytest.mark.parametrize(
    "bad_line",
    [
        "#⚠️Missing space",
        "## Wrong marker",
        "// Not allowed",
        "/* Not allowed */",
    ],
)
def test_invalid_comment_markers_raise(bad_line: str) -> None:
    source = f"""
app "Test".
{bad_line}
page "Home" at "/":
  show text "Hello"
"""
    with pytest.raises(N3SyntaxError):
        parse_module(source)
