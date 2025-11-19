from namel3ss.errors import N3SyntaxError, N3TypeError


def test_error_format_includes_metadata() -> None:
    err = N3TypeError(
        "Type mismatch",
        path="demo.n3",
        line=4,
        column=2,
        code="TYPE_MISMATCH",
        hint="Ensure operands use compatible types.",
    )
    formatted = err.format()
    assert "Type mismatch" in formatted
    assert "demo.n3:4:2" in formatted
    assert "TYPE_MISMATCH" in formatted
    assert "Ensure operands" in formatted


def test_error_format_handles_missing_location() -> None:
    err = N3SyntaxError("Unexpected token")
    formatted = err.format()
    assert formatted.startswith("Unexpected token")
    assert "(" not in formatted