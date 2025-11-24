"""Smoke tests for the React/Vite frontend generator."""

import json
import re
import subprocess

from namel3ss.ast import App
from namel3ss.codegen.frontend.react_vite import generate_react_vite_site
from namel3ss.parser import Parser


def _ts_helpers_to_js(snippet: str) -> str:
    """Convert the helper functions emitted in n3Client.ts to runnable JS."""

    transformed = snippet
    transformed = re.sub(r"type\s+UnknownRecord\s*=\s*[^;]+;\n", "", transformed)
    transformed = re.sub(r"\s+as\s+[A-Za-z0-9_<\>\[\]\s\|\.]+", "", transformed)
    transformed = re.sub(r"Map<[^>]+>", "Map", transformed)

    def _strip_signature(match: re.Match[str]) -> str:
        name = match.group("name")
        raw_params = match.group("params")
        params = []
        for param in raw_params.split(","):
            text = param.strip()
            if not text:
                continue
            if ":" in text:
                text = text.split(":", 1)[0].strip()
            if text.endswith("?"):
                text = text[:-1]
            params.append(text)
        return f"function {name}({', '.join(params)}) {{"

    transformed = re.sub(
        r"function\s+(?P<name>[A-Za-z0-9_]+)\s*\((?P<params>[^)]*)\)\s*:[^\{]+\{",
        _strip_signature,
        transformed,
    )

    transformed = re.sub(r"const\s+([A-Za-z0-9_]+)\s*:\s*UnknownRecord\[\]", r"const \1", transformed)
    transformed = re.sub(r"const\s+([A-Za-z0-9_]+)\s*:\s*UnknownRecord", r"const \1", transformed)
    transformed = re.sub(r"let\s+([A-Za-z0-9_]+)\s*:\s*UnknownRecord\[\]", r"let \1", transformed)
    transformed = re.sub(r"let\s+([A-Za-z0-9_]+)\s*:\s*UnknownRecord", r"let \1", transformed)
    transformed += "\nmodule.exports = { mergeOptimisticData };\n"
    return transformed


def _build_sample_app() -> App:
    source = (
        'app "Optimistic UI" connects to postgres "APP_DB".\n\n'
        'page "Home" at "/":\n'
        '  show table "Orders" from table orders\n'
        '    columns: id, status\n'
        '  show form "Add Order":\n'
        '    fields: customer_name\n'
        '    on submit:\n'
        '      show toast "Saved"\n'
    )
    return Parser(source).parse_app()


def test_react_vite_generates_pending_indicators(tmp_path) -> None:
    app = _build_sample_app()
    output_dir = tmp_path / "frontend"
    generate_react_vite_site(app, str(output_dir))

    form_widget = (output_dir / "src" / "components" / "FormWidget.tsx").read_text(encoding="utf-8")
    assert "setSubmitting(true)" in form_widget
    assert "disabled={submitting}" in form_widget
    assert "submitting ? \"Submitting...\" : \"Submit\"" in form_widget

    page_component = (output_dir / "src" / "pages" / "index.tsx").read_text(encoding="utf-8")
    assert "resolveWidgetData" in page_component
    assert "PAGE_DEFINITION.preview" in page_component

    table_widget = (output_dir / "src" / "components" / "TableWidget.tsx").read_text(encoding="utf-8")
    assert "Array.isArray((data as any)?.rows)" in table_widget
    assert "JSON.stringify(data ?? widget, null, 2)" in table_widget


def test_resolve_widget_data_merges_optimistic_overlays(tmp_path) -> None:
        app = _build_sample_app()
        output_dir = tmp_path / "frontend"
        generate_react_vite_site(app, str(output_dir))

        client_lib = (output_dir / "src" / "lib" / "n3Client.ts").read_text(encoding="utf-8")
        assert "function mergeOptimisticData" in client_lib

        match = re.search(
                r"type UnknownRecord[\s\S]+?return mutated \? baseRecord : base;\n\s*}\n",
                client_lib,
        )
        assert match, "Expected optimistic helpers block in n3Client.ts"

        js_runtime = _ts_helpers_to_js(match.group(0))

        script = (
                js_runtime
                + """
const basePayload = {
    rows: [
        { id: 1, status: "pending", amount: 10 },
        { id: 2, status: "done", amount: 20 }
    ],
    summary: { total: 30 },
    metadata: { cursor: "abc" },
    datasets: [
        {
            id: "recent",
            rows: [{ id: "r-1", status: "processing" }],
            summary: { total: 1 },
            metadata: { cursor: "r0" }
        }
    ]
};

const optimisticOverlay = {
    pending: true,
    data: {
        optimisticRows: [
            { id: 2, status: "processing" }
        ],
        appendRows: [
            { id: 3, status: "new", amount: 5 }
        ],
        replaceRows: {
            "1": { status: "complete", amount: 11 }
        },
        optimisticSummary: { delta: 1 },
        metadata: { cursor: "def", marker: "opt" },
        optimisticDatasets: [
            {
                id: "recent",
                optimisticRows: [
                    { id: "r-1", status: "done" }
                ],
                appendRows: [
                    { id: "r-2", status: "queued" }
                ],
                metadata: { cursor: "r1" }
            },
            {
                id: "next",
                rows: [
                    { id: "n-1", status: "seed" }
                ],
                summary: { total: 1 }
            }
        ],
        status: "optimistic"
    }
};

const result = mergeOptimisticData(basePayload, optimisticOverlay);
console.log(JSON.stringify({
    pending: result && result.pending,
    rows: result && result.rows,
    summary: result && result.summary,
    metadata: result && result.metadata,
    datasets: result && result.datasets,
    status: result && result.status
}));
"""
        )

        completed = subprocess.run(
                ["node", "-e", script],
                check=True,
                capture_output=True,
                text=True,
        )
        payload = json.loads(completed.stdout.strip())

        assert payload["pending"] is True
        assert [row["id"] for row in payload["rows"]] == [1, 2, 3]

        row1 = next(row for row in payload["rows"] if row["id"] == 1)
        assert row1["status"] == "complete"
        assert row1["amount"] == 11

        row2 = next(row for row in payload["rows"] if row["id"] == 2)
        assert row2["status"] == "processing"

        row3 = next(row for row in payload["rows"] if row["id"] == 3)
        assert row3["status"] == "new"
        assert row3["amount"] == 5

        assert payload["summary"] == {"total": 30, "delta": 1}
        assert payload["metadata"] == {"cursor": "def", "marker": "opt"}

        assert isinstance(payload["datasets"], list)
        assert sorted(dataset["id"] for dataset in payload["datasets"]) == ["next", "recent"]

        recent = next(dataset for dataset in payload["datasets"] if dataset["id"] == "recent")
        assert [row["id"] for row in recent["rows"]] == ["r-1", "r-2"]
        recent_r1 = next(row for row in recent["rows"] if row["id"] == "r-1")
        assert recent_r1["status"] == "done"
        recent_r2 = next(row for row in recent["rows"] if row["id"] == "r-2")
        assert recent_r2["status"] == "queued"
        assert recent["metadata"] == {"cursor": "r1"}

        next_dataset = next(dataset for dataset in payload["datasets"] if dataset["id"] == "next")
        assert [row["id"] for row in next_dataset["rows"]] == ["n-1"]
        assert next_dataset["summary"] == {"total": 1}

        assert payload["status"] == "optimistic"
