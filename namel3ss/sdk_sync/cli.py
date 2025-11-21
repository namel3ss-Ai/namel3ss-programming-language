"""
CLI command for SDK generation.

Usage:
    namel3ss sdk-sync python --backend http://localhost:8000 --out ./sdk
    namel3ss sdk-sync python --schema-dir ./schemas --out ./sdk
    namel3ss sdk-sync python --spec-file ./spec.json --out ./sdk

Example:
    Generate Python SDK from running backend:
    ```bash
    namel3ss sdk-sync python \\
        --backend http://localhost:8000 \\
        --out ./my_sdk \\
        --package-name my_n3_sdk \\
        --base-url http://api.example.com
    ```
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional
import sys

from .registry import SchemaRegistry, SchemaExporter
from .generators.python import PythonSDKGenerator
from .ir import SchemaVersion, IRSpec
from .errors import SDKSyncError


async def generate_python_sdk(
    backend_url: Optional[str] = None,
    schema_dir: Optional[Path] = None,
    spec_file: Optional[Path] = None,
    output_dir: Path = Path("./sdk"),
    package_name: str = "n3_sdk",
    base_url: str = "http://localhost:8000",
    version: str = "1.0.0",
) -> None:
    """
    Generate Python SDK from N3 schemas.
    
    Args:
        backend_url: URL of running N3 backend
        schema_dir: Directory containing schema files
        spec_file: Path to IR spec JSON file
        output_dir: Output directory for SDK
        package_name: Name for generated package
        base_url: Default API base URL for clients
        version: SDK version
    """
    print(f"🚀 Generating Python SDK: {package_name}")
    print()

    # Step 1: Export schemas to IR
    print("📊 Step 1/4: Exporting schemas...")
    spec: Optional[IRSpec] = None

    try:
        if backend_url:
            print(f"   Source: Backend at {backend_url}")
            exporter = SchemaExporter(backend_url=backend_url)
            spec = await exporter.export_from_backend()
        elif schema_dir:
            print(f"   Source: Schema directory {schema_dir}")
            exporter = SchemaExporter()
            spec = exporter.export_from_directory(schema_dir)
        elif spec_file:
            print(f"   Source: Spec file {spec_file}")
            import json

            with open(spec_file) as f:
                spec_data = json.load(f)
            # Parse spec from JSON
            spec = IRSpec(
                version=SchemaVersion.parse(spec_data.get("version", version)),
                api_version=spec_data.get("api_version", "1.0"),
                metadata=spec_data.get("metadata", {}),
            )
        else:
            raise SDKSyncError(
                "Must provide one of: --backend, --schema-dir, or --spec-file",
                code="SDK009",
            )

        if not spec:
            raise SDKSyncError("Failed to export schemas", code="SDK010")

        print(f"   ✓ Exported {len(spec.models)} models, {len(spec.tools)} tools")
        print()

    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        sys.exit(1)

    # Step 2: Register schemas
    print("📝 Step 2/4: Registering schemas...")
    try:
        registry = SchemaRegistry()
        registry.import_spec(spec)
        print(f"   ✓ Registered all schemas")
        print()
    except Exception as e:
        print(f"   ✗ Registration failed: {e}")
        sys.exit(1)

    # Step 3: Generate Python SDK
    print("⚙️  Step 3/4: Generating Python code...")
    try:
        generator = PythonSDKGenerator(
            spec=spec,
            package_name=package_name,
            output_dir=output_dir,
            base_url=base_url,
        )
        await generator.generate()
        print(f"   ✓ Generated SDK at {output_dir / package_name}")
        print()
    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        sys.exit(1)

    # Step 4: Summary
    print("✅ Step 4/4: Complete!")
    print()
    print("Generated SDK:")
    print(f"   Package: {package_name}")
    print(f"   Location: {output_dir}")
    print(f"   Schema version: {spec.version}")
    print(f"   API version: {spec.api_version}")
    print()
    print("Next steps:")
    print(f"   cd {output_dir}")
    print(f"   pip install -e .")
    print()


def sdk_sync_command(args: argparse.Namespace) -> None:
    """
    Execute sdk-sync command.
    
    Args:
        args: Parsed command-line arguments
    """
    if args.language != "python":
        print(f"Error: Unsupported language: {args.language}")
        print("Supported languages: python")
        sys.exit(1)

    # Run async generation
    asyncio.run(
        generate_python_sdk(
            backend_url=args.backend,
            schema_dir=Path(args.schema_dir) if args.schema_dir else None,
            spec_file=Path(args.spec_file) if args.spec_file else None,
            output_dir=Path(args.out),
            package_name=args.package_name,
            base_url=args.base_url,
            version=args.version,
        )
    )


def add_sdk_sync_command(subparsers) -> None:
    """
    Add sdk-sync subcommand to CLI.
    
    Args:
        subparsers: Argparse subparsers object
    """
    sdk_parser = subparsers.add_parser(
        "sdk-sync",
        help="Generate typed SDK from N3 schemas",
        description="Generate fully typed client SDKs from N3 schemas and tool specifications",
    )

    sdk_parser.add_argument(
        "language",
        choices=["python"],
        help="Target language for SDK generation",
    )

    # Input sources (mutually exclusive)
    input_group = sdk_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--backend",
        help="URL of running N3 backend to export schemas from",
    )
    input_group.add_argument(
        "--schema-dir",
        help="Directory containing N3 schema files",
    )
    input_group.add_argument(
        "--spec-file",
        help="Path to IR spec JSON file",
    )

    # Output options
    sdk_parser.add_argument(
        "--out",
        "-o",
        default="./sdk",
        help="Output directory for generated SDK (default: ./sdk)",
    )
    sdk_parser.add_argument(
        "--package-name",
        default="n3_sdk",
        help="Name for generated package (default: n3_sdk)",
    )
    sdk_parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Default API base URL for clients (default: http://localhost:8000)",
    )
    sdk_parser.add_argument(
        "--version",
        default="1.0.0",
        help="SDK version (default: 1.0.0)",
    )

    # Advanced options
    sdk_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (fail on warnings)",
    )
    sdk_parser.add_argument(
        "--no-format",
        action="store_true",
        help="Skip formatting generated code with ruff",
    )
    sdk_parser.add_argument(
        "--save-spec",
        help="Save IR spec to file for reuse",
    )

    sdk_parser.set_defaults(func=sdk_sync_command)


# For standalone usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate typed SDK from N3 schemas"
    )
    add_sdk_sync_command(parser.add_subparsers(dest="command"))
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
