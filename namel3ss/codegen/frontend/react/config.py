"""Configuration file generators for Vite + React + TypeScript project."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .utils import write_file


def write_package_json(out: Path) -> None:
    """Generate package.json with React, Vite, and TypeScript dependencies."""
    package = {
        "name": "namel3ss-react-frontend",
        "private": True,
        "version": "0.1.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "tsc && vite build",
            "preview": "vite preview",
        },
        "dependencies": {
            "react": "^18.3.1",
            "react-dom": "^18.3.1",
            "react-router-dom": "^6.28.0",
        },
        "devDependencies": {
            "@types/node": "^20.11.30",
            "@types/react": "^18.2.73",
            "@types/react-dom": "^18.2.24",
            "@vitejs/plugin-react": "^4.2.1",
            "typescript": "^5.4.5",
            "vite": "^5.3.1",
        },
    }
    write_file(out / "package.json", json.dumps(package, indent=2) + "\n")


def write_tsconfig(out: Path) -> None:
    """Generate TypeScript configuration for the React project."""
    tsconfig = {
        "compilerOptions": {
            "target": "ESNext",
            "useDefineForClassFields": True,
            "module": "ESNext",
            "moduleResolution": "Node",
            "strict": True,
            "jsx": "react-jsx",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
        },
        "include": ["src"],
        "references": [{"path": "./tsconfig.node.json"}],
    }
    write_file(out / "tsconfig.json", json.dumps(tsconfig, indent=2) + "\n")


def write_tsconfig_node(out: Path) -> None:
    """Generate TypeScript configuration for Vite config file."""
    tsconfig_node = {
        "compilerOptions": {
            "composite": True,
            "module": "ESNext",
            "moduleResolution": "Node",
            "allowSyntheticDefaultImports": True,
        },
        "include": ["vite.config.ts"],
    }
    write_file(out / "tsconfig.node.json", json.dumps(tsconfig_node, indent=2) + "\n")


def write_vite_config(out: Path) -> None:
    """Generate Vite configuration with React plugin and API proxy."""
    content = textwrap.dedent(
        """
        import { defineConfig } from "vite";
        import react from "@vitejs/plugin-react";

        export default defineConfig({
          plugins: [react()],
          server: {
            proxy: {
              "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
              },
              "/ws": {
                target: "http://localhost:8000",
                ws: true,
              },
            },
          },
        });
        """
    ).strip() + "\n"
    write_file(out / "vite.config.ts", content)
