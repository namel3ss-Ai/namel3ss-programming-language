"""
Dependency File Generator

Generates requirements.txt and package.json from detected features while
preserving user customizations.

Strategy:
--------
1. Read existing dependency files (if any)
2. Determine required dependencies from features
3. Merge with existing, preserving user additions
4. Generate clean, deterministic output

Non-Destructive Updates:
-----------------------
- Preserves user-added packages
- Respects existing version pins
- Maintains alphabetical ordering
- Adds comments to separate sections
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional
import json
import re

from .spec import get_python_packages_for_features, get_npm_packages_for_features, PythonPackage, NPMPackage


@dataclass
class GeneratedDeps:
    """Result of dependency generation"""
    requirements_txt: str
    package_json: str  # JSON string
    added_python: List[str] = field(default_factory=list)
    added_npm: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DependencyGenerator:
    """
    Generates dependency files from feature sets.
    
    Usage:
        generator = DependencyGenerator()
        deps = generator.generate({'openai', 'chat', 'postgres'})
        Path('requirements.txt').write_text(deps.requirements_txt)
    """
    
    def __init__(self):
        self.warnings: List[str] = []
    
    def generate(
        self,
        python_packages_or_features,
        npm_packages_or_existing_requirements=None,
        existing_package_json: Optional[Dict] = None,
        existing_requirements_path: Optional[Path] = None,
        existing_package_json_path: Optional[Path] = None,
    ) -> GeneratedDeps:
        """
        Generate dependency files.
        
        Two calling modes:
        1. generate(features: Set[str], existing_requirements=..., existing_package_json=...)
        2. generate(python_packages: List[PythonPackage], npm_packages: List[NPMPackage])
        
        Args:
            python_packages_or_features: Either Set[str] of features OR List[PythonPackage]
            npm_packages_or_existing_requirements: Either List[NPMPackage] OR existing requirements string
            existing_package_json: Existing package.json dict (mode 1 only)
            existing_requirements_path: Path to existing requirements.txt (mode 1 only)
            existing_package_json_path: Path to existing package.json (mode 1 only)
            
        Returns:
            GeneratedDeps with file contents
        """
        # Determine calling mode
        if isinstance(python_packages_or_features, (list, tuple)):
            # Mode 2: Direct package lists
            python_packages = python_packages_or_features
            npm_packages = npm_packages_or_existing_requirements if isinstance(npm_packages_or_existing_requirements, (list, tuple)) else []
            existing_requirements = None
            existing_package_json = None
            
            # Load existing files from paths if provided (mode 2 can also use paths)
            if existing_requirements_path and Path(existing_requirements_path).exists():
                existing_requirements = Path(existing_requirements_path).read_text(encoding='utf-8')
            if existing_package_json_path and Path(existing_package_json_path).exists():
                try:
                    existing_package_json = json.loads(Path(existing_package_json_path).read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    # Malformed JSON, ignore and start fresh
                    existing_package_json = None
        else:
            # Mode 1: Feature-based
            features = python_packages_or_features
            existing_requirements = npm_packages_or_existing_requirements
            
            # Load existing files from paths if provided
            if existing_requirements_path and Path(existing_requirements_path).exists():
                existing_requirements = Path(existing_requirements_path).read_text(encoding='utf-8')
            if existing_package_json_path and Path(existing_package_json_path).exists():
                try:
                    existing_package_json = json.loads(Path(existing_package_json_path).read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    # Malformed JSON, ignore and start fresh
                    existing_package_json = None
            
            # Get required packages from features
            python_packages = get_python_packages_for_features(features)
            npm_packages = get_npm_packages_for_features(features)
        
        # Generate requirements.txt
        requirements_txt, added_python = self._generate_requirements_txt(
            python_packages,
            existing_requirements
        )
        
        # Generate package.json
        package_json_dict, added_npm = self._generate_package_json(
            npm_packages,
            existing_package_json
        )
        
        return GeneratedDeps(
            requirements_txt=requirements_txt,
            package_json=json.dumps(package_json_dict, indent=2),
            added_python=added_python,
            added_npm=added_npm,
            warnings=self.warnings.copy()
        )
    
    def _generate_requirements_txt(
        self,
        packages: List[PythonPackage],
        existing: Optional[str]
    ) -> tuple[str, List[str]]:
        """Generate requirements.txt content"""
        # Parse existing requirements
        existing_packages = self._parse_requirements_txt(existing or "")
        
        # Build new requirements
        required_packages = {}
        for pkg in packages:
            req_line = pkg.to_requirement()
            required_packages[pkg.name] = req_line
        
        # Merge: keep existing versions if present, add new packages
        merged = {}
        added = []
        
        # First, add all existing (preserves user customizations)
        for name, line in existing_packages.items():
            merged[name] = line
        
        # Then add/update required packages
        for name, line in required_packages.items():
            if name not in merged:
                merged[name] = line
                added.append(name)
            # If already in merged, keep existing version (user customization)
        
        # Sort alphabetically for deterministic output
        sorted_names = sorted(merged.keys())
        
        # If no packages, return minimal content
        if not sorted_names:
            return "# Generated by namel3ss sync-deps\n", []
        
        # Build output
        lines = [
            "# Generated by namel3ss sync-deps",
            "# Core dependencies (required)",
            ""
        ]
        
        # Separate core from features
        core_packages = ['fastapi', 'uvicorn', 'pydantic', 'httpx']
        feature_packages = [n for n in sorted_names if n not in core_packages]
        
        # Add core first
        for name in sorted_names:
            if name in core_packages:
                lines.append(merged[name])
        
        if feature_packages:
            lines.extend(["", "# Feature dependencies", ""])
            for name in feature_packages:
                lines.append(merged[name])
        
        lines.append("")  # Trailing newline
        
        return "\n".join(lines), added
    
    def _generate_package_json(
        self,
        packages: List[NPMPackage],
        existing: Optional[Dict]
    ) -> tuple[Dict, List[str]]:
        """Generate package.json content"""
        # Start with existing or template
        if existing:
            package_json = existing.copy()
        else:
            package_json = {
                "name": "namel3ss-app",
                "private": True,
                "version": "0.1.0",
                "type": "module",
                "scripts": {
                    "dev": "vite",
                    "build": "tsc && vite build",
                    "preview": "vite preview"
                }
            }
        
        # Ensure dependencies and devDependencies exist
        if "dependencies" not in package_json:
            package_json["dependencies"] = {}
        if "devDependencies" not in package_json:
            package_json["devDependencies"] = {}
        
        deps = package_json["dependencies"]
        dev_deps = package_json["devDependencies"]
        added = []
        
        # Add required packages
        for pkg in packages:
            name, version = pkg.to_package_json_entry()
            target = dev_deps if pkg.dev else deps
            
            if name not in target and name not in deps and name not in dev_deps:
                target[name] = version
                added.append(name)
            # If already present, preserve existing version
        
        # Sort dependencies alphabetically
        package_json["dependencies"] = dict(sorted(deps.items()))
        package_json["devDependencies"] = dict(sorted(dev_deps.items()))
        
        return package_json, added
    
    def _parse_requirements_txt(self, content: str) -> Dict[str, str]:
        """Parse requirements.txt into package name → requirement line"""
        packages = {}
        
        for line in content.splitlines():
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip -e editable installs (preserve separately)
            if line.startswith('-e'):
                continue
            
            # Extract package name
            # Handle: package, package==1.0, package>=1.0,<2.0, package[extras]>=1.0
            match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*?\])?(.*)?$', line)
            if match:
                pkg_name = match.group(1)
                packages[pkg_name] = line
        
        return packages
    
    def generate_requirements_file(
        self,
        features: Set[str],
        output_path: str | Path,
        preserve_existing: bool = True,
        existing_path: Optional[Path] = None
    ) -> List[str]:
        """
        Generate requirements.txt file.
        
        Args:
            features: Set of feature IDs
            output_path: Where to write requirements.txt
            preserve_existing: Whether to preserve existing content
            existing_path: Optional path to existing requirements.txt (overrides preserve_existing logic)
            
        Returns:
            List of added package names
        """
        output_path = Path(output_path)
        
        # Read existing if preserve
        existing = None
        if existing_path and existing_path.exists():
            existing = existing_path.read_text(encoding='utf-8')
        elif preserve_existing and output_path.exists():
            existing = output_path.read_text(encoding='utf-8')
        
        # Generate
        python_packages = get_python_packages_for_features(features)
        content, added = self._generate_requirements_txt(python_packages, existing)
        
        # Write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
        
        return added
    
    def generate_package_json_file(
        self,
        features: Set[str],
        output_path: str | Path,
        preserve_existing: bool = True,
        existing_path: Optional[Path] = None
    ) -> List[str]:
        """
        Generate package.json file.
        
        Args:
            features: Set of feature IDs
            output_path: Where to write package.json
            preserve_existing: Whether to preserve existing content
            existing_path: Optional path to existing package.json (overrides preserve_existing logic)
            
        Returns:
            List of added package names
        """
        output_path = Path(output_path)
        
        # Read existing if preserve
        existing = None
        if existing_path and existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                existing = None
        elif preserve_existing and output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                self.warnings.append(f"Could not parse existing {output_path}")
        
        # Generate
        npm_packages = get_npm_packages_for_features(features)
        content, added = self._generate_package_json(npm_packages, existing)
        
        # Write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(content, indent=2) + "\n",
            encoding='utf-8'
        )
        
        return added


def generate_requirements_txt(features: Set[str], existing: Optional[str] = None) -> str:
    """Convenience function to generate requirements.txt"""
    generator = DependencyGenerator()
    deps = generator.generate(features, existing_requirements=existing)
    return deps.requirements_txt


def generate_package_json(features: Set[str], existing: Optional[Dict] = None) -> Dict:
    """Convenience function to generate package.json"""
    generator = DependencyGenerator()
    deps = generator.generate(features, existing_package_json=existing)
    return deps.package_json
