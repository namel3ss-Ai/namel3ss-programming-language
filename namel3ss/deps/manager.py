"""
Dependency Manager - High-Level API

Orchestrates feature detection and dependency generation for Namel3ss projects.

This is the main entry point for dependency management operations.
"""

from pathlib import Path
from typing import Optional, Set, Dict, List
import json

from .detector import FeatureDetector, DetectedFeatures
from .generator import DependencyGenerator, GeneratedDeps
from .spec import get_dependency_spec


class DependencyManager:
    """
    High-level API for dependency management.
    
    Usage:
        manager = DependencyManager()
        
        # Sync entire project
        manager.sync_project('./my_project')
        
        # Or sync specific files
        manager.sync_from_file('app.ai', output_dir='./backend')
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.detector = FeatureDetector()
        self.generator = DependencyGenerator()
    
    def sync_project(
        self,
        project_root: str | Path,
        requirements_path: Optional[str | Path] = None,
        package_json_path: Optional[str | Path] = None,
    ) -> Dict:
        """
        Analyze project and sync dependency files.
        
        Args:
            project_root: Root directory of project
            requirements_path: Path to requirements.txt (defaults to project_root/requirements.txt)
            package_json_path: Path to package.json (defaults to project_root/package.json)
            
        Returns:
            Dict with sync results:
            {
                'features': set of detected features,
                'added_python': list of added Python packages,
                'added_npm': list of added NPM packages,
                'warnings': list of warning messages,
            }
        """
        project_root = Path(project_root)
        
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        
        # Detect features
        if self.verbose:
            print(f"📦 Analyzing project: {project_root}")
        
        detected = self.detector.detect_from_directory(project_root)
        
        if self.verbose:
            print(f"✨ Detected features: {', '.join(sorted(detected.features)) or 'none'}")
        
        # Determine output paths
        if requirements_path is None:
            # Try backend/requirements.txt first, fall back to root
            backend_reqs = project_root / "backend" / "requirements.txt"
            root_reqs = project_root / "requirements.txt"
            requirements_path = backend_reqs if backend_reqs.parent.exists() else root_reqs
        else:
            requirements_path = Path(requirements_path)
        
        if package_json_path is None:
            # Try frontend/package.json first, fall back to root
            frontend_pkg = project_root / "frontend" / "package.json"
            root_pkg = project_root / "package.json"
            package_json_path = frontend_pkg if frontend_pkg.parent.exists() else root_pkg
        else:
            package_json_path = Path(package_json_path)
        
        # Generate and write dependencies
        added_python = self.generator.generate_requirements_file(
            detected.features,
            requirements_path,
            preserve_existing=True
        )
        
        added_npm = self.generator.generate_package_json_file(
            detected.features,
            package_json_path,
            preserve_existing=True
        )
        
        if self.verbose:
            if added_python:
                print(f"➕ Added Python packages: {', '.join(added_python)}")
            if added_npm:
                print(f"➕ Added NPM packages: {', '.join(added_npm)}")
            
            if not added_python and not added_npm:
                print("✅ Dependencies up to date")
        
        # Show warnings
        if detected.warnings:
            for warning in detected.warnings:
                print(f"⚠️  {warning}")
        
        return {
            'features': detected.features,
            'added_python': added_python,
            'added_npm': added_npm,
            'warnings': detected.warnings + self.generator.warnings,
        }
    
    def sync_from_file(
        self,
        ai_file: str | Path,
        output_dir: Optional[str | Path] = None,
        preview: bool = False,
    ) -> Dict:
        """
        Analyze single .ai file and generate dependencies.
        
        Args:
            ai_file: Path to .ai file
            output_dir: Output directory (defaults to file's directory)
            preview: If True, detect features but don't write files
            
        Returns:
            Dict with sync results
        """
        ai_file = Path(ai_file)
        
        # Determine output directory
        if output_dir is None:
            output_dir = ai_file.parent if ai_file.exists() else Path.cwd()
        else:
            output_dir = Path(output_dir)
        
        # Detect features (detector handles missing files gracefully)
        detected = self.detector.detect_from_file(ai_file)
        
        # If preview mode, skip generation
        if preview:
            return {
                'features': detected.features,
                'added_python': [],
                'added_npm': [],
                'warnings': detected.warnings,
            }
        
        # Generate dependencies
        requirements_path = output_dir / "requirements.txt"
        package_json_path = output_dir / "package.json"
        
        added_python = self.generator.generate_requirements_file(
            detected.features,
            requirements_path,
            preserve_existing=True
        )
        
        added_npm = self.generator.generate_package_json_file(
            detected.features,
            package_json_path,
            preserve_existing=True
        )
        
        return {
            'features': detected.features,
            'added_python': added_python,
            'added_npm': added_npm,
            'warnings': detected.warnings + self.generator.warnings,
        }
    
    def get_features_for_source(self, source: str) -> Set[str]:
        """
        Get required features for source code (useful for testing).
        
        Args:
            source: Namel3ss source code
            
        Returns:
            Set of feature IDs
        """
        detected = self.detector.detect_from_source(source)
        return detected.features
    
    def list_available_features(self) -> Dict[str, Dict]:
        """
        List all available features and their dependencies.
        
        Returns:
            Dict mapping feature IDs to their specs
        """
        specs = get_dependency_spec()
        return {
            feature_id: {
                'category': spec.category.value,
                'description': spec.description,
                'python_packages': [pkg.name for pkg in spec.python_packages],
                'npm_packages': [pkg.name for pkg in spec.npm_packages],
                'requires': spec.requires,
            }
            for feature_id, spec in specs.items()
        }
    
    def preview_dependencies(
        self,
        project_root: str | Path,
    ) -> Dict:
        """
        Preview what dependencies would be added without modifying files.
        
        Args:
            project_root: Root directory of project
            
        Returns:
            Dict with preview information (same format as sync methods)
        """
        project_root = Path(project_root)
        
        # Detect features
        detected = self.detector.detect_from_directory(project_root)
        
        # Get what would be generated
        deps = self.generator.generate(detected.features)
        
        # Extract package names from requirements.txt
        python_packages = [
            line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            for line in deps.requirements_txt.splitlines()
            if line and not line.startswith('#') and not line.strip().startswith('-')
        ]
        
        # Extract package names from package.json (it's now a JSON string)
        import json
        package_json_dict = json.loads(deps.package_json)
        npm_packages = list(package_json_dict.get('dependencies', {}).keys())
        npm_dev_packages = list(package_json_dict.get('devDependencies', {}).keys())
        
        return {
            'features': detected.features,
            'added_python': python_packages,
            'added_npm': npm_packages + npm_dev_packages,
            'warnings': detected.warnings + deps.warnings,
        }
