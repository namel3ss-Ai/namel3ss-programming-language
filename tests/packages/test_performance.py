"""Test performance and scalability of the module and package system."""

import pytest
import time
from pathlib import Path
from textwrap import dedent

from namel3ss.loader import load_workspace_program, get_workspace_info
from namel3ss.packages.discovery import PackageDiscovery, load_workspace_config


@pytest.fixture
def large_workspace(tmp_path):
    """Create a large workspace with many packages and modules for performance testing."""
    workspace = tmp_path / "large_workspace"
    workspace.mkdir()
    
    # Workspace configuration
    config_content = dedent("""
        [workspace]
        name = "large-performance-test"
        package_paths = ["packages", "external", "libraries"]
        module_paths = ["src", "apps", "tools"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    return workspace


def create_large_package(workspace_path, package_name, module_count=20, dependency_count=3):
    """Helper function to create a package with many modules."""
    packages_dir = workspace_path / "packages"
    packages_dir.mkdir(exist_ok=True)
    
    pkg_dir = packages_dir / package_name.replace(".", "_")
    pkg_dir.mkdir()
    
    # Create dependencies section
    deps_section = "[dependencies]\n"
    for i in range(dependency_count):
        if i < dependency_count - 1:  # Avoid self-dependency
            deps_section += f'"test.package_{i}" = ">=1.0.0"\n'
    
    # Create modules list
    modules_list = [f"module_{i}" for i in range(module_count)]
    modules_section = f'modules = {modules_list}'
    
    manifest_content = dedent(f"""
        [package]
        name = "{package_name}"
        version = "1.0.0"
        description = "Large test package with {module_count} modules"
        
        {deps_section}
        
        {modules_section}
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Create module files
    for i in range(module_count):
        module_content = dedent(f"""
            module module_{i}
            
            prompt "prompt_{i}_1" {{
                template: "This is prompt {i}_1 with data: {{{{data}}}}"
            }}
            
            prompt "prompt_{i}_2" {{
                template: "This is prompt {i}_2 for {{{{purpose}}}}"
            }}
            
            tool "tool_{i}" {{
                type: "utility"
                config: "config_{i}.json"
            }}
            
            llm "model_{i}" {{
                provider: "openai"
                model: "gpt-3.5-turbo"
                temperature: 0.{i % 10}
            }}
        """)
        (pkg_dir / f"module_{i}.ai").write_text(module_content)
    
    return pkg_dir


def create_many_workspace_modules(workspace_path, module_count=50):
    """Helper to create many workspace modules."""
    src_dir = workspace_path / "src"
    src_dir.mkdir(exist_ok=True)
    
    apps_dir = workspace_path / "apps"
    apps_dir.mkdir(exist_ok=True)
    
    # Create modules in src directory
    for i in range(module_count // 2):
        module_content = dedent(f"""
            module src.component_{i}
            
            prompt "component_prompt_{i}" {{
                template: "Component {i}: {{{{data}}}}"
            }}
            
            app "ComponentApp_{i}" {{
                description: "App for component {i}"
            }}
        """)
        (src_dir / f"component_{i}.ai").write_text(module_content)
    
    # Create modules in apps directory
    for i in range(module_count // 2):
        module_content = dedent(f"""
            module apps.application_{i}
            
            use src.component_{i % 10}
            
            app "Application_{i}" {{
                description: "Application {i}"
                components: ["component_{i % 10}"]
            }}
            
            workflow "workflow_{i}" {{
                steps: ["step1", "step2", "step3"]
            }}
        """)
        (apps_dir / f"application_{i}.ai").write_text(module_content)


@pytest.mark.performance
def test_large_workspace_discovery_performance(large_workspace):
    """Test performance of discovering a large workspace."""
    # Create many packages
    package_count = 20
    modules_per_package = 15
    
    for i in range(package_count):
        create_large_package(large_workspace, f"test.package_{i}", modules_per_package, min(i, 3))
    
    # Create many workspace modules
    create_many_workspace_modules(large_workspace, 100)
    
    # Measure discovery time
    start_time = time.time()
    
    config = load_workspace_config(large_workspace)
    discovery = PackageDiscovery(large_workspace, config)
    packages, workspace_modules = discovery.discover_workspace()
    
    discovery_time = time.time() - start_time
    
    # Verify results
    assert len(packages) == package_count
    assert len(workspace_modules) == 100  # 50 src + 50 apps
    
    # Performance assertion - discovery should complete in reasonable time
    # Adjust threshold based on acceptable performance
    assert discovery_time < 10.0, f"Discovery took {discovery_time:.2f}s, expected < 10s"
    
    print(f"Discovery of {package_count} packages and 100 modules took {discovery_time:.2f}s")


@pytest.mark.performance
def test_large_dependency_resolution_performance(large_workspace):
    """Test performance of dependency resolution with many packages."""
    package_count = 15
    
    # Create packages with complex dependency chains
    for i in range(package_count):
        dependency_count = min(i, 5)  # Each package depends on previous packages
        create_large_package(large_workspace, f"chain.package_{i}", 10, dependency_count)
    
    start_time = time.time()
    
    config = load_workspace_config(large_workspace)
    discovery = PackageDiscovery(large_workspace, config)
    packages, _ = discovery.discover_workspace()
    
    # Resolve all packages
    from namel3ss.packages.discovery import DependencyResolver
    resolver = DependencyResolver(packages)
    
    all_package_names = list(packages.keys())
    resolved = resolver.resolve_dependencies(all_package_names)
    
    resolution_time = time.time() - start_time
    
    # Verify resolution
    assert len(resolved) == package_count
    
    # Performance assertion
    assert resolution_time < 5.0, f"Resolution took {resolution_time:.2f}s, expected < 5s"
    
    print(f"Dependency resolution of {package_count} packages took {resolution_time:.2f}s")


@pytest.mark.performance
def test_large_workspace_loading_performance(large_workspace):
    """Test performance of loading a complete large workspace."""
    # Create moderate number of packages with dependencies
    package_count = 10
    modules_per_package = 20
    
    for i in range(package_count):
        deps = min(i, 2)  # Each package depends on up to 2 previous packages
        create_large_package(large_workspace, f"load.package_{i}", modules_per_package, deps)
    
    create_many_workspace_modules(large_workspace, 50)
    
    # Measure complete loading time
    start_time = time.time()
    
    program, packages = load_workspace_program(large_workspace)
    
    loading_time = time.time() - start_time
    
    # Verify loading
    total_expected_modules = (package_count * modules_per_package) + 50
    assert len(program.modules) == total_expected_modules
    assert len(packages) == package_count
    
    # Performance assertion
    assert loading_time < 15.0, f"Loading took {loading_time:.2f}s, expected < 15s"
    
    print(f"Loading {len(program.modules)} modules from {package_count} packages took {loading_time:.2f}s")


@pytest.mark.performance
def test_workspace_info_performance(large_workspace):
    """Test performance of workspace info generation."""
    # Create workspace with many packages
    for i in range(25):
        create_large_package(large_workspace, f"info.package_{i}", 8, 2)
    
    create_many_workspace_modules(large_workspace, 30)
    
    start_time = time.time()
    
    info = get_workspace_info(large_workspace)
    
    info_time = time.time() - start_time
    
    # Verify info completeness
    assert info['total_packages'] == 25
    assert info['total_modules'] == (25 * 8) + 30  # package modules + workspace modules
    assert len(info['packages']) == 25
    
    # Performance assertion
    assert info_time < 8.0, f"Workspace info took {info_time:.2f}s, expected < 8s"
    
    print(f"Workspace info generation took {info_time:.2f}s")


@pytest.mark.performance
def test_memory_usage_large_workspace(large_workspace):
    """Test memory usage with large workspaces."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large workspace
    for i in range(30):
        create_large_package(large_workspace, f"memory.package_{i}", 12, 2)
    
    create_many_workspace_modules(large_workspace, 75)
    
    # Load workspace
    program, packages = load_workspace_program(large_workspace)
    
    # Check memory usage after loading
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Verify loading worked
    assert len(packages) == 30
    assert len(program.modules) == (30 * 12) + 75
    
    # Memory usage should be reasonable for the amount of data loaded
    # This is a rough guideline - adjust based on acceptable memory usage
    modules_loaded = len(program.modules)
    memory_per_module = memory_increase / modules_loaded if modules_loaded > 0 else 0
    
    print(f"Memory usage: {memory_increase:.1f}MB for {modules_loaded} modules ({memory_per_module:.2f}MB/module)")
    
    # Assert reasonable memory usage (adjust threshold as needed)
    assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"


def test_concurrent_workspace_operations(large_workspace):
    """Test concurrent operations on workspace (basic concurrency simulation)."""
    import threading
    import queue
    
    # Create workspace with packages
    for i in range(10):
        create_large_package(large_workspace, f"concurrent.package_{i}", 5, 1)
    
    results_queue = queue.Queue()
    errors_queue = queue.Queue()
    
    def load_workspace_thread():
        """Thread function to load workspace."""
        try:
            start_time = time.time()
            program, packages = load_workspace_program(large_workspace)
            load_time = time.time() - start_time
            results_queue.put(('load', load_time, len(program.modules), len(packages)))
        except Exception as e:
            errors_queue.put(('load', e))
    
    def get_workspace_info_thread():
        """Thread function to get workspace info."""
        try:
            start_time = time.time()
            info = get_workspace_info(large_workspace)
            info_time = time.time() - start_time
            results_queue.put(('info', info_time, info['total_modules'], info['total_packages']))
        except Exception as e:
            errors_queue.put(('info', e))
    
    def discover_packages_thread():
        """Thread function to discover packages."""
        try:
            start_time = time.time()
            config = load_workspace_config(large_workspace)
            discovery = PackageDiscovery(large_workspace, config)
            packages, modules = discovery.discover_workspace()
            discovery_time = time.time() - start_time
            results_queue.put(('discovery', discovery_time, len(modules), len(packages)))
        except Exception as e:
            errors_queue.put(('discovery', e))
    
    # Start threads
    threads = [
        threading.Thread(target=load_workspace_thread),
        threading.Thread(target=get_workspace_info_thread),
        threading.Thread(target=discover_packages_thread)
    ]
    
    for thread in threads:
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check for errors
    while not errors_queue.empty():
        operation, error = errors_queue.get()
        pytest.fail(f"Error in {operation}: {error}")
    
    # Verify results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    assert len(results) == 3  # All three operations completed
    
    # All operations should report consistent counts
    operations = {result[0]: result for result in results}
    
    if 'load' in operations and 'info' in operations:
        load_modules = operations['load'][2]
        info_modules = operations['info'][2]
        assert load_modules == info_modules, "Module counts should be consistent"
    
    print("Concurrent operations completed successfully")


@pytest.mark.performance
def test_incremental_package_discovery(large_workspace):
    """Test incremental discovery when packages are added."""
    # Start with some packages
    initial_count = 5
    for i in range(initial_count):
        create_large_package(large_workspace, f"incremental.package_{i}", 8, 1)
    
    # Initial discovery
    start_time = time.time()
    config = load_workspace_config(large_workspace)
    discovery = PackageDiscovery(large_workspace, config)
    packages, _ = discovery.discover_workspace()
    initial_time = time.time() - start_time
    
    assert len(packages) == initial_count
    
    # Add more packages
    additional_count = 5
    for i in range(initial_count, initial_count + additional_count):
        create_large_package(large_workspace, f"incremental.package_{i}", 8, 1)
    
    # Rediscover - should find new packages
    start_time = time.time()
    discovery = PackageDiscovery(large_workspace, config)  # Fresh discovery instance
    packages, _ = discovery.discover_workspace()
    incremental_time = time.time() - start_time
    
    assert len(packages) == initial_count + additional_count
    
    # Incremental discovery should be reasonably fast
    print(f"Initial discovery: {initial_time:.2f}s, Incremental: {incremental_time:.2f}s")
    
    # The time difference should be reasonable
    # (This assumes no significant caching optimizations)
    assert incremental_time < initial_time * 3, "Incremental discovery took too long"