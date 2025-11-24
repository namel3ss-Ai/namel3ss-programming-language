"""
Test Configuration and Utilities for Namel3ss Test Suite.

This module provides configuration management and utilities for running
the comprehensive test suite efficiently.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class TestConfig:
    """Configuration for test execution."""
    
    # Test suite selection
    run_unit_tests: bool = True
    run_integration_tests: bool = True 
    run_performance_tests: bool = True
    run_edge_case_tests: bool = True
    run_fault_tolerance_tests: bool = True
    run_security_tests: bool = True
    
    # Performance test configuration
    performance_quick_mode: bool = False
    max_concurrency_test: int = 20
    large_scale_test_size: int = 200
    
    # Fault tolerance test configuration
    fault_injection_rate: float = 0.3
    recovery_timeout_seconds: float = 5.0
    
    # Security test configuration
    security_audit_enabled: bool = True
    attack_simulation_enabled: bool = True
    
    # Output configuration
    verbose_output: bool = True
    save_detailed_reports: bool = True
    report_directory: str = "test_reports"
    
    # Execution configuration
    parallel_test_execution: bool = False
    test_timeout_seconds: float = 300.0  # 5 minutes per test
    retry_failed_tests: bool = True
    max_retries: int = 2


class TestLogger:
    """Centralized logging for test execution."""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger("namel3ss_tests")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class TestEnvironment:
    """Test environment setup and management."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = TestLogger()
        self.setup_complete = False
    
    def setup_test_environment(self):
        """Set up test environment."""
        try:
            # Create report directory
            if self.config.save_detailed_reports:
                os.makedirs(self.config.report_directory, exist_ok=True)
                self.logger.info(f"Created report directory: {self.config.report_directory}")
            
            # Set environment variables for testing
            os.environ['NAMEL3SS_TEST_MODE'] = 'true'
            os.environ['NAMEL3SS_LOG_LEVEL'] = 'DEBUG'
            
            # Configure test-specific settings
            if self.config.performance_quick_mode:
                os.environ['NAMEL3SS_PERF_QUICK_MODE'] = 'true'
            
            self.setup_complete = True
            self.logger.info("Test environment setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            raise
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        try:
            # Remove test environment variables
            test_env_vars = [
                'NAMEL3SS_TEST_MODE',
                'NAMEL3SS_LOG_LEVEL', 
                'NAMEL3SS_PERF_QUICK_MODE'
            ]
            
            for var in test_env_vars:
                if var in os.environ:
                    del os.environ[var]
            
            self.logger.info("Test environment cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup test environment: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.setup_test_environment()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_test_environment()


class TestReportManager:
    """Manage test reports and results."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.reports: Dict[str, Any] = {}
    
    def save_report(self, test_suite: str, report: Dict[str, Any]):
        """Save individual test suite report."""
        self.reports[test_suite] = report
        
        if self.config.save_detailed_reports:
            report_file = os.path.join(
                self.config.report_directory,
                f"{test_suite}_report.json"
            )
            
            try:
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"📄 {test_suite} report saved to: {report_file}")
            except Exception as e:
                print(f"❌ Failed to save {test_suite} report: {e}")
    
    def save_master_report(self, master_report: Dict[str, Any]):
        """Save master test report."""
        self.reports['master'] = master_report
        
        if self.config.save_detailed_reports:
            report_file = os.path.join(
                self.config.report_directory,
                "master_test_report.json"
            )
            
            try:
                with open(report_file, 'w') as f:
                    json.dump(master_report, f, indent=2, default=str)
                print(f"📄 Master report saved to: {report_file}")
            except Exception as e:
                print(f"❌ Failed to save master report: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        if 'master' not in self.reports:
            return "No master report available"
        
        master = self.reports['master']
        stats = master.get('aggregate_statistics', {})
        readiness = master.get('production_readiness', {})
        
        summary = f"""
🎯 NAMEL3SS TEST SUITE SUMMARY
{'='*50}

📊 Overall Results:
   Total Tests: {stats.get('total_tests', 0)}
   Passed: {stats.get('passed_tests', 0)}
   Failed: {stats.get('failed_tests', 0)}
   Success Rate: {stats.get('overall_success_rate', 0):.1f}%

🚀 Production Readiness:
   Ready: {'✅ YES' if readiness.get('ready_for_production', False) else '❌ NO'}
   Confidence: {readiness.get('confidence_score', 0):.1f}%

💡 Key Recommendations:
"""
        
        recommendations = master.get('deployment_recommendations', [])
        for rec in recommendations[:3]:  # Top 3 recommendations
            summary += f"   {rec}\n"
        
        return summary
    
    def export_results_csv(self, filename: Optional[str] = None):
        """Export test results to CSV format."""
        if not filename:
            filename = os.path.join(
                self.config.report_directory,
                "test_results.csv"
            )
        
        try:
            import csv
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Test Suite', 'Test Name', 'Status', 'Duration (ms)', 'Error Message'
                ])
                
                for suite_name, report in self.reports.items():
                    if suite_name == 'master':
                        continue
                    
                    results = report.get('results', [])
                    for result in results:
                        writer.writerow([
                            suite_name,
                            result.get('test_name', ''),
                            result.get('status', ''),
                            result.get('duration_ms', 0),
                            result.get('error_message', '')
                        ])
            
            print(f"📊 CSV results exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")


def load_test_config(config_file: Optional[str] = None) -> TestConfig:
    """Load test configuration from file or create default."""
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            return TestConfig(**config_data)
            
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_file}: {e}")
            print("📋 Using default configuration")
    
    return TestConfig()


def save_test_config(config: TestConfig, config_file: str = "test_config.json"):
    """Save test configuration to file."""
    try:
        config_data = {
            'run_unit_tests': config.run_unit_tests,
            'run_integration_tests': config.run_integration_tests,
            'run_performance_tests': config.run_performance_tests,
            'run_edge_case_tests': config.run_edge_case_tests,
            'run_fault_tolerance_tests': config.run_fault_tolerance_tests,
            'run_security_tests': config.run_security_tests,
            'performance_quick_mode': config.performance_quick_mode,
            'max_concurrency_test': config.max_concurrency_test,
            'large_scale_test_size': config.large_scale_test_size,
            'fault_injection_rate': config.fault_injection_rate,
            'recovery_timeout_seconds': config.recovery_timeout_seconds,
            'security_audit_enabled': config.security_audit_enabled,
            'attack_simulation_enabled': config.attack_simulation_enabled,
            'verbose_output': config.verbose_output,
            'save_detailed_reports': config.save_detailed_reports,
            'report_directory': config.report_directory,
            'parallel_test_execution': config.parallel_test_execution,
            'test_timeout_seconds': config.test_timeout_seconds,
            'retry_failed_tests': config.retry_failed_tests,
            'max_retries': config.max_retries
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Test configuration saved to: {config_file}")
        
    except Exception as e:
        print(f"❌ Failed to save config: {e}")


# Predefined configurations for common scenarios
QUICK_TEST_CONFIG = TestConfig(
    performance_quick_mode=True,
    run_performance_tests=False,
    run_edge_case_tests=False,
    max_concurrency_test=5,
    large_scale_test_size=50
)

COMPREHENSIVE_TEST_CONFIG = TestConfig(
    run_unit_tests=True,
    run_integration_tests=True,
    run_performance_tests=True,
    run_edge_case_tests=True,
    run_fault_tolerance_tests=True,
    run_security_tests=True,
    performance_quick_mode=False,
    max_concurrency_test=50,
    large_scale_test_size=500
)

SECURITY_FOCUSED_CONFIG = TestConfig(
    run_unit_tests=True,
    run_integration_tests=True,
    run_performance_tests=False,
    run_edge_case_tests=True,
    run_fault_tolerance_tests=True,
    run_security_tests=True,
    security_audit_enabled=True,
    attack_simulation_enabled=True
)

PERFORMANCE_FOCUSED_CONFIG = TestConfig(
    run_unit_tests=False,
    run_integration_tests=True,
    run_performance_tests=True,
    run_edge_case_tests=True,
    run_fault_tolerance_tests=False,
    run_security_tests=False,
    performance_quick_mode=False,
    max_concurrency_test=100,
    large_scale_test_size=1000
)