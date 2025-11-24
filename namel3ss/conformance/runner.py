"""
Conformance test runner that executes tests against the Namel3ss implementation.

This runner can be used by this implementation to validate conformance, and
the test format can be adopted by other implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from namel3ss.conformance.models import (
    ConformanceTestDescriptor,
    TestPhase,
    TestStatus,
    Diagnostic,
    discover_conformance_tests,
)
from namel3ss.parser import Parser
from namel3ss.lang.parser.errors import N3Error, N3SyntaxError, N3SemanticError
from namel3ss.ast import Module
from namel3ss.resolver import resolve_program, ModuleResolutionError
from namel3ss.ast import Program


class TestResult(str, Enum):
    """Result of running a conformance test."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class PhaseResult:
    """Result of executing a single test phase."""
    phase: TestPhase
    result: TestResult
    message: Optional[str] = None
    actual: Optional[Dict[str, Any]] = None
    expected: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class ConformanceTestResult:
    """Complete result of running a conformance test."""
    test_id: str
    test_name: str
    category: str
    result: TestResult
    phase_results: List[PhaseResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category,
            "result": self.result.value,
            "phase_results": [
                {
                    "phase": pr.phase.value,
                    "result": pr.result.value,
                    "message": pr.message,
                    "duration_ms": pr.duration_ms
                }
                for pr in self.phase_results
            ],
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message
        }


class ConformanceRunner:
    """
    Runner that executes conformance tests against the Namel3ss implementation.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ConformanceTestResult] = []
    
    def run_test(self, descriptor: ConformanceTestDescriptor) -> ConformanceTestResult:
        """
        Run a single conformance test.
        
        Args:
            descriptor: Test descriptor to execute
        
        Returns:
            Result of test execution
        """
        start_time = time.time()
        
        try:
            # Load source code
            sources = []
            for source_spec in descriptor.sources:
                content = source_spec.get_content(descriptor.get_base_path())
                sources.append(content)
            
            if not sources:
                return ConformanceTestResult(
                    test_id=descriptor.test_id,
                    test_name=descriptor.name,
                    category=descriptor.category,
                    result=TestResult.ERROR,
                    error_message="No sources provided"
                )
            
            # We'll primarily work with the first source
            source_code = sources[0]
            
            # Execute each phase
            phase_results = []
            overall_result = TestResult.PASS
            
            for phase in descriptor.phases:
                phase_result = self._execute_phase(
                    phase, source_code, descriptor
                )
                phase_results.append(phase_result)
                
                if phase_result.result == TestResult.FAIL:
                    overall_result = TestResult.FAIL
                elif phase_result.result == TestResult.ERROR:
                    overall_result = TestResult.ERROR
                    break  # Stop on error
            
            duration = (time.time() - start_time) * 1000
            
            return ConformanceTestResult(
                test_id=descriptor.test_id,
                test_name=descriptor.name,
                category=descriptor.category,
                result=overall_result,
                phase_results=phase_results,
                total_duration_ms=duration
            )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ConformanceTestResult(
                test_id=descriptor.test_id,
                test_name=descriptor.name,
                category=descriptor.category,
                result=TestResult.ERROR,
                error_message=str(e),
                total_duration_ms=duration
            )
    
    def _execute_phase(
        self,
        phase: TestPhase,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute a single test phase."""
        start_time = time.time()
        
        try:
            if phase == TestPhase.PARSE:
                result = self._execute_parse_phase(source_code, descriptor)
            elif phase == TestPhase.RESOLVE:
                result = self._execute_resolve_phase(source_code, descriptor)
            elif phase == TestPhase.TYPECHECK:
                result = self._execute_typecheck_phase(source_code, descriptor)
            elif phase == TestPhase.CODEGEN:
                result = self._execute_codegen_phase(source_code, descriptor)
            elif phase == TestPhase.RUNTIME:
                result = self._execute_runtime_phase(source_code, descriptor)
            else:
                result = PhaseResult(
                    phase=phase,
                    result=TestResult.SKIP,
                    message=f"Phase {phase.value} not implemented"
                )
            
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return PhaseResult(
                phase=phase,
                result=TestResult.ERROR,
                message=f"Unexpected error: {e}",
                duration_ms=duration
            )
    
    def _execute_parse_phase(
        self,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute parse phase and compare with expectations."""
        expectation = descriptor.expect.parse
        if not expectation:
            return PhaseResult(
                phase=TestPhase.PARSE,
                result=TestResult.ERROR,
                message="No parse expectation provided"
            )
        
        try:
            # Attempt to parse
            parser = Parser(source_code, path="test.ai")
            module = parser.parse()
            
            # If we expected success
            if expectation.status == TestStatus.SUCCESS:
                # Optionally check AST structure
                if expectation.ast:
                    actual_ast = self._module_to_dict(module)
                    if not self._compare_ast(expectation.ast, actual_ast, descriptor.config.strict_ast_match):
                        return PhaseResult(
                            phase=TestPhase.PARSE,
                            result=TestResult.FAIL,
                            message="AST structure mismatch",
                            expected=expectation.ast,
                            actual=actual_ast
                        )
                
                return PhaseResult(
                    phase=TestPhase.PARSE,
                    result=TestResult.PASS,
                    message="Parse successful"
                )
            else:
                # Expected error but got success
                return PhaseResult(
                    phase=TestPhase.PARSE,
                    result=TestResult.FAIL,
                    message="Expected parse error but parsing succeeded"
                )
        
        except (N3SyntaxError, N3SemanticError) as e:
            # Catch both syntax and semantic errors from parser
            # If we expected error
            if expectation.status == TestStatus.ERROR:
                # TODO: Compare error details with expectation.errors
                return PhaseResult(
                    phase=TestPhase.PARSE,
                    result=TestResult.PASS,
                    message=f"Parse failed as expected: {e}"
                )
            else:
                # Unexpected error
                return PhaseResult(
                    phase=TestPhase.PARSE,
                    result=TestResult.FAIL,
                    message=f"Unexpected parse error: {e}"
                )
        
        except Exception as e:
            return PhaseResult(
                phase=TestPhase.PARSE,
                result=TestResult.ERROR,
                message=f"Parse phase error: {e}"
            )
    
    def _execute_resolve_phase(
        self,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute resolve phase."""
        # First parse
        try:
            parser = Parser(source_code, path="test.ai")
            module = parser.parse()
            program = Program(modules=[module])
            
            # Attempt resolution
            try:
                resolved = resolve_program(program)
                
                # Check if we have typecheck expectations
                expectation = descriptor.expect.typecheck
                if expectation and expectation.status == TestStatus.SUCCESS:
                    return PhaseResult(
                        phase=TestPhase.RESOLVE,
                        result=TestResult.PASS,
                        message="Resolution successful"
                    )
                elif expectation and expectation.status == TestStatus.ERROR:
                    return PhaseResult(
                        phase=TestPhase.RESOLVE,
                        result=TestResult.FAIL,
                        message="Expected resolution error but succeeded"
                    )
                else:
                    return PhaseResult(
                        phase=TestPhase.RESOLVE,
                        result=TestResult.PASS,
                        message="Resolution successful"
                    )
            
            except ModuleResolutionError as e:
                expectation = descriptor.expect.typecheck
                if expectation and expectation.status == TestStatus.ERROR:
                    return PhaseResult(
                        phase=TestPhase.RESOLVE,
                        result=TestResult.PASS,
                        message=f"Resolution failed as expected: {e}"
                    )
                else:
                    return PhaseResult(
                        phase=TestPhase.RESOLVE,
                        result=TestResult.FAIL,
                        message=f"Unexpected resolution error: {e}"
                    )
        
        except Exception as e:
            return PhaseResult(
                phase=TestPhase.RESOLVE,
                result=TestResult.ERROR,
                message=f"Resolve phase error: {e}"
            )
    
    def _execute_typecheck_phase(
        self,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute typecheck phase (currently integrated with resolve)."""
        # For now, typecheck is done during resolution
        return self._execute_resolve_phase(source_code, descriptor)
    
    def _execute_codegen_phase(
        self,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute code generation phase."""
        return PhaseResult(
            phase=TestPhase.CODEGEN,
            result=TestResult.SKIP,
            message="Codegen phase not yet implemented in conformance runner"
        )
    
    def _execute_runtime_phase(
        self,
        source_code: str,
        descriptor: ConformanceTestDescriptor
    ) -> PhaseResult:
        """Execute runtime phase."""
        return PhaseResult(
            phase=TestPhase.RUNTIME,
            result=TestResult.SKIP,
            message="Runtime phase not yet implemented in conformance runner"
        )
    
    def _module_to_dict(self, module: Module) -> Dict[str, Any]:
        """Convert Module AST to dictionary for comparison."""
        result = {
            "type": "Module",
            "name": module.name,
            "body": []
        }
        
        for node in module.body:
            result["body"].append(self._ast_node_to_dict(node))
        
        return result
    
    def _ast_node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Convert AST node to dictionary representation."""
        node_type = type(node).__name__
        result = {"type": node_type}
        
        # Add key attributes based on node type
        if hasattr(node, "name"):
            result["name"] = node.name
        if hasattr(node, "agents") and node_type == "App":
            result["agents"] = [self._ast_node_to_dict(a) for a in node.agents]
        if hasattr(node, "tools") and node_type == "App":
            result["tools"] = [self._ast_node_to_dict(t) for t in node.tools]
        if hasattr(node, "llms") and node_type == "App":
            result["llms"] = [self._ast_node_to_dict(l) for l in node.llms]
        
        return result
    
    def _compare_ast(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        strict: bool
    ) -> bool:
        """
        Compare expected and actual AST structures.
        
        Args:
            expected: Expected AST structure
            actual: Actual AST structure
            strict: If True, require exact match. If False, structural match only.
        
        Returns:
            True if match, False otherwise
        """
        # Check type match
        if expected.get("type") != actual.get("type"):
            return False
        
        # For non-strict comparison, only check keys present in expected
        keys_to_check = expected.keys() if not strict else set(expected.keys()) | set(actual.keys())
        
        for key in keys_to_check:
            if key == "type":
                continue  # Already checked
            
            exp_val = expected.get(key)
            act_val = actual.get(key)
            
            # If key not in expected and we're not strict, skip
            if not strict and key not in expected:
                continue
            
            # Compare values
            if isinstance(exp_val, dict) and isinstance(act_val, dict):
                if not self._compare_ast(exp_val, act_val, strict):
                    return False
            elif isinstance(exp_val, list) and isinstance(act_val, list):
                if len(exp_val) != len(act_val):
                    return False
                for e, a in zip(exp_val, act_val):
                    if isinstance(e, dict) and isinstance(a, dict):
                        if not self._compare_ast(e, a, strict):
                            return False
                    elif e != a:
                        return False
            elif exp_val != act_val:
                return False
        
        return True
    
    def run_all_tests(
        self,
        test_dir: Path,
        category: Optional[str] = None,
        test_id: Optional[str] = None
    ) -> List[ConformanceTestResult]:
        """
        Run all conformance tests in a directory.
        
        Args:
            test_dir: Directory containing conformance tests
            category: Optional category filter
            test_id: Optional specific test ID
        
        Returns:
            List of test results
        """
        descriptors = discover_conformance_tests(test_dir, category, test_id)
        
        results = []
        for descriptor in descriptors:
            if self.verbose:
                print(f"\nRunning test: {descriptor.test_id} - {descriptor.name}")
            
            result = self.run_test(descriptor)
            results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        self.results = results
        return results
    
    def _print_result(self, result: ConformanceTestResult):
        """Print test result in human-readable format."""
        symbol = {
            TestResult.PASS: "✓",
            TestResult.FAIL: "✗",
            TestResult.SKIP: "○",
            TestResult.ERROR: "E"
        }[result.result]
        
        print(f"  {symbol} {result.result.value.upper()}")
        
        if result.error_message:
            print(f"    Error: {result.error_message}")
        
        for phase_result in result.phase_results:
            phase_symbol = {
                TestResult.PASS: "✓",
                TestResult.FAIL: "✗",
                TestResult.SKIP: "○",
                TestResult.ERROR: "E"
            }[phase_result.result]
            print(f"    {phase_symbol} {phase_result.phase.value}: {phase_result.message or phase_result.result.value}")
    
    def print_summary(self):
        """Print summary of all test results."""
        if not self.results:
            print("No tests run")
            return
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.result == TestResult.PASS)
        failed = sum(1 for r in self.results if r.result == TestResult.FAIL)
        skipped = sum(1 for r in self.results if r.result == TestResult.SKIP)
        errors = sum(1 for r in self.results if r.result == TestResult.ERROR)
        
        print("\n" + "=" * 60)
        print("CONFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Total:   {total}")
        print(f"Passed:  {passed} ({passed/total*100:.1f}%)")
        print(f"Failed:  {failed}")
        print(f"Skipped: {skipped}")
        print(f"Errors:  {errors}")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if result.result == TestResult.FAIL:
                    print(f"  - {result.test_id}: {result.test_name}")
        
        if errors > 0:
            print("\nTests with errors:")
            for result in self.results:
                if result.result == TestResult.ERROR:
                    print(f"  - {result.test_id}: {result.error_message}")


__all__ = [
    "TestResult",
    "PhaseResult",
    "ConformanceTestResult",
    "ConformanceRunner",
]
