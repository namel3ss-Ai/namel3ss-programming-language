"""
Core test execution engine for namel3ss application testing.

This module provides the main test runner that:
1. Loads and parses .ai application modules
2. Configures mock LLM and tool providers  
3. Executes specific targets (prompts, agents, chains)
4. Evaluates assertions against outputs
5. Reports test results

The runner integrates with the existing namel3ss parser, resolver,
typechecker, and runtime components to provide authentic test execution.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from namel3ss.ast import App, Prompt, Chain
from namel3ss.ast.agents import AgentDefinition
from namel3ss.loader import load_app_from_file
from namel3ss.resolver import resolve_app
from namel3ss.types.checker import TypeChecker
from namel3ss.providers.integration import run_chain_with_provider, run_agent_with_provider
from namel3ss.providers.base import N3Provider, ProviderMessage

from namel3ss.testing import (
    TestSuite, TestCase, TestAssertion, AssertionType, 
    TargetType, MockLLMSpec, MockToolSpec
)
from namel3ss.testing.mocks import MockLLMProvider, MockN3Provider, create_mock_llm_from_spec
from namel3ss.testing.tools import MockToolRegistry, create_mock_registry_from_specs


@dataclass
class TestResult:
    """Result of executing a single test case."""
    
    case_name: str
    success: bool
    execution_time_ms: float
    target_output: Any = None
    assertion_results: List[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AssertionResult:
    """Result of evaluating a single assertion."""
    
    assertion: TestAssertion
    passed: bool
    actual_value: Any = None
    error: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Result of executing a complete test suite."""
    
    suite_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    execution_time_ms: float
    case_results: List[TestResult] = field(default_factory=list)
    setup_error: Optional[str] = None
    

class TestRunner:
    """
    Core test execution engine for namel3ss applications.
    
    Orchestrates the complete testing workflow:
    1. Load and validate .ai application modules
    2. Set up mock providers and tools
    3. Execute test targets with mocked dependencies
    4. Evaluate assertions and collect results
    
    Example usage:
        >>> runner = TestRunner()
        >>> suite = load_test_suite("tests/content_analyzer.test.yaml")
        >>> result = await runner.run_test_suite(suite)
        >>> result.passed_cases
        2
        >>> result.failed_cases  
        1
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize test runner.
        
        Args:
            verbose: Enable verbose logging during test execution
        """
        self.verbose = verbose
        self.mock_llm_registry: Dict[str, MockLLMProvider] = {}
        self.mock_tool_registry: Optional[MockToolRegistry] = None
        self.loaded_apps: Dict[str, App] = {}
        
    async def run_test_suite(self, suite: TestSuite) -> TestSuiteResult:
        """
        Execute a complete test suite.
        
        Args:
            suite: Test suite to execute
            
        Returns:
            TestSuiteResult with execution details and case results
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Running test suite: {suite.name}")
            print(f"Application: {suite.app_module}")
        
        # Load and validate application
        try:
            app = await self._load_application(suite.app_module)
        except Exception as e:
            return TestSuiteResult(
                suite_name=suite.name,
                total_cases=len(suite.cases),
                passed_cases=0,
                failed_cases=len(suite.cases),
                execution_time_ms=0,
                setup_error=f"Failed to load application: {e}"
            )
        
        # Setup global mocks
        self._setup_global_mocks(suite.global_mocks)
        
        # Execute test cases
        case_results = []
        for case in suite.cases:
            if self.verbose:
                print(f"  Running test case: {case.name}")
                
            result = await self._run_test_case(case, app)
            case_results.append(result)
            
            if self.verbose:
                status = "PASS" if result.success else "FAIL"
                print(f"    {status} ({result.execution_time_ms:.1f}ms)")
                if not result.success and result.error:
                    print(f"    Error: {result.error}")
        
        execution_time = (time.time() - start_time) * 1000
        passed_cases = sum(1 for result in case_results if result.success)
        failed_cases = len(case_results) - passed_cases
        
        return TestSuiteResult(
            suite_name=suite.name,
            total_cases=len(suite.cases),
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            execution_time_ms=execution_time,
            case_results=case_results
        )
    
    async def _load_application(self, app_module_path: str) -> App:
        """
        Load and validate .ai application module.
        
        Args:
            app_module_path: Path to .ai file
            
        Returns:
            Parsed and resolved App AST
            
        Raises:
            Exception: If loading, parsing, or validation fails
        """
        if app_module_path in self.loaded_apps:
            return self.loaded_apps[app_module_path]
        
        # Load using existing namel3ss infrastructure
        app_path = Path(app_module_path)
        if not app_path.exists():
            raise FileNotFoundError(f"Application file not found: {app_module_path}")
        
        # Parse application using real namel3ss parser
        app = load_app_from_file(app_path)
        
        # Resolve references and validate types
        resolved_app = resolve_app(app)
        
        # Run type checker
        type_checker = TypeChecker()
        type_checker.check_app(resolved_app)
        
        self.loaded_apps[app_module_path] = resolved_app
        return resolved_app
    
    def _setup_global_mocks(self, global_mocks: Dict[str, Any]) -> None:
        """
        Configure global mock providers and tools.
        
        Args:
            global_mocks: Global mock configuration from test suite
        """
        # Setup mock LLMs
        llm_mocks = global_mocks.get('llms', [])
        for llm_mock_config in llm_mocks:
            spec = MockLLMSpec(**llm_mock_config)
            mock_llm = create_mock_llm_from_spec(spec)
            self.mock_llm_registry[spec.model_name] = mock_llm
        
        # Setup mock tools
        tool_mocks = global_mocks.get('tools', [])
        tool_specs = [MockToolSpec(**tool_config) for tool_config in tool_mocks]
        self.mock_tool_registry = create_mock_registry_from_specs(tool_specs)
    
    async def _run_test_case(self, case: TestCase, app: App) -> TestResult:
        """
        Execute a single test case.
        
        Args:
            case: Test case configuration
            app: Loaded application AST
            
        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        
        try:
            # Setup case-specific mocks
            self._setup_case_mocks(case.mocks)
            
            # Execute target
            target_output = await self._execute_target(case.target, case.inputs, app)
            
            # Evaluate assertions
            assertion_results = []
            for assertion in case.assertions:
                result = self._evaluate_assertion(assertion, target_output)
                assertion_results.append(result)
            
            # Determine overall success
            success = all(result.passed for result in assertion_results)
            
            execution_time = (time.time() - start_time) * 1000
            
            return TestResult(
                case_name=case.name,
                success=success,
                execution_time_ms=execution_time,
                target_output=target_output,
                assertion_results=assertion_results
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
                
            return TestResult(
                case_name=case.name,
                success=False,
                execution_time_ms=execution_time,
                error=error_msg
            )
    
    def _setup_case_mocks(self, case_mocks: Dict[str, Any]) -> None:
        """
        Setup mocks specific to this test case.
        
        Args:
            case_mocks: Mock configuration for this case
        """
        # Add case-specific LLM mocks
        llm_mocks = case_mocks.get('llms', [])
        for llm_mock_config in llm_mocks:
            spec = MockLLMSpec(**llm_mock_config)
            if spec.model_name not in self.mock_llm_registry:
                self.mock_llm_registry[spec.model_name] = MockLLMProvider()
            
            mock_llm = self.mock_llm_registry[spec.model_name]
            if spec.response:
                mock_llm.add_response_mapping(
                    model=spec.model_name,
                    prompt_pattern=spec.prompt_pattern,
                    response=spec.response,
                    priority=100  # Case-specific mocks have higher priority
                )
        
        # Add case-specific tool mocks
        tool_mocks = case_mocks.get('tools', [])
        if tool_mocks and self.mock_tool_registry:
            for tool_config in tool_mocks:
                spec = MockToolSpec(**tool_config)
                self.mock_tool_registry.register_mock(
                    tool_name=spec.tool_name,
                    input_pattern=spec.input_pattern,
                    response=spec.response,
                    priority=100  # Higher priority for case-specific
                )
    
    async def _execute_target(
        self, 
        target_config: Dict[str, Any], 
        inputs: Dict[str, Any], 
        app: App
    ) -> Any:
        """
        Execute the specified test target.
        
        Args:
            target_config: Target specification {type: "prompt", name: "analyze_content"}
            inputs: Input parameters for the target
            app: Application AST
            
        Returns:
            Output from target execution
        """
        target_type = TargetType(target_config['type'])
        target_name = target_config['name']
        
        if target_type == TargetType.PROMPT:
            return await self._execute_prompt_target(target_name, inputs, app)
        elif target_type == TargetType.AGENT:
            return await self._execute_agent_target(target_name, inputs, app)
        elif target_type == TargetType.CHAIN:
            return await self._execute_chain_target(target_name, inputs, app)
        elif target_type == TargetType.APP:
            return await self._execute_app_target(inputs, app)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
    
    async def _execute_prompt_target(
        self, 
        prompt_name: str, 
        inputs: Dict[str, Any], 
        app: App
    ) -> Any:
        """Execute a prompt target with mock LLM."""
        # Find prompt in app
        prompt = None
        for p in app.prompts:
            if p.name == prompt_name:
                prompt = p
                break
        
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found in application")
        
        # Get mock LLM for prompt's model
        model_name = prompt.model
        if model_name not in self.mock_llm_registry:
            raise ValueError(f"No mock LLM configured for model '{model_name}'")
        
        mock_llm = self.mock_llm_registry[model_name]
        
        # Render prompt template with inputs
        rendered_prompt = self._render_prompt_template(prompt.template, inputs)
        
        # Execute with mock LLM
        response = await mock_llm.agenerate(rendered_prompt, model=model_name)
        
        return {
            "output_text": response.output_text,
            "metadata": response.metadata,
            "rendered_prompt": rendered_prompt
        }
    
    async def _execute_agent_target(
        self, 
        agent_name: str, 
        inputs: Dict[str, Any], 
        app: App
    ) -> Any:
        """Execute an agent target with mock providers."""
        # Find agent in app
        agent = None
        for a in app.agents:
            if a.name == agent_name:
                agent = a
                break
        
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in application")
        
        # Get mock provider for agent's LLM
        llm_name = agent.llm_name
        if llm_name not in self.mock_llm_registry:
            raise ValueError(f"No mock LLM configured for agent LLM '{llm_name}'")
        
        # Create mock N3Provider from mock LLM
        mock_provider = MockN3Provider()
        mock_llm = self.mock_llm_registry[llm_name]
        
        # Copy response mappings from LLM mock to provider mock
        for mapping in mock_llm.response_mappings:
            mock_provider.add_response_mapping(
                model=mapping.model,
                prompt_pattern=mapping.prompt_pattern.pattern if mapping.prompt_pattern else None,
                response=mapping.response.output_text,
                priority=mapping.priority
            )
        
        # Execute agent with mock provider
        user_input = inputs.get('user_input', inputs.get('input', ''))
        
        result = await run_agent_with_provider(
            agent_def=agent,
            provider=mock_provider,
            user_input=user_input,
            max_turns=inputs.get('max_turns', agent.max_turns)
        )
        
        return result
    
    async def _execute_chain_target(
        self, 
        chain_name: str, 
        inputs: Dict[str, Any], 
        app: App
    ) -> Any:
        """Execute a chain target with mock providers."""
        # Find chain in app
        chain = None
        for c in app.chains:
            if c.name == chain_name:
                chain = c
                break
        
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found in application")
        
        # Create mock provider (use first available mock LLM)
        if not self.mock_llm_registry:
            raise ValueError("No mock LLM providers configured for chain execution")
        
        mock_provider = MockN3Provider()
        
        # Setup provider with mappings from all mock LLMs
        for mock_llm in self.mock_llm_registry.values():
            for mapping in mock_llm.response_mappings:
                mock_provider.add_response_mapping(
                    model=mapping.model,
                    prompt_pattern=mapping.prompt_pattern.pattern if mapping.prompt_pattern else None,
                    response=mapping.response.output_text,
                    priority=mapping.priority
                )
        
        # Execute chain with mock provider
        result = await run_chain_with_provider(
            chain_steps=chain.steps,
            provider=mock_provider,
            initial_input=inputs
        )
        
        return result
    
    async def _execute_app_target(self, inputs: Dict[str, Any], app: App) -> Any:
        """Execute full application target (placeholder for future implementation)."""
        # This would involve setting up the full runtime environment
        # with mocked providers and executing a complete application flow
        raise NotImplementedError("Full application testing not yet implemented")
    
    def _render_prompt_template(self, template: str, inputs: Dict[str, Any]) -> str:
        """
        Render a prompt template with input variables.
        
        Args:
            template: Prompt template with {{variable}} placeholders
            inputs: Input variables
            
        Returns:
            Rendered prompt text
        """
        # Simple template rendering (could use Jinja2 or similar for production)
        rendered = template
        for key, value in inputs.items():
            placeholder = f"{{{{{key}}}}}"
            rendered = rendered.replace(placeholder, str(value))
        
        return rendered
    
    def _evaluate_assertion(self, assertion: TestAssertion, output: Any) -> AssertionResult:
        """
        Evaluate a single assertion against the target output.
        
        Args:
            assertion: Assertion to evaluate
            output: Target execution output
            
        Returns:
            AssertionResult with evaluation details
        """
        try:
            if assertion.type == AssertionType.EQUALS:
                passed = output == assertion.value
                
            elif assertion.type == AssertionType.NOT_EQUALS:
                passed = output != assertion.value
                
            elif assertion.type == AssertionType.CONTAINS:
                output_str = str(output)
                passed = assertion.value in output_str
                
            elif assertion.type == AssertionType.NOT_CONTAINS:
                output_str = str(output)
                passed = assertion.value not in output_str
                
            elif assertion.type == AssertionType.MATCHES:
                import re
                output_str = str(output)
                passed = bool(re.search(assertion.value, output_str))
                
            elif assertion.type == AssertionType.NOT_MATCHES:
                import re
                output_str = str(output)
                passed = not bool(re.search(assertion.value, output_str))
                
            elif assertion.type == AssertionType.HAS_KEYS:
                if isinstance(output, dict):
                    passed = all(key in output for key in assertion.value)
                else:
                    passed = False
                    
            elif assertion.type == AssertionType.MISSING_KEYS:
                if isinstance(output, dict):
                    passed = all(key not in output for key in assertion.value)
                else:
                    passed = True
                    
            elif assertion.type == AssertionType.HAS_LENGTH:
                try:
                    passed = len(output) == assertion.value
                except TypeError:
                    passed = False
                    
            elif assertion.type == AssertionType.TYPE_IS:
                type_name = type(output).__name__
                passed = type_name == assertion.value
                
            elif assertion.type == AssertionType.JSON_PATH:
                # Parse output as JSON and check path
                try:
                    if isinstance(output, str):
                        data = json.loads(output)
                    elif isinstance(output, dict) and 'output_text' in output:
                        data = json.loads(output['output_text'])
                    else:
                        data = output
                    
                    # Simple JSONPath implementation
                    path_value = self._get_json_path_value(data, assertion.path)
                    passed = path_value == assertion.value
                    
                except (json.JSONDecodeError, KeyError, TypeError):
                    passed = False
                    
            elif assertion.type == AssertionType.FIELD_EXISTS:
                if isinstance(output, dict):
                    passed = assertion.value in output
                else:
                    try:
                        if isinstance(output, str):
                            data = json.loads(output)
                        elif isinstance(output, dict) and 'output_text' in output:
                            data = json.loads(output['output_text'])
                        else:
                            data = output
                        passed = assertion.value in data
                    except:
                        passed = False
                        
            elif assertion.type == AssertionType.FIELD_MISSING:
                if isinstance(output, dict):
                    passed = assertion.value not in output
                else:
                    try:
                        if isinstance(output, str):
                            data = json.loads(output)
                        elif isinstance(output, dict) and 'output_text' in output:
                            data = json.loads(output['output_text'])
                        else:
                            data = output
                        passed = assertion.value not in data
                    except:
                        passed = True
                        
            else:
                return AssertionResult(
                    assertion=assertion,
                    passed=False,
                    error=f"Unsupported assertion type: {assertion.type}"
                )
            
            return AssertionResult(
                assertion=assertion,
                passed=passed,
                actual_value=output
            )
            
        except Exception as e:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                actual_value=output,
                error=f"Assertion evaluation failed: {e}"
            )
    
    def _get_json_path_value(self, data: Any, path: str) -> Any:
        """
        Simple JSONPath implementation for basic path expressions.
        
        Args:
            data: JSON data to query
            path: JSONPath expression (e.g., "$.sentiment", "$.results[0].score")
            
        Returns:
            Value at the specified path
            
        Raises:
            KeyError: If path doesn't exist
        """
        if not path.startswith('$.'):
            raise ValueError(f"Invalid JSONPath: {path}")
        
        # Remove $. prefix
        path = path[2:]
        
        # Split path and navigate
        current = data
        for part in path.split('.'):
            if '[' in part and ']' in part:
                # Handle array indexing like "results[0]"
                key = part.split('[')[0]
                index = int(part.split('[')[1].rstrip(']'))
                current = current[key][index]
            else:
                current = current[part]
        
        return current


# Add missing import
from dataclasses import dataclass, field


__all__ = [
    "TestResult",
    "AssertionResult", 
    "TestSuiteResult",
    "TestRunner"
]