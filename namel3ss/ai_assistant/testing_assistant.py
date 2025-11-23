"""
AI-Powered Testing Assistant for Namel3ss
Generates comprehensive tests, edge cases, and mock data.
"""

import ast
import asyncio
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Mock provider for demo
class MockAIProvider:
    def __init__(self, name, model):
        self.name = name
        self.model = model
    
    async def generate_completion(self, prompt):
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse("Generated test content")


class TestingAssistant:
    """AI-powered testing assistant for generating comprehensive test suites."""
    
    def __init__(self, ai_provider: str = "openai", model: str = "gpt-4"):
        try:
            from namel3ss.providers import get_provider
            self.provider = get_provider(ai_provider, model)
        except ImportError:
            # Fallback to mock provider for demo
            self.provider = MockAIProvider(ai_provider, model)
            self.provider = MockAIProvider(ai_provider, model)
    
    async def generate_test_suite(
        self, 
        code: str, 
        test_type: str = "unit",
        coverage_targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive test suite for given code.
        
        Args:
            code: Source code to test
            test_type: Type of tests (unit, integration, e2e)
            coverage_targets: Specific functions/classes to target
            
        Returns:
            Generated test suite with multiple test cases
        """
        
        # Analyze the code structure
        code_analysis = await self._analyze_code_structure(code)
        
        # Generate test cases
        test_cases = await self._generate_test_cases(
            code_analysis, test_type, coverage_targets
        )
        
        # Generate edge cases
        edge_cases = await self._generate_edge_cases(code_analysis)
        
        # Generate mock data
        mock_data = await self._generate_mock_data(code_analysis)
        
        # Generate test fixtures
        fixtures = await self._generate_fixtures(code_analysis)
        
        return {
            "code_analysis": code_analysis,
            "test_cases": test_cases,
            "edge_cases": edge_cases,
            "mock_data": mock_data,
            "fixtures": fixtures,
            "full_test_file": await self._assemble_test_file(
                test_cases, edge_cases, mock_data, fixtures
            )
        }
    
    async def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure to understand what needs testing."""
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}", "functions": [], "classes": []}
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "line_number": node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "is_async": isinstance(item, ast.AsyncFunctionDef),
                            "is_property": any("property" in ast.unparse(d) for d in item.decorator_list)
                        })
                
                classes.append({
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": methods,
                    "docstring": ast.get_docstring(node),
                    "line_number": node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                else:
                    module = node.module or ""
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        return {
            "functions": functions,
            "classes": classes, 
            "imports": imports,
            "complexity_score": len(functions) + len(classes) * 2
        }
    
    async def _generate_test_cases(
        self, 
        analysis: Dict[str, Any], 
        test_type: str,
        coverage_targets: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases."""
        
        prompt = f"""
Generate comprehensive {test_type} test cases for the following code structure:

Functions: {analysis['functions']}
Classes: {analysis['classes']}
Imports: {analysis['imports']}

Focus on: {coverage_targets if coverage_targets else 'all components'}

For each function/method, generate test cases that cover:
1. Normal operation with valid inputs
2. Boundary conditions
3. Error conditions and exception handling
4. Type validation
5. Side effects and state changes

Return a JSON array of test cases with this structure:
{{
    "function_name": "name_of_function_being_tested",
    "test_name": "descriptive_test_name", 
    "description": "what this test validates",
    "setup": "any setup code needed",
    "test_code": "the actual test code",
    "assertions": ["list of assertions being made"],
    "mock_requirements": ["any mocks needed"],
    "test_data": "sample input data"
}}

Focus on practical, runnable tests using pytest conventions.
"""
        
        response = await self.provider.generate_completion(prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: generate basic test cases
        test_cases = []
        for func in analysis['functions']:
            test_cases.append({
                "function_name": func['name'],
                "test_name": f"test_{func['name']}_basic_functionality",
                "description": f"Test basic functionality of {func['name']}",
                "setup": "",
                "test_code": f"result = {func['name']}()",
                "assertions": ["assert result is not None"],
                "mock_requirements": [],
                "test_data": {}
            })
        
        return test_cases
    
    async def _generate_edge_cases(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge case scenarios."""
        
        prompt = f"""
Generate edge case test scenarios for:

Functions: {analysis['functions']}
Classes: {analysis['classes']}

Focus on:
1. Empty inputs, None values, zero values
2. Very large inputs, overflow conditions
3. Invalid types, malformed data
4. Network timeouts, connection failures
5. File system errors, permission issues
6. Race conditions, concurrent access
7. Memory constraints, resource exhaustion

Return JSON array of edge cases:
{{
    "scenario": "brief_description",
    "target": "function_or_class_name",
    "condition": "what makes this an edge case",
    "test_code": "test implementation",
    "expected_behavior": "what should happen",
    "risk_level": "low|medium|high"
}}
"""
        
        response = await self.provider.generate_completion(prompt)
        
        # Fallback edge cases
        edge_cases = []
        for func in analysis['functions']:
            edge_cases.extend([
                {
                    "scenario": "null_input",
                    "target": func['name'],
                    "condition": "Function called with None",
                    "test_code": f"with pytest.raises(TypeError): {func['name']}(None)",
                    "expected_behavior": "Should raise TypeError",
                    "risk_level": "medium"
                },
                {
                    "scenario": "empty_input", 
                    "target": func['name'],
                    "condition": "Function called with empty values",
                    "test_code": f"result = {func['name']}('')",
                    "expected_behavior": "Should handle gracefully",
                    "risk_level": "low"
                }
            ])
        
        return edge_cases
    
    async def _generate_mock_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic mock data for testing."""
        
        prompt = f"""
Generate realistic mock data for testing these components:

Functions: {analysis['functions']}
Classes: {analysis['classes']}

Create mock data that includes:
1. Valid sample inputs for each function
2. Mock objects for complex dependencies  
3. Sample responses for external APIs
4. Fixture data for database operations
5. File content samples for I/O operations

Return JSON with structure:
{{
    "sample_inputs": {{"function_name": ["input1", "input2"]}},
    "mock_objects": {{"class_name": {{"property": "value"}}}},
    "api_responses": {{"endpoint": {{"success": {{}}, "error": {{}}}}}},
    "file_contents": {{"filename": "content"}},
    "database_records": [{{}}]
}}
"""
        
        response = await self.provider.generate_completion(prompt)
        
        # Fallback mock data
        mock_data = {
            "sample_inputs": {},
            "mock_objects": {},
            "api_responses": {},
            "file_contents": {
                "test.txt": "test content",
                "config.json": '{"key": "value"}'
            },
            "database_records": [{"id": 1, "name": "test"}]
        }
        
        for func in analysis['functions']:
            mock_data["sample_inputs"][func['name']] = ["test_input", 123, True]
            
        return mock_data
    
    async def _generate_fixtures(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate pytest fixtures for common test setup."""
        
        fixtures = [
            {
                "name": "mock_llm_provider",
                "scope": "function", 
                "code": '''@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    class MockProvider:
        async def generate_completion(self, prompt):
            return MockResponse("Mock LLM response")
    return MockProvider()'''
            },
            {
                "name": "sample_data",
                "scope": "session",
                "code": '''@pytest.fixture(scope="session") 
def sample_data():
    """Sample data for tests."""
    return {
        "test_string": "Hello World",
        "test_number": 42,
        "test_list": [1, 2, 3],
        "test_dict": {"key": "value"}
    }'''
            },
            {
                "name": "temp_file",
                "scope": "function",
                "code": '''@pytest.fixture
def temp_file(tmp_path):
    """Create temporary file for testing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    return test_file'''
            }
        ]
        
        # Add class-specific fixtures
        for cls in analysis['classes']:
            fixtures.append({
                "name": f"mock_{cls['name'].lower()}",
                "scope": "function",
                "code": f'''@pytest.fixture
def mock_{cls['name'].lower()}():
    """Mock {cls['name']} for testing."""
    return Mock(spec={cls['name']})'''
            })
        
        return fixtures
    
    async def _assemble_test_file(
        self,
        test_cases: List[Dict[str, Any]],
        edge_cases: List[Dict[str, Any]], 
        mock_data: Dict[str, Any],
        fixtures: List[Dict[str, Any]]
    ) -> str:
        """Assemble complete test file."""
        
        imports = '''"""Auto-generated test file."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import tempfile

'''
        
        # Add fixtures
        fixtures_code = "\n\n".join([f["code"] for f in fixtures])
        
        # Add test cases
        test_methods = []
        for test in test_cases:
            method_code = f'''
def {test['test_name']}():
    """{test['description']}"""
    {test.get('setup', '')}
    {test['test_code']}
    # {test.get('assertions', [])}
'''
            test_methods.append(method_code)
        
        # Add edge case tests
        for edge in edge_cases:
            method_code = f'''
def test_{edge['target']}_edge_case_{edge['scenario']}():
    """Edge case: {edge['condition']}"""
    {edge['test_code']}
'''
            test_methods.append(method_code)
        
        return imports + "\n" + fixtures_code + "\n" + "\n".join(test_methods)
    
    async def analyze_test_coverage(self, test_file: str, source_file: str) -> Dict[str, Any]:
        """Analyze test coverage and suggest improvements."""
        
        prompt = f"""
Analyze test coverage for:

Test file content: {test_file[:2000]}...
Source file content: {source_file[:2000]}...

Identify:
1. Functions/methods not covered by tests
2. Missing edge cases 
3. Insufficient error handling tests
4. Missing integration tests
5. Recommendations for improvement

Return analysis as JSON with coverage percentage and suggestions.
"""
        
        response = await self.provider.generate_completion(prompt)
        
        return {
            "coverage_analysis": response.text,
            "missing_tests": [],
            "recommendations": ["Add more edge case tests", "Improve error handling coverage"],
            "coverage_score": 75  # Placeholder
        }
    
    async def suggest_test_improvements(self, existing_tests: str) -> List[str]:
        """Suggest improvements for existing test suite."""
        
        prompt = f"""
Review this test suite and suggest improvements:

{existing_tests}

Focus on:
1. Test organization and structure
2. Better assertions and test data
3. Missing test scenarios
4. Performance test considerations
5. Maintainability improvements

Provide specific, actionable suggestions.
"""
        
        response = await self.provider.generate_completion(prompt)
        
        # Parse suggestions from response
        suggestions = []
        lines = response.text.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                suggestions.append(line.strip())
        
        return suggestions or [
            "Add more comprehensive assertions",
            "Include performance benchmarks", 
            "Add integration test scenarios",
            "Improve test data variety",
            "Add negative test cases"
        ]


# CLI interface for testing assistant
async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Testing Assistant")
    parser.add_argument("file", help="Python file to generate tests for")
    parser.add_argument("--output", "-o", help="Output test file name")
    parser.add_argument("--type", choices=["unit", "integration", "e2e"], default="unit")
    parser.add_argument("--coverage", nargs="+", help="Specific functions to target")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File {args.file} not found")
        return
    
    # Read source file
    with open(args.file, 'r') as f:
        source_code = f.read()
    
    # Generate tests
    assistant = TestingAssistant()
    result = await assistant.generate_test_suite(
        source_code, 
        test_type=args.type,
        coverage_targets=args.coverage
    )
    
    # Write output
    output_file = args.output or f"test_{Path(args.file).stem}.py"
    with open(output_file, 'w') as f:
        f.write(result["full_test_file"])
    
    print(f"✅ Generated test file: {output_file}")
    print(f"📊 Generated {len(result['test_cases'])} test cases")
    print(f"🔍 Generated {len(result['edge_cases'])} edge cases") 
    print(f"🎭 Generated {len(result['fixtures'])} fixtures")


if __name__ == "__main__":
    asyncio.run(main())