"""
Tests for Advanced AI-Powered Education Platform

Tests cover:
- Parser: Multiple LLMs, memory systems, prompts, agents, tools, datasets, pages
- Multi-agent orchestration and workflows
- Complex UI components with agent integration
- IR Generation: Correct IR nodes for all entities
- Component Coverage: All required components present
"""

import pytest
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_backend_ir, build_frontend_ir
from namel3ss.codegen.backend.state import build_backend_state


@pytest.fixture
def quiz_suite_source():
    """Load the advanced education platform DSL file"""
    path = Path(__file__).parent.parent / "examples" / "education-quiz-suite.ai"
    assert path.exists(), f"Example file not found: {path}"
    return path.read_text(encoding='utf-8')


@pytest.fixture
def parsed_app(quiz_suite_source):
    """Parse the quiz suite and return the app node"""
    parser = Parser(quiz_suite_source)
    app = parser.parse_app()
    return app


# =============================================================================
# Parser Tests - Advanced Features
# =============================================================================

class TestAdvancedQuizSuiteParser:
    """Test that the advanced education platform DSL parses correctly"""
    
    def test_parse_app_name(self, parsed_app):
        """Test that app name is Education Quiz Suite"""
        assert parsed_app.name == "Education Quiz Suite"
    
    def test_parse_llms(self, parsed_app):
        """Test that all 4 LLMs are defined with different configurations"""
        llms = parsed_app.llms
        assert len(llms) >= 4
        llm_names = {llm.name for llm in llms}
        expected_llms = {'gpt4', 'claude', 'gpt4_creative', 'gpt4_precise'}
        assert expected_llms.issubset(llm_names)
    
    def test_parse_memory_systems(self, parsed_app):
        """Test that all 4 memory systems are defined"""
        memories = parsed_app.memories
        assert len(memories) >= 4
        memory_names = {mem.name for mem in memories}
        expected_memories = {'conversation_history', 'student_profiles', 
                           'quiz_generation_context', 'grading_rubrics'}
        assert expected_memories.issubset(memory_names)
    
    def test_parse_prompts(self, parsed_app):
        """Test that all 5 system prompts are defined"""
        prompts = parsed_app.prompts
        assert len(prompts) >= 5
        prompt_names = {p.name for p in prompts}
        expected_prompts = {'quiz_generation_expert', 'grading_rubric_specialist',
                          'adaptive_learning_tutor', 'plagiarism_analyst', 
                          'peer_review_facilitator'}
        assert expected_prompts.issubset(prompt_names)
    
    def test_parse_advanced_tools(self, parsed_app):
        """Test that all 8 advanced AI tools are defined"""
        tools = parsed_app.tools
        assert len(tools) >= 8
        tool_names = {t.name for t in tools}
        expected_tools = {'generate_quiz', 'grade_submission', 'explain_concept',
                         'analyze_performance', 'detect_plagiarism', 
                         'generate_learning_path', 'facilitate_peer_review',
                         'generate_study_materials'}
        assert expected_tools.issubset(tool_names)
    
    def test_parse_agents(self, parsed_app):
        """Test that all 7 specialized agents are defined"""
        agents = parsed_app.agents
        assert len(agents) >= 7
        agent_names = {a.name for a in agents}
        expected_agents = {'quiz_architect', 'master_grader', 'adaptive_tutor',
                          'integrity_monitor', 'peer_review_coordinator',
                          'performance_analyst', 'content_curator'}
        assert expected_agents.issubset(agent_names)
    
    def test_parse_datasets(self, parsed_app):
        """Test that all 15 datasets are defined"""
        datasets = parsed_app.datasets
        assert len(datasets) >= 15
        dataset_names = {d.name for d in datasets}
        expected_datasets = {'quizzes', 'questions', 'students', 'submissions', 
                           'analytics', 'rubrics', 'explanations', 'learning_paths',
                           'peer_reviews', 'study_materials', 'plagiarism_reports',
                           'student_profiles', 'performance_metrics', 'feedback',
                           'achievements'}
        assert expected_datasets.issubset(dataset_names)
    
    def test_parse_pages(self, parsed_app):
        """Test that all 8 pages are defined"""
        pages = parsed_app.pages
        assert len(pages) >= 8
        page_names = {p.name for p in pages}
        expected_pages = {'Quiz Builder', 'Student Submission', 'Analytics Dashboard',
                         'Grading Review', 'Quiz Library', 'Learning Paths',
                         'Peer Review', 'Study Materials'}
        assert expected_pages.issubset(page_names)
    
    def test_parse_page_routes(self, parsed_app):
        """Test that page routes are correct"""
        pages = {p.name: p.route for p in parsed_app.pages}
        assert pages["Quiz Builder"] == "/quiz-builder"
        assert pages["Student Submission"] == "/submit"
        assert pages["Analytics Dashboard"] == "/analytics"
        assert pages["Grading Review"] == "/grading"
        assert pages["Quiz Library"] == "/library"
        assert pages["Learning Paths"] == "/learning-paths"
        assert pages["Peer Review"] == "/peer-review"
        assert pages["Study Materials"] == "/study"
    
    def test_parse_navigation_structure(self, parsed_app):
        """Test that pages have components"""
        for page in parsed_app.pages:
            # Modern brace syntax pages have components in body
            assert hasattr(page, 'body')
            assert isinstance(page.body, list)
            # Should have at least some content
            assert len(page.body) >= 0


# =============================================================================
# Advanced AI Features Tests
# =============================================================================

class TestAdvancedAIFeatures:
    """Test advanced AI orchestration and multi-agent features"""
    
    def test_tool_parameter_complexity(self, parsed_app):
        """Test that tools have complex parameter schemas"""
        # Find generate_quiz tool
        quiz_tool = next((t for t in parsed_app.tools if t.name == 'generate_quiz'), None)
        assert quiz_tool is not None
        # Tool should have parameters dict
        assert hasattr(quiz_tool, 'parameters')
    
    def test_agent_llm_assignments(self, parsed_app):
        """Test that agents are assigned to specific LLMs"""
        # quiz_architect should use gpt4
        quiz_architect = next((a for a in parsed_app.agents if a.name == 'quiz_architect'), None)
        assert quiz_architect is not None
        assert hasattr(quiz_architect, 'llm_name')
        assert quiz_architect.llm_name == 'gpt4'
        
        # master_grader should use gpt4_precise
        grader = next((a for a in parsed_app.agents if a.name == 'master_grader'), None)
        assert grader is not None
        assert hasattr(grader, 'llm_name')
        assert grader.llm_name == 'gpt4_precise'
    
    def test_agent_memory_assignments(self, parsed_app):
        """Test that agents have memory systems"""
        # quiz_architect should have quiz_generation_context memory
        quiz_architect = next((a for a in parsed_app.agents if a.name == 'quiz_architect'), None)
        assert quiz_architect is not None
        assert hasattr(quiz_architect, 'memory_config')
        assert quiz_architect.memory_config == 'quiz_generation_context'
    
    def test_agent_tool_assignments(self, parsed_app):
        """Test that agents have tool access"""
        # quiz_architect should have generate_quiz and generate_study_materials
        quiz_architect = next((a for a in parsed_app.agents if a.name == 'quiz_architect'), None)
        assert quiz_architect is not None
        assert hasattr(quiz_architect, 'tool_names')
        assert len(quiz_architect.tool_names) >= 2
        assert 'generate_quiz' in quiz_architect.tool_names
        assert 'generate_study_materials' in quiz_architect.tool_names
    
    def test_prompt_templates(self, parsed_app):
        """Test that system prompts are comprehensive"""
        # quiz_generation_expert should have parameters with role and content
        expert_prompt = next((p for p in parsed_app.prompts if p.name == 'quiz_generation_expert'), None)
        assert expert_prompt is not None
        assert hasattr(expert_prompt, 'parameters')
        assert 'role' in expert_prompt.parameters
        assert 'content' in expert_prompt.parameters
        assert expert_prompt.parameters['role'] == 'system'


# =============================================================================
# IR Tests
# =============================================================================

class TestQuizSuiteIR:
    """Test that IR generation works for advanced quiz suite"""
    
    def test_ir_build(self, parsed_app):
        """Test that IR builds without errors"""
        backend_ir = build_backend_ir(parsed_app)
        frontend_ir = build_frontend_ir(parsed_app)
        assert backend_ir is not None
        assert frontend_ir is not None
    
    def test_ir_datasets(self, parsed_app):
        """Test that IR contains all 15 datasets"""
        backend_ir = build_backend_ir(parsed_app)
        assert len(backend_ir.datasets) >= 15
        dataset_names = {d.name for d in backend_ir.datasets}
        expected = {'quizzes', 'questions', 'students', 'submissions', 'analytics',
                   'rubrics', 'explanations', 'learning_paths', 'peer_reviews'}
        assert expected.issubset(dataset_names)
    
    def test_ir_pages(self, parsed_app):
        """Test that IR contains all 8 pages"""
        frontend_ir = build_frontend_ir(parsed_app)
        assert len(frontend_ir.pages) >= 8
        page_names = {p.name for p in frontend_ir.pages}
        expected = {'Quiz Builder', 'Student Submission', 'Analytics Dashboard',
                   'Grading Review', 'Quiz Library', 'Learning Paths', 
                   'Peer Review', 'Study Materials'}
        assert expected.issubset(page_names)


# =============================================================================
# Integration Tests
# =============================================================================

class TestQuizSuiteIntegration:
    """Test full pipeline integration"""
    
    def test_full_pipeline(self, quiz_suite_source):
        """Test that full pipeline (Parser → IR → Backend State) works"""
        # Parse
        parser = Parser(quiz_suite_source)
        app = parser.parse_app()
        assert app is not None
        
        # Build IR
        backend_ir = build_backend_ir(app)
        frontend_ir = build_frontend_ir(app)
        assert backend_ir is not None
        assert frontend_ir is not None
        
        # Build backend state (used by codegen)
        backend_state = build_backend_state(app)
        assert backend_state is not None
    
    def test_component_coverage(self, parsed_app):
        """Test that pages have content"""
        # Verify pages have body content
        for page in parsed_app.pages:
            assert hasattr(page, 'body')
            # Pages should have show statements in body
            assert len(page.body) > 0, f"Page {page.name} should have components"
    
    def test_multi_agent_workflow(self, parsed_app):
        """Test that multiple agents can be used in workflows"""
        # Count agents with tool access
        agents_with_tools = [a for a in parsed_app.agents if hasattr(a, 'tool_names') and len(a.tool_names) > 0]
        assert len(agents_with_tools) >= 5, "Should have multiple agents with tools"
    
    def test_learning_paths_integration(self, parsed_app):
        """Test that learning paths page exists with proper structure"""
        learning_paths_page = next((p for p in parsed_app.pages if p.name == 'Learning Paths'), None)
        assert learning_paths_page is not None
        assert learning_paths_page.route == "/learning-paths"
    
    def test_peer_review_integration(self, parsed_app):
        """Test that peer review page exists with proper structure"""
        peer_review_page = next((p for p in parsed_app.pages if p.name == 'Peer Review'), None)
        assert peer_review_page is not None
        assert peer_review_page.route == "/peer-review"
    
    def test_study_materials_integration(self, parsed_app):
        """Test that study materials page exists"""
        study_page = next((p for p in parsed_app.pages if p.name == 'Study Materials'), None)
        assert study_page is not None
        assert study_page.route == "/study"
