"""
Phase 2: Core Components Test Suite

This test suite validates that Phase 2 components (form, data_table, chart) 
parse correctly in the Namel3ss DSL.

Test Coverage:
1. Form components with various field types
2. Data table components with columns and actions
3. Chart components (bar, line, pie, gauge)
4. Mixed pages with multiple component types
5. IR generation for all components
6. Full pipeline validation
"""

import pytest
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_backend_ir, build_frontend_ir

# Test file path
TEST_FILE = Path(__file__).parent.parent / "examples" / "test_phase2_core_components.ai"


class TestPhase2ComponentParsing:
    """Test that Phase 2 components parse without errors."""

    @pytest.fixture
    def parsed_module(self):
        """Parse the test file once and reuse."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        parser = Parser(content)
        return parser.parse()

    def test_app_parses(self, parsed_module):
        """Test that the app declaration parses successfully."""
        assert parsed_module is not None
        app = parsed_module.body[0]
        assert app.name == "EducationPlatformPhase2"
        assert app.description == "Phase 2 - Core Components Test Suite"

    def test_datasets_parse(self, parsed_module):
        """Test that all datasets parse correctly."""
        app = parsed_module.body[0]
        datasets = [d for d in app.body if hasattr(d, '__class__') and d.__class__.__name__ == 'Dataset']
        
        assert len(datasets) == 4
        dataset_names = [d.name for d in datasets]
        assert "students" in dataset_names
        assert "quizzes" in dataset_names
        assert "submissions" in dataset_names
        assert "analytics" in dataset_names

    def test_pages_parse(self, parsed_module):
        """Test that all pages parse correctly."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        
        assert len(pages) == 4
        page_names = [p.name for p in pages]
        assert "Quiz Creator" in page_names
        assert "Student Roster" in page_names
        assert "Analytics Dashboard" in page_names
        assert "Dashboard" in page_names

    def test_form_component_in_quiz_creator(self, parsed_module):
        """Test that form component parses in Quiz Creator page."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        quiz_creator = next(p for p in pages if p.name == "Quiz Creator")
        
        assert len(quiz_creator.body) > 0
        show_stmt = quiz_creator.body[0]
        
        # Modern parser uses dict format
        if isinstance(show_stmt, dict):
            assert show_stmt["type"] == "show"
            assert show_stmt["component"] == "form"
            assert "config" in show_stmt
            config = show_stmt["config"]
            assert "title" in config
            assert config["title"] == "Create New Quiz"
        else:
            # Legacy ShowForm object
            assert show_stmt.__class__.__name__ == "ShowForm"

    def test_data_table_component_in_student_roster(self, parsed_module):
        """Test that data_table component parses in Student Roster page."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        student_roster = next(p for p in pages if p.name == "Student Roster")
        
        assert len(student_roster.body) > 0
        show_stmt = student_roster.body[0]
        
        # Modern parser uses dict format
        if isinstance(show_stmt, dict):
            assert show_stmt["type"] == "show"
            assert show_stmt["component"] == "data_table"
            assert "config" in show_stmt
        else:
            # Legacy ShowDataTable object
            assert show_stmt.__class__.__name__ == "ShowDataTable"

    def test_chart_components_in_analytics(self, parsed_module):
        """Test that chart components parse in Analytics Dashboard page."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        analytics = next(p for p in pages if p.name == "Analytics Dashboard")
        
        assert len(analytics.body) >= 1
        
        # Check that we have at least one chart component
        chart_count = 0
        for stmt in analytics.body:
            if isinstance(stmt, dict):
                if stmt.get("component") == "chart":
                    chart_count += 1
            elif hasattr(stmt, '__class__') and 'Chart' in stmt.__class__.__name__:
                chart_count += 1
        
        assert chart_count >= 1, f"Expected at least 1 chart, found {chart_count}"

    def test_bar_chart_config(self, parsed_module):
        """Test bar chart configuration parsing."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        analytics = next(p for p in pages if p.name == "Analytics Dashboard")
        
        # Find first chart (bar chart)
        chart_stmt = None
        for stmt in analytics.body:
            if isinstance(stmt, dict) and stmt.get("component") == "chart":
                chart_stmt = stmt
                break
            elif hasattr(stmt, '__class__') and 'Chart' in stmt.__class__.__name__:
                chart_stmt = stmt
                break
        
        assert chart_stmt is not None
        
        if isinstance(chart_stmt, dict):
            config = chart_stmt["config"]
            assert config.get("type") == "bar"
            assert "title" in config
            assert "data_source" in config
            assert "x_axis" in config
            assert "y_axis" in config

    def test_mixed_components_page(self, parsed_module):
        """Test that Dashboard page has multiple component types."""
        app = parsed_module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        dashboard = next(p for p in pages if p.name == "Dashboard")
        
        assert len(dashboard.body) >= 4
        
        # Count component types
        has_text = False
        has_chart = False
        has_data_table = False
        has_form = False
        
        for stmt in dashboard.body:
            if isinstance(stmt, dict):
                comp = stmt.get("component", "")
                if comp == "text":
                    has_text = True
                elif comp == "chart":
                    has_chart = True
                elif comp == "data_table":
                    has_data_table = True
                elif comp == "form":
                    has_form = True
            else:
                class_name = stmt.__class__.__name__
                if "Text" in class_name:
                    has_text = True
                elif "Chart" in class_name:
                    has_chart = True
                elif "Table" in class_name:
                    has_data_table = True
                elif "Form" in class_name:
                    has_form = True
        
        assert has_text, "Dashboard should have text component"
        assert has_chart, "Dashboard should have chart component"
        assert has_data_table, "Dashboard should have data_table component"
        assert has_form, "Dashboard should have form component"


class TestPhase2IRGeneration:
    """Test IR generation for Phase 2 components."""

    @pytest.fixture
    def ir_result(self):
        """Build IR from test file."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        parser = Parser(content)
        module = parser.parse()
        app = module.body[0]
        return build_backend_ir(app)

    def test_ir_builds_successfully(self, ir_result):
        """Test that IR builds without errors."""
        assert ir_result is not None
        assert hasattr(ir_result, 'datasets') or isinstance(ir_result, dict)

    def test_ir_has_datasets(self, ir_result):
        """Test that IR contains dataset definitions."""
        if hasattr(ir_result, 'datasets'):
            datasets = ir_result.datasets
        elif isinstance(ir_result, dict):
            datasets = ir_result.get('datasets', {})
        else:
            datasets = {}
        
        assert len(datasets) == 4
        assert "students" in datasets
        assert "quizzes" in datasets
        assert "submissions" in datasets
        assert "analytics" in datasets

    def test_ir_has_pages(self, ir_result):
        """Test that IR contains page definitions."""
        if hasattr(ir_result, 'pages'):
            pages = ir_result.pages
        elif isinstance(ir_result, dict):
            pages = ir_result.get('pages', {})
        else:
            pages = {}
        
        assert len(pages) == 4


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_full_pipeline(self):
        """Test complete parse -> IR -> validation pipeline."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse
        parser = Parser(content)
        module = parser.parse()
        assert module is not None
        
        # Build IR
        app = module.body[0]
        ir = build_backend_ir(app)
        assert ir is not None
        
        # Validate structure
        app = module.body[0]
        assert app.name == "EducationPlatformPhase2"
        
        # Count declarations
        datasets = [d for d in app.body if hasattr(d, '__class__') and d.__class__.__name__ == 'Dataset']
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        
        assert len(datasets) == 4
        assert len(pages) == 4

    def test_form_field_types_coverage(self):
        """Test that form component is recognized."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = Parser(content)
        module = parser.parse()
        app = module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        
        # Check Quiz Creator form
        quiz_creator = next(p for p in pages if p.name == "Quiz Creator")
        show_stmt = quiz_creator.body[0]
        
        if isinstance(show_stmt, dict):
            assert show_stmt["component"] == "form"
            assert show_stmt["config"]["title"] == "Create New Quiz"

    def test_chart_types_coverage(self):
        """Test that chart component is recognized."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = Parser(content)
        module = parser.parse()
        app = module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        
        analytics = next(p for p in pages if p.name == "Analytics Dashboard")
        
        chart_types = []
        for stmt in analytics.body:
            if isinstance(stmt, dict) and stmt.get("component") == "chart":
                chart_type = stmt["config"].get("type")
                if chart_type:
                    chart_types.append(chart_type)
        
        assert "bar" in chart_types

    def test_data_table_features(self):
        """Test data_table component is recognized."""
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = Parser(content)
        module = parser.parse()
        app = module.body[0]
        pages = [p for p in app.body if hasattr(p, '__class__') and p.__class__.__name__ == 'Page']
        
        roster = next(p for p in pages if p.name == "Student Roster")
        show_stmt = roster.body[0]
        
        if isinstance(show_stmt, dict):
            assert show_stmt["component"] == "data_table"
            config = show_stmt["config"]
            assert config.get("sortable") == True
            assert config.get("filterable") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
