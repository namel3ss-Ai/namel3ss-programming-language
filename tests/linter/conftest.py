"""Test configuration and fixtures for linter tests."""

import pytest


# Sample apps for testing various linting scenarios
CLEAN_APP = '''app "CleanApp" {
    model "gpt_4" {
        provider: "openai"
        model_name: "gpt-4"
    }
    
    prompt "greeting" {
        model: "gpt_4"
        template: "Hello {{name}}"
    }
    
    chain "support_flow" {
        prompt "greeting"
    }
    
    dataset "user_data" {
        connector: "postgres"
        query: "SELECT id, name FROM users WHERE active = true"
    }
    
    page "/home" {
        show text: "Welcome"
        show table: user_data
        
        action "greet_user" {
            run chain: support_flow
        }
    }
}'''

PROBLEMATIC_APP = '''app "ProblematicApp" {
    prompt "UnusedCamelCase" {
        model: "gpt-4"
        template: "This prompt is never used"
    }
    
    prompt "used_prompt" {
        model: "gpt-4"
        template: "This one is used"
    }
    
    chain "EmptyChain" {
    }
    
    chain "LongChainWithManySteps" {
        prompt "step1"
        prompt "step2"
        prompt "step3"
        prompt "step4"
        prompt "step5"
        prompt "step6"
        prompt "step7"
        prompt "step8"
        prompt "step9"
        prompt "step10"
        prompt "step11"
        prompt "step12"
        prompt "step13"
        prompt "step14"
        prompt "step15"
        prompt "step16"
        prompt "step17"
        prompt "step18"
        prompt "step19"
        prompt "step20"
        prompt "step21"
        prompt "step22"
        prompt "step23"
    }
    
    dataset "inefficient_dataset" {
        connector: "postgres"
        query: "SELECT * FROM large_table"
    }
    
    config "security_issue" {
        password: "hardcoded123"
        api_key: "sk-1234567890"
        secret: "my-secret-token"
    }
    
    page "/empty_page" {
    }
    
    page "/normal_page" {
        show text: "This page uses the prompt"
        
        action "test" {
            run prompt: used_prompt
        }
    }
}'''

SYNTAX_ERROR_APP = '''app "SyntaxError" 
    missing opening brace
    invalid: syntax here
}'''

@pytest.fixture
def clean_app():
    """Well-formed app with no issues."""
    return CLEAN_APP

@pytest.fixture  
def problematic_app():
    """App with various linting issues."""
    return PROBLEMATIC_APP

@pytest.fixture
def syntax_error_app():
    """App with syntax errors."""
    return SYNTAX_ERROR_APP

@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    def _create_temp_file(content: str, name: str = "test.ai"):
        file_path = tmp_path / name
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)
    return _create_temp_file