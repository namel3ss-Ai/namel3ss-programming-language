"""Test configuration and fixtures for formatting tests."""

import pytest


# Basic test apps for formatting tests
SIMPLE_APP = '''app "TestApp" {
    page "/home" {
        show text: "Hello World"
    }
}'''

COMPLEX_APP = '''app "AdvancedApp" {
    theme: "modern"
    
    model "gpt-4" {
        provider: "openai"
        model_name: "gpt-4"
    }
    
    prompt "greeting" {
        model: "gpt-4"
        template: """
        Hello {{name}}, welcome to our app!
        How can I help you today?
        """
    }
    
    chain "support_flow" {
        prompt "classify"
        prompt "respond"
    }
    
    dataset "users" {
        connector: "postgres"
        query: """
        SELECT id, name, email 
        FROM users 
        WHERE active = true
        """
    }
    
    page "/dashboard" {
        show text: "Welcome to the dashboard"
        show table: users
        
        action "help" {
            run chain: support_flow
        }
    }
}'''

MALFORMED_APP = '''app "BadApp" 
    missing opening brace
    invalid syntax here
}'''

@pytest.fixture
def simple_app():
    """Simple app for basic tests."""
    return SIMPLE_APP

@pytest.fixture
def complex_app():
    """Complex app for comprehensive tests.""" 
    return COMPLEX_APP

@pytest.fixture
def malformed_app():
    """Malformed app for error handling tests."""
    return MALFORMED_APP