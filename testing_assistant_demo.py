"""
Demo of AI-Powered Testing Assistant
Shows automated test generation capabilities.
"""

import asyncio
from pathlib import Path

from namel3ss.ai_assistant.testing_assistant import TestingAssistant


async def demo_testing_assistant():
    """Demo the testing assistant capabilities."""
    
    print("🧪 AI-Powered Testing Assistant Demo")
    print("=" * 50)
    
    # Sample code to test
    sample_code = '''
class UserManager:
    """Manages user operations."""
    
    def __init__(self, database):
        self.database = database
        self.cache = {}
    
    async def create_user(self, username: str, email: str) -> dict:
        """Create a new user."""
        if not username or not email:
            raise ValueError("Username and email are required")
        
        if "@" not in email:
            raise ValueError("Invalid email format")
            
        user_id = await self.database.insert({
            "username": username,
            "email": email,
            "created_at": "2024-01-01"
        })
        
        user = {"id": user_id, "username": username, "email": email}
        self.cache[user_id] = user
        return user
    
    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        if user_id in self.cache:
            return self.cache[user_id]
        
        user = self.database.get(user_id)
        if user:
            self.cache[user_id] = user
        return user
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        return "@" in email and "." in email.split("@")[1]

def calculate_discount(price: float, user_type: str, quantity: int = 1) -> float:
    """Calculate discount based on user type and quantity."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    base_discount = 0
    if user_type == "premium":
        base_discount = 0.15
    elif user_type == "standard":
        base_discount = 0.05
    
    quantity_discount = min(quantity * 0.01, 0.20)
    total_discount = min(base_discount + quantity_discount, 0.50)
    
    return price * (1 - total_discount)
'''
    
    # Initialize testing assistant
    print("🤖 Initializing AI Testing Assistant...")
    try:
        assistant = TestingAssistant(ai_provider="mock", model="gpt-4")
        print("✅ Assistant initialized")
    except Exception as e:
        print(f"⚠️  Using fallback mode: {e}")
        assistant = TestingAssistant(ai_provider="mock", model="mock")
    
    # Generate comprehensive test suite
    print("\\n📝 Analyzing code structure...")
    result = await assistant.generate_test_suite(
        sample_code,
        test_type="unit",
        coverage_targets=["UserManager", "calculate_discount"]
    )
    
    print("\\n🔍 Code Analysis Results:")
    analysis = result["code_analysis"]
    print(f"   Functions found: {len(analysis['functions'])}")
    for func in analysis['functions']:
        print(f"     - {func['name']} ({len(func['args'])} args)")
    
    print(f"   Classes found: {len(analysis['classes'])}")
    for cls in analysis['classes']:
        print(f"     - {cls['name']} ({len(cls['methods'])} methods)")
        for method in cls['methods'][:3]:  # Show first 3 methods
            print(f"       • {method['name']}")
    
    print(f"   Complexity score: {analysis['complexity_score']}")
    
    # Show generated test cases
    print("\\n🧪 Generated Test Cases:")
    test_cases = result["test_cases"]
    for i, test in enumerate(test_cases[:5]):  # Show first 5
        print(f"   {i+1}. {test['test_name']}")
        print(f"      Target: {test['function_name']}")
        print(f"      Description: {test['description']}")
    
    if len(test_cases) > 5:
        print(f"   ... and {len(test_cases) - 5} more test cases")
    
    # Show edge cases
    print("\\n⚡ Generated Edge Cases:")
    edge_cases = result["edge_cases"]
    for i, edge in enumerate(edge_cases[:5]):  # Show first 5
        print(f"   {i+1}. {edge['scenario']} ({edge['risk_level']} risk)")
        print(f"      Target: {edge['target']}")
        print(f"      Condition: {edge['condition']}")
    
    if len(edge_cases) > 5:
        print(f"   ... and {len(edge_cases) - 5} more edge cases")
    
    # Show mock data
    print("\\n🎭 Generated Mock Data:")
    mock_data = result["mock_data"]
    print(f"   Sample inputs: {len(mock_data['sample_inputs'])} functions covered")
    print(f"   Mock objects: {len(mock_data['mock_objects'])} objects")
    print(f"   File contents: {len(mock_data['file_contents'])} files")
    print(f"   Database records: {len(mock_data['database_records'])} records")
    
    # Show fixtures
    print("\\n🔧 Generated Fixtures:")
    fixtures = result["fixtures"]
    for fixture in fixtures:
        print(f"   - {fixture['name']} (scope: {fixture['scope']})")
    
    # Generate test improvements
    print("\\n💡 Test Improvement Suggestions:")
    suggestions = await assistant.suggest_test_improvements(result["full_test_file"][:1000])
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"   {i}. {suggestion}")
    
    # Show a sample of the generated test file
    print("\\n📄 Sample Generated Test Code:")
    print("-" * 40)
    test_preview = result["full_test_file"][:800]
    print(test_preview)
    if len(result["full_test_file"]) > 800:
        print("\\n... (truncated)")
    print("-" * 40)
    
    # Performance summary
    print("\\n📊 Generation Summary:")
    print(f"   ✅ {len(test_cases)} comprehensive test cases")
    print(f"   ⚡ {len(edge_cases)} edge case scenarios") 
    print(f"   🎭 {len(fixtures)} reusable fixtures")
    print(f"   📝 {len(result['full_test_file'].split('def ')) - 1} total test methods")
    print(f"   🔧 Complete test file generated ({len(result['full_test_file'])} characters)")
    
    print("\\n🎉 Testing Assistant Demo Complete!")
    return result


async def demo_coverage_analysis():
    """Demo test coverage analysis."""
    
    print("\\n🔍 Test Coverage Analysis Demo")
    print("=" * 40)
    
    # Sample existing test
    existing_test = '''
def test_calculate_discount_basic():
    """Test basic discount calculation."""
    result = calculate_discount(100.0, "premium")
    assert result == 85.0
'''
    
    # Sample source code
    source_code = '''
def calculate_discount(price, user_type, quantity=1):
    if price < 0:
        raise ValueError("Price cannot be negative")
    # ... implementation
'''
    
    assistant = TestingAssistant(ai_provider="mock", model="mock")
    analysis = await assistant.analyze_test_coverage(existing_test, source_code)
    
    print("📈 Coverage Analysis:")
    print(f"   Coverage score: {analysis['coverage_score']}%")
    print("   Recommendations:")
    for rec in analysis['recommendations']:
        print(f"     • {rec}")
    
    print("✅ Coverage analysis complete!")


async def main():
    """Run the complete testing assistant demo."""
    await demo_testing_assistant()
    await demo_coverage_analysis()


if __name__ == "__main__":
    asyncio.run(main())