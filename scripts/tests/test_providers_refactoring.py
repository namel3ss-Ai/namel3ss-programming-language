#!/usr/bin/env python3
"""
Test script to verify Providers refactoring - imports, validation, and error handling.
"""

def test_imports():
    """Test that all new types can be imported."""
    print("Testing imports...")
    
    try:
        from namel3ss.providers import (
            # Base types
            N3Provider,
            ProviderMessage,
            ProviderResponse,
            ProviderError,
            BaseLLMAdapter,
            # Errors
            ProviderValidationError,
            ProviderAuthError,
            ProviderRateLimitError,
            ProviderAPIError,
            ProviderTimeoutError,
            # Validation
            validate_provider_name,
            validate_model_name,
            validate_temperature,
            validate_max_tokens,
            validate_top_p,
            validate_api_key,
            validate_endpoint,
            validate_message,
            validate_messages,
            validate_provider_config,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_validation_functions():
    """Test validation functions."""
    print("\nTesting validation functions...")
    
    from namel3ss.providers import (
        validate_provider_name,
        validate_model_name,
        validate_temperature,
        validate_max_tokens,
        validate_top_p,
        validate_api_key,
        validate_endpoint,
        validate_message,
        validate_messages,
        validate_provider_config,
        ProviderValidationError,
    )
    
    # Test valid inputs
    try:
        validate_provider_name("openai")
        validate_model_name("gpt-4")
        validate_temperature(0.7)
        validate_max_tokens(1000)
        validate_top_p(0.9)
        validate_api_key("sk-1234567890")
        validate_endpoint("https://api.openai.com/v1")
        validate_message({"role": "user", "content": "Hello"})
        validate_messages([{"role": "user", "content": "Hello"}])
        print("✓ Valid inputs pass validation")
    except ProviderValidationError as e:
        print(f"✗ Valid inputs failed: [{e.code}] {e.message}")
        return False
    
    # Test invalid provider name
    try:
        validate_provider_name("")
        print("✗ Empty provider name should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV001":
            print(f"✓ Empty provider name rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test invalid temperature
    try:
        validate_temperature(3.0)
        print("✗ Temperature > 2 should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV006":
            print(f"✓ Invalid temperature rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test invalid max_tokens
    try:
        validate_max_tokens(-1)
        print("✗ Negative max_tokens should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV008":
            print(f"✓ Invalid max_tokens rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test invalid top_p
    try:
        validate_top_p(1.5)
        print("✗ top_p > 1 should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV010":
            print(f"✓ Invalid top_p rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test short API key
    try:
        validate_api_key("short")
        print("✗ Short API key should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV013":
            print(f"✓ Short API key rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test non-HTTPS endpoint
    try:
        validate_endpoint("http://insecure.com")
        print("✗ HTTP endpoint should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV015":
            print(f"✓ HTTP endpoint rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test invalid message role
    try:
        validate_message({"role": "invalid", "content": "Hello"})
        print("✗ Invalid role should fail")
        return False
    except ProviderValidationError as e:
        if e.code == "PROV020":
            print(f"✓ Invalid message role rejected: [{e.code}]")
        else:
            print(f"✗ Wrong error code: {e.code}")
            return False
    
    # Test validate_provider_config
    try:
        validate_provider_config(
            name="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        print("✓ Valid config passes validation")
    except ProviderValidationError as e:
        print(f"✗ Valid config failed: [{e.code}] {e.message}")
        return False
    
    return True


def test_error_types():
    """Test error types and formatting."""
    print("\nTesting error types...")
    
    from namel3ss.providers import (
        ProviderValidationError,
        ProviderAuthError,
        ProviderRateLimitError,
        ProviderAPIError,
        ProviderTimeoutError,
    )
    
    # Test ProviderValidationError
    try:
        error = ProviderValidationError(
            message="Invalid temperature",
            code="PROV006",
            provider="openai",
            field="temperature",
            value=3.0,
            expected="0 <= temperature <= 2",
        )
        formatted = error.format()
        assert "PROV006" in formatted
        assert "temperature" in formatted
        assert "3.0" in formatted
        print(f"✓ ProviderValidationError: {formatted}")
    except Exception as e:
        print(f"✗ ProviderValidationError failed: {e}")
        return False
    
    # Test ProviderAuthError
    try:
        error = ProviderAuthError(
            message="Invalid API key",
            code="PROV017",
            provider="openai",
            auth_type="api_key",
            status_code=401,
        )
        formatted = error.format()
        assert "PROV017" in formatted
        assert "api_key" in formatted
        assert "401" in formatted
        print(f"✓ ProviderAuthError: {formatted}")
    except Exception as e:
        print(f"✗ ProviderAuthError failed: {e}")
        return False
    
    # Test ProviderRateLimitError
    try:
        error = ProviderRateLimitError(
            message="Rate limit exceeded",
            code="PROV026",
            provider="openai",
            limit_type="requests_per_minute",
            retry_after=60,
        )
        formatted = error.format()
        assert "PROV026" in formatted
        assert "60" in formatted
        assert error.retry_after == 60
        print(f"✓ ProviderRateLimitError: {formatted}")
    except Exception as e:
        print(f"✗ ProviderRateLimitError failed: {e}")
        return False
    
    # Test ProviderAPIError
    try:
        error = ProviderAPIError(
            message="Service unavailable",
            code="PROV039",
            provider="openai",
            error_type="service_unavailable",
            status_code=503,
            retryable=True,
        )
        formatted = error.format()
        assert "PROV039" in formatted
        assert "503" in formatted
        assert error.retryable is True
        print(f"✓ ProviderAPIError: {formatted}")
    except Exception as e:
        print(f"✗ ProviderAPIError failed: {e}")
        return False
    
    # Test ProviderTimeoutError
    try:
        error = ProviderTimeoutError(
            message="Request timed out",
            code="PROV046",
            provider="openai",
            operation="generation",
            timeout_seconds=30.0,
        )
        formatted = error.format()
        assert "PROV046" in formatted
        assert "30.0" in formatted
        assert error.timeout_seconds == 30.0
        print(f"✓ ProviderTimeoutError: {formatted}")
    except Exception as e:
        print(f"✗ ProviderTimeoutError failed: {e}")
        return False
    
    return True


def test_backward_compatibility():
    """Test that existing imports still work."""
    print("\nTesting backward compatibility...")
    
    try:
        from namel3ss.providers import (
            N3Provider,
            ProviderResponse,
            ProviderMessage,
            ProviderError,
            create_provider_from_spec,
            load_openai_config,
            OpenAIProvider,
        )
        print("✓ Existing imports still work")
        return True
    except ImportError as e:
        print(f"✗ Backward compatibility broken: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Providers Refactoring Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Validation", test_validation_functions()))
    results.append(("Error Types", test_error_types()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
