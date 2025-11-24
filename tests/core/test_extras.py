"""
Tests for optional extras functionality.

These tests verify that when optional extras are installed, the corresponding
features work correctly. These tests will be skipped if the required extras
are not installed, so they should pass in both minimal and full installations.
"""

import pytest

from namel3ss.features import (
    has_openai,
    has_anthropic,
    has_redis,
    has_sqlalchemy,
    has_mongo,
    has_opentelemetry,
    has_websockets,
)


# Markers for conditional test execution
requires_openai = pytest.mark.skipif(
    not has_openai(),
    reason="Requires namel3ss[openai] or namel3ss[ai]"
)
requires_anthropic = pytest.mark.skipif(
    not has_anthropic(),
    reason="Requires namel3ss[anthropic] or namel3ss[ai]"
)
requires_redis = pytest.mark.skipif(
    not has_redis(),
    reason="Requires namel3ss[redis]"
)
requires_sql = pytest.mark.skipif(
    not has_sqlalchemy(),
    reason="Requires namel3ss[sql], namel3ss[postgres], or namel3ss[mysql]"
)
requires_mongo = pytest.mark.skipif(
    not has_mongo(),
    reason="Requires namel3ss[mongo]"
)
requires_otel = pytest.mark.skipif(
    not has_opentelemetry(),
    reason="Requires namel3ss[otel]"
)
requires_websockets = pytest.mark.skipif(
    not has_websockets(),
    reason="Requires namel3ss[websockets] or namel3ss[realtime]"
)


class TestAIExtras:
    """Test AI/LLM provider extras."""
    
    @requires_openai
    def test_openai_imports(self):
        """Test that OpenAI SDK can be imported when extra installed."""
        from openai import OpenAI
        assert OpenAI is not None
    
    @requires_openai
    def test_openai_adapter_setup(self):
        """Test ModelAdapter can set up OpenAI client."""
        from namel3ss.adapters.model import ModelAdapter, ModelAdapterConfig
        
        config = ModelAdapterConfig(
            name="test_openai",
            provider="openai",
            model="gpt-4",
            api_key="sk-test",  # Fake key for testing
        )
        
        adapter = ModelAdapter(config)
        # Should not raise error with openai installed
        adapter._setup_openai()
        assert adapter._client is not None
    
    @requires_anthropic
    def test_anthropic_imports(self):
        """Test that Anthropic SDK can be imported when extra installed."""
        from anthropic import Anthropic
        assert Anthropic is not None
    
    @requires_anthropic
    def test_anthropic_adapter_setup(self):
        """Test ModelAdapter can set up Anthropic client."""
        from namel3ss.adapters.model import ModelAdapter, ModelAdapterConfig
        
        config = ModelAdapterConfig(
            name="test_anthropic",
            provider="anthropic",
            model="claude-3-sonnet",
            api_key="sk-ant-test",  # Fake key for testing
        )
        
        adapter = ModelAdapter(config)
        # Should not raise error with anthropic installed
        adapter._setup_anthropic()
        assert adapter._client is not None
    
    @pytest.mark.skipif(
        not (has_openai() and has_anthropic()),
        reason="Requires both OpenAI and Anthropic (namel3ss[ai])"
    )
    def test_full_ai_extra_installed(self):
        """Test that full ai extra includes both OpenAI and Anthropic."""
        from openai import OpenAI
        from anthropic import Anthropic
        
        assert OpenAI is not None
        assert Anthropic is not None


class TestDatabaseExtras:
    """Test database extras."""
    
    @requires_sql
    def test_sqlalchemy_imports(self):
        """Test SQLAlchemy can be imported when SQL extra installed."""
        import sqlalchemy
        from sqlalchemy.ext.asyncio import AsyncSession
        
        assert sqlalchemy is not None
        assert AsyncSession is not None
    
    @requires_sql
    def test_postgres_driver_imports(self):
        """Test PostgreSQL drivers can be imported."""
        # asyncpg should be available with postgres or sql extra
        try:
            import asyncpg
            assert asyncpg is not None
        except ImportError:
            pytest.skip("asyncpg not installed")
        
        # psycopg should be available with postgres or sql extra
        try:
            import psycopg
            assert psycopg is not None
        except ImportError:
            pytest.skip("psycopg not installed")
    
    @requires_mongo
    def test_mongo_imports(self):
        """Test MongoDB drivers can be imported when mongo extra installed."""
        import motor
        import pymongo
        
        assert motor is not None
        assert pymongo is not None


class TestCachingExtras:
    """Test caching/queue extras."""
    
    @requires_redis
    def test_redis_imports(self):
        """Test Redis client can be imported when redis extra installed."""
        import redis
        from redis import Redis
        
        assert redis is not None
        assert Redis is not None
    
    @requires_redis
    def test_redis_queue_adapter(self):
        """Test QueueAdapter works with Redis installed."""
        from namel3ss.adapters.queue import QueueAdapter, QueueAdapterConfig, QueueBackend
        
        config = QueueAdapterConfig(
            name="test_queue",
            backend=QueueBackend.RQ,
            broker_url="redis://localhost:6379/0",
            queue_name="test",
            task_name="test.task",
        )
        
        # Should not raise error with redis installed
        # (actual connection will fail, but adapter should initialize)
        try:
            adapter = QueueAdapter(config)
            assert adapter is not None
        except Exception as e:
            # Connection errors are OK, but not import errors
            assert "redis" not in str(e).lower() or "connection" in str(e).lower()


class TestRealtimeExtras:
    """Test real-time communication extras."""
    
    @requires_websockets
    def test_websockets_imports(self):
        """Test websockets library can be imported when extra installed."""
        import websockets
        assert websockets is not None


class TestObservabilityExtras:
    """Test observability extras."""
    
    @requires_otel
    def test_opentelemetry_imports(self):
        """Test OpenTelemetry can be imported when otel extra installed."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        
        assert trace is not None
        assert TracerProvider is not None
        assert FastAPIInstrumentor is not None
    
    @requires_otel
    def test_otel_tracing_setup(self):
        """Test that OpenTelemetry tracing can be configured."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        
        tracer = trace.get_tracer("test")
        assert tracer is not None
        
        with tracer.start_as_current_span("test_span") as span:
            assert span is not None


class TestExtrasCombinations:
    """Test combinations of extras work together."""
    
    @pytest.mark.skipif(
        not (has_openai() and has_sqlalchemy()),
        reason="Requires namel3ss[ai] and namel3ss[sql]"
    )
    def test_ai_plus_sql(self):
        """Test AI and SQL extras work together."""
        from openai import OpenAI
        import sqlalchemy
        
        assert OpenAI is not None
        assert sqlalchemy is not None
    
    @pytest.mark.skipif(
        not (has_redis() and has_websockets()),
        reason="Requires namel3ss[realtime] or both redis and websockets"
    )
    def test_realtime_extra(self):
        """Test realtime extra includes both Redis and WebSockets."""
        import redis
        import websockets
        
        assert redis is not None
        assert websockets is not None
    
    @pytest.mark.skipif(
        not (has_openai() and has_anthropic() and has_sqlalchemy() and 
             has_redis() and has_opentelemetry()),
        reason="Requires namel3ss[all]"
    )
    def test_all_extras_installed(self):
        """Test that 'all' extra includes all optional dependencies."""
        # AI
        from openai import OpenAI
        from anthropic import Anthropic
        
        # Databases
        import sqlalchemy
        import motor
        
        # Caching
        import redis
        
        # Real-time
        import websockets
        
        # Observability
        from opentelemetry import trace
        
        assert all([
            OpenAI is not None,
            Anthropic is not None,
            sqlalchemy is not None,
            motor is not None,
            redis is not None,
            websockets is not None,
            trace is not None,
        ])


class TestFeatureDetectionAccuracy:
    """Test that feature detection accurately reflects installation state."""
    
    def test_has_functions_return_bool(self):
        """Test all has_* functions return boolean values."""
        from namel3ss.features import get_available_features
        
        features = get_available_features()
        
        for feature_name, is_available in features.items():
            assert isinstance(is_available, bool), \
                f"{feature_name} detection should return bool, got {type(is_available)}"
    
    def test_feature_consistency(self):
        """Test that feature detection is consistent with actual imports."""
        from namel3ss.features import has_openai, has_redis, has_sqlalchemy
        
        # If has_openai returns True, import should work
        if has_openai():
            try:
                import openai
                assert openai is not None
            except ImportError:
                pytest.fail("has_openai() returned True but import failed")
        
        # If has_redis returns True, import should work
        if has_redis():
            try:
                import redis
                assert redis is not None
            except ImportError:
                pytest.fail("has_redis() returned True but import failed")
        
        # If has_sqlalchemy returns True, import should work
        if has_sqlalchemy():
            try:
                import sqlalchemy
                assert sqlalchemy is not None
            except ImportError:
                pytest.fail("has_sqlalchemy() returned True but import failed")


def test_print_feature_status():
    """Test print_feature_status function works."""
    from namel3ss.features import print_feature_status
    
    # Should not raise exception
    import io
    import sys
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        print_feature_status()
        output = sys.stdout.getvalue()
        
        # Should contain some expected text
        assert "Optional Features" in output or "Features" in output
        assert len(output) > 0
    finally:
        sys.stdout = old_stdout
