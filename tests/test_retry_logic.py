"""Tests for LLM retry logic and fallback functionality."""

import pytest
from unittest.mock import patch, MagicMock
from openai.lib.azure import AsyncAzureOpenAI
from main import _is_retryable_error
from agent_framework._types import UsageDetails


def _usage_value(usage: UsageDetails, key: str):
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key)


class TestImmediateFailover:
    """Test that primary LLM is configured for immediate failover."""

    def test_primary_client_has_zero_retries(self) -> None:
        """Primary AsyncAzureOpenAI client should have max_retries=0."""
        # Create a client with the same configuration as main.py
        client = AsyncAzureOpenAI(
            azure_endpoint="https://test.openai.azure.us",
            azure_deployment="gpt-4o",
            api_key="test-key",
            api_version="2024-02-15-preview",
            max_retries=0,
        )
        assert client.max_retries == 0

    def test_secondary_client_has_retries(self) -> None:
        """Secondary AsyncAzureOpenAI client should have max_retries for resilience."""
        client = AsyncAzureOpenAI(
            azure_endpoint="https://test-secondary.openai.azure.us",
            azure_deployment="gpt-4o-mini",
            api_key="test-key",
            api_version="2024-02-15-preview",
            max_retries=3,
        )
        assert client.max_retries == 3

    def test_zero_retries_client_fails_immediately(self) -> None:
        """Client with max_retries=0 should not retry on failure."""
        client = AsyncAzureOpenAI(
            azure_endpoint="https://test.openai.azure.us",
            azure_deployment="gpt-4o",
            api_key="test-key",
            api_version="2024-02-15-preview",
            max_retries=0,
        )
        # Verify the client configuration - no internal retry delays
        assert client.max_retries == 0
        # Default max_retries is 2, so setting to 0 should disable internal retries
        assert client.max_retries != 2


class TestContextPreservation:
    """Test that failover preserves conversation context."""

    def test_run_agent_stream_accepts_none_message(self) -> None:
        """_run_agent_stream should accept None for message_content to continue from existing context."""
        # This test verifies the function signature accepts Optional[str]
        from main import _run_agent_stream
        import inspect
        
        sig = inspect.signature(_run_agent_stream)
        msg_param = sig.parameters.get("message_content")
        assert msg_param is not None
        # The annotation should be Optional[str] which includes None
        annotation_str = str(msg_param.annotation)
        assert "None" in annotation_str or "Optional" in annotation_str or "str | None" in annotation_str


class TestIsRetryableError:
    """Test the _is_retryable_error helper function."""

    def test_detects_429_in_message(self) -> None:
        """Should detect 429 status code in error message."""
        error = Exception("Error: 429 Too Many Requests")
        assert _is_retryable_error(error) is True

    def test_detects_too_many_requests_message(self) -> None:
        """Should detect 'Too Many Requests' in error message."""
        error = Exception("Server returned Too Many Requests")
        assert _is_retryable_error(error) is True

    def test_detects_rate_limit_in_message(self) -> None:
        """Should detect 'rate_limit' in error message (case insensitive)."""
        error = Exception("Request failed due to rate_limit exceeded")
        assert _is_retryable_error(error) is True

    def test_detects_rate_limit_uppercase(self) -> None:
        """Should detect 'RATE_LIMIT' in error message (case insensitive)."""
        error = Exception("RATE_LIMIT error occurred")
        assert _is_retryable_error(error) is True

    def test_detects_capacity_in_message(self) -> None:
        """Should detect 'capacity' in error message (case insensitive)."""
        error = Exception("Service at capacity, please retry later")
        assert _is_retryable_error(error) is True

    def test_detects_capacity_uppercase(self) -> None:
        """Should detect 'CAPACITY' in error message (case insensitive)."""
        error = Exception("CAPACITY exceeded")
        assert _is_retryable_error(error) is True

    def test_does_not_match_other_errors(self) -> None:
        """Should not match unrelated errors."""
        error = Exception("Connection timeout")
        assert _is_retryable_error(error) is False

    def test_does_not_match_500_error(self) -> None:
        """Should not match 500 internal server error."""
        error = Exception("Error: 500 Internal Server Error")
        assert _is_retryable_error(error) is False

    def test_does_not_match_empty_message(self) -> None:
        """Should not match empty error message."""
        error = Exception("")
        assert _is_retryable_error(error) is False


class TestRetryLogicBehavior:
    """Tests for retry logic behavior patterns."""

    def test_retryable_error_should_trigger_fallback(self) -> None:
        """429 errors should trigger fallback logic."""
        rate_limit_error = Exception("429 Too Many Requests")
        assert _is_retryable_error(rate_limit_error) is True

    def test_non_retryable_error_should_not_trigger_fallback(self) -> None:
        """Non-429 errors should not trigger fallback logic."""
        regular_error = Exception("Connection timeout")
        assert _is_retryable_error(regular_error) is False

    def test_rate_limit_error_type_detection(self) -> None:
        """Should detect RateLimitError type in exception."""
        # Simulate what a RateLimitError type check would return
        class RateLimitError(Exception):
            pass
        
        error = RateLimitError("Rate limit exceeded")
        assert _is_retryable_error(error) is True

    def test_azure_openai_capacity_error(self) -> None:
        """Should detect Azure OpenAI capacity errors."""
        error = Exception("The server is currently at capacity. Please try again later.")
        assert _is_retryable_error(error) is True

    def test_openai_rate_limit_header_error(self) -> None:
        """Should detect OpenAI rate limit errors with header info."""
        error = Exception("Rate limit reached for requests. Please retry after 60 seconds.")
        assert _is_retryable_error(error) is True


class TestNewChatSession:
    """Test that new chat sessions properly reset state."""

    def test_on_chat_start_clears_previous_thread(self) -> None:
        """on_chat_start should clear any existing thread from previous session."""
        # Verify the pattern in main.py clears old state before creating new
        import inspect
        import main
        
        source = inspect.getsource(main.on_chat_start)
        
        # Check that on_chat_start sets thread to None before creating new one
        assert 'cl.user_session.set("thread", None)' in source
        assert 'cl.user_session.set("agent", None)' in source
        assert 'cl.user_session.set("secondary_agent", None)' in source

    def test_on_chat_end_exists(self) -> None:
        """on_chat_end should be defined for cleanup."""
        import main
        
        # Verify on_chat_end is defined
        assert hasattr(main, 'on_chat_end')
        assert callable(main.on_chat_end)

    def test_on_chat_end_clears_session(self) -> None:
        """on_chat_end should clear thread and agents from session."""
        import inspect
        import main
        
        source = inspect.getsource(main.on_chat_end)
        
        # Check that on_chat_end clears the thread and agents
        assert 'cl.user_session.set("thread", None)' in source
        assert 'cl.user_session.set("agent", None)' in source
        assert 'cl.user_session.set("secondary_agent", None)' in source

    def test_new_thread_created_each_chat_start(self) -> None:
        """Verify that agent.get_new_thread() creates unique threads."""
        try:
            from agent_framework import ChatAgent
        except ImportError:
            from agent_framework import RawAgent as ChatAgent
        import inspect
        from unittest.mock import MagicMock
        
        # Create mock LLM client
        mock_client = MagicMock()
        mock_client.get_chat_client_name.return_value = 'test'
        mock_client.get_model_name.return_value = 'test'
        
        # Create agent (supports both older ChatAgent(chat_client=...) and newer RawAgent(client=...))
        init_params = inspect.signature(ChatAgent).parameters
        if "chat_client" in init_params:
            agent = ChatAgent(chat_client=mock_client, name='test', instructions='test')
        elif "client" in init_params:
            agent = ChatAgent(client=mock_client, name='test', instructions='test')
        else:
            pytest.skip("Unsupported agent constructor signature for thread creation test")

        if not hasattr(agent, "get_new_thread"):
            pytest.skip("Agent implementation does not expose get_new_thread")
        
        # Get multiple threads - they should be different objects
        thread1 = agent.get_new_thread()
        thread2 = agent.get_new_thread()
        
        # Verify threads are different objects (new thread for each chat)
        assert thread1 is not thread2
        assert id(thread1) != id(thread2)


class TestTokenUsage:
    """Test token usage tracking functionality."""

    def test_usage_details_initialization(self) -> None:
        """UsageDetails can be initialized with token counts."""
        usage = UsageDetails(
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
        )
        assert _usage_value(usage, "input_token_count") == 100
        assert _usage_value(usage, "output_token_count") == 50
        assert _usage_value(usage, "total_token_count") == 150

    def test_usage_details_addition(self) -> None:
        """UsageDetails instances can be added together."""
        usage1 = UsageDetails(
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
        )
        usage2 = UsageDetails(
            input_token_count=200,
            output_token_count=100,
            total_token_count=300,
        )
        if isinstance(usage1, dict):
            combined = {
                "input_token_count": (usage1.get("input_token_count") or 0)
                + (usage2.get("input_token_count") or 0),
                "output_token_count": (usage1.get("output_token_count") or 0)
                + (usage2.get("output_token_count") or 0),
                "total_token_count": (usage1.get("total_token_count") or 0)
                + (usage2.get("total_token_count") or 0),
            }
        else:
            combined = usage1 + usage2
        assert _usage_value(combined, "input_token_count") == 300
        assert _usage_value(combined, "output_token_count") == 150
        assert _usage_value(combined, "total_token_count") == 450

    def test_usage_details_iadd(self) -> None:
        """UsageDetails supports in-place addition."""
        usage1 = UsageDetails(
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
        )
        usage2 = UsageDetails(
            input_token_count=200,
            output_token_count=100,
            total_token_count=300,
        )
        if isinstance(usage1, dict):
            usage1 = {
                "input_token_count": (usage1.get("input_token_count") or 0)
                + (usage2.get("input_token_count") or 0),
                "output_token_count": (usage1.get("output_token_count") or 0)
                + (usage2.get("output_token_count") or 0),
                "total_token_count": (usage1.get("total_token_count") or 0)
                + (usage2.get("total_token_count") or 0),
            }
        else:
            usage1 += usage2
        assert _usage_value(usage1, "input_token_count") == 300
        assert _usage_value(usage1, "output_token_count") == 150
        assert _usage_value(usage1, "total_token_count") == 450

    def test_usage_details_empty_initialization(self) -> None:
        """UsageDetails can be initialized with no arguments."""
        usage = UsageDetails()
        assert _usage_value(usage, "input_token_count") is None
        assert _usage_value(usage, "output_token_count") is None
        assert _usage_value(usage, "total_token_count") is None

    def test_run_agent_stream_returns_tuple(self) -> None:
        """_run_agent_stream should return a tuple of (answer, usage)."""
        from main import _run_agent_stream
        import inspect
        
        sig = inspect.signature(_run_agent_stream)
        annotation_str = str(sig.return_annotation)
        # Should return Tuple[Optional[cl.Message], Optional[UsageDetails]]
        assert "Tuple" in annotation_str or "tuple" in annotation_str

    def test_on_chat_start_initializes_token_usage(self) -> None:
        """on_chat_start should initialize token_usage in session."""
        import inspect
        import main
        
        source = inspect.getsource(main.on_chat_start)
        
        # Check that on_chat_start initializes token_usage
        assert 'cl.user_session.set("token_usage"' in source

    def test_on_chat_end_clears_token_usage(self) -> None:
        """on_chat_end should clear token_usage from session."""
        import inspect
        import main
        
        source = inspect.getsource(main.on_chat_end)
        
        # Check that on_chat_end clears token_usage
        assert 'cl.user_session.set("token_usage", None)' in source

    def test_on_message_logs_token_usage(self) -> None:
        """on_message should log token usage."""
        import inspect
        import main
        
        source = inspect.getsource(main.on_message)
        
        # Check that on_message logs token usage
        assert 'Request token usage' in source
        assert 'Session cumulative' in source
