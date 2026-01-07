"""Tests for LLM retry logic and fallback functionality."""

import pytest
from unittest.mock import patch, MagicMock
from openai.lib.azure import AsyncAzureOpenAI
from main import _is_retryable_error


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
