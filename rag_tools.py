import asyncio
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure.core.exceptions import HttpResponseError
from azure.identity import AzureAuthorityHosts, DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_DIRECTORY = Path.cwd() / ".azure"
_ENV_PREFIX = "avcoe-*"
_DEFAULT_SELECT_FIELDS = ["title", "chunk"]


def _make_jsonable(value: Any) -> Any:
    """Convert Azure SDK values into JSON-serialisable primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(item) for item in value]
    return str(value)


def _escape_filter_value(value: str) -> str:
    """Escape single quotes for OData filter expressions."""

    return value.replace("'", "''")


@lru_cache(maxsize=1)
def _load_environment() -> Optional[Path]:
    """Load the first matching env file so credentials resolve via DefaultAzureCredential.

    Only loads from .env file when running locally. In deployed environments (Azure),
    environment variables should already be set via App Settings.
    """
    # Check if we're running in Azure (common Azure environment variables)
    if os.getenv("WEBSITE_INSTANCE_ID") or os.getenv("WEBSITE_SITE_NAME"):
        logger.info("Running in Azure - using existing environment variables")
        return None

    load_dotenv()
    logger.info("Loaded environment variables from .env for local development")
    return None


@lru_cache(maxsize=1)
def _get_search_client(index_name: Optional[str] = None) -> SearchClient:
    """Construct a SearchClient configured for the current cloud."""

    _load_environment()
    search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
    logger.info(f"Endpoint: {search_endpoint}")
    if not search_endpoint:
        raise RuntimeError("SEARCH_SERVICE_ENDPOINT is not defined in the environment")

    resolved_index = index_name or os.getenv("SEARCH_INDEX_NAME")
    if not resolved_index:
        raise RuntimeError("SEARCH_INDEX_NAME is not defined in the environment")

    if "azure.us" in search_endpoint:
        authority_host = AzureAuthorityHosts.AZURE_GOVERNMENT
        audience = "https://search.azure.us"
    else:
        authority_host = AzureAuthorityHosts.AZURE_PUBLIC_CLOUD
        audience = "https://search.azure.com"

    logger.info(
        "Creating SearchClient for endpoint %s, index %s, authority %s",
        search_endpoint,
        resolved_index,
        authority_host,
    )

    credential = DefaultAzureCredential(authority=authority_host)
    return SearchClient(
        endpoint=search_endpoint,
        index_name=resolved_index,
        credential=credential,
        audience=audience,
    )


async def list_facets(
    facet_name: str = "title",
    search_text: str = "*",
) -> Dict[str, Any]:
    """Retrieve faceted navigation values from the Azure AI Search index.

    This tool returns aggregated counts of distinct values for a specified field in the search index,
    commonly used to build filters or understand the distribution of values across documents.

    Args:
        facet_name: The name of the field to facet on (e.g., "title", "category", "author").
                   Must be a facetable field in your search index schema. Defaults to "title".
        search_text: Optional search query to scope the facet results. Use "*" (default) to
                    retrieve facets across all documents, or provide a query string to only
                    facet within matching documents.

    Returns:
        A dictionary containing:
        - facet: The name of the field that was faceted
        - search_text: The query used to scope results
        - values: List of facet value objects, each containing:
          - value: The distinct field value
          - count: Number of documents with this value

    Example use cases:
        - List all unique document titles: list_facets(facet_name="title")
        - Get category distribution: list_facets(facet_name="category")
        - Find authors in security-related docs: list_facets(facet_name="author", search_text="security")

    Raises:
        RuntimeError: If the search service is unreachable or the field is not facetable.
    """

    client = _get_search_client()

    def _run() -> List[Dict[str, Any]]:
        results = client.search(
            search_text,
            facets=[facet_name],
            top=0,
        )
        facets = results.get_facets().get(facet_name, [])
        return [_make_jsonable(facet) for facet in facets]

    try:
        values = await asyncio.to_thread(_run)
    except HttpResponseError as exc:
        logger.exception("Facet retrieval failed for %s", facet_name)
        raise RuntimeError(
            f"Failed to retrieve facets for '{facet_name}': {str(exc)}"
        ) from exc

    return {
        "facet": facet_name,
        "search_text": search_text,
        "values": values,
    }


async def semantic_search(
    query: str,
    top: int = 3,
    facet_value: Optional[str] = None,
    select_fields: Optional[List[str]] = None,
    query_type: str = "semantic",
) -> Dict[str, Any]:
    """Execute a search query against the Azure AI Search index with semantic ranking.

    This tool performs searches using Azure's semantic search capabilities, which understand natural
    language queries and rank results by semantic relevance rather than just keyword matching. It's
    ideal for question-answering scenarios, document retrieval, and RAG (Retrieval Augmented Generation).

    Args:
        query: Natural language search query (e.g., "what is an example of a group 2 UAS" or
              "security best practices for containers"). The semantic ranker will interpret
              the query's intent and return the most contextually relevant documents.

        top: Maximum number of documents to return (default: 3). Higher values provide more
            context but may include less relevant results. Typical range: 1-10 for RAG scenarios.

        facet_value: Optional filter to restrict search to documents with a specific title value.
                    Use this to scope searches within a particular document or category.
                    Example: "DoD Unmanned Systems Roadmap 2020" to search only within that document.
                    The value is automatically escaped for safe OData filter expressions.

        select_fields: List of field names to include in results (default: ["title", "chunk"]).
                      Reduces payload size and focuses on relevant fields. Common fields include:
                      "title", "chunk", "content", "metadata", "url", etc.

        query_type: Search algorithm to use. Options:
                   - "semantic" (default): Uses AI-powered semantic ranking for best relevance
                   - "simple": Basic keyword matching without semantic understanding

    Returns:
        A dictionary containing:
        - query: The original search query
        - query_type: The search algorithm used
        - top: Number of results requested
        - filter: The OData filter expression applied (if facet_value was specified)
        - select: List of fields included in results
        - total_count: Total number of matching documents (may exceed 'top')
        - documents: List of matching documents, each containing the requested fields
                    plus a @search.score indicating relevance (higher is better)

    Example use cases:
        - Answer questions: semantic_search("what are the safety requirements?")
        - Find specific info in a document: semantic_search("launch procedures", facet_value="Flight Manual")
        - Get diverse results: semantic_search("drone regulations", top=10)
        - Retrieve full documents: semantic_search("policy overview", select_fields=["title", "content", "metadata"])

    Best practices:
        - Use semantic search (default) for natural language queries and Q&A
        - Increase 'top' to 5-10 when you need comprehensive context
        - Apply facet_value filters when you know which document to search within
        - Request only necessary fields via select_fields to minimize latency
        - Check total_count to understand if you're seeing all relevant results

    Raises:
        ValueError: If top <= 0 or query_type is invalid
        RuntimeError: If the search request fails or the service is unreachable
    """

    if top <= 0:
        raise ValueError("top must be greater than zero")

    client = _get_search_client()
    select = select_fields or _DEFAULT_SELECT_FIELDS
    allowed_query_types = {"semantic", "simple"}
    if query_type not in allowed_query_types:
        raise ValueError(f"query_type must be one of {sorted(allowed_query_types)}")

    filter_expression: Optional[str] = None
    if facet_value:
        filter_expression = f"title eq '{_escape_filter_value(facet_value)}'"

    def _run() -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {
            "include_total_count": True,
            "top": top,
            "select": select,
            "query_type": query_type,
        }
        if filter_expression:
            search_kwargs["filter"] = filter_expression

        results = client.search(query, **search_kwargs)
        documents = [_make_jsonable(dict(hit)) for hit in results]
        total = results.get_count()
        return {
            "query": query,
            "query_type": query_type,
            "top": top,
            "filter": filter_expression,
            "select": select,
            "total_count": total,
            "documents": documents,
        }

    try:
        payload = await asyncio.to_thread(_run)
    except HttpResponseError as exc:
        logger.exception("Search request failed for query '%s'", query)
        raise RuntimeError(f"Search request failed: {str(exc)}") from exc

    return payload
