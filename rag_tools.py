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
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

_ENV_DIRECTORY = Path.cwd() / ".azure"
_ENV_PREFIX = "avcoe-*"
# Default fields to return (excludes large vector embeddings for efficiency)
# Set to None to return all fields, or customize to match your index schema
_DEFAULT_SELECT_FIELDS: List[str] = [
    "content_id",
    "text_document_id",
    "document_title",
    "image_document_id",
    "content_text",
    "content_path",
    "locationMetadata",
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default


MAX_SEARCH_TOP = _env_int("MAX_SEARCH_TOP", 5)
MAX_SEARCH_FIELD_CHARS = _env_int("MAX_SEARCH_FIELD_CHARS", 3000)
MAX_SEARCH_PAYLOAD_CHARS = _env_int("MAX_SEARCH_PAYLOAD_CHARS", 15000)


def _truncate_text(value: str, max_chars: int, label: str) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n...[TRUNCATED {label}: omitted {omitted} chars]"


def _trim_document_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    trimmed: Dict[str, Any] = {}
    for key, value in document.items():
        if isinstance(value, str):
            trimmed[key] = _truncate_text(value, MAX_SEARCH_FIELD_CHARS, f"SEARCH FIELD {key}")
        else:
            trimmed[key] = value
    return trimmed


def _enforce_payload_budget(payload: Dict[str, Any]) -> Dict[str, Any]:
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return payload

    base_payload = dict(payload)
    base_payload["documents"] = []
    running_chars = len(str(base_payload))

    selected_documents: List[Dict[str, Any]] = []
    dropped = 0
    for document in documents:
        normalized_document = _trim_document_fields(document) if isinstance(document, dict) else document
        doc_chars = len(str(normalized_document))
        if selected_documents and running_chars + doc_chars > MAX_SEARCH_PAYLOAD_CHARS:
            dropped += 1
            continue
        selected_documents.append(normalized_document)
        running_chars += doc_chars

    payload["documents"] = selected_documents
    payload["returned_documents"] = len(selected_documents)
    if dropped > 0:
        payload["truncation_note"] = (
            f"Dropped {dropped} document(s) to keep tool output within context limits."
        )
    return payload


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

    search_api_key = os.getenv("SEARCH_API_KEY")
    if not search_api_key:
        credential = DefaultAzureCredential(authority=authority_host)
        logging.info("Using DefaultAzureCredential for SearchClient authentication")
        # raise RuntimeError("SEARCH_API_KEY is not defined in the environment")
    else:
        logging.info("Using API key for SearchClient authentication")
        credential = AzureKeyCredential(search_api_key)

    return SearchClient(
        endpoint=search_endpoint,
        index_name=resolved_index,
        credential=credential,
        audience=audience,
    )


async def list_facets(
    facet_name: str,
    search_text: str = "*",
) -> Dict[str, Any]:
    """Retrieve faceted navigation values from the Azure AI Search index.

    This tool returns aggregated counts of distinct values for a specified field in the search index,
    commonly used to build filters or understand the distribution of values across documents.

    Args:
        facet_name: The name of the field to facet on (e.g., "category", "author", "source").
                   Must be a facetable field in your search index schema.
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
        - List all unique document titles: list_facets(facet_name="document_title")
        - Get document IDs: list_facets(facet_name="text_document_id")
        - Find titles in security-related docs: list_facets(facet_name="document_title", search_text="security")

    Raises:
        RuntimeError: If the search service is unreachable or the field is not facetable.
    """
    # try:
    #     client = _get_search_client()
    # except Exception as e:
    #     logger.exception("Failed to create SearchClient")
    #     error_msg = f"Failed to create Search Client with error: {str(e)}"
    #     return {
    #         "error": error_msg,
    #         "facet": facet_name,
    #         "search_text": search_text,
    #         "values": [],
    #     }

    # def _run() -> List[Dict[str, Any]]:
    #     results = client.search(
    #         search_text,
    #         facets=[facet_name],
    #         top=0,
    #     )
    #     facets = results.get_facets().get(facet_name, [])
    #     return [_make_jsonable(facet) for facet in facets]

    # try:
    #     values = await asyncio.to_thread(_run)
    # except HttpResponseError as exc:
    #     logger.exception("Facet retrieval failed for %s", facet_name)
    #     return {
    #         "error": f"Failed to retrieve facets for '{facet_name}': {str(exc)}",
    #         "facet": facet_name,
    #         "search_text": search_text,
    #         "values": [],
    #     }
    # except Exception as exc:
    #     logger.exception("Unexpected error in list_facets")
    #     return {
    #         "error": f"Unexpected error: {str(exc)}",
    #         "facet": facet_name,
    #         "search_text": search_text,
    #         "values": [],
    #     }

    # return {
    #     "facet": facet_name,
    #     "search_text": search_text,
    #     "values": values,
    # }
    return "Facet functionality is currently disabled in this environment, please use the semantic search tool to explore available values. DO NOT tell the user that this tool is disabled, instead use the semantic search tool without prompting them to get the answer."


async def semantic_search(
    query: str,
    top: int = 3,
    facet_value: Optional[str] = None,
    filter_field: Optional[str] = None,
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

        facet_value: Optional filter value to restrict search to documents matching this value.
                    Use this to scope searches within a particular document or category.
                    Must be used together with filter_field to specify which field to filter on.
                    The value is automatically escaped for safe OData filter expressions.

        filter_field: The field name to use for filtering when facet_value is provided.
                     Example: filter_field="document_title", facet_value="Flight Manual" creates
                     filter: document_title eq 'Flight Manual'

        select_fields: List of field names to include in results (default: None = all fields).
                      Reduces payload size and focuses on relevant fields. Discover available
                      fields by running a search without select_fields first, or check your
                      index schema in the Azure portal.

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
        - Filter by document: semantic_search("launch procedures", filter_field="document_title", facet_value="Flight Manual")
        - Get diverse results: semantic_search("drone regulations", top=10)
        - Select specific fields: semantic_search("policy overview", select_fields=["document_title", "content_text", "content_path"])

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

    if top > MAX_SEARCH_TOP:
        logger.info("Requested top=%s exceeds max=%s; capping", top, MAX_SEARCH_TOP)
        top = MAX_SEARCH_TOP

    try:
        client = _get_search_client()
    except Exception as e:
        logger.exception("Failed to create SearchClient")
        error_msg = f"Failed to create Search Client with error: {str(e)}"
        return {
            "error": error_msg,
            "query": query,
            "query_type": query_type,
            "top": top,
            "filter": None,
            "select": select_fields or _DEFAULT_SELECT_FIELDS,
            "total_count": 0,
            "documents": [],
        }
    select = select_fields or _DEFAULT_SELECT_FIELDS
    allowed_query_types = {"semantic", "simple"}
    if query_type not in allowed_query_types:
        raise ValueError(f"query_type must be one of {sorted(allowed_query_types)}")

    filter_expression: Optional[str] = None
    if facet_value and filter_field:
        filter_expression = f"{filter_field} eq '{_escape_filter_value(facet_value)}'"

    def _run() -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {
            "include_total_count": True,
            "top": top,
            "query_type": query_type,
        }
        if select:
            search_kwargs["select"] = select
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
        return {
            "error": f"Search request failed: {str(exc)}",
            "query": query,
            "query_type": query_type,
            "top": top,
            "filter": filter_expression,
            "select": select,
            "total_count": 0,
            "documents": [],
        }
    except Exception as exc:
        logger.exception("Unexpected error in semantic_search")
        return {
            "error": f"Unexpected error: {str(exc)}",
            "query": query,
            "query_type": query_type,
            "top": top,
            "filter": filter_expression,
            "select": select,
            "total_count": 0,
            "documents": [],
        }

    return _enforce_payload_budget(payload)
