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
MAX_FACET_SAMPLE_DOCS = _env_int("MAX_FACET_SAMPLE_DOCS", 100)
DEFAULT_SEARCH_FILTERABLE_FIELDS = {"text_document_id", "image_document_id"}


def _env_csv_set(name: str, default: Optional[set[str]] = None) -> set[str]:
    raw = os.getenv(name, "")
    if not raw:
        return {item.lower() for item in (default or set())}
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


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


def _document_matches_field_value(document: Dict[str, Any], field: str, value: str) -> bool:
    """Return True when a document field matches the requested filter value.

    Supports scalar string values and list fields.
    """

    if field not in document:
        return False

    candidate = document.get(field)
    if candidate is None:
        return False

    target = str(value).strip()
    if isinstance(candidate, list):
        return any(str(item).strip() == target for item in candidate)
    return str(candidate).strip() == target


def _is_non_filterable_error(error_text: str) -> bool:
    lower = error_text.lower()
    return (
        "not filterable" in lower
        or "not a filterable" in lower
        or "isn't filterable" in lower
        or "filterable field" in lower
    )


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

        Army maintenance-document usage guidance:
                - Use this tool for technical publication content (TM, LO, TB, SB, MWO, SC) and maintenance procedures.
                - If a user provides a publication identifier (example: TM 9-2320-280-10), search using the exact number
                    and then broaden with related terms only if results are empty.
                - Interpret TM suffixes for maintenance scope when explaining findings:
                    - -10 operator level
                    - -20/-30 field level (combined category)
                    - -40/-50 sustainment level
                    - &P parts manual, -HR hand receipt
                - Prefer answers grounded in retrieved excerpts; if no relevant document is returned, explicitly state
                    that no supporting publication text was found.
                - For procedural questions, prioritize warnings/cautions, initial setup conditions, and follow-on tasks
                    when present in retrieved content.

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
    configured_filterable_fields = _env_csv_set(
        "SEARCH_FILTERABLE_FIELDS",
        default=DEFAULT_SEARCH_FILTERABLE_FIELDS,
    )
    preemptive_client_side_filter = False
    if facet_value and filter_field:
        normalized_filter_field = filter_field.lower()
        if configured_filterable_fields and normalized_filter_field not in configured_filterable_fields:
            preemptive_client_side_filter = True
            logger.info(
                "Field '%s' not in SEARCH_FILTERABLE_FIELDS; using client-side sampled filter",
                filter_field,
            )
        else:
            filter_expression = f"{filter_field} eq '{_escape_filter_value(facet_value)}'"

    requested_filter_expression = filter_expression

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

    def _run_client_side_filter_fallback() -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {
            "include_total_count": True,
            "top": max(top, MAX_FACET_SAMPLE_DOCS),
            "query_type": query_type,
        }
        if select:
            search_kwargs["select"] = select

        results = client.search(query, **search_kwargs)
        documents = [_make_jsonable(dict(hit)) for hit in results]
        matched_documents = [
            document
            for document in documents
            if filter_field and facet_value and _document_matches_field_value(document, filter_field, facet_value)
        ]

        return {
            "query": query,
            "query_type": query_type,
            "top": top,
            "filter": None,
            "requested_filter": (
                requested_filter_expression
                or (f"{filter_field} eq '{_escape_filter_value(facet_value)}'" if filter_field and facet_value else None)
            ),
            "filter_mode": "client_side_sampled",
            "filter_note": (
                f"Field '{filter_field}' is not filterable in the index schema. "
                "Filter applied client-side on sampled search results, so matches may be incomplete."
            ),
            "select": select,
            "total_count": len(matched_documents),
            "sampled_documents": len(documents),
            "documents": matched_documents[:top],
        }

    def _run_unfiltered_basic_fallback(reason: str) -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {
            "include_total_count": True,
            "top": top,
            "query_type": "simple",
        }
        if select:
            search_kwargs["select"] = select

        results = client.search(query, **search_kwargs)
        documents = [_make_jsonable(dict(hit)) for hit in results]
        total = results.get_count()
        return {
            "query": query,
            "query_type": "simple",
            "top": top,
            "filter": None,
            "select": select,
            "total_count": total,
            "documents": documents,
            "fallback_executed": "simple_unfiltered",
            "fallback_reason": reason,
            "model_guidance": (
                "Semantic search returned no results. Use this unfiltered basic search result set; "
                "if still empty, answer that no matching content was found and ask the user to broaden the query."
            ),
        }

    if preemptive_client_side_filter:
        try:
            payload = await asyncio.to_thread(_run_client_side_filter_fallback)
            return _enforce_payload_budget(payload)
        except Exception as fallback_exc:
            logger.exception("Preemptive client-side filter fallback failed")
            return {
                "error": f"Client-side filter fallback failed: {str(fallback_exc)}",
                "query": query,
                "query_type": query_type,
                "top": top,
                "filter": None,
                "select": select,
                "total_count": 0,
                "documents": [],
            }

    try:
        payload = await asyncio.to_thread(_run)
    except HttpResponseError as exc:
        error_text = str(exc)
        if requested_filter_expression and _is_non_filterable_error(error_text):
            logger.info(
                "Field '%s' is not filterable in index schema; using client-side sampled filter fallback",
                filter_field,
            )

            try:
                payload = await asyncio.to_thread(_run_client_side_filter_fallback)
                return _enforce_payload_budget(payload)
            except Exception as fallback_exc:
                logger.exception("Client-side filter fallback failed")
                return {
                    "error": f"Search request failed and fallback could not complete: {str(fallback_exc)}",
                    "query": query,
                    "query_type": query_type,
                    "top": top,
                    "filter": requested_filter_expression,
                    "select": select,
                    "total_count": 0,
                    "documents": [],
                }

        logger.exception("Search request failed for query '%s'", query)
        return {
            "error": f"Search request failed: {error_text}",
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

    has_no_documents = not payload.get("documents")
    if query_type == "semantic" and has_no_documents:
        try:
            fallback_payload = await asyncio.to_thread(
                _run_unfiltered_basic_fallback,
                "semantic_search_returned_no_documents",
            )
            return _enforce_payload_budget(fallback_payload)
        except Exception as fallback_exc:
            logger.exception("Unfiltered basic fallback failed after empty semantic results")
            payload["model_guidance"] = (
                "Semantic search returned no results. Retry with query_type='simple' and no filters. "
                f"Automatic fallback failed: {str(fallback_exc)}"
            )

    return _enforce_payload_budget(payload)
