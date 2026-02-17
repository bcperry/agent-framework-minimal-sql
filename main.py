import os
import json
import chainlit as cl
import logging
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from openai.lib.azure import AsyncAzureOpenAI
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ChatAgent, AgentThread
from agent_framework._types import ChatMessage, UsageContent, UsageDetails
from tools import SqlDatabase
from rag_tools import semantic_search, list_facets
from custom_oauth import AzureGovOAuthProvider, AzureGovHybridOAuthProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AI_SEARCH_BOT_PROFILE = "Technical Maintenance AI"
SQL_BOT_PROFILE = "Tactical Readiness AI"

DEFAULT_MAX_THREAD_MESSAGES = int(os.getenv("MAX_THREAD_MESSAGES", "18"))
DEFAULT_MAX_THREAD_CHARS = int(os.getenv("MAX_THREAD_CHARS", "48000"))
DEFAULT_MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "12000"))
DEFAULT_MAX_USER_INPUT_CHARS = int(os.getenv("MAX_USER_INPUT_CHARS", "8000"))
MIN_THREAD_CONTEXT_CHARS = 4000

# Track users who have agreed to disclaimer (persists across profile changes)
# Key: user identifier, Value: True if agreed
## TODO: Enable Azure Government OAuth providers once we have the creds

# Register Azure Government OAuth providers by adding them to the module
# These will be automatically discovered by Chainlit
azure_gov_provider = AzureGovOAuthProvider()
azure_gov_hybrid_provider = AzureGovHybridOAuthProvider()

if azure_gov_provider.is_configured():
    # Register the provider by importing it into the module namespace
    import chainlit.oauth_providers as oauth_module

    oauth_module.providers.append(azure_gov_provider)
    logger.info("Registered Azure Government OAuth provider")

if azure_gov_hybrid_provider.is_configured():
    import chainlit.oauth_providers as oauth_module

    oauth_module.providers.append(azure_gov_hybrid_provider)
    logger.info("Registered Azure Government Hybrid OAuth provider")

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    # Customize user metadata for Azure Government users if needed
    if provider_id in ["azure-gov", "azure-gov-hybrid"]:
        default_user.metadata = {
            **default_user.metadata,
            "azure_gov_user": True,
        }
    return default_user



@cl.set_chat_profiles
async def chat_profile():
    base_starter = [
        cl.Starter(
            label="What can this Agent do?",
            message="What can you do, and what tools do you have available?",
            icon="/public/azure.png",
        ),
    ]
    return [
        cl.ChatProfile(
            name=AI_SEARCH_BOT_PROFILE,
            markdown_description="Get responses grounded on Maintenance data.",
            icon="public/logo_dark.png",
            starters=base_starter
            + [
                cl.Starter(
                    label="Yearly maintenance requirements",
                    message="What yearly maintenance is required for the M1 Abrams?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="How to replace parts",
                    message="Can you describe how to replace the oil filter on a HMMWV?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="What maintenance manuals are available",
                    message="What maintenance manuals are available for the UH-60 Helicopter?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="What documents are in the repository",
                    message="Give me an overview of the documents in the Azure AI Search repository.",
                    icon="/public/logo_light.png",
                ),
            ],
        ),
        cl.ChatProfile(
            name=SQL_BOT_PROFILE,
            markdown_description="Get responses grounded on data in Azure Synapse.",
            icon="public/logo_dark.png",
            starters=base_starter
            + [
                cl.Starter(
                    label="List all database tables",
                    message="What tables are available in the database?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="Show table schema",
                    message="Can you describe the schema of the first table you find?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="Query top records",
                    message="Show me the first 10 records from any table in the database.",
                    icon="/public/logo_light.png",
                ),
            ],
        ),
        cl.ChatProfile(
            name="Combat LogiGuard AI",
            markdown_description="Get responses grounded on all available data sources.",
            icon="public/logo_dark.png",
            default=True,
            starters=base_starter
            + [
                cl.Starter(
                    label="Based on my database, what vehicles need maintenance, and what documents are relevant?",
                    message="Based on my database, what vehicles need maintenance, and what documents are relevant?",
                    icon="/public/logo_light.png",
                ),
            ],
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    # Clear any existing session data to ensure fresh state for new chat
    # This prevents thread leakage between chat sessions
    old_thread = cl.user_session.get("thread")
    if old_thread:
        logger.info("Clearing existing thread from previous chat session")
    cl.user_session.set("thread", None)
    cl.user_session.set("agent", None)
    cl.user_session.set("secondary_agent", None)
    
    # Reset token usage counter for new chat session
    cl.user_session.set("token_usage", UsageDetails())
    cl.user_session.set(
        "context_usage",
        {
            "request_count": 0,
            "sum_context_chars": 0,
            "max_context_chars": 0,
            "last_context_chars": 0,
        },
    )
    
    # Setup Semantic Kernel
    app_user = cl.user_session.get("user")
    if app_user:
        logger.info("App user identifier: %s", getattr(app_user, "identifier", "unknown"))
    else:
        logger.info("App user identifier: anonymous")
    # await cl.Message(f"Hello {app_user.identifier}").send()

    logger.info("Initializing new chat session")

    # Primary LLM configuration
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment_name = os.getenv("AZURE_OPENAI_MODEL", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # Secondary/Fallback LLM configuration
    secondary_endpoint = os.getenv("AZURE_OPENAI_SECONDARY_ENDPOINT", "")
    secondary_deployment_name = os.getenv("AZURE_OPENAI_SECONDARY_MODEL", "")
    secondary_api_key = os.getenv("AZURE_OPENAI_SECONDARY_API_KEY", "")
    # Use secondary API version, fallback to primary, or use default
    secondary_api_version = (
        os.getenv("AZURE_OPENAI_SECONDARY_API_VERSION")
        or api_version
        or "2024-02-15-preview"
    )

    logger.info(
        "Initializing Azure OpenAI chat client (deployment=%s)", deployment_name
    )

    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_MODEL": deployment_name,
            "AZURE_OPENAI_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        logger.warning(
            "Missing Azure OpenAI settings: %s. Please set them in your environment or .env file.",
            ", ".join(missing),
        )

    # Check for secondary LLM configuration
    has_secondary_llm = all([
        secondary_endpoint,
        secondary_deployment_name,
        secondary_api_key,
    ])
    if has_secondary_llm:
        logger.info(
            "Secondary LLM configured (deployment=%s)", secondary_deployment_name
        )
    else:
        logger.info("No secondary LLM configured - fallback will not be available")

    # Get connection string from environment
    connection_string = os.environ.get("AZURE_SQL_CONNECTIONSTRING")
    logger.info("SQL connection string configured: %s", bool(connection_string))
    if not connection_string:
        raise ValueError("AZURE_SQL_CONNECTIONSTRING environment variable is required")

    db_tool = SqlDatabase(connection_string)

    # Configure primary OpenAI client with NO retries for immediate fallback
    # We must create the AsyncAzureOpenAI client directly with max_retries=0
    # because the AzureOpenAIChatClient doesn't pass max_retries to the underlying client
    primary_async_client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=api_version or "2024-02-15-preview",
        max_retries=0,   # No retries - fail immediately to trigger fallback
        timeout=120.0,   # Total timeout for request
    )

    llm = AzureOpenAIChatClient(
        endpoint=endpoint or None,
        deployment_name=deployment_name or None,
        api_key=api_key or None,
        api_version=api_version or None,
        async_client=primary_async_client,  # Use pre-configured client with no retries
    )

    # Configure secondary/fallback OpenAI client if available
    secondary_llm: Optional[AzureOpenAIChatClient] = None
    if has_secondary_llm:
        logger.info(
            "Initializing secondary Azure OpenAI client (endpoint=%s, deployment=%s)",
            secondary_endpoint,
            secondary_deployment_name,
        )
        secondary_llm = AzureOpenAIChatClient(
            endpoint=secondary_endpoint,
            deployment_name=secondary_deployment_name,
            api_key=secondary_api_key,
            api_version=secondary_api_version,
            max_retries=3,
            timeout=120.0,
        )

    agent = ChatAgent(
        chat_client=llm,
        name="agent_name",
        instructions="You are a helpful assistant with database tools. Follow these rules strictly:\n\
1. ALWAYS use your tools to answer questions - never rely on assumptions or general knowledge\n\
2. NEVER make up information - if you don't have data from a tool, say so\n\
3. Follow ALL step-by-step instructions in tool docstrings exactly - including STEP 3a before STEP 3b\n\
4. If a tool says REQUIRED or DO NOT skip, you MUST comply with that instruction\n\
5. Provide clear, concise responses based only on verified tool results",
        temperature=0.1,
    )

    # Create secondary agent if fallback LLM is configured
    secondary_agent: Optional[ChatAgent] = None
    if secondary_llm:
        secondary_agent = ChatAgent(
            chat_client=secondary_llm,
            name="agent_name_secondary",
            instructions="You are a helpful assistant with database tools. Follow these rules strictly:\n\
1. ALWAYS use your tools to answer questions - never rely on assumptions or general knowledge\n\
2. NEVER make up information - if you don't have data from a tool, say so\n\
3. Follow ALL step-by-step instructions in tool docstrings exactly - including STEP 3a before STEP 3b\n\
4. If a tool says REQUIRED or DO NOT skip, you MUST comply with that instruction\n\
5. Provide clear, concise responses based only on verified tool results",
            temperature=0.1,
        )

    db_tools = [
        db_tool.list_tables,
        db_tool.list_views,
        db_tool.describe_table,
        db_tool.read_query,
    ]
    search_tools = [semantic_search, list_facets]


    thread = agent.get_new_thread()
    logger.info("Created new thread for chat session")

    cl.user_session.set("agent", agent)
    cl.user_session.set("secondary_agent", secondary_agent)
    cl.user_session.set("db_tools", db_tools)
    cl.user_session.set("ai_search", search_tools)
    cl.user_session.set("thread", thread)


@cl.on_chat_end
async def on_chat_end():
    """Clean up session data when chat ends."""
    logger.info("Chat session ending, cleaning up resources")
    # Log final token usage for the session
    token_usage: Optional[UsageDetails] = cl.user_session.get("token_usage")
    context_usage = cl.user_session.get("context_usage") or {}
    if token_usage:
        logger.info(
            "Session token usage - Input: %s, Output: %s, Total: %s | "
            "Context chars - Last: %s, Max: %s, Avg: %s",
            token_usage.input_token_count or 0,
            token_usage.output_token_count or 0,
            token_usage.total_token_count or 0,
            context_usage.get("last_context_chars", 0),
            context_usage.get("max_context_chars", 0),
            int(
                (context_usage.get("sum_context_chars", 0) / max(context_usage.get("request_count", 1), 1))
            ),
        )
    # Clear thread and agents to ensure no state leaks to next session
    cl.user_session.set("thread", None)
    cl.user_session.set("agent", None)
    cl.user_session.set("secondary_agent", None)
    cl.user_session.set("token_usage", None)
    cl.user_session.set("context_usage", None)


def _is_retryable_error(e: Exception) -> bool:
    """Check if an error is retryable (429 rate limit or similar transient errors)."""
    error_message = str(e)
    error_type = str(type(e))
    error_lower = error_message.lower()
    return (
        "429" in error_message
        or "Too Many Requests" in error_message
        or "RateLimitError" in error_type
        or "rate_limit" in error_lower
        or "rate limit" in error_lower
        or "capacity" in error_lower
    )


def _is_context_length_error(e: Exception) -> bool:
    """Check if an error indicates the prompt/context is too large."""
    error_text = str(e).lower()
    return any(
        phrase in error_text
        for phrase in [
            "context length",
            "maximum context length",
            "token limit",
            "too many tokens",
            "prompt is too long",
            "maximum prompt",
        ]
    )


def _estimate_message_chars(message: ChatMessage) -> int:
    """Approximate message size in characters via serialized payload."""
    try:
        return len(json.dumps(message.to_dict(), ensure_ascii=False))
    except Exception:
        return len(str(message))


def _estimate_thread_context_chars(thread: Optional[AgentThread]) -> int:
    """Estimate total serialized size of thread messages in characters."""
    if not thread or not thread.message_store or not hasattr(thread.message_store, "messages"):
        return 0
    messages = thread.message_store.messages or []
    return sum(_estimate_message_chars(message) for message in messages)


def _extract_tool_call_ids(message: ChatMessage) -> tuple[set[str], set[str]]:
    """Return (function_call_ids, function_result_ids) present in a message."""
    function_calls: set[str] = set()
    function_results: set[str] = set()

    contents = getattr(message, "contents", []) or []
    for content in contents:
        content_type = getattr(content, "type", None)
        call_id = getattr(content, "call_id", None)
        if not call_id:
            continue
        if content_type == "function_call":
            function_calls.add(call_id)
        elif content_type == "function_result":
            function_results.add(call_id)

    return function_calls, function_results


def _is_message_tool_pair_consistent(message: ChatMessage, matched_call_ids: set[str]) -> bool:
    """Check whether a message containing tool call/result content is pair-consistent."""
    function_calls, function_results = _extract_tool_call_ids(message)
    if function_calls and not (function_calls & matched_call_ids):
        return False
    if function_results and not (function_results & matched_call_ids):
        return False
    return True


def _trim_user_input(user_input: str, max_chars: int = DEFAULT_MAX_USER_INPUT_CHARS) -> str:
    """Trim oversized user input to reduce immediate prompt size."""
    if len(user_input) <= max_chars:
        return user_input
    marker = "\n\n[Input truncated to fit model context window.]"
    return user_input[: max_chars - len(marker)] + marker


def _trim_thread_context(
    thread: Optional[AgentThread],
    reserve_for_input_chars: int = 0,
    aggressive: bool = False,
) -> None:
    """Trim local thread history to a bounded rolling window.

    Keeps newest messages first, while enforcing:
    1) max message count
    2) max per-message size (drops oversized historical messages)
    3) total character budget
    """
    if not thread or not thread.message_store or not hasattr(thread.message_store, "messages"):
        return

    messages = list(thread.message_store.messages or [])
    if not messages:
        return

    max_messages = 8 if aggressive else DEFAULT_MAX_THREAD_MESSAGES
    max_message_chars = 6000 if aggressive else DEFAULT_MAX_MESSAGE_CHARS
    configured_budget = 18000 if aggressive else DEFAULT_MAX_THREAD_CHARS
    max_thread_chars = max(configured_budget - reserve_for_input_chars, MIN_THREAD_CONTEXT_CHARS)

    # Keep only the newest messages by count first
    messages = messages[-max_messages:]

    # Drop oversized historical messages (commonly large tool outputs)
    filtered_messages = [
        message for message in messages if _estimate_message_chars(message) <= max_message_chars
    ]
    if filtered_messages:
        messages = filtered_messages

    # Enforce total thread character budget from newest to oldest
    selected: list[ChatMessage] = []
    running_chars = 0
    for message in reversed(messages):
        msg_chars = _estimate_message_chars(message)
        if msg_chars > max_message_chars:
            continue
        if running_chars + msg_chars > max_thread_chars:
            if selected:
                break
            continue
        if msg_chars <= 0:
            continue
        selected.append(message)
        running_chars += msg_chars

    trimmed_messages = list(reversed(selected))

    # Keep only matched function_call/function_result pairs to avoid invalid tool_call history.
    all_function_calls: set[str] = set()
    all_function_results: set[str] = set()
    for message in trimmed_messages:
        calls, results = _extract_tool_call_ids(message)
        all_function_calls.update(calls)
        all_function_results.update(results)

    matched_call_ids = all_function_calls & all_function_results
    pair_safe_messages = [
        message
        for message in trimmed_messages
        if _is_message_tool_pair_consistent(message, matched_call_ids)
    ]

    thread.message_store.messages = pair_safe_messages


async def _run_agent_stream(
    agent: ChatAgent,
    message_content: Optional[str],
    tools: list,
    thread: AgentThread,
    parent_message_id: str,
) -> Tuple[Optional[cl.Message], Optional[UsageDetails]]:
    """Run agent stream and handle response rendering. Returns the answer message and usage or None.
    
    Args:
        agent: The ChatAgent to run
        message_content: The message to send, or None to continue from existing thread context
        tools: The tools available to the agent
        thread: The conversation thread (may contain prior context from failed primary agent)
        parent_message_id: The parent message ID for UI steps
    
    Returns:
        A tuple of (answer message, usage details) - either may be None
    """
    answer: Optional[cl.Message] = None
    active_steps: dict = {}
    current_call_id: Optional[str] = None
    request_usage: Optional[UsageDetails] = None

    async for msg in agent.run_stream(message_content, tools=tools, thread=thread):
        msg_dict = msg.to_dict()
        if msg_dict.get("type") == "agent_run_response_update":
            for content in msg_dict.get("contents", []):
                content_type = content.get("type")

                if content_type == "function_call":
                    if answer:
                        await answer.update()
                        answer = None

                    call_id = content.get("call_id")
                    name = content.get("name")
                    arguments = content.get("arguments", "")

                    if name:
                        step = cl.Step(name=name, type="tool", parent_id=parent_message_id)
                        step.input = arguments
                        if call_id:
                            active_steps[call_id] = step
                            current_call_id = call_id
                        await step.send()
                    else:
                        target_id = call_id if call_id else current_call_id
                        if target_id and target_id in active_steps:
                            active_steps[target_id].input += arguments
                            await active_steps[target_id].update()

                elif content_type == "function_result":
                    call_id = content.get("call_id")
                    result = content.get("result")
                    if call_id in active_steps:
                        active_steps[call_id].output = result
                        await active_steps[call_id].update()
                        del active_steps[call_id]

                elif content_type == "usage":
                    # Extract token usage from the response
                    details = content.get("details", {})
                    usage = UsageDetails(
                        input_token_count=details.get("input_token_count"),
                        output_token_count=details.get("output_token_count"),
                        total_token_count=details.get("total_token_count"),
                    )
                    request_usage = (request_usage + usage) if request_usage else usage

        if getattr(msg, "text", None):
            if answer is None:
                answer = cl.Message(content="")
            await answer.stream_token(msg.text)

    return answer, request_usage


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    secondary_agent = cl.user_session.get("secondary_agent")
    tools = cl.user_session.get("tools")
    thread = cl.user_session.get("thread")

    profile = cl.user_session.get("chat_profile")

    logging.info(f"Current chat profile: {profile if profile else 'None'}")
    if profile == AI_SEARCH_BOT_PROFILE:
        tools = cl.user_session.get("ai_search")

    elif profile == SQL_BOT_PROFILE:
        tools = cl.user_session.get("db_tools")

    else:
        db_tools = cl.user_session.get("db_tools")
        ai_search = cl.user_session.get("ai_search")
        tools = db_tools + ai_search

    answer: Optional[cl.Message] = None
    request_usage: Optional[UsageDetails] = None
    used_fallback = False
    prepared_message = _trim_user_input(message.content)

    # Proactively trim thread context before every model call
    _trim_thread_context(thread, reserve_for_input_chars=len(prepared_message))
    current_context_chars = _estimate_thread_context_chars(thread) + len(prepared_message)

    context_usage = cl.user_session.get("context_usage") or {
        "request_count": 0,
        "sum_context_chars": 0,
        "max_context_chars": 0,
        "last_context_chars": 0,
    }
    context_usage["request_count"] = context_usage.get("request_count", 0) + 1
    context_usage["sum_context_chars"] = context_usage.get("sum_context_chars", 0) + current_context_chars
    context_usage["max_context_chars"] = max(context_usage.get("max_context_chars", 0), current_context_chars)
    context_usage["last_context_chars"] = current_context_chars
    cl.user_session.set("context_usage", context_usage)

    try:
        # Try primary agent first
        answer, request_usage = await _run_agent_stream(
            agent, prepared_message, tools, thread, message.id
        )
    except Exception as e:
        # If context window is exceeded, trim aggressively and retry once
        if _is_context_length_error(e):
            logger.warning(
                "Context length exceeded. Retrying with aggressively trimmed thread context."
            )
            _trim_thread_context(
                thread,
                reserve_for_input_chars=len(prepared_message),
                aggressive=True,
            )
            try:
                answer, request_usage = await _run_agent_stream(
                    agent, prepared_message, tools, thread, message.id
                )
            except Exception as retry_error:
                e = retry_error
            else:
                e = None

        if e is None:
            pass
        # Check if we should retry with secondary agent
        elif _is_retryable_error(e) and secondary_agent:
            logger.warning(
                f"Primary LLM failed with retryable error: {e}. Retrying with secondary model..."
            )
            # Notify user about retry
            await cl.Message(
                content="⏳ Primary model unavailable, retrying with backup model..."
            ).send()

            try:
                # Reuse the existing thread to preserve any tool calls and results
                # from the primary agent's partial execution. Pass None for message
                # since it's already in the thread context.
                answer, request_usage = await _run_agent_stream(
                    secondary_agent, None, tools, thread, message.id
                )
                used_fallback = True
                logger.info("Successfully processed request using secondary model with preserved context")
            except Exception as secondary_e:
                logger.error(
                    f"Secondary LLM also failed: {secondary_e}", exc_info=True
                )
                await cl.Message(
                    content="❌ Both primary and backup AI models are unavailable. Please try again later."
                ).send()
                cl.user_session.set("thread", thread)
                return
        elif _is_retryable_error(e):
            # Rate limit but no secondary agent configured
            logger.error(f"Rate limit error (429) with no fallback: {e}")
            await cl.Message(
                content="⚠️ The AI service is currently experiencing high demand (rate limit exceeded). Please try again in a moment."
            ).send()
            cl.user_session.set("thread", thread)
            return
        else:
            # Non-retryable error
            logger.error(f"Error processing message: {e}", exc_info=True)
            await cl.Message(
                content=f"❌ An error occurred while processing your request: {str(e)}"
            ).send()
            cl.user_session.set("thread", thread)
            return

    # Update cumulative token usage for the session
    if request_usage:
        session_usage: Optional[UsageDetails] = cl.user_session.get("token_usage")
        if session_usage:
            session_usage += request_usage
        else:
            session_usage = request_usage
        cl.user_session.set("token_usage", session_usage)
        
        # Log token usage for this request (not shown to user)
        logger.info(
            "Request token usage - Input: %s, Output: %s, Total: %s | "
            "Session cumulative - Input: %s, Output: %s, Total: %s | "
            "Context chars - Current: %s, Max: %s, Avg: %s",
            request_usage.input_token_count or 0,
            request_usage.output_token_count or 0,
            request_usage.total_token_count or 0,
            session_usage.input_token_count or 0,
            session_usage.output_token_count or 0,
            session_usage.total_token_count or 0,
            context_usage.get("last_context_chars", 0),
            context_usage.get("max_context_chars", 0),
            int(
                (context_usage.get("sum_context_chars", 0) / max(context_usage.get("request_count", 1), 1))
            ),
        )

    # Send the final message if not already sent
    if answer:
        if used_fallback:
            answer.content += "\n\n_Response generated using backup model._"
        await answer.send()

    cl.user_session.set("thread", thread)
