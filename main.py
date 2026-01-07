import os
import chainlit as cl
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from openai.lib.azure import AsyncAzureOpenAI
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ChatAgent, AgentThread
from tools import SqlDatabase
from rag_tools import semantic_search, list_facets
from custom_oauth import AzureGovOAuthProvider, AzureGovHybridOAuthProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AI_SEARCH_BOT_PROFILE = "Technical Maintenance AI"
SQL_BOT_PROFILE = "Tactical Readiness AI"


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

# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#     # Customize user metadata for Azure Government users if needed
#     if provider_id in ["azure-gov", "azure-gov-hybrid"]:
#         default_user.metadata = {
#             **default_user.metadata,
#             "azure_gov_user": True,
#         }
#     return default_user


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
    
    # Setup Semantic Kernel
    app_user = cl.user_session.get("user")
    logger.info(f"App user: {app_user}")
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
    print(f"Connected using: {connection_string}")
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
    # Clear thread and agents to ensure no state leaks to next session
    cl.user_session.set("thread", None)
    cl.user_session.set("agent", None)
    cl.user_session.set("secondary_agent", None)


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


async def _run_agent_stream(
    agent: ChatAgent,
    message_content: Optional[str],
    tools: list,
    thread: AgentThread,
    parent_message_id: str,
) -> Optional[cl.Message]:
    """Run agent stream and handle response rendering. Returns the answer message or None.
    
    Args:
        agent: The ChatAgent to run
        message_content: The message to send, or None to continue from existing thread context
        tools: The tools available to the agent
        thread: The conversation thread (may contain prior context from failed primary agent)
        parent_message_id: The parent message ID for UI steps
    """
    answer: Optional[cl.Message] = None
    active_steps: dict = {}
    current_call_id: Optional[str] = None

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

        if getattr(msg, "text", None):
            if answer is None:
                answer = cl.Message(content="")
            await answer.stream_token(msg.text)

    return answer


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
    used_fallback = False

    try:
        # Try primary agent first
        answer = await _run_agent_stream(
            agent, message.content, tools, thread, message.id
        )
    except Exception as e:
        # Check if we should retry with secondary agent
        if _is_retryable_error(e) and secondary_agent:
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
                answer = await _run_agent_stream(
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

    # Send the final message if not already sent
    if answer:
        if used_fallback:
            answer.content += "\n\n_Response generated using backup model._"
        await answer.send()

    cl.user_session.set("thread", thread)
