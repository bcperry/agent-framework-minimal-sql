import os
import chainlit as cl
import logging
from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ChatAgent
from tools import SqlDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Maintenance Data Bot",
            markdown_description="Get responses grounded on Maintenance data.",
            icon="public/logo_dark.png",
            starters=[
                cl.Starter(
                    label="List maintenance records",
                    message="What maintenance records are available in the database?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="Show maintenance schema",
                    message="Can you describe the schema of the maintenance table?",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="Query recent maintenance",
                    message="Show me the last 10 maintenance records from the database.",
                    icon="/public/logo_light.png",
                ),
                cl.Starter(
                    label="Maintenance overview",
                    message="Give me an overview of the maintenance data structure including key columns.",
                    icon="/public/logo_light.png",
                ),
            ],
        ),
        cl.ChatProfile(
            name="Synapse Data Bot",
            markdown_description="Get responses grounded on Azure Synapse Data.",
            icon="public/logo_dark.png",
            starters=[
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
                cl.Starter(
                    label="Database overview",
                    message="Give me an overview of the database structure including all tables and their key columns.",
                    icon="/public/logo_light.png",
                ),
            ],
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    # Setup Semantic Kernel

    logger.info("Initializing app")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment_name = os.getenv("AZURE_OPENAI_MODEL", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

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

    # Get connection string from environment
    connection_string = os.environ.get("AZURE_SQL_CONNECTIONSTRING")
    print(f"Connected using: {connection_string}")
    if not connection_string:
        raise ValueError("AZURE_SQL_CONNECTIONSTRING environment variable is required")

    db_tool = SqlDatabase(connection_string)

    llm = AzureOpenAIChatClient(
        endpoint=endpoint or None,
        deployment_name=deployment_name or None,
        api_key=api_key or None,
        api_version=api_version or None,
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
        tools=[db_tool.list_tables, db_tool.describe_table, db_tool.read_query],
        temperature=0.1,
    )

    thread = agent.get_new_thread()

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread", thread)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread = cl.user_session.get("thread")

    # Create a Chainlit message for the response stream
    answer = None

    active_steps = {}
    current_call_id = None

    async for msg in agent.run_stream(message.content, thread=thread):
        logging.info(f"Agent message: {msg.to_json()}")

        msg_dict = msg.to_dict()
        if msg_dict.get("type") == "agent_run_response_update":
            for content in msg_dict.get("contents", []):
                content_type = content.get("type")

                if content_type == "function_call":
                    # If we were streaming an answer, finalize it and reset
                    if answer:
                        await answer.update()
                        answer = None

                    call_id = content.get("call_id")
                    name = content.get("name")
                    arguments = content.get("arguments", "")

                    if name:
                        step = cl.Step(name=name, type="tool", parent_id=message.id)
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

    # Send the final message if not already sent
    if answer:
        await answer.send()

    cl.user_session.set("thread", thread)
