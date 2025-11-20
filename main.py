import os
import logging
import asyncio
from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ChatAgent
from tools import SqlDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(stream: bool = True) -> None:
    logger.info("Hello from agent-framework-minimal-sql!")

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
        instructions="Your instructions here",
        tools=[db_tool.list_tables, db_tool.describe_table, db_tool.read_query],
        temperature=0.7,
    )

    thread = agent.get_new_thread()

    chat_with_agent(agent, thread)


def chat_with_agent(agent: ChatAgent, thread, stream=True) -> None:
    print("=== Command Line Agent ===")
    print("Type 'exit' or 'quit' to stop chatting.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        try:
            asyncio.run(stream_response(agent, thread, user_input))
        except Exception as exc:  # Surface connection issues without crashing the loop
            logger.exception("Streaming response failed")
            print(f"[Error: {exc.__class__.__name__}: {exc}]")
        print()  # Newline after agent response


async def stream_response(agent: ChatAgent, thread: str, user_input: str) -> None:
    # Consume the agent's async stream and print tokens as they arrive.
    function_calls = {}  # Track function calls by call_id

    async for chunk in agent.run_stream(user_input, thread=thread):
        if chunk.text:
            print(chunk.text, end="", flush=True)
        else:
            # Extract function call info from contents
            chunk_dict = chunk.to_dict()
            contents = chunk_dict.get("contents", [])

            for content in contents:
                if content.get("type") == "function_call":
                    call_id = content.get("call_id", "")
                    name = content.get("name", "")
                    arguments = content.get("arguments", "")

                    # Initialize function call tracking with a real call_id
                    if call_id and call_id not in function_calls:
                        function_calls[call_id] = {
                            "name": name,
                            "arguments": "",
                            "printed": False,
                        }

                    # Update existing call with streaming data
                    if call_id:
                        if name:
                            function_calls[call_id]["name"] = name
                        if arguments:
                            function_calls[call_id]["arguments"] += arguments
                    # Empty call_id means streaming is complete for the last call
                    elif arguments and function_calls:
                        # Add to the most recent function call
                        last_call_id = list(function_calls.keys())[-1]
                        function_calls[last_call_id]["arguments"] += arguments

            # After processing all contents, print any completed calls
            for call_id, call_info in function_calls.items():
                if call_info["name"] and not call_info["printed"]:
                    # Check if this call is complete (next chunk has empty contents or different call_id)
                    # For now, print when we have both name and some indication of completion
                    if (
                        call_id
                        and len(call_info["arguments"]) > 0
                        and call_info["arguments"].endswith("}")
                    ):
                        print(
                            f"\n[Tool: {call_info['name']}({call_info['arguments']})]",
                            end="",
                            flush=True,
                        )
                        call_info["printed"] = True
                        print()


if __name__ == "__main__":
    main()
