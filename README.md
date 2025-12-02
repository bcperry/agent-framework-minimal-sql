# Agent Framework Minimal SQL

A minimal example project demonstrating how to use the [agent-framework](https://pypi.org/project/agent-framework/) with Azure OpenAI and Azure SQL Database. This project includes both a command-line interface (`main.py`) and an interactive Jupyter notebook (`notebook.ipynb`).

## Overview

This project creates an AI agent that can:
- List tables in an Azure SQL Database
- Describe table schemas (columns, data types, constraints)
- Execute SQL queries based on natural language requests

## Prerequisites

- Python 3.12 or higher
- Azure OpenAI account with deployed model
- Azure SQL Database (or SQL dedicated pool in Azure Synapse)
- ODBC Driver for SQL Server installed on your system

## Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. Install uv if you haven't already:
   ```powershell
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install dependencies:
   ```powershell
   uv sync
   ```

   **Note:** With uv, you don't need to manually create or activate a virtual environment. Simply use `uv run` to run commands (see usage below).

### Option 2: Using pip

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

   Or install from `pyproject.toml`:
   ```powershell
   pip install -e .
   ```

## Configuration

Create a `.env` file in the project root with the following environment variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Azure SQL Database Configuration
AZURE_SQL_CONNECTIONSTRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server.database.windows.net,1433;Database=your-database;Uid=your-username;Pwd=your-password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
```

### Connection String Format

The Azure SQL connection string should follow this format:
```
Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=<database>;Uid=<username>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
```

## Deployment to Azure App Service

This application can be deployed to Azure App Service using the Azure Developer CLI (azd).

### Prerequisites for Deployment

- [Azure Developer CLI (azd)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) installed
- Azure subscription
- Azure OpenAI resource deployed
- Azure SQL Database configured
- Azure AI Search service configured

### Deployment with Azure Developer CLI

**1. Initialize the environment:**

```powershell
azd init
```

When prompted, use an existing environment name or create a new one (e.g., `dev`).

**2. Provision and deploy in one command:**

```powershell
azd up
```

This will:
- Create an Azure Resource Group
- Create an App Service Plan (Linux, B1 SKU)
- Create an App Service with Python 3.12 runtime
- Configure managed identity
- Enable WebSocket support (required for Chainlit)
- Deploy your application code

**3. Configure environment variables:**

After deployment, set the required environment variables in the Azure Portal or using azd:

```powershell
azd env set AZURE_OPENAI_ENDPOINT "https://your-resource.openai.azure.com/"
azd env set AZURE_OPENAI_MODEL "your-deployment-name"
azd env set AZURE_OPENAI_API_KEY "your-api-key"
azd env set AZURE_OPENAI_API_VERSION "2024-08-01-preview"
azd env set AZURE_SQL_CONNECTIONSTRING "Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server.database.windows.net,1433;Database=your-database;Uid=your-username;Pwd=your-password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
azd env set SEARCH_SERVICE_ENDPOINT "https://your-search-service.search.windows.net"
azd env set SEARCH_INDEX_NAME "your-index-name"
```

Then deploy again to apply the settings:

```powershell
azd deploy
```

### Alternative: Step-by-Step Deployment

**Provision infrastructure only:**
```powershell
azd provision
```

**Deploy application code only:**
```powershell
azd deploy
```

### Managing Your Deployment

**View deployment logs:**
```powershell
azd monitor
```

**Open the deployed app:**
```powershell
azd browse
```

**View all resources:**
```powershell
azd show
```

**Clean up resources:**
```powershell
azd down
```

### Important Notes

- **ODBC Drivers**: Azure App Service Linux includes ODBC Driver 18 for SQL Server by default
- **WebSockets**: WebSocket support is automatically enabled for Chainlit
- **Managed Identity**: A user-assigned managed identity is created and can be used for secure access to Azure resources
- **Environment Variables**: Configure sensitive values using `azd env set` or Azure Portal App Settings

## Usage

### Running the Command-Line Interface (main.py)

The CLI provides an interactive chat interface where you can ask questions about your SQL database.

**Using uv:**
```powershell
uv run main.py
```

**Using pip/venv:**
```powershell
python main.py
```

**Example interaction:**
```
=== Command Line Agent ===
Type 'exit' or 'quit' to stop chatting.

You: List the top 5 airlines by total aircraft.
Agent: [Thinking and using tools...]
Agent: Here are the top 5 airlines by total aircraft:
1. American Airlines - 956 aircraft
2. Delta Air Lines - 889 aircraft
...

You: exit
Goodbye!
```

**Features:**
- Interactive chat loop with streaming responses
- Automatic tool calling (list_tables, describe_table, read_query)
- Error handling and graceful exit
- Type 'exit' or 'quit' to end the session

### Running the Jupyter Notebook (notebook.ipynb)

The notebook provides a step-by-step interactive environment for working with the agent.

1. **Start Jupyter:**
   
   **Using uv:**
   ```powershell
   uv run jupyter notebook
   ```
   
   **Using pip/venv:**
   ```powershell
   jupyter notebook
   ```
   
   Or simply open `notebook.ipynb` directly in VS Code.

2. **Run the cells in order:**
   - **Cell 1:** Load necessary modules and set up logging
   - **Cell 2:** Read environment variables from `.env`
   - **Cell 3:** Create the SqlDatabase tool, Azure OpenAI client, and ChatAgent
   - **Cell 4:** Execute a sample query (e.g., "List the top 5 airlines by total aircraft")

3. **Modify the query:**
   In the last cell, change the `query` variable to ask different questions:
   ```python
   query = "What tables are available?"
   query = "Show me the schema of the flights table"
   query = "Find all flights departing from LAX"
   ```

**Note:** The notebook uses `await agent.run()` so cells must be run in an async context (Jupyter automatically provides this).

### Installing Development Dependencies

If you want to contribute or run pre-commit hooks:

**Using uv:**
```powershell
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
```

**Using pip:**
```powershell
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
agent-framework-minimal-sql/
├── main.py              # Command-line interface with streaming chat
├── notebook.ipynb       # Interactive Jupyter notebook
├── tools.py             # SqlDatabase class with list_tables, describe_table, read_query
├── pyproject.toml       # Project dependencies (uv/pip compatible)
├── requirements.txt     # Locked dependencies (auto-generated by uv)
├── .env                 # Environment variables (create this file)
└── README.md            # This file
```

## How It Works

1. **SqlDatabase Tool** (`tools.py`):
   - Connects to Azure SQL Database using pyodbc
   - Exposes three methods as agent tools:
     - `list_tables()`: Discover available tables
     - `describe_table(table_name)`: Get schema information
     - `read_query(query)`: Execute SELECT queries

2. **ChatAgent**:
   - Uses Azure OpenAI for language understanding
   - Automatically decides when to call tools based on user input
   - Maintains conversation history via threads

3. **Streaming** (`main.py`):
   - Responses stream token-by-token for better UX
   - Tool calls are displayed as they're made

## Troubleshooting

**Import Error: No module named 'agent_framework'**
- Make sure you've activated the virtual environment
- Run `uv sync` or `pip install -r requirements.txt`

**Database Connection Error**
- Verify your `AZURE_SQL_CONNECTIONSTRING` is correct
- Ensure the ODBC Driver 18 for SQL Server is installed
- Check firewall rules allow your IP address

**Azure OpenAI Error**
- Verify all Azure OpenAI environment variables are set correctly
- Check that your deployment name matches the model you've deployed
- Ensure your API key is valid

## License

This is a minimal example project for demonstration purposes.
