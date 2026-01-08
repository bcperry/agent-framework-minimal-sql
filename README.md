# Agent Framework Minimal SQL

AI agent for Azure SQL Database using [agent-framework](https://pypi.org/project/agent-framework/) with Azure OpenAI. Includes CLI and Jupyter notebook.

## Overview

AI agent capabilities:
- List tables in an Azure SQL Database
- Describe table schemas (columns, data types, constraints)
- Execute SQL queries based on natural language requests

## Prerequisites

- Python 3.12+
- Azure OpenAI account with deployed model
- Azure SQL Database
- ODBC Driver for SQL Server

## Installation

**Using uv (Recommended):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

**Using pip:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.us/
AZURE_OPENAI_MODEL=gpt-4o
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Azure SQL Database (local development with interactive auth)
AZURE_SQL_CONNECTIONSTRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server.database.usgovcloudapi.net,1433;Database=your-database;Encrypt=yes;Uid=your-user@your-tenant.onmicrosoft.us;TrustServerCertificate=no;Connection Timeout=30;Authentication=ActiveDirectoryInteractive

# Azure SQL Database (App Service with Managed Identity)
# AZURE_SQL_CONNECTIONSTRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server.database.usgovcloudapi.net,1433;Database=your-database;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;Authentication=ActiveDirectoryMsi;UID=your-managed-identity-client-id

# Azure AI Search
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.azure.us
SEARCH_INDEX_NAME=your-index-name
SEARCH_API_KEY=your-search-api-key

# Azure Government OAuth (optional)
OAUTH_AZURE_GOV_AD_CLIENT_ID=your-client-id
OAUTH_AZURE_GOV_AD_CLIENT_SECRET=your-client-secret
OAUTH_AZURE_GOV_AD_TENANT_ID=your-tenant-id

# Chainlit
CHAINLIT_AUTH_SECRET=your-auth-secret
```

## Usage

**CLI:**
```powershell
uv run chainlit run main.py
```

**Jupyter:**
```powershell
uv run jupyter notebook
```

Or open `notebook.ipynb` in VS Code.

## Azure Deployment

### Azure Government Login

**Azure Developer CLI (azd):**
```powershell
azd config set cloud.name AzureUSGovernment
azd auth login
```

**Azure CLI (az):**
```powershell
az cloud set --name AzureUSGovernment
az login
```

### Deploy using Azure Developer CLI

```powershell
azd up
```

Configure environment variables:

```powershell
azd env set AZURE_OPENAI_ENDPOINT "https://your-resource.openai.azure.com/"
azd env set AZURE_OPENAI_MODEL "your-deployment-name"
azd env set AZURE_OPENAI_API_KEY "your-api-key"
azd env set AZURE_SQL_CONNECTIONSTRING "your-connection-string"
azd deploy
```

### Managed Identity Database Access

Grant managed identity access to SQL Database:

```sql
CREATE USER [<app-service-name>] FROM EXTERNAL PROVIDER;
ALTER ROLE db_datareader ADD MEMBER [<app-service-name>];
```

See `sql_queries/` for helper scripts.

### Deployment Commands

```powershell
azd provision  # Infrastructure only
azd deploy     # Code only
azd down       # Clean up
```

### Alternative: Azure CLI Deployment (Existing Resources)

If you already have an App Service and want to deploy without `azd` infrastructure provisioning, you can use the Azure CLI directly.

**Deploy the application code:**

```powershell
az webapp up --resource-group <your-resource-group> --name <your-app-service-name> --runtime "PYTHON:3.11"
```

See [az webapp up documentation](https://learn.microsoft.com/en-us/cli/azure/webapp?view=azure-cli-latest#az-webapp-up) for more options.

**App Settings Import/Export:**

1. Export current settings:
   ```bash
   az webapp config appsettings list --resource-group <your-resource-group> --name <your-app-service-name> > settings.json
   ```

2. Convert to import format (the export format isn't directly importable):
   ```bash
   jq 'map({(.name): .value}) | add' settings.json > settings_import.json
   ```

3. Import settings to another App Service:
   ```bash
   az webapp config appsettings set --resource-group <your-resource-group> --name <your-app-service-name> --settings @settings_import.json
   ```

> **Note:** The `list` command outputs `[{"name": "KEY", "value": "VAL"}, ...]` but `set` requires `{"KEY": "VAL", ...}`. The `jq` command handles this conversion.

See [Edit app settings in bulk](https://learn.microsoft.com/en-us/azure/app-service/configure-common?tabs=cli#edit-app-settings-in-bulk) for more details.

## Development

```powershell
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
```

## Troubleshooting

- **Import errors**: Run `uv sync` or `pip install -r requirements.txt`
- **Database errors**: Check connection string, ODBC driver, and firewall rules
- **Azure OpenAI errors**: Verify endpoint, model name, and API key

## Azure Government OAuth

This application supports Azure Government OAuth authentication via custom providers in `custom_oauth.py`.

### Setup

1. Register an app in **Azure Active Directory → App registrations** in the Azure Government Portal
2. Set redirect URI to `https://your-app.azurewebsites.us/auth/oauth/azure-gov/callback`
3. Create a client secret and note the client ID, tenant ID, and secret value

### Environment Variables

```env
OAUTH_AZURE_GOV_AD_CLIENT_ID=your-client-id
OAUTH_AZURE_GOV_AD_CLIENT_SECRET=your-client-secret
OAUTH_AZURE_GOV_AD_TENANT_ID=your-tenant-id
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
