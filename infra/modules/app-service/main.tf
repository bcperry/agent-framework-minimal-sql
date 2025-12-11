# App Service Plan
resource "azurerm_service_plan" "app_service_plan" {
  name                = var.app_service_plan_name
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = coalesce(var.app_service_plan_tags, var.tags)

  os_type  = "Linux"
  sku_name = var.app_service_plan_sku
}

# App Service
resource "azurerm_linux_web_app" "app_service" {
  name                = var.name
  location            = var.location
  resource_group_name = var.resource_group_name
  service_plan_id     = azurerm_service_plan.app_service_plan.id
  tags                = coalesce(var.app_service_tags, var.tags)

  https_only = true

  identity {
    type = "UserAssigned"
    identity_ids = [
      var.managed_identity_id
    ]
  }

  site_config {
    always_on           = true
    websockets_enabled  = true
    ftps_state          = "Disabled"
    minimum_tls_version = "1.2"

    application_stack {
      python_version = var.python_version
    }

    app_command_line = "python -m chainlit run main.py --host 0.0.0.0 --port 8000"
  }

  app_settings = {
    SCM_DO_BUILD_DURING_DEPLOYMENT = "true"
    CHAINLIT_HOST                  = "0.0.0.0"
    CHAINLIT_PORT                  = "8000"
    CHAINLIT_URL                   = "https://${var.name}.azurewebsites.us"
    AZURE_CLIENT_ID                = var.managed_identity_client_id
    WEBSITES_PORT                  = "8000"
    
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT     = var.azure_openai_endpoint
    AZURE_OPENAI_MODEL        = var.azure_openai_model
    AZURE_OPENAI_API_KEY      = var.azure_openai_api_key
    AZURE_OPENAI_API_VERSION  = var.azure_openai_api_version
    
    # Azure SQL settings
    AZURE_SQL_CONNECTIONSTRING = var.azure_sql_connectionstring
    
    # Azure AI Search settings
    SEARCH_SERVICE_ENDPOINT = var.search_service_endpoint
    SEARCH_INDEX_NAME       = var.search_index_name
    SEARCH_API_KEY          = var.search_api_key
  }

  logs {
    detailed_error_messages = true
    failed_request_tracing  = true

    http_logs {
      file_system {
        retention_in_days = 1
        retention_in_mb   = 35
      }
    }

    application_logs {
      file_system_level = "Verbose"
    }
  }
}
