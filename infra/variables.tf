variable "environment_name" {
  description = "Name of the environment that can be used as part of naming resource convention"
  type        = string

  validation {
    condition     = length(var.environment_name) >= 1 && length(var.environment_name) <= 64
    error_message = "Environment name must be between 1 and 64 characters."
  }
}

variable "location" {
  description = "Primary location for all resources"
  type        = string

  validation {
    condition     = length(var.location) >= 1
    error_message = "Location must not be empty."
  }
}

variable "principal_id" {
  description = "Id of the user or app to assign application roles"
  type        = string
  default     = ""
}

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "app_service_plan_sku" {
  description = "App Service Plan SKU"
  type        = string
  default     = "B1"
}

variable "python_version" {
  description = "Python version for the App Service"
  type        = string
  default     = "3.12"
}

variable "existing_resource_group_name" {
  description = "Name of an existing resource group to use (leave empty to create a new one)"
  type        = string
}

variable "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint URL: 'https://{your-custom-endpoint}.openai.azure.com/'"
  type        = string
}

variable "azure_openai_model" {
  description = "Azure OpenAI model deployment name: 'your-deployment-name'"
  type        = string
}

variable "azure_openai_api_key" {
  description = "Azure OpenAI API key (leave empty to use managed identity)"
  type        = string
  sensitive   = true
}

variable "azure_openai_api_version" {
  description = "Azure OpenAI API version: '2024-02-15-preview'"
  type        = string
}

variable "azure_sql_connectionstring" {
  description = "Azure SQL connection string: 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:<yourserver>.database.usgovcloudapi.net,1433;Database=<yourdatabase>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;Authentication=ActiveDirectoryMsi'"
  type        = string
  sensitive   = true
}

variable "search_service_endpoint" {
  description = "Azure AI Search service endpoint: 'https://{your-custom-endpoint}.search.azure.us'"
  type        = string
}

variable "search_index_name" {
  description = "Azure AI Search index name: 'your-index-name'"
  type        = string
}

variable "search_api_key" {
  description = "Azure AI Search API key (leave empty to use managed identity)"
  type        = string
  sensitive   = true
}
