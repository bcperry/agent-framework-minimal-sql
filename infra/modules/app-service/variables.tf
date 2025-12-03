variable "name" {
  description = "The name of the App Service"
  type        = string
}

variable "location" {
  description = "The location of the App Service"
  type        = string
}

variable "tags" {
  description = "Tags to apply to the App Service (deprecated, use app_service_tags)"
  type        = map(string)
  default     = {}
}

variable "app_service_tags" {
  description = "Tags to apply to the App Service"
  type        = map(string)
  default     = null
}

variable "app_service_plan_tags" {
  description = "Tags to apply to the App Service Plan"
  type        = map(string)
  default     = null
}

variable "app_service_plan_name" {
  description = "The name of the App Service Plan"
  type        = string
}

variable "app_service_plan_sku" {
  description = "The SKU of the App Service Plan"
  type        = string
  default     = "B1"
}

variable "python_version" {
  description = "The Python version to use"
  type        = string
  default     = "3.12"
}

variable "managed_identity_id" {
  description = "The managed identity resource ID"
  type        = string
}

variable "managed_identity_client_id" {
  description = "The managed identity client ID"
  type        = string
}

variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
}

variable "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint URL"
  type        = string
}

variable "azure_openai_model" {
  description = "Azure OpenAI model deployment name"
  type        = string
}

variable "azure_openai_api_key" {
  description = "Azure OpenAI API key"
  type        = string
  sensitive   = true
}

variable "azure_openai_api_version" {
  description = "Azure OpenAI API version"
  type        = string
}

variable "azure_sql_connectionstring" {
  description = "Azure SQL connection string"
  type        = string
  sensitive   = true
}

variable "search_service_endpoint" {
  description = "Azure AI Search service endpoint"
  type        = string
}

variable "search_index_name" {
  description = "Azure AI Search index name"
  type        = string
}

variable "search_api_key" {
  description = "Azure AI Search API key"
  type        = string
  sensitive   = true
}
