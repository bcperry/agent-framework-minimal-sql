output "app_service_id" {
  description = "The ID of the App Service"
  value       = azurerm_linux_web_app.app_service.id
}

output "app_service_name" {
  description = "The name of the App Service"
  value       = azurerm_linux_web_app.app_service.name
}

output "app_service_url" {
  description = "The URL of the App Service"
  value       = "https://${azurerm_linux_web_app.app_service.default_hostname}"
}

output "app_service_plan_id" {
  description = "The ID of the App Service Plan"
  value       = azurerm_service_plan.app_service_plan.id
}
