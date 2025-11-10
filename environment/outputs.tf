# Output values after successful deployment

output "resource_group_name" {
  description = "Name of the created Resource Group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Location of the created Resource Group"
  value       = azurerm_resource_group.main.location
}

output "static_web_app_name" {
  description = "Name of the created Static Web App"
  value       = azurerm_static_web_app.main.name
}

output "static_web_app_default_host_name" {
  description = "Default hostname of the Static Web App"
  value       = azurerm_static_web_app.main.default_host_name
}

output "static_web_app_url" {
  description = "URL of the Static Web App"
  value       = "https://${azurerm_static_web_app.main.default_host_name}"
}

output "static_web_app_api_key" {
  description = "API key for the Static Web App (sensitive)"
  value       = azurerm_static_web_app.main.api_key
  sensitive   = true
}

# Azure Container Registry outputs
output "acr_name" {
  description = "Name of the Azure Container Registry"
  value       = azurerm_container_registry.main.name
}

output "acr_login_server" {
  description = "Login server URL for the Azure Container Registry"
  value       = azurerm_container_registry.main.login_server
}

output "acr_admin_username" {
  description = "Admin username for the Azure Container Registry"
  value       = azurerm_container_registry.main.admin_username
}

output "acr_admin_password" {
  description = "Admin password for the Azure Container Registry (sensitive)"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "docker_login_command" {
  description = "Docker login command for the Azure Container Registry"
  value       = "docker login ${azurerm_container_registry.main.login_server} -u ${azurerm_container_registry.main.admin_username} -p ${azurerm_container_registry.main.admin_password}"
  sensitive   = true
}