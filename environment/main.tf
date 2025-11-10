# Configure the Azure Provider
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
}

# Create a resource group
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = var.common_tags
}

# Create a static web app
resource "azurerm_static_web_app" "main" {
  name                = var.static_web_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku_tier            = var.static_web_app_sku_tier
  sku_size            = var.static_web_app_sku_size

  tags = var.common_tags
}

# Create an Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = var.acr_admin_enabled

  # Network access rules
  public_network_access_enabled = var.acr_public_network_access_enabled
  network_rule_bypass_option    = var.acr_network_rule_bypass

  tags = var.common_tags
}