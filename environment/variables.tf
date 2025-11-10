# Variable definitions for Azure Static Web App deployment

variable "resource_group_name" {
  description = "Name of the Azure Resource Group"
  type        = string
  # This will prompt the user to enter a value when running terraform plan/apply
}

variable "location" {
  description = "Azure region where resources will be deployed"
  type        = string
  default     = "East US 2"
}

variable "static_web_app_name" {
  description = "Name of the Static Web App"
  type        = string
  default     = "fiap-prediction-swa"
}

variable "static_web_app_sku_tier" {
  description = "SKU tier for the Static Web App"
  type        = string
  default     = "Free"
  
  validation {
    condition     = contains(["Free", "Standard"], var.static_web_app_sku_tier)
    error_message = "SKU tier must be either 'Free' or 'Standard'."
  }
}

variable "static_web_app_sku_size" {
  description = "SKU size for the Static Web App"
  type        = string
  default     = "Free"
  
  validation {
    condition     = contains(["Free", "Standard"], var.static_web_app_sku_size)
    error_message = "SKU size must be either 'Free' or 'Standard'."
  }
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default = {
    Environment = "Development"
    Project     = "FIAP_TechChallenge4_Prediction"
    ManagedBy   = "Terraform"
  }
}

# Azure Container Registry variables
variable "acr_name" {
  description = "Name of the Azure Container Registry"
  type        = string
  default     = "fiappredictionacr"
  
  validation {
    condition     = can(regex("^[a-zA-Z0-9]*$", var.acr_name)) && length(var.acr_name) >= 5 && length(var.acr_name) <= 50
    error_message = "ACR name must be 5-50 characters long and contain only alphanumeric characters."
  }
}

variable "acr_sku" {
  description = "SKU for the Azure Container Registry"
  type        = string
  default     = "Basic"
  
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be 'Basic', 'Standard', or 'Premium'."
  }
}

variable "acr_admin_enabled" {
  description = "Enable admin user for the Azure Container Registry"
  type        = bool
  default     = true
}

variable "acr_public_network_access_enabled" {
  description = "Enable public network access for the Azure Container Registry"
  type        = bool
  default     = true
}

variable "acr_network_rule_bypass" {
  description = "Network rule bypass option for the Azure Container Registry"
  type        = string
  default     = "AzureServices"
  
  validation {
    condition     = contains(["AzureServices", "None"], var.acr_network_rule_bypass)
    error_message = "Network rule bypass must be 'AzureServices' or 'None'."
  }
}