# Azure Infrastructure with Terraform

This directory contains Terraform configuration files to deploy Azure resources for the FIAP TechChallenge4 Prediction project.

## Resources Created

- **Azure Resource Group**: Container for all project resources
- **Azure Static Web App**: Hosting platform for the frontend application

## Prerequisites

1. **Azure CLI**: Install and authenticate with Azure
   ```powershell
   az login
   ```

2. **Terraform**: Install Terraform (if not already installed)
   ```powershell
   winget install Hashicorp.Terraform
   ```

## Files Structure

- `main.tf`: Main Terraform configuration with provider and resource definitions
- `variables.tf`: Variable definitions with default values
- `outputs.tf`: Output values displayed after deployment
- `README.md`: This documentation file

## Deployment Instructions

1. **Initialize Terraform** (if not already done):
   ```powershell
   terraform init
   ```

2. **Validate the configuration**:
   ```powershell
   terraform validate
   ```

3. **Plan the deployment**:
   ```powershell
   terraform plan
   ```
   You will be prompted to enter a name for the resource group.

4. **Apply the configuration**:
   ```powershell
   terraform apply -auto-approve
   ```

## Variables

### Required Variables (will prompt for input)
- `resource_group_name`: Name of the Azure Resource Group

### Optional Variables (with defaults)
- `location`: Azure region (default: "East US")
- `static_web_app_name`: Name of the Static Web App (default: "fiap-prediction-swa")
- `static_web_app_sku_tier`: SKU tier (default: "Free")
- `static_web_app_sku_size`: SKU size (default: "Free")
- `common_tags`: Tags applied to all resources

## Customizing Variables

You can override default values by:

1. **Creating a terraform.tfvars file**:
   ```hcl
   resource_group_name = "my-resource-group"
   location = "West Europe"
   static_web_app_name = "my-custom-swa"
   ```

2. **Using command line flags**:
   ```powershell
   terraform apply -var="resource_group_name=my-rg" -var="location=West Europe"
   ```

3. **Using environment variables**:
   ```powershell
   $env:TF_VAR_resource_group_name = "my-resource-group"
   terraform apply
   ```

## Outputs

After successful deployment, you'll see:
- Resource Group name and location
- Static Web App name and URL
- API key for deployment (marked as sensitive)

## Cleanup

To destroy all created resources:
```powershell
terraform destroy
```