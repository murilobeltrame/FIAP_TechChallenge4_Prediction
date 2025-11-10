# GitHub Actions CI/CD Setup

This repository contains automated CI/CD pipelines split into focused workflows for each component. Each workflow only triggers when changes are detected in its respective folder.

## Workflows

### 1. `deploy-infrastructure.yml` - Infrastructure Deployment
- **Triggers**: Changes in `environment/` folder
- **Features**:
  - Terraform validation and deployment
  - Creates Azure Resource Group, ACR, and Static Web App
  - Only applies changes on push (not PRs)
  - Displays Azure Portal links

### 2. `deploy-backend.yml` - Backend Deployment  
- **Triggers**: Changes in `backend/` folder
- **Features**:
  - Python testing with pytest
  - Security scanning with Snyk
  - Docker image build and push to ACR
  - Requires existing infrastructure

### 3. `deploy-frontend.yml` - Frontend Deployment
- **Triggers**: Changes in `frontend/` folder  
- **Features**:
  - Node.js build with npm
  - Linting and security scanning
  - Deployment to Azure Static Web Apps
  - Requires existing infrastructure

### 4. `ci-cd.yml` - Deployment Overview
- **Triggers**: All pushes and PRs
- **Features**:
  - Shows which components changed
  - Provides deployment status overview
  - Links to individual workflows

### 5. `destroy.yml` - Infrastructure Cleanup
- **Triggers**: Manual workflow dispatch
- **Features**:
  - Terraform destroy with confirmation
  - Fallback Azure CLI cleanup
  - Requires typing "destroy" to confirm

## Required Secrets

Set up these secrets in your GitHub repository:

### Azure Authentication
```
AZURE_CREDENTIALS
```
Create this using:
```bash
az ad sp create-for-rbac --name "github-actions" --role contributor \
  --scopes /subscriptions/{subscription-id} \
  --sdk-auth
```

### Static Web App Token
```
AZURE_STATIC_WEB_APPS_API_TOKEN
```
Get this from Azure Portal → Static Web App → Manage deployment token

### Snyk Security Scanning (Optional)
```
SNYK_TOKEN
```
Get from [Snyk.io](https://snyk.io/) for security vulnerability scanning

## Deployment Flow

### 📁 Folder-Based Triggers
Each folder triggers its specific workflow:

```bash
# Infrastructure changes
git add environment/
git commit -m "Update Terraform config"
git push origin main
# → Triggers: Infrastructure Deployment

# Backend changes  
git add backend/
git commit -m "Update API endpoints"
git push origin main
# → Triggers: Backend Deployment

# Frontend changes
git add frontend/
git commit -m "Update UI components" 
git push origin main
# → Triggers: Frontend Deployment
```

### 🔄 Workflow Dependencies
- **Backend & Frontend** require existing infrastructure
- Workflows automatically detect existing Azure resources
- If no infrastructure exists, workflows will fail with helpful error messages

## Deployment Order

### First Time Setup:
1. **Deploy Infrastructure**: Change something in `environment/` folder
2. **Deploy Backend**: Change something in `backend/` folder  
3. **Deploy Frontend**: Change something in `frontend/` folder

### Ongoing Development:
- Make changes to any folder independently
- Only the affected component will redeploy
- Other components remain untouched

## Resource Naming Convention

- **Infrastructure**: `rg-fiap-prediction-{run-number}`
- **Docker Images**: `fiap-prediction-backend:{sha}` and `fiap-prediction-backend:latest`

## Security Features

- ✅ **Folder Isolation**: Changes in one folder don't affect others
- ✅ **Secrets Management**: All sensitive data stored as GitHub secrets
- ✅ **Vulnerability Scanning**: Snyk integration for each component
- ✅ **SARIF Upload**: Security results uploaded to GitHub Security tab
- ✅ **Test Gates**: Backend tests must pass before deployment

## Troubleshooting

### Common Issues

1. **"No infrastructure found" Error**
   - Deploy infrastructure first by making changes to `environment/` folder
   - Verify Azure resources exist in your subscription

2. **ACR Authentication Failed**
   - Verify Azure credentials are correct
   - Ensure service principal has ACR push permissions

3. **Static Web App Deployment Failed**
   - Check if `AZURE_STATIC_WEB_APPS_API_TOKEN` is set correctly
   - Verify the token hasn't expired

### Debug Steps

1. Check the **Deployment Overview** workflow for change detection
2. Look at individual workflow logs for detailed error messages  
3. Verify all required secrets are set in repository settings
4. Test Terraform/builds locally first

## Production Considerations

1. **Remote State**: Configure Terraform backend for state management
2. **Environments**: Create separate branches/workflows for dev/staging/prod
3. **Approval Gates**: Add manual approval steps for production deployments
4. **Rollback**: Implement rollback strategies in workflows
5. **Monitoring**: Add deployment notifications and health checks

## Workflow Benefits

- 🎯 **Focused Deployments**: Only deploy what changed
- ⚡ **Faster CI/CD**: No unnecessary builds or deployments  
- 🔍 **Clear Separation**: Each component has its own pipeline
- 🛡️ **Isolated Testing**: Component-specific tests and security scans
- 📊 **Better Visibility**: Easy to see what's deploying and why

## Workflows

### 1. `ci-cd.yml` - Automatic Deployment
- **Triggers**: Push to `main`/`develop` branches, Pull Requests to `main`
- **Features**:
  - Detects changes in `frontend/`, `backend/`, and `environment/` folders
  - Only deploys components that have changes
  - Runs security scanning with Snyk
  - Parallel execution where possible

### 2. `manual-deploy.yml` - Manual Deployment
- **Triggers**: Manual workflow dispatch
- **Features**:
  - Selective deployment of components
  - Custom resource group naming
  - Override automatic change detection

## Required Secrets

Set up these secrets in your GitHub repository:

### Azure Authentication
```
AZURE_CREDENTIALS
```
Create this using:
```bash
az ad sp create-for-rbac --name "github-actions" --role contributor \
  --scopes /subscriptions/{subscription-id} \
  --sdk-auth
```

### Individual Azure Secrets (Alternative)
```
AZURE_SUBSCRIPTION_ID
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET
AZURE_TENANT_ID
```

### Static Web App Token
```
AZURE_STATIC_WEB_APPS_API_TOKEN
```
Get this from Azure Portal → Static Web App → Manage deployment token

### Snyk Security Scanning (Optional)
```
SNYK_TOKEN
```
Get from [Snyk.io](https://snyk.io/) for security vulnerability scanning

## Deployment Flow

### Infrastructure (environment/)
1. **Setup**: Terraform initialization and validation
2. **Plan**: Creates deployment plan with dynamic resource group naming
3. **Apply**: Deploys Azure resources (Resource Group, ACR, Static Web App)
4. **Output**: Captures resource details for subsequent deployments

### Backend (backend/)
1. **Setup**: Python environment and Poetry dependencies
2. **Test**: Runs pytest test suite
3. **Build**: Creates Docker image
4. **Deploy**: Pushes to Azure Container Registry

### Frontend (frontend/)
1. **Setup**: Node.js environment and npm dependencies
2. **Build**: Creates production build with Vite
3. **Deploy**: Uploads to Azure Static Web Apps

## Usage Examples

### Automatic Deployment
Just push changes to the respective folders:
```bash
# Deploy only frontend
git add frontend/
git commit -m "Update frontend components"
git push origin main

# Deploy only backend
git add backend/
git commit -m "Update API endpoints"
git push origin main

# Deploy infrastructure
git add environment/
git commit -m "Update Terraform configuration"
git push origin main
```

### Manual Deployment
1. Go to Actions tab in GitHub
2. Select "Manual Deployment" workflow
3. Click "Run workflow"
4. Choose components to deploy
5. Optionally specify custom resource group name

## Resource Naming Convention

- **Automatic**: `rg-fiap-prediction-{run-number}`
- **Manual**: `rg-fiap-prediction-manual` (default) or custom name

## Security Features

- ✅ **Secrets Management**: All sensitive data stored as GitHub secrets
- ✅ **Vulnerability Scanning**: Snyk integration for dependency scanning
- ✅ **SARIF Upload**: Security results uploaded to GitHub Security tab
- ✅ **Least Privilege**: Service Principal with minimal required permissions

## Troubleshooting

### Common Issues

1. **ACR Authentication Failed**
   - Verify Azure credentials are correct
   - Ensure service principal has ACR push permissions

2. **Static Web App Deployment Failed**
   - Check if `AZURE_STATIC_WEB_APPS_API_TOKEN` is set correctly
   - Verify the token hasn't expired

3. **Terraform State Conflicts**
   - Consider using remote state storage for production
   - Current setup uses local state (for simplicity)

### Debug Steps

1. Check Actions logs for detailed error messages
2. Verify all required secrets are set
3. Ensure Azure service principal has correct permissions
4. Test Terraform locally first

## Production Considerations

1. **Remote State**: Configure Terraform backend for state management
2. **Environments**: Separate workflows for dev/staging/prod
3. **Approval Gates**: Add manual approval steps for production
4. **Rollback**: Implement rollback strategies
5. **Monitoring**: Add deployment notifications and health checks

## Security Best Practices

- Use least privilege access for service principals
- Rotate secrets regularly
- Enable branch protection rules
- Review and approve PRs before merging
- Monitor security scan results