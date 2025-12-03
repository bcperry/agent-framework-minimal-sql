# Terraform Infrastructure

This directory contains Terraform configuration files for deploying the agent framework to Azure.

## Structure

```
infra/
├── main.tf                          # Main infrastructure configuration
├── variables.tf                     # Input variable definitions
├── outputs.tf                       # Output values
├── provider.tf                      # Provider configurations
├── terraform.tfvars.example         # Example variable values
└── modules/
    ├── managed-identity/            # Managed Identity module
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── app-service/                 # App Service module
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Resources Created

- **Resource Group**: Container for all resources
- **Managed Identity**: User-assigned identity for the app
- **App Service Plan**: Linux-based hosting plan
- **App Service**: Web app for running the Chainlit application

## Deployment with Azure Developer CLI

The Azure Developer CLI (azd) automatically uses Terraform when configured in `azure.yaml`:

```bash
# Initialize the environment
azd init

# Provision and deploy
azd up
```

The `azure.yaml` file is configured to use Terraform:

```yaml
infra:
  provider: terraform
  path: infra
```

## Manual Terraform Deployment

If you prefer to use Terraform directly:

```bash
# Navigate to the infra directory
cd infra

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var="environment_name=dev" -var="location=eastus" -var="subscription_id=YOUR_SUBSCRIPTION_ID"

# Apply the configuration
terraform apply -var="environment_name=dev" -var="location=eastus" -var="subscription_id=YOUR_SUBSCRIPTION_ID"
```

## Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `environment_name` | Name of the environment | - | Yes |
| `location` | Azure region | - | Yes |
| `subscription_id` | Azure subscription ID | "" | No* |
| `principal_id` | User/app principal ID | "" | No |
| `app_service_plan_sku` | App Service Plan SKU | "B1" | No |
| `python_version` | Python version | "3.12" | No |

*Required for azd deployments, handled automatically

## Outputs

- `AZURE_LOCATION`: Deployment region
- `AZURE_TENANT_ID`: Azure tenant ID
- `RESOURCE_GROUP_NAME`: Resource group name
- `WEB_APP_NAME`: App Service name
- `WEB_APP_URL`: App Service URL
- `MANAGED_IDENTITY_CLIENT_ID`: Managed identity client ID
- `MANAGED_IDENTITY_PRINCIPAL_ID`: Managed identity principal ID
