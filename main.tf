# Configure the Azure Provider
provider "azurerm" {
  features {}
}

# Create a Resource Group
resource "azurerm_resource_group" "project_rg" {
  name     = "rg-optitask-enterprise"
  location = "East US"
}

# Create an App Service Plan (F1 is free for students)
resource "azurerm_service_plan" "plan" {
  name                = "plan-optitask"
  resource_group_name = azurerm_resource_group.project_rg.name
  location            = azurerm_resource_group.project_rg.location
  os_type             = "Linux"
  sku_name            = "F1"
}

# Create the Web App for Containers
resource "azurerm_linux_web_app" "app" {
  name                = "optitask-app"
  resource_group_name = azurerm_resource_group.project_rg.name
  location            = azurerm_service_plan.plan.location
  service_plan_id     = azurerm_service_plan.plan.id

  site_config {
    application_stack {
      docker_image     = "your-docker-hub-username/optitask"
      docker_image_tag = "latest"
    }
  }
}