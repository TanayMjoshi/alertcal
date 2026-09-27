#!/usr/bin/env bash
# Deploy the alertcal API to Azure Container Apps. Day 2. Needs `az login` first.
#
# What happens, in order:
#   1. resource group in germanywestcentral
#   2. Azure Container Registry (Basic)
#   3. `az acr build`: the image is built IN the registry from this folder; no local Docker needed
#   4. Container Apps environment (creates a Log Analytics workspace for you)
#   5. the container app: external HTTPS ingress on port 8000, scale 0..1, pulls with a managed identity
#   6. prints the public URL; /docs is the Swagger UI
#
# Override any name with an env var, e.g.  ACR=myregistry123 azure/deploy.sh
# Registry names must be globally unique, 5-50 lowercase alphanumerics.
set -euo pipefail

RG="${RG:-alertcal-rg}"
LOCATION="${LOCATION:-germanywestcentral}"
ACR="${ACR:-alertcal$(whoami | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')}"
ENV_NAME="${ENV_NAME:-alertcal-env}"
APP="${APP:-alertcal}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE="$ACR.azurecr.io/alertcal:$TAG"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

step() { printf '\n==> %s\n' "$*"; }

step "checking az login"
az account show --query '{subscription:name, user:user.name}' -o table

step "container apps extension + resource providers (one-off, safe to repeat)"
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

step "1/6 resource group $RG in $LOCATION"
az group create --name "$RG" --location "$LOCATION" -o none

step "2/6 container registry $ACR"
az acr create --name "$ACR" --resource-group "$RG" --sku Basic -o none

step "3/6 build image $IMAGE in the registry (no local push)"
az acr build --registry "$ACR" --image "alertcal:$TAG" "$ROOT"

step "4/6 container apps environment $ENV_NAME"
az containerapp env create --name "$ENV_NAME" --resource-group "$RG" --location "$LOCATION" -o none

step "5/6 container app $APP"
if az containerapp show --name "$APP" --resource-group "$RG" -o none 2>/dev/null; then
  az containerapp update --name "$APP" --resource-group "$RG" --image "$IMAGE" -o none
else
  az containerapp create \
    --name "$APP" --resource-group "$RG" --environment "$ENV_NAME" \
    --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-identity system \
    --target-port 8000 --ingress external \
    --cpu 0.25 --memory 0.5Gi --min-replicas 0 --max-replicas 1 \
    -o none
fi

step "6/6 public URL"
FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "https://$FQDN/docs"
echo "https://$FQDN/health"

# ---------------------------------------------------------------------------------------------
# Teardown (everything above lives in one resource group, so one command removes it all):
#   az group delete --name alertcal-rg --yes --no-wait
# ---------------------------------------------------------------------------------------------
