#!/usr/bin/env bash
set -euo pipefail

WEIGHT="${1:-10}"

if [[ "$WEIGHT" =~ ^[0-9]+$ ]] && (( WEIGHT >= 0 && WEIGHT <= 100 )); then
  :
else
  echo "Usage: $0 <weight-0-100>"
  exit 1
fi

kubectl patch ingress fraud-api-ingress-canary \
  -n default \
  --type merge \
  -p "{\"metadata\":{\"annotations\":{\"nginx.ingress.kubernetes.io/canary-weight\":\"${WEIGHT}\"}}}"

echo "Canary traffic set to ${WEIGHT}% for fraud-api-ingress-canary"

if (( WEIGHT == 100 )); then
  echo "Canary is fully promoted. You can now scale down the stable deployment if needed."
fi
