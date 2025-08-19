#!/bin/bash
# Grafana Dashboard Snapshot Tool
#
# This script pulls current Grafana dashboards and stores them in the 
# repository for version tracking. It should be configured to run:
# 1. On CI/CD pipeline after dashboard changes
# 2. After merging a Kaizen ticket
# 3. Weekly to catch manual changes
#
# Usage: ./grafana-snapshot.sh [--all|--uid UID1,UID2,...]

# Configuration
GRAFANA_URL=${GRAFANA_URL:-"http://localhost:3000"}
GRAFANA_API_KEY=${GRAFANA_API_KEY:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"docs/dashboards"}
COMMIT_MESSAGE=${COMMIT_MESSAGE:-"Update Grafana dashboard snapshots"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Default to phasemonitor dashboard if no parameters
DASHBOARD_UIDS=${1:-"phasemonitor"}

# Function to get dashboard
get_dashboard() {
  local uid=$1
  local output_file="$OUTPUT_DIR/$uid.json"
  
  echo "Fetching dashboard $uid..."
  
  # Call Grafana API to get dashboard
  curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" \
       "$GRAFANA_URL/api/dashboards/uid/$uid" | jq '.dashboard' > "$output_file"
  
  if [ $? -eq 0 ] && [ -s "$output_file" ]; then
    echo "Successfully saved dashboard $uid to $output_file"
    return 0
  else
    echo "Failed to fetch dashboard $uid"
    return 1
  fi
}

# Function to get all dashboards
get_all_dashboards() {
  echo "Fetching all dashboards..."
  
  # Get list of dashboard UIDs
  local uids=$(curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" \
              "$GRAFANA_URL/api/search?type=dash-db" | jq -r '.[].uid')
  
  for uid in $uids; do
    get_dashboard "$uid"
  done
}

# Main logic
if [ "$DASHBOARD_UIDS" == "--all" ]; then
  get_all_dashboards
else
  # Split comma-separated UIDs
  IFS=',' read -ra UIDS <<< "$DASHBOARD_UIDS"
  for uid in "${UIDS[@]}"; do
    get_dashboard "$uid"
  done
fi

# Create a timestamp file to track last update
date -u > "$OUTPUT_DIR/.last_updated"

# If running in CI with KAIZEN_TICKET environment variable, add to commit message
if [ -n "$KAIZEN_TICKET" ]; then
  COMMIT_MESSAGE="$COMMIT_MESSAGE for $KAIZEN_TICKET"
fi

# If git is available and we're in a git repo, commit changes
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
  echo "Committing changes to git..."
  git add "$OUTPUT_DIR"
  git commit -m "$COMMIT_MESSAGE"
  echo "Changes committed with message: $COMMIT_MESSAGE"
else
  echo "Not in a git repository or git not available. Skipping commit."
fi

echo "Dashboard snapshot complete."
