#!/bin/bash
# 
# Tag Release Script for Kaizen Loop & Phase Metrics
#
# This script creates a Git tag for the current release version
# and pushes it to the remote repository
#
# Usage: ./scripts/tag_release.sh [optional custom message]

# Default values
TAG_NAME="v1.1-kaizen"
DEFAULT_MESSAGE="Kaizen Loop & Phase Metrics prod release"

# Use custom message if provided
if [ $# -gt 0 ]; then
  MESSAGE="$*"
else
  MESSAGE="$DEFAULT_MESSAGE"
fi

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating release tag: ${TAG_NAME}${NC}"
echo -e "Message: ${MESSAGE}"
echo

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
  echo -e "${RED}Warning: You have uncommitted changes.${NC}"
  echo "It's recommended to commit all changes before tagging."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 1
  fi
fi

# Create annotated tag
git tag -a "$TAG_NAME" -m "$MESSAGE"

if [ $? -ne 0 ]; then
  echo -e "${RED}Error: Failed to create tag.${NC}"
  exit 1
fi

echo -e "${GREEN}Tag created successfully.${NC}"

# Push tag to remote
read -p "Push tag to remote repository? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git push origin "$TAG_NAME"
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to push tag.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Tag pushed successfully to remote repository.${NC}"
fi

# Print next steps
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Complete the Phase-0 burn-in checklist"
echo "2. Capture Grafana snapshot after 24 hours:"
echo "   ./scripts/grafana_snapshot.sh kaizen-baseline.json"
echo "3. Schedule the Chaos Drill for Week 2"
echo
echo -e "${GREEN}Kaizen Loop & Phase Metrics v1.1 is now ready for deployment.${NC}"
