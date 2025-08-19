#!/bin/bash
# Quick Start Script for Enhanced TORI Holographic System

echo "🚀 TORI Holographic Enhancement Quick Start"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found!${NC}"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo -e "${GREEN}✓ Node.js found${NC}"

# Make integration script executable
chmod +x integrate_enhancements.js

# Run integration
echo -e "\n${BLUE}Starting integration...${NC}"
node integrate_enhancements.js

# Check if integration succeeded
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Integration successful!${NC}"
    
    # Offer to start dev server
    echo -e "\n${YELLOW}Would you like to start the development server? (y/n)${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cd ../tori_ui_svelte
        echo -e "${BLUE}Starting development server...${NC}"
        npm run dev
    else
        echo -e "\n${BLUE}To start manually:${NC}"
        echo "  cd ../tori_ui_svelte"
        echo "  npm run dev"
    fi
else
    echo -e "\n${RED}❌ Integration failed!${NC}"
    echo "Please check the error messages above."
fi
