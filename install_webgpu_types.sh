#!/bin/bash

# Install WebGPU Types
echo -e "\033[36mInstalling @webgpu/types for TORI Project...\033[0m"

# Save current directory
ORIGINAL_DIR=$(pwd)

# Function to install in a directory
install_in_dir() {
    local dir=$1
    echo -e "\n\033[33mInstalling in $dir...\033[0m"
    cd "$dir" || exit
    npm install --save-dev @webgpu/types
}

# Install in all directories
install_in_dir "D:/Dev/kha"
install_in_dir "D:/Dev/kha/tori_ui_svelte"
install_in_dir "D:/Dev/kha/frontend"
install_in_dir "D:/Dev/kha/frontend/hybrid"

echo -e "\n\033[32m✅ WebGPU types installed successfully!\033[0m"

# Test type checking
echo -e "\n\033[36mTesting TypeScript compilation...\033[0m"
cd "D:/Dev/kha" || exit
npx tsc --noEmit

if [ $? -eq 0 ]; then
    echo -e "\033[32m✅ TypeScript compilation successful - No errors!\033[0m"
else
    echo -e "\033[33m⚠️ TypeScript compilation has some issues\033[0m"
fi

# Return to original directory
cd "$ORIGINAL_DIR" || exit

echo -e "\n\033[36mInstallation complete! You can now run:\033[0m"
echo "  npm run type-check"
echo "  npm run build"
