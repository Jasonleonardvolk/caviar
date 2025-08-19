#!/bin/bash
# Quick fix script for common TypeScript errors in the KHA project
# Generated from photoMorphPipeline.ts fix analysis

echo "KHA TypeScript Error Quick Fix Script"
echo "======================================"
echo ""

# Function to fix GPUTexture.size references
fix_gpu_texture_size() {
    echo "Fixing GPUTexture.size references..."
    
    # Common patterns to replace
    # field.size[0] -> field.width
    # field.size[1] -> field.height
    # texture.size -> [texture.width, texture.height]
    
    find . -name "*.ts" -type f -exec sed -i.bak \
        -e 's/\.size\[0\]/.width/g' \
        -e 's/\.size\[1\]/.height/g' \
        -e 's/size: \([a-zA-Z]*\)\.size,/size: [\1.width, \1.height],/g' \
        {} \;
}

# Function to remove writeTimestamp calls
fix_write_timestamp() {
    echo "Removing deprecated writeTimestamp calls..."
    
    # Comment out writeTimestamp lines
    find . -name "*.ts" -type f -exec sed -i.bak \
        -e 's/^\(.*\)\.writeTimestamp/\/\/ DEPRECATED: \1.writeTimestamp/g' \
        {} \;
}

# Function to add fragment targets
fix_fragment_targets() {
    echo "Adding missing fragment targets..."
    
    # This is more complex and might need manual review
    # Pattern: fragment: { module: X, entryPoint: Y }
    # Should become: fragment: { module: X, entryPoint: Y, targets: [{ format: 'rgba8unorm' }] }
    
    echo "Note: Fragment targets need manual review - check render pipelines"
}

# Function to fix uninitialized properties
fix_uninitialized_properties() {
    echo "Fixing uninitialized properties..."
    echo "Note: Add '!' to properties initialized in async methods:"
    echo "  private myProperty!: Type;"
}

# Main execution
echo "Starting fixes..."
echo ""

# Backup original files
echo "Creating backups..."
mkdir -p ./backup_$(date +%Y%m%d_%H%M%S)
cp -r ./frontend ./backup_$(date +%Y%m%d_%H%M%S)/
cp -r ./tori_ui_svelte ./backup_$(date +%Y%m%d_%H%M%S)/

# Apply fixes
fix_gpu_texture_size
fix_write_timestamp
fix_fragment_targets
fix_uninitialized_properties

echo ""
echo "Quick fixes applied!"
echo "Please review the changes and run:"
echo "  npm run build"
echo "  npm run type-check"
echo ""
echo "Backups created in ./backup_* directory"
