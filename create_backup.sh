#!/bin/bash
# Backup script for enhanced_launcher.py

echo "Creating backup of enhanced_launcher.py..."
cp enhanced_launcher.py enhanced_launcher.py.original_prajna_backup
echo "Backup created: enhanced_launcher.py.original_prajna_backup"

echo ""
echo "To revert to the original (slow) Prajna API, run:"
echo "  cp enhanced_launcher.py.original_prajna_backup enhanced_launcher.py"
echo ""
echo "Current configuration uses quick_api_server.py for faster startup."
