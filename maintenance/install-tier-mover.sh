#!/bin/bash
# Installation script for tori-tier-mover

set -e

echo "Installing Tori Tier Mover..."

# Create necessary directories
sudo mkdir -p /opt/tori/maintenance
sudo mkdir -p /opt/tori/mesh/{hot,warm,cold}
sudo mkdir -p /var/log

# Copy the tier mover script
sudo cp tier_mover.py /opt/tori/maintenance/
sudo chmod +x /opt/tori/maintenance/tier_mover.py

# Copy systemd files
sudo cp tori-tier-mover.service /etc/systemd/system/
sudo cp tori-tier-mover.timer /etc/systemd/system/

# Create tori user if it doesn't exist
if ! id "tori" &>/dev/null; then
    echo "Creating tori user..."
    sudo useradd -r -s /bin/false -d /opt/tori tori
fi

# Set ownership
sudo chown -R tori:tori /opt/tori/mesh
sudo chown tori:tori /opt/tori/maintenance/tier_mover.py

# Reload systemd and enable timer
sudo systemctl daemon-reload
sudo systemctl enable tori-tier-mover.timer
sudo systemctl start tori-tier-mover.timer

echo "Installation complete!"
echo ""
echo "Status check commands:"
echo "  sudo systemctl status tori-tier-mover.timer"
echo "  sudo systemctl list-timers tori-tier-mover.timer"
echo "  sudo journalctl -u tori-tier-mover.service"
echo ""
echo "Manual run command:"
echo "  sudo systemctl start tori-tier-mover.service"
