#!/bin/bash
# Install Dickbox systemd services and timers

set -euo pipefail

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${SCRIPT_DIR}/../systemd"

echo "Installing Dickbox systemd services..."

# Install service files
for service in "$SERVICE_DIR"/*.service; do
    if [ -f "$service" ]; then
        echo "Installing $(basename "$service")"
        cp "$service" /etc/systemd/system/
        chmod 644 "/etc/systemd/system/$(basename "$service")"
    fi
done

# Install timer files
for timer in "$SERVICE_DIR"/*.timer; do
    if [ -f "$timer" ]; then
        echo "Installing $(basename "$timer")"
        cp "$timer" /etc/systemd/system/
        chmod 644 "/etc/systemd/system/$(basename "$timer")"
    fi
done

# Create required directories
echo "Creating directories..."
mkdir -p /opt/tori/{bin,lib,releases}
mkdir -p /var/log/tori
mkdir -p /var/run/tori
mkdir -p /etc/tori/{keys,zmq_keys}
mkdir -p /var/tmp/tori_energy

# Set permissions
echo "Setting permissions..."
if id "tori" &>/dev/null; then
    chown -R tori:tori /opt/tori
    chown -R tori:tori /var/log/tori
    chown -R tori:tori /var/run/tori
    chown -R tori:tori /etc/tori
    chown -R tori:tori /var/tmp/tori_energy
else
    echo "Warning: 'tori' user not found. Please create it and set permissions."
fi

# Install Python scripts
echo "Installing Python scripts..."
cp "${SCRIPT_DIR}/../zmq_key_rotation.py" /opt/tori/bin/rotate_zmq_keys.py
chmod 755 /opt/tori/bin/rotate_zmq_keys.py

# Create soliton-mps-keepalive from soliton_mps.py
python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/..')
from soliton_mps import PYTHON_KEEPALIVE
with open('/opt/tori/bin/soliton-mps-keepalive', 'w') as f:
    f.write('#!/usr/bin/env python3\\n' + PYTHON_KEEPALIVE)
"
chmod 755 /opt/tori/bin/soliton-mps-keepalive

# Create systemd slice hierarchy
echo "Creating systemd slices..."
cat > /etc/systemd/system/tori.slice << EOF
[Unit]
Description=TORI Parent Slice
Documentation=man:systemd.slice(5)
Before=slices.target

[Slice]
# No limits on parent slice
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable timers (but don't start)
echo "Enabling timers..."
systemctl enable zmq-key-rotate.timer || true

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Create the 'tori' user if it doesn't exist:"
echo "   useradd -r -s /bin/false -d /opt/tori tori"
echo ""
echo "2. Generate minisign keys for capsule signing:"
echo "   minisign -G -p /etc/tori/keys/minisign.pub -s /etc/tori/keys/minisign.key"
echo ""
echo "3. Start the ZMQ key rotation timer:"
echo "   systemctl start zmq-key-rotate.timer"
echo ""
echo "4. If using GPUs, ensure NVIDIA drivers and CUDA are installed"
echo ""
