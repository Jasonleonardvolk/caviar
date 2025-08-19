#!/bin/bash
# Installation script for build-vector-index

set -e

echo "Installing Tori Vector Index Builder..."

# Create necessary directories
sudo mkdir -p /opt/tori/maintenance
sudo mkdir -p /opt/tori/mesh/hot
sudo mkdir -p /opt/tori/.cache/huggingface
sudo mkdir -p /var/log

# Copy the vector index builder script
sudo cp build_vector_index.py /opt/tori/maintenance/
sudo chmod +x /opt/tori/maintenance/build_vector_index.py

# Copy systemd files
sudo cp tori-vector-index.service /etc/systemd/system/
sudo cp tori-vector-index.timer /etc/systemd/system/

# Create tori user if it doesn't exist
if ! id "tori" &>/dev/null; then
    echo "Creating tori user..."
    sudo useradd -r -s /bin/false -d /opt/tori tori
fi

# Set ownership
sudo chown -R tori:tori /opt/tori/mesh
sudo chown -R tori:tori /opt/tori/.cache
sudo chown tori:tori /opt/tori/maintenance/build_vector_index.py

# Install Python dependencies for tori user
echo "Installing Python dependencies..."
sudo -u tori pip install --user sentence-transformers faiss-cpu numpy

# Reload systemd and enable timer
sudo systemctl daemon-reload
sudo systemctl enable tori-vector-index.timer
sudo systemctl start tori-vector-index.timer

echo "Installation complete!"
echo ""
echo "Status check commands:"
echo "  sudo systemctl status tori-vector-index.timer"
echo "  sudo systemctl list-timers tori-vector-index.timer"
echo "  sudo journalctl -u tori-vector-index.service"
echo ""
echo "Manual run commands:"
echo "  sudo systemctl start tori-vector-index.service"
echo "  sudo -u tori /opt/tori/maintenance/build_vector_index.py"
echo ""
echo "The vector index will be rebuilt every Sunday at 02:00 CST"
