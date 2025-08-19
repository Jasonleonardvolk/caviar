#!/bin/bash
# Schedule TORI compaction using cron
# Run as: ./schedule_compaction.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/compact_all_meshes.py"
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Find Python executable
if command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
elif command -v python &> /dev/null; then
    PYTHON_EXE="python"
else
    echo "Error: Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Script path: $PYTHON_SCRIPT"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Compaction script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Function to add cron job if not exists
add_cron_job() {
    local schedule="$1"
    local description="$2"
    local job="$schedule cd $SCRIPT_DIR && $PYTHON_EXE $PYTHON_SCRIPT >> $LOG_DIR/compaction.log 2>&1"
    
    # Check if job already exists
    if crontab -l 2>/dev/null | grep -q "$PYTHON_SCRIPT"; then
        echo "[EXISTS] Cron job already scheduled: $description"
    else
        # Add job
        (crontab -l 2>/dev/null; echo "# $description"; echo "$job") | crontab -
        if [ $? -eq 0 ]; then
            echo "[OK] Added cron job: $description"
        else
            echo "[ERROR] Failed to add cron job: $description"
            return 1
        fi
    fi
    return 0
}

echo -e "\nScheduling TORI compaction tasks..."

# Add midnight compaction
add_cron_job "0 0 * * *" "TORI Compaction - Midnight"

# Add noon compaction
add_cron_job "0 12 * * *" "TORI Compaction - Noon"

# Add weekly full compaction (Sunday 3am)
add_cron_job "0 3 * * 0" "TORI Compaction - Weekly full (Sunday 3am)"

# Show current cron jobs
echo -e "\nCurrent TORI compaction jobs:"
crontab -l 2>/dev/null | grep -A1 "TORI Compaction" || echo "No TORI jobs found"

# Create systemd timer as alternative (if systemd is available)
if command -v systemctl &> /dev/null; then
    echo -e "\nWould you like to create systemd timers as well? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        # Create service file
        SERVICE_FILE="/etc/systemd/system/tori-compaction.service"
        TIMER_FILE="/etc/systemd/system/tori-compaction.timer"
        
        # Check if running with sudo
        if [ "$EUID" -ne 0 ]; then
            echo "Need sudo to create systemd files. Re-run with: sudo $0"
        else
            # Create service
            cat > "$SERVICE_FILE" << EOF
[Unit]
Description=TORI Concept Mesh Compaction
After=network.target

[Service]
Type=oneshot
User=$SUDO_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_EXE $PYTHON_SCRIPT
StandardOutput=append:$LOG_DIR/compaction.log
StandardError=append:$LOG_DIR/compaction.log

[Install]
WantedBy=multi-user.target
EOF

            # Create timer
            cat > "$TIMER_FILE" << EOF
[Unit]
Description=TORI Compaction Timer
Requires=tori-compaction.service

[Timer]
# Run at midnight and noon
OnCalendar=00:00
OnCalendar=12:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

            # Enable and start timer
            systemctl daemon-reload
            systemctl enable tori-compaction.timer
            systemctl start tori-compaction.timer
            
            echo "[OK] Created systemd timer"
            echo "Status: systemctl status tori-compaction.timer"
            echo "Logs: journalctl -u tori-compaction.service"
        fi
    fi
fi

echo -e "\nSetup complete!"
echo -e "\nUseful commands:"
echo "  View cron jobs:  crontab -l"
echo "  Edit cron jobs:  crontab -e"
echo "  View logs:       tail -f $LOG_DIR/compaction.log"
echo "  Run manually:    cd $SCRIPT_DIR && $PYTHON_EXE $PYTHON_SCRIPT"
echo "  Force run:       cd $SCRIPT_DIR && $PYTHON_EXE $PYTHON_SCRIPT --force"

# Test run option
echo -e "\nWould you like to test run the compaction now? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    echo "Running compaction test..."
    cd "$SCRIPT_DIR" && $PYTHON_EXE "$PYTHON_SCRIPT" --force
fi
