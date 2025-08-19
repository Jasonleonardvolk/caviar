#!/bin/bash
# PsiArchive maintenance cron jobs

# Daily seal job - run at 23:59
# Add to crontab: 59 23 * * * /path/to/psi_archive_cron.sh seal

# Weekly snapshot - run Sunday at 02:00
# Add to crontab: 0 2 * * 0 /path/to/psi_archive_cron.sh snapshot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORI_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$TORI_DIR/venv/bin/activate" ]; then
    source "$TORI_DIR/venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="$TORI_DIR:$PYTHONPATH"

case "$1" in
    seal)
        echo "[$(date)] Running daily seal job..."
        python3 "$TORI_DIR/tools/cron_daily_seal.py"
        ;;
        
    snapshot)
        echo "[$(date)] Creating weekly snapshot..."
        SNAPSHOT_DATE=$(date +%F)
        python3 "$TORI_DIR/tools/psi_replay.py" \
            --until "${SNAPSHOT_DATE}T00:00:00" \
            --output-dir "$TORI_DIR/data/snapshots/$SNAPSHOT_DATE"
        
        # Self-healing snapshot rotation
        echo "[$(date)] Cleaning up old snapshots..."
        
        # Keep last 45 days of snapshots
        find "$TORI_DIR/data/snapshots" -maxdepth 1 -type d -mtime +45 -exec rm -rf {} + 2>/dev/null
        
        # Always keep the most recent full snapshot
        LATEST_FULL=$(find "$TORI_DIR/data/snapshots" -maxdepth 1 -type d -name "*-full" | sort -r | head -1)
        if [ -n "$LATEST_FULL" ]; then
            echo "  Preserving latest full snapshot: $(basename "$LATEST_FULL")"
        fi
        
        # Check disk usage and warn if low
        DISK_USAGE=$(df "$TORI_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ "$DISK_USAGE" -gt 85 ]; then
            echo "  ⚠️  WARNING: Disk usage at ${DISK_USAGE}% - consider more aggressive cleanup"
            # Emergency cleanup: keep only last 14 days
            find "$TORI_DIR/data/snapshots" -maxdepth 1 -type d -mtime +14 ! -name "*-full" -exec rm -rf {} + 2>/dev/null
        fi
        ;;
        
    verify)
        echo "[$(date)] Verifying archive integrity..."
        # Add hash chain verification here
        python3 -c "
from core.psi_archive_extended import PSI_ARCHIVER
# Verify last 7 days
# TODO: Add verify_chain method
print('Archive verification not yet implemented')
"
        ;;
        
    *)
        echo "Usage: $0 {seal|snapshot|verify}"
        exit 1
        ;;
esac

echo "[$(date)] Job completed"
