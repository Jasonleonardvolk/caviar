#!/bin/bash
# beyond_metal_install.sh - One-click Beyond Metacognition setup for bare metal

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TORI_BASE="/opt/tori"
VENV_PATH="$TORI_BASE/venv"

echo "🌌 Beyond Metacognition - Bare Metal Installation"
echo "================================================"
echo

# Check if running as appropriate user
if [ "$USER" != "tori" ] && [ "$USER" != "root" ]; then
    echo "⚠️  Run as 'tori' user or root with sudo"
    exit 1
fi

# Step 1: Check prerequisites
echo "1️⃣ Checking prerequisites..."

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ TORI virtualenv not found at $VENV_PATH"
    exit 1
fi

if ! command -v ansible &> /dev/null; then
    echo "❌ Ansible not found. Install with: apt install ansible / yum install ansible"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Install with: apt install git / yum install git"
    exit 1
fi

echo "✅ Prerequisites OK"
echo

# Step 2: Install Python dependencies
echo "2️⃣ Installing Python dependencies..."

cat > /tmp/requirements-beyond.txt << EOF
numpy>=1.21.0
pyyaml
psutil
asyncio
EOF

if [ "$USER" = "root" ]; then
    sudo -u tori $VENV_PATH/bin/pip install -r /tmp/requirements-beyond.txt
else
    $VENV_PATH/bin/pip install -r /tmp/requirements-beyond.txt
fi

echo "✅ Python dependencies installed"
echo

# Step 3: Generate deployment package
echo "3️⃣ Generating deployment package..."

cd "$SCRIPT_DIR"
$VENV_PATH/bin/python beyond_deploy_bare_metal.py

# Get the latest release directory
LATEST_RELEASE=$(ls -dt $TORI_BASE/releases/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_RELEASE" ]; then
    # If releases dir doesn't exist yet, create it
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    LATEST_RELEASE="$TORI_BASE/releases/${GIT_SHA}_${TIMESTAMP}"
    
    if [ "$USER" = "root" ]; then
        sudo -u tori mkdir -p "$LATEST_RELEASE"
    else
        mkdir -p "$LATEST_RELEASE"
    fi
fi

echo "✅ Deployment package created: $LATEST_RELEASE"
echo

# Step 4: Copy Beyond files to release
echo "4️⃣ Copying Beyond Metacognition files..."

# List of files to copy
FILES=(
    "alan_backend/origin_sentry.py"
    "alan_backend/braid_aggregator.py"
    "python/core/braid_buffers.py"
    "python/core/observer_synthesis.py"
    "python/core/creative_feedback.py"
    "python/core/topology_tracker.py"
    "apply_beyond_patches.py"
    "verify_beyond_integration.py"
    "beyond_demo.py"
    "torictl.py"
    "beyond_diagnostics.py"
    "beyond_prometheus_exporter.py"
    "beyond_logging_config.py"
)

for file in "${FILES[@]}"; do
    src="$SCRIPT_DIR/$file"
    dst="$LATEST_RELEASE/kha/$file"
    
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        if [ "$USER" = "root" ]; then
            sudo -u tori cp "$src" "$dst"
        else
            cp "$src" "$dst"
        fi
        echo "  ✅ $file"
    else
        echo "  ⚠️  Missing: $file"
    fi
done

echo

# Step 5: Create default configs
echo "5️⃣ Creating default configuration..."

if [ "$USER" = "root" ]; then
    sudo -u tori mkdir -p "$TORI_BASE/conf"
else
    mkdir -p "$TORI_BASE/conf"
fi

# Create runtime.yaml if it doesn't exist
if [ ! -f "$TORI_BASE/conf/runtime.yaml" ]; then
    cat > "$TORI_BASE/conf/runtime.yaml" << EOF
beyond_metacognition:
  observer_synthesis:
    reflex_budget: 30
    measurement_cooldown_ms: 100
  creative_feedback:
    novelty_threshold_high: 0.8
    emergency_threshold: 0.06
  safety:
    auto_rollback_threshold: 0.08
EOF
    echo "  ✅ Created runtime.yaml"
fi

# Create data bootstrap
if [ "$USER" = "root" ]; then
    sudo -u tori mkdir -p "$TORI_BASE/data/bootstrap"
else
    mkdir -p "$TORI_BASE/data/bootstrap"
fi

echo '{"format_version": "1.0", "entries": []}' > "$TORI_BASE/data/bootstrap/lyapunov_watchlist.json"
echo '[]' > "$TORI_BASE/data/bootstrap/spectral_db.json"

echo "✅ Configuration created"
echo

# Step 6: Apply patches
echo "6️⃣ Applying Beyond Metacognition patches..."

cd "$LATEST_RELEASE"
$VENV_PATH/bin/python kha/apply_beyond_patches.py --verify

if [ $? -eq 0 ]; then
    echo "✅ Patches applied successfully"
else
    echo "❌ Patch application failed"
    exit 1
fi
echo

# Step 7: Setup systemd (if root)
if [ "$USER" = "root" ]; then
    echo "7️⃣ Installing systemd services..."
    
    # Generate systemd files
    $VENV_PATH/bin/python "$SCRIPT_DIR/beyond_deploy_bare_metal.py"
    
    # Install services
    cp "$LATEST_RELEASE/systemd/"*.service /etc/systemd/system/
    systemctl daemon-reload
    
    echo "✅ Systemd services installed"
    echo
fi

# Step 8: Create convenience scripts
echo "8️⃣ Creating convenience scripts..."

cat > "$TORI_BASE/torictl_beyond" << 'EOF'
#!/bin/bash
# Beyond Metacognition control script

TORI_BASE=/opt/tori
VENV=$TORI_BASE/venv/bin/python
CURRENT=$TORI_BASE/current/kha

case "$1" in
    verify)
        $VENV $CURRENT/verify_beyond_integration.py
        ;;
    monitor)
        $VENV $CURRENT/torictl.py monitor
        ;;
    demo)
        $VENV $CURRENT/torictl.py demo ${2:-emergence}
        ;;
    status)
        $VENV $CURRENT/torictl.py status
        ;;
    diagnose)
        $VENV $CURRENT/beyond_diagnostics.py
        ;;
    *)
        echo "Usage: $0 {verify|monitor|demo|status|diagnose}"
        exit 1
        ;;
esac
EOF

chmod +x "$TORI_BASE/torictl_beyond"
echo "✅ Created $TORI_BASE/torictl_beyond"
echo

# Step 9: Final summary
echo "🎉 Beyond Metacognition Installation Complete!"
echo "============================================="
echo
echo "Next steps:"
echo
echo "1. Update symlink to activate:"
echo "   ln -sfn $LATEST_RELEASE $TORI_BASE/current"
echo
echo "2. Restart TORI services:"
echo "   systemctl restart tori-api"
echo "   systemctl start tori-braid-aggregator tori-beyond-monitor"
echo
echo "3. Verify installation:"
echo "   $TORI_BASE/torictl_beyond verify"
echo
echo "4. Run a demo:"
echo "   $TORI_BASE/torictl_beyond demo emergence"
echo
echo "5. Monitor live:"
echo "   $TORI_BASE/torictl_beyond monitor"
echo
echo "📚 Documentation:"
echo "   - Deployment guide: BARE_METAL_DEPLOYMENT_GUIDE.md"
echo "   - Quick start: BEYOND_QUICKSTART.md"
echo "   - Full docs: BEYOND_METACOGNITION_COMPLETE.md"
echo
echo "🛡️ Safety notes:"
echo "   - Auto-rollback monitors λ_max > 0.08 for 3+ minutes"
echo "   - Manual rollback: ln -sfn <old-release> $TORI_BASE/current"
echo "   - Conservative settings enabled by default"
echo
echo "Your bare-metal TORI is ready to transcend! 🌌🔧"
