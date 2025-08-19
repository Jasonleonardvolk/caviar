#!/usr/bin/env python3
"""
beyond_deploy_bare_metal.py - Deployment tools for bare-metal TORI setup
Adapted for virtualenv + systemd + Ansible environment
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Deployment paths matching your setup
TORI_BASE = Path("/opt/tori")
RELEASES_DIR = TORI_BASE / "releases"
CURRENT_LINK = TORI_BASE / "current"
VENV_PATH = TORI_BASE / "venv"
CONFIG_DIR = TORI_BASE / "conf"

class BareMetalDeployer:
    """Deploy Beyond Metacognition to bare-metal TORI setup"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.git_sha = self._get_git_sha()
        self.release_dir = RELEASES_DIR / f"{self.git_sha}_{self.timestamp}"
        
    def _get_git_sha(self) -> str:
        """Get current git SHA"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def create_release_structure(self):
        """Create release directory structure"""
        print(f"📁 Creating release: {self.release_dir}")
        
        # Create directories
        for subdir in ["kha", "conf", "data/bootstrap", "scripts", "ansible"]:
            (self.release_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Copy Beyond Metacognition files
        src_files = [
            # Core components
            "kha/alan_backend/origin_sentry.py",
            "kha/alan_backend/braid_aggregator.py",
            "kha/python/core/braid_buffers.py",
            "kha/python/core/observer_synthesis.py",
            "kha/python/core/creative_feedback.py",
            "kha/python/core/topology_tracker.py",
            
            # Tools
            "kha/apply_beyond_patches.py",
            "kha/verify_beyond_integration.py",
            "kha/beyond_demo.py",
            "kha/torictl.py",
            "kha/beyond_diagnostics.py",
            
            # Configs
            "kha/conf/beyond_config_templates.yaml"
        ]
        
        for src in src_files:
            src_path = Path(src)
            if src_path.exists():
                dst = self.release_dir / src
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst)
                print(f"  ✅ Copied {src}")
            else:
                print(f"  ⚠️  Missing {src}")
    
    def generate_systemd_units(self):
        """Generate systemd unit files for Beyond components"""
        
        # Main TORI API service (patched)
        tori_api_service = f"""[Unit]
Description=TORI API with Beyond Metacognition
After=network.target
Wants=tori-braid-aggregator.service

[Service]
Type=simple
User=tori
Group=tori
WorkingDirectory={CURRENT_LINK}
Environment="PATH={VENV_PATH}/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONPATH={CURRENT_LINK}:{CURRENT_LINK}/kha"
Environment="BEYOND_REFLEX_BUDGET=30"
Environment="BEYOND_NOVELTY_THRESHOLD=0.8"
ExecStart={VENV_PATH}/bin/python {CURRENT_LINK}/tori_master.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tori-api

[Install]
WantedBy=multi-user.target
"""
        
        # Braid Aggregator service (new)
        braid_service = f"""[Unit]
Description=TORI Braid Aggregator Service
After=tori-api.service
PartOf=tori-api.service

[Service]
Type=simple
User=tori
Group=tori
WorkingDirectory={CURRENT_LINK}
Environment="PATH={VENV_PATH}/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONPATH={CURRENT_LINK}:{CURRENT_LINK}/kha"
ExecStart={VENV_PATH}/bin/python -c "
import asyncio
from kha.alan_backend.braid_aggregator import start_braid_aggregator
asyncio.run(start_braid_aggregator())
"
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tori-braid

[Install]
WantedBy=multi-user.target
"""
        
        # Monitoring service for rollback
        monitor_service = f"""[Unit]
Description=TORI Beyond Metacognition Safety Monitor
After=tori-api.service

[Service]
Type=simple
User=tori
Group=tori
WorkingDirectory={CURRENT_LINK}
Environment="ROLLBACK_THRESHOLD=0.08"
Environment="ROLLBACK_DURATION_MIN=3"
ExecStart=/bin/bash {CURRENT_LINK}/scripts/beyond_monitor.sh
Restart=always
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tori-monitor

[Install]
WantedBy=multi-user.target
"""
        
        # Save unit files
        units_dir = self.release_dir / "systemd"
        units_dir.mkdir(exist_ok=True)
        
        with open(units_dir / "tori-api.service", "w") as f:
            f.write(tori_api_service)
        
        with open(units_dir / "tori-braid-aggregator.service", "w") as f:
            f.write(braid_service)
            
        with open(units_dir / "tori-beyond-monitor.service", "w") as f:
            f.write(monitor_service)
        
        print("✅ Generated systemd unit files")
    
    def generate_ansible_playbook(self):
        """Generate Ansible playbook for deployment"""
        
        playbook = f"""---
# beyond_metacognition_deploy.yml
- name: Deploy Beyond Metacognition to TORI nodes
  hosts: tori_nodes
  become: yes
  vars:
    tori_base: /opt/tori
    release_dir: {self.release_dir}
    git_sha: {self.git_sha}
    
  tasks:
    - name: Ensure TORI user exists
      user:
        name: tori
        group: tori
        home: /opt/tori
        shell: /bin/bash
        
    - name: Create release directory
      file:
        path: "{{{{ release_dir }}}}"
        state: directory
        owner: tori
        group: tori
        mode: '0755'
        
    - name: Sync Beyond Metacognition files
      synchronize:
        src: "{{{{ release_dir }}}}/"
        dest: "{{{{ release_dir }}}}"
        delete: no
        recursive: yes
        
    - name: Install Python dependencies
      pip:
        requirements: "{{{{ release_dir }}}}/requirements-beyond.txt"
        virtualenv: "{{{{ tori_base }}}}/venv"
        virtualenv_python: python3.10
      become_user: tori
      
    - name: Apply Beyond patches
      command: |
        {{{{ tori_base }}}}/venv/bin/python \\
        {{{{ release_dir }}}}/kha/apply_beyond_patches.py --verify
      args:
        chdir: "{{{{ release_dir }}}}"
      become_user: tori
      register: patch_result
      
    - name: Update Beyond configuration
      template:
        src: "{{{{ item }}}}"
        dest: "{{{{ tori_base }}}}/conf/{{{{ item | basename }}}}"
        owner: tori
        group: tori
        backup: yes
      loop:
        - conf/runtime.yaml
        - conf/braid.yaml
        - conf/origin.yaml
        
    - name: Create data bootstrap directory
      file:
        path: "{{{{ tori_base }}}}/data/bootstrap"
        state: directory
        owner: tori
        group: tori
        
    - name: Install bootstrap data files
      copy:
        src: "{{{{ item }}}}"
        dest: "{{{{ tori_base }}}}/data/bootstrap/"
        owner: tori
        group: tori
      loop:
        - data/bootstrap/lyapunov_watchlist.json
        - data/bootstrap/spectral_db.json
        
    - name: Update systemd unit files
      copy:
        src: "{{{{ release_dir }}}}/systemd/{{{{ item }}}}"
        dest: /etc/systemd/system/
        owner: root
        group: root
      loop:
        - tori-api.service
        - tori-braid-aggregator.service
        - tori-beyond-monitor.service
      notify:
        - reload systemd
        
    - name: Update current symlink
      file:
        src: "{{{{ release_dir }}}}"
        dest: "{{{{ tori_base }}}}/current"
        state: link
        owner: tori
        group: tori
      notify:
        - restart tori services
        
  handlers:
    - name: reload systemd
      systemd:
        daemon_reload: yes
        
    - name: restart tori services
      systemd:
        name: "{{{{ item }}}}"
        state: restarted
        enabled: yes
      loop:
        - tori-api
        - tori-braid-aggregator
        - tori-beyond-monitor
        
    - name: check service health
      uri:
        url: http://localhost:8002/api/health
        timeout: 30
      retries: 3
      delay: 10
"""
        
        with open(self.release_dir / "ansible" / "beyond_deploy.yml", "w") as f:
            f.write(playbook)
            
        # Also create inventory template
        inventory = """[tori_nodes]
tori-node-1 ansible_host=192.168.1.10
tori-node-2 ansible_host=192.168.1.11
tori-node-3 ansible_host=192.168.1.12

[tori_nodes:vars]
ansible_user=ansible
ansible_python_interpreter=/usr/bin/python3
"""
        
        with open(self.release_dir / "ansible" / "inventory.ini", "w") as f:
            f.write(inventory)
            
        print("✅ Generated Ansible playbook")
    
    def generate_rollback_script(self):
        """Generate rollback script for bare metal"""
        
        rollback_script = f"""#!/bin/bash
# beyond_rollback.sh - Rollback Beyond Metacognition

set -e

TORI_BASE=/opt/tori
ROLLBACK_TO="${{1:-1}}"  # Default rollback 1 version

echo "🔄 Rolling back Beyond Metacognition..."

# Find previous release
RELEASES=($(ls -dt $TORI_BASE/releases/*/ | head -n $((ROLLBACK_TO + 1))))
if [ -z "${{RELEASES[$ROLLBACK_TO]}}" ]; then
    echo "❌ No release found at position $ROLLBACK_TO"
    exit 1
fi

ROLLBACK_RELEASE="${{RELEASES[$ROLLBACK_TO]}}"
echo "Rolling back to: $ROLLBACK_RELEASE"

# Stop services
echo "Stopping services..."
sudo systemctl stop tori-api tori-braid-aggregator tori-beyond-monitor || true

# Update symlink
echo "Updating symlink..."
sudo ln -sfn "$ROLLBACK_RELEASE" "$TORI_BASE/current"

# Remove Beyond patches from the rollback release
echo "Removing Beyond patches..."
cd "$ROLLBACK_RELEASE"
for file in alan_backend/eigensentry_guard.py \\
           python/core/chaos_control_layer.py \\
           tori_master.py \\
           services/metrics_ws.py; do
    if [ -f "$file.backup"* ]; then
        # Restore from backup
        cp "$file.backup"* "$file"
        echo "  ✅ Restored $file from backup"
    fi
done

# Start services (without Beyond components)
echo "Starting services..."
sudo systemctl start tori-api

# Health check
sleep 10
if curl -s http://localhost:8002/api/health > /dev/null; then
    echo "✅ Rollback complete - services healthy"
else
    echo "❌ Services not responding after rollback"
    exit 1
fi
"""
        
        script_path = self.release_dir / "scripts" / "beyond_rollback.sh"
        with open(script_path, "w") as f:
            f.write(rollback_script)
        os.chmod(script_path, 0o755)
        
        print("✅ Generated rollback script")
    
    def generate_monitor_script(self):
        """Generate monitoring script for auto-rollback"""
        
        monitor_script = f"""#!/bin/bash
# beyond_monitor.sh - Monitor λ_max and trigger rollback if needed

THRESHOLD=${{ROLLBACK_THRESHOLD:-0.08}}
DURATION_MIN=${{ROLLBACK_DURATION_MIN:-3}}
METRICS_URL="http://localhost:9090/metrics"
ROLLBACK_SCRIPT="{CURRENT_LINK}/scripts/beyond_rollback.sh"

echo "[MONITOR] Starting Beyond Metacognition safety monitor"
echo "[MONITOR] Threshold: λ_max > $THRESHOLD for $DURATION_MIN minutes"

HIGH_LAMBDA_START=""

while true; do
    # Get current lambda_max
    LAMBDA_MAX=$(curl -s "$METRICS_URL" | grep "beyond_lambda_max" | awk '{{print $2}}' || echo "0")
    
    if (( $(echo "$LAMBDA_MAX > $THRESHOLD" | bc -l) )); then
        if [ -z "$HIGH_LAMBDA_START" ]; then
            HIGH_LAMBDA_START=$(date +%s)
            echo "[MONITOR] High λ_max detected: $LAMBDA_MAX at $(date)"
        else
            # Check duration
            DURATION=$(($(date +%s) - HIGH_LAMBDA_START))
            if [ $DURATION -gt $((DURATION_MIN * 60)) ]; then
                echo "[MONITOR] CRITICAL: λ_max=$LAMBDA_MAX sustained for $((DURATION/60)) minutes"
                echo "[MONITOR] Triggering automatic rollback!"
                
                # Send alert
                logger -t tori-monitor "CRITICAL: Auto-rollback triggered due to sustained high λ_max"
                
                # Execute rollback
                bash "$ROLLBACK_SCRIPT" 1
                
                # Exit monitor after rollback
                exit 0
            fi
        fi
    else
        # Reset timer if lambda drops
        if [ -n "$HIGH_LAMBDA_START" ]; then
            echo "[MONITOR] λ_max returned to normal: $LAMBDA_MAX"
            HIGH_LAMBDA_START=""
        fi
    fi
    
    sleep 30
done
"""
        
        script_path = self.release_dir / "scripts" / "beyond_monitor.sh"
        with open(script_path, "w") as f:
            f.write(monitor_script)
        os.chmod(script_path, 0o755)
        
        print("✅ Generated monitor script")
    
    def generate_nginx_config(self):
        """Generate Nginx config snippet for WebSocket and metrics"""
        
        nginx_config = """# Add to your TORI nginx config

# WebSocket endpoint for Beyond metrics
location /ws/eigensentry {
    proxy_pass http://localhost:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}

# Metrics endpoint (restrict access)
location /metrics {
    proxy_pass http://localhost:9090;
    allow 127.0.0.1;
    allow 192.168.1.0/24;  # Prometheus server
    deny all;
}

# Beyond Metacognition health check
location /api/beyond/health {
    proxy_pass http://localhost:8002/api/beyond/health;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
"""
        
        with open(self.release_dir / "nginx" / "beyond_locations.conf", "w") as f:
            f.write(nginx_config)
            
        print("✅ Generated Nginx config")

def create_torictl_wrapper():
    """Create wrapper for torictl to handle Beyond deployment"""
    
    wrapper = '''#!/bin/bash
# torictl_beyond - Extension for Beyond Metacognition

TORI_BASE=/opt/tori
VENV_PATH=$TORI_BASE/venv

case "$1" in
    deploy-beyond)
        echo "🚀 Deploying Beyond Metacognition..."
        
        # Create release
        python3 beyond_deploy_bare_metal.py
        
        # Run Ansible
        cd $TORI_BASE/current/ansible
        ansible-playbook -i inventory.ini beyond_deploy.yml
        ;;
        
    verify-beyond)
        echo "🔍 Verifying Beyond Metacognition..."
        $VENV_PATH/bin/python $TORI_BASE/current/kha/verify_beyond_integration.py
        ;;
        
    monitor-beyond)
        echo "📊 Monitoring Beyond Metacognition..."
        $VENV_PATH/bin/python $TORI_BASE/current/kha/torictl.py monitor
        ;;
        
    rollback-beyond)
        echo "🔄 Rolling back Beyond Metacognition..."
        bash $TORI_BASE/current/scripts/beyond_rollback.sh ${2:-1}
        ;;
        
    *)
        echo "Usage: torictl_beyond {deploy-beyond|verify-beyond|monitor-beyond|rollback-beyond [n]}"
        exit 1
        ;;
esac
'''
    
    with open("torictl_beyond", "w") as f:
        f.write(wrapper)
    os.chmod("torictl_beyond", 0o755)
    
    print("✅ Created torictl_beyond wrapper")

def main():
    """Main deployment preparation"""
    deployer = BareMetalDeployer()
    
    print("🚀 Preparing Beyond Metacognition for bare-metal deployment")
    print(f"   Git SHA: {deployer.git_sha}")
    print(f"   Release: {deployer.release_dir}")
    print()
    
    # Generate all deployment artifacts
    deployer.create_release_structure()
    deployer.generate_systemd_units()
    deployer.generate_ansible_playbook()
    deployer.generate_rollback_script()
    deployer.generate_monitor_script()
    deployer.generate_nginx_config()
    
    # Create convenience wrapper
    create_torictl_wrapper()
    
    print("\n✅ Deployment package ready!")
    print("\nNext steps:")
    print("1. Review generated files in:", deployer.release_dir)
    print("2. Run: ./torictl_beyond deploy-beyond")
    print("3. Verify: ./torictl_beyond verify-beyond")
    print("4. Monitor: ./torictl_beyond monitor-beyond")
    
    # Create requirements file
    requirements = """# requirements-beyond.txt
numpy>=1.21.0
asyncio
psutil
pyyaml
"""
    
    with open(deployer.release_dir / "requirements-beyond.txt", "w") as f:
        f.write(requirements)

if __name__ == "__main__":
    main()
