#!/usr/bin/env python3
"""
beyond_deploy_tools.py - Production deployment utilities for Beyond Metacognition

Addresses:
1. Branch management
2. Data migration
3. Config loading
4. Metrics setup
5. Rollback safety
"""

import os
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ============================================================================
# 1. BRANCH MANAGEMENT HELPER
# ============================================================================

def create_feature_branch():
    """Create feature branch with proper naming"""
    branch_name = f"feat/beyond-metacog-v2-{datetime.now().strftime('%Y%m%d')}"
    
    commands = [
        "git checkout staging",
        "git pull origin staging",
        f"git checkout -b {branch_name}",
    ]
    
    print(f"🌿 Creating feature branch: {branch_name}")
    for cmd in commands:
        print(f"  $ {cmd}")
        result = subprocess.run(cmd.split(), capture_output=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr.decode()}")
            return False
    
    print(f"✅ Branch created: {branch_name}")
    return True

def commit_incrementally(files_to_patch):
    """Commit each patch incrementally with tests"""
    test_commands = {
        "alan_backend/eigensentry_guard.py": "pytest -m bdg",
        "python/core/chaos_control_layer.py": "pytest -m chaos",
        "tori_master.py": "pytest -m integration",
        "services/metrics_ws.py": "pytest -m websocket"
    }
    
    for file_path, test_cmd in test_commands.items():
        print(f"\n📝 Patching {file_path}...")
        
        # Run patch
        patch_result = subprocess.run(
            ["python", "apply_beyond_patches.py", "--single", file_path],
            capture_output=True
        )
        
        if patch_result.returncode != 0:
            print(f"❌ Patch failed for {file_path}")
            continue
        
        # Run tests
        print(f"  🧪 Running: {test_cmd}")
        test_result = subprocess.run(test_cmd.split(), capture_output=True)
        
        if test_result.returncode == 0:
            # Commit
            subprocess.run(["git", "add", file_path])
            subprocess.run([
                "git", "commit", "-m", 
                f"feat(beyond): patch {Path(file_path).name} for Beyond Metacognition"
            ])
            print(f"  ✅ Committed {file_path}")
        else:
            # Revert
            subprocess.run(["git", "checkout", "--", file_path])
            print(f"  ⚠️ Tests failed, reverted {file_path}")

# ============================================================================
# 2. DATA MIGRATION HELPERS  
# ============================================================================

def setup_data_bootstrap():
    """Create bootstrap templates for data files"""
    bootstrap_dir = Path("data/bootstrap")
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    
    # Lyapunov watchlist template
    lyapunov_template = {
        "format_version": "1.0",
        "created": datetime.now().isoformat(),
        "entries": []
    }
    
    with open(bootstrap_dir / "lyapunov_watchlist.json", "w") as f:
        json.dump(lyapunov_template, f, indent=2)
    
    # Spectral DB template
    spectral_template = []
    
    with open(bootstrap_dir / "spectral_db.json", "w") as f:
        json.dump(spectral_template, f)
    
    print("✅ Created bootstrap data templates")

def patch_origin_sentry_loader():
    """Add migration safety to OriginSentry"""
    origin_path = Path("alan_backend/origin_sentry.py")
    
    # Read current file
    content = origin_path.read_text()
    
    # Find _load method
    load_patch = '''
    def _load(self):
        """Load from disk with migration safety"""
        if not self.storage_path.exists():
            # Check for bootstrap template
            bootstrap_path = Path("data/bootstrap") / self.storage_path.name
            if bootstrap_path.exists():
                shutil.copy(bootstrap_path, self.storage_path)
                logger.info(f"Initialized from bootstrap: {self.storage_path}")
            else:
                return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Migration logic for older formats
            if isinstance(data, dict) and 'entries' in data:
                # New format
                data = data['entries']
            
            for item in data:
                sig = SpectralSignature(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    eigenvalues=np.array(item['eigenvalues']),
                    hash_id=item['hash_id'],
                    coherence_state=item.get('coherence_state', 'local'),
                    gaps=item.get('gaps', [])
                )
                self.signatures.append(sig)
                self.hash_index[sig.hash_id] = sig
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load spectral DB: {e}")
            # Continue with empty DB
        except Exception as e:
            logger.error(f"Failed to load spectral DB: {e}")
'''
    
    # Replace the _load method
    # (In practice, use proper AST manipulation)
    print("📝 Add migration safety to SpectralDB._load()")

# ============================================================================
# 3. CONFIG LOADER WITH DEFAULTS
# ============================================================================

class BeyondConfig:
    """Configuration loader with defaults"""
    
    DEFAULT_CONFIG = {
        "beyond_metacognition": {
            "observer_synthesis": {
                "reflex_budget": 60,
                "measurement_cooldown_ms": 100,
                "oscillation_detection_window": 10
            },
            "creative_feedback": {
                "novelty_threshold_high": 0.7,
                "novelty_threshold_low": 0.2,
                "max_exploration_steps": 1000,
                "emergency_threshold": 0.08,
                "max_lambda_allowed": 0.1
            },
            "temporal_braiding": {
                "micro_capacity": 10000,
                "meso_capacity": 1000,
                "macro_capacity": 100
            },
            "origin_sentry": {
                "novelty_threshold": 0.7,
                "gap_min": 0.02,
                "enable_topology": False
            }
        }
    }
    
    @classmethod
    def load(cls, config_dir: Path = Path("conf")) -> Dict[str, Any]:
        """Load config with defaults"""
        config = cls.DEFAULT_CONFIG.copy()
        
        # Try to load runtime.yaml
        runtime_path = config_dir / "runtime.yaml"
        if runtime_path.exists():
            try:
                with open(runtime_path) as f:
                    runtime_config = yaml.safe_load(f)
                    # Deep merge
                    config = cls._deep_merge(config, runtime_config)
            except Exception as e:
                print(f"⚠️ Could not load runtime.yaml: {e}")
        
        # Environment overrides
        if reflex_budget := os.getenv("BEYOND_REFLEX_BUDGET"):
            config["beyond_metacognition"]["observer_synthesis"]["reflex_budget"] = int(reflex_budget)
        
        return config
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = BeyondConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

# ============================================================================
# 4. METRICS SETUP
# ============================================================================

PROMETHEUS_METRICS_TEMPLATE = '''
# Add to your prometheus_metrics.py

from prometheus_client import Counter, Gauge, Histogram

# Beyond Metacognition metrics
origin_dim_expansions_total = Counter(
    'origin_dim_expansions_total',
    'Total dimensional expansions detected by OriginSentry'
)

origin_gap_births_total = Counter(
    'origin_gap_births_total', 
    'Total spectral gap births detected'
)

braid_retrocoherence_events_total = Counter(
    'braid_retrocoherence_events_total',
    'Total retro-coherent labeling events'
)

beyond_novelty_score = Gauge(
    'beyond_novelty_score',
    'Current novelty score from OriginSentry'
)

beyond_lambda_max = Gauge(
    'beyond_lambda_max',
    'Current maximum eigenvalue'
)

creative_mode = Gauge(
    'creative_mode',
    'Current creative feedback mode (0=stable, 1=exploring, 2=consolidating, -1=emergency)'
)

# Histograms for performance
origin_classify_duration = Histogram(
    'origin_classify_duration_seconds',
    'Time to classify spectral state'
)

braid_aggregation_duration = Histogram(
    'braid_aggregation_duration_seconds',
    'Time to aggregate temporal buffers',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)
'''

def setup_metrics():
    """Setup Prometheus metrics"""
    metrics_file = Path("services/prometheus_metrics.py")
    
    if metrics_file.exists():
        print("📊 Adding Beyond Metacognition metrics...")
        # In practice, properly parse and add to existing file
        print(PROMETHEUS_METRICS_TEMPLATE)
    else:
        print("⚠️ prometheus_metrics.py not found - create it first")

# ============================================================================
# 5. LOGGING SETUP
# ============================================================================

def setup_enhanced_logging():
    """Add [BEYOND] tags to all new log lines"""
    
    logging_config = '''
import logging

class BeyondFormatter(logging.Formatter):
    """Add [BEYOND] tag to Beyond Metacognition logs"""
    
    def format(self, record):
        # Check if this is a Beyond component
        if any(comp in record.name for comp in [
            'origin_sentry', 'braid_buffers', 'observer_synthesis', 
            'creative_feedback', 'braid_aggregator'
        ]):
            record.msg = f"[BEYOND] [SPECTRAL] {record.msg}"
        return super().format(record)

# Apply to all Beyond components
beyond_handler = logging.StreamHandler()
beyond_handler.setFormatter(BeyondFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

for logger_name in [
    'alan_backend.origin_sentry',
    'python.core.braid_buffers',
    'python.core.observer_synthesis',
    'python.core.creative_feedback',
    'alan_backend.braid_aggregator'
]:
    logger = logging.getLogger(logger_name)
    logger.addHandler(beyond_handler)
'''
    
    print("📝 Enhanced logging configuration:")
    print(logging_config)

# ============================================================================
# 6. ROLLBACK SAFETY
# ============================================================================

def create_rollback_script():
    """Create automated rollback script"""
    
    rollback_script = '''#!/bin/bash
# auto_rollback.sh - Safety rollback for Beyond Metacognition

THRESHOLD=0.08
DURATION_MIN=3
OLD_IMAGE="tori:stable-pre-beyond"
NEW_IMAGE="tori:beyond-metacog"

# Monitor lambda_max
while true; do
    # Get current lambda_max from metrics endpoint
    LAMBDA_MAX=$(curl -s http://localhost:9090/metrics | grep beyond_lambda_max | awk '{print $2}')
    
    if (( $(echo "$LAMBDA_MAX > $THRESHOLD" | bc -l) )); then
        echo "[ROLLBACK] High lambda detected: $LAMBDA_MAX"
        
        # Check duration
        sleep 180  # 3 minutes
        
        LAMBDA_MAX_2=$(curl -s http://localhost:9090/metrics | grep beyond_lambda_max | awk '{print $2}')
        
        if (( $(echo "$LAMBDA_MAX_2 > $THRESHOLD" | bc -l) )); then
            echo "[ROLLBACK] Sustained high lambda - initiating rollback!"
            
            # Scale up old image
            kubectl scale deployment tori-stable --replicas=3
            
            # Update service selector
            kubectl patch service tori -p '{"spec":{"selector":{"version":"stable"}}}'
            
            # Alert ops
            curl -X POST $SLACK_WEBHOOK -d '{"text":"⚠️ Beyond Metacognition auto-rollback triggered!"}'
            
            # Scale down new image
            kubectl scale deployment tori-beyond --replicas=0
            
            break
        fi
    fi
    
    sleep 30
done
'''
    
    with open("scripts/auto_rollback.sh", "w") as f:
        f.write(rollback_script)
    
    os.chmod("scripts/auto_rollback.sh", 0o755)
    print("✅ Created auto_rollback.sh")

# ============================================================================
# 7. POST-DEPLOY VERIFICATION
# ============================================================================

def create_health_check():
    """Create post-deploy health check script"""
    
    health_check = '''#!/usr/bin/env python3
"""
quick_health.py - Post-deploy verification for Beyond Metacognition
"""

import sys
import time
import argparse
import requests
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def check_beyond_health(expect_beyond=True):
    """Quick health check for Beyond components"""
    
    checks = {
        "origin_sentry": False,
        "braid_buffers": False,
        "creative_feedback": False,
        "metrics_endpoint": False
    }
    
    print("🔍 Beyond Metacognition Health Check")
    print("=" * 60)
    
    # 1. Check if components initialize
    try:
        from alan_backend.origin_sentry import OriginSentry
        origin = OriginSentry()
        
        # Check coherence state
        import numpy as np
        result = origin.classify(np.random.randn(5) * 0.02)
        
        if result['coherence'] in ['local', 'global', 'critical']:
            checks['origin_sentry'] = True
            print("✅ OriginSentry: coherence state =", result['coherence'])
        else:
            print("❌ OriginSentry: invalid coherence state")
            
    except Exception as e:
        print(f"❌ OriginSentry failed: {e}")
    
    # 2. Check braid buffers
    try:
        from python.core.braid_buffers import get_braiding_engine, TimeScale
        
        engine = get_braiding_engine()
        depths = {
            scale: len(engine.buffers[scale].buffer)
            for scale in TimeScale
        }
        
        checks['braid_buffers'] = True
        print(f"✅ Braid buffers: {depths}")
        
    except Exception as e:
        print(f"❌ Braid buffers failed: {e}")
    
    # 3. Check creative feedback
    try:
        from python.core.creative_feedback import get_creative_feedback
        
        creative = get_creative_feedback()
        if creative.mode.value in ['stable', 'exploring', 'consolidating', 'emergency']:
            checks['creative_feedback'] = True
            print(f"✅ Creative feedback: mode = {creative.mode.value}")
        else:
            print("❌ Creative feedback: invalid mode")
            
    except Exception as e:
        print(f"❌ Creative feedback failed: {e}")
    
    # 4. Check metrics endpoint
    try:
        response = requests.get("http://localhost:9090/metrics", timeout=5)
        if "origin_dim_expansions_total" in response.text:
            checks['metrics_endpoint'] = True
            print("✅ Metrics endpoint: Beyond metrics exposed")
        else:
            print("⚠️ Metrics endpoint: Beyond metrics not found")
            
    except Exception as e:
        print(f"⚠️ Metrics endpoint not reachable: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks.values())
    total = len(checks)
    
    if expect_beyond and passed == total:
        print(f"✅ HEALTHY: All {total} Beyond components operational")
        return 0
    elif not expect_beyond and passed == 0:
        print("✅ HEALTHY: Beyond components correctly absent")
        return 0
    else:
        print(f"❌ UNHEALTHY: Only {passed}/{total} components working")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-beyond", action="store_true",
                       help="Expect Beyond components to be active")
    args = parser.parse_args()
    
    sys.exit(check_beyond_health(args.expect_beyond))
'''
    
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)
    
    with open(tools_dir / "quick_health.py", "w") as f:
        f.write(health_check)
    
    print("✅ Created tools/quick_health.py")

# ============================================================================
# 8. MAIN DEPLOYMENT HELPER
# ============================================================================

def main():
    """Run deployment preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Beyond Metacognition deployment tools"
    )
    parser.add_argument("--branch", action="store_true",
                       help="Create feature branch")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Setup data bootstrap")
    parser.add_argument("--metrics", action="store_true",
                       help="Show metrics setup")
    parser.add_argument("--rollback", action="store_true",
                       help="Create rollback script")
    parser.add_argument("--health", action="store_true",
                       help="Create health check")
    parser.add_argument("--all", action="store_true",
                       help="Do all setup steps")
    
    args = parser.parse_args()
    
    if args.all or args.branch:
        create_feature_branch()
    
    if args.all or args.bootstrap:
        setup_data_bootstrap()
        patch_origin_sentry_loader()
    
    if args.all or args.metrics:
        setup_metrics()
        setup_enhanced_logging()
    
    if args.all or args.rollback:
        create_rollback_script()
    
    if args.all or args.health:
        create_health_check()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
