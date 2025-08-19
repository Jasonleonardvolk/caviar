#!/usr/bin/env python3
"""
apply_beyond_patches.py  ·  v0.3  (2025-07-03)

Automates all code insertions required for the Beyond-Metacognition upgrade.
Run with:
    python apply_beyond_patches.py           # normal run
    python apply_beyond_patches.py --dry     # show changes only
    python apply_beyond_patches.py --verify  # patch then launch verification script
"""

import argparse, re, sys, textwrap, json, time
from pathlib import Path
from datetime import datetime
import shutil, difflib, subprocess

# --------------------------------------------------------------------------- #
#  PATCH TABLE (full patches from our implementation)
# --------------------------------------------------------------------------- #

PATCHES = {
    "alan_backend/eigensentry_guard.py": {
        "imports": """from alan_backend.origin_sentry import OriginSentry
from python.core.braid_buffers import get_braiding_engine
from python.core.observer_synthesis import get_observer_synthesis
from python.core.creative_feedback import get_creative_feedback""",
        
        "init_additions": """        # Beyond Metacognition components
        self.origin_sentry = OriginSentry()
        self.braiding_engine = get_braiding_engine()
        self.observer_synthesis = get_observer_synthesis()
        self.creative_feedback = get_creative_feedback()
        
        # Integration flags
        self.enable_self_measurement = True
        self.enable_creative_feedback = True""",
        
        "check_eigenvalues_additions": """        # Classify with OriginSentry
        origin_classification = self.origin_sentry.classify(
            eigenvalues, 
            betti_numbers=None  # TODO: Add topology tracking
        )
        
        # Record in temporal braid
        self.braiding_engine.record_event(
            kind='eigenmode',
            lambda_max=float(max_eigenvalue),
            data={'origin_classification': origin_classification}
        )
        
        # Self-measurement (stochastic)
        if self.enable_self_measurement:
            measurement = self.observer_synthesis.apply_stochastic_measurement(
                eigenvalues,
                origin_classification['coherence'],
                origin_classification['novelty_score'],
                base_probability=0.1
            )
            
            if measurement:
                # Record measurement in braid
                self.braiding_engine.record_event(
                    kind='self_measurement',
                    lambda_max=float(max_eigenvalue),
                    data={'measurement': measurement.to_dict()}
                )
        
        # Creative feedback
        if self.enable_creative_feedback:
            creative_action = self.creative_feedback.update({
                'novelty_score': origin_classification['novelty_score'],
                'lambda_max': float(max_eigenvalue),
                'coherence_state': origin_classification['coherence']
            })
            
            # Apply creative action
            if creative_action['action'] == 'inject_entropy':
                self.current_threshold *= creative_action['lambda_factor']
                logger.info(f"🎲 Creative entropy injection: threshold *= {creative_action['lambda_factor']}")
            elif creative_action['action'] == 'emergency_damping':
                self.current_threshold *= creative_action['lambda_factor']
                logger.warning(f"⚠️ Emergency creative damping: threshold *= {creative_action['lambda_factor']}")
        
        # Update metrics with origin data
        self.metrics.update({
            'origin_dimension': origin_classification['metrics']['current_dimension'],
            'dimension_expansions': origin_classification['metrics']['dimension_expansions'],
            'creative_mode': self.creative_feedback.mode.value,
            'self_measurements': len(self.observer_synthesis.measurements)
        })"""
    },
    
    "python/core/chaos_control_layer.py": {
        "imports": """from python.core.braid_buffers import get_braiding_engine""",
        
        "init_additions": """        # Temporal braiding integration
        self.braiding_engine = get_braiding_engine()""",
        
        "propagate_additions": """            # Record spectral snapshot in braid
            if hasattr(self, 'braiding_engine'):
                # Compute approximate eigenvalues from field
                field_fft = np.fft.fft(current)
                power_spectrum = np.abs(field_fft)**2
                # Top k modes as pseudo-eigenvalues
                top_modes = np.sort(power_spectrum)[-8:][::-1]
                
                self.braiding_engine.record_event(
                    kind='soliton_evolution',
                    lambda_max=float(np.max(top_modes)),
                    data={'step': _, 'dt': dt}
                )"""
    },
    
    "tori_master.py": {
        "imports": """from alan_backend.braid_aggregator import BraidAggregator
from python.core.observer_synthesis import get_observer_synthesis""",
        
        "init_additions": """        # Beyond Metacognition components
        self.braid_aggregator = None
        self.observer_synthesis = None""",
        
        "start_additions": """            # Start Braid Aggregator
            logger.info("Starting Temporal Braid Aggregator...")
            self.braid_aggregator = BraidAggregator()
            aggregator_task = asyncio.create_task(self.braid_aggregator.start())
            self.tasks.append(aggregator_task)
            
            # Initialize Observer-Observed Synthesis
            self.observer_synthesis = get_observer_synthesis()
            logger.info("✅ Observer-Observed Synthesis initialized")
            
            # Log Beyond Metacognition status
            logger.info("🚀 Beyond Metacognition components active:")
            logger.info("   ✅ OriginSentry: Dimensional emergence detection")
            logger.info("   ✅ Temporal Braiding: Multi-scale cognitive traces")
            logger.info("   ✅ Observer Synthesis: Self-measurement operators")
            logger.info("   ✅ Creative Feedback: Entropy injection control")""",
        
        "check_health_additions": """        # Check creative metrics
        if 'eigen_guard' in self.components:
            guard = self.components['eigen_guard']
            if hasattr(guard, 'creative_feedback'):
                creative_metrics = guard.creative_feedback.get_creative_metrics()
                if creative_metrics['current_mode'] == 'emergency':
                    logger.error("⚠️ CREATIVE EMERGENCY MODE ACTIVE!")
                elif creative_metrics['current_mode'] == 'exploring':
                    logger.info(f"🎨 Creative exploration active (step {creative_metrics['steps_in_mode']})")""",
        
        "handle_integrations_additions": """                # Generate metacognitive context
                if self.observer_synthesis:
                    meta_context = self.observer_synthesis.generate_metacognitive_context()
                    
                    # Log if reflexive patterns detected
                    if 'REFLEXIVE_OSCILLATION_DETECTED' in meta_context.get('warning', ''):
                        logger.warning("🔄 Reflexive oscillation detected - reducing self-measurement")
                    
                    # TODO: Feed meta_context into next reasoning cycle"""
    },
    
    "services/metrics_ws.py": {
        "imports": """from python.core.braid_buffers import TimeScale""",
        
        "broadcast_additions": """            'beyond_metacognition': {
                'origin_sentry': {
                    'dimension': self.guard.origin_sentry.metrics['current_dimension'],
                    'dimension_births': self.guard.origin_sentry.metrics['dimension_expansions'],
                    'coherence': self.guard.origin_sentry.metrics['coherence_state'],
                    'novelty': self.guard.origin_sentry.metrics['novelty_score']
                },
                'creative_feedback': {
                    'mode': self.guard.creative_feedback.mode.value,
                    'total_injections': self.guard.creative_feedback.metrics['total_injections'],
                    'success_rate': self.guard.creative_feedback.metrics.get('success_rate', 0.0)
                },
                'temporal_braid': {
                    'micro_events': len(self.guard.braiding_engine.buffers[TimeScale.MICRO].buffer),
                    'meso_events': len(self.guard.braiding_engine.buffers[TimeScale.MESO].buffer),
                    'macro_events': len(self.guard.braiding_engine.buffers[TimeScale.MACRO].buffer)
                },
                'self_measurement': {
                    'total_measurements': len(self.guard.observer_synthesis.measurements),
                    'reflex_remaining': self.guard.observer_synthesis._get_reflex_budget_remaining()
                }
            },"""
    }
}

# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #

BACKUP_SUFFIX = ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
STATUS_FILE = "BEYOND_METACOGNITION_STATUS.json"

def backup(path: Path) -> None:
    """Create timestamped backup of file"""
    if path.exists():
        backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        shutil.copy2(path, backup_path)
        print(f"✅ Created backup: {backup_path.name}")

def already_contains(block: str, text: str) -> bool:
    """Check if text already contains key lines from block"""
    # Check for distinctive lines from the addition
    key_lines = [line.strip() for line in text.strip().splitlines() 
                 if line.strip() and not line.strip().startswith('#')]
    for key_line in key_lines[:3]:  # Check first 3 non-comment lines
        if key_line in block:
            return True
    return False

def insert_imports(src: list[str], addition: str) -> list[str]:
    """Insert imports after existing import statements"""
    if already_contains("".join(src), addition):
        return src
    
    # Find last import line
    last_import_idx = 0
    for i, line in enumerate(src):
        if line.strip().startswith(('import ', 'from ')):
            last_import_idx = i
    
    # Insert after last import
    insert_idx = last_import_idx + 1
    return src[:insert_idx] + [addition + "\n\n"] + src[insert_idx:]

def insert_into_method(src: list[str], method_name: str, addition: str, 
                      insert_location: str = "end") -> list[str]:
    """Insert code into a method"""
    if already_contains("".join(src), addition):
        return src
    
    pattern = re.compile(rf"^\s*def {method_name}\s*\(")
    method_start = None
    
    for i, line in enumerate(src):
        if pattern.match(line):
            method_start = i
            break
    
    if method_start is None:
        print(f"⚠️  Method {method_name} not found")
        return src
    
    # Find method indentation
    indent_level = len(src[method_start]) - len(src[method_start].lstrip())
    method_indent = " " * (indent_level + 4)
    
    # Find end of method
    for j in range(method_start + 1, len(src)):
        line = src[j]
        if line.strip():  # Non-empty line
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                # Found dedent - insert before this line
                indented_addition = textwrap.indent(addition.rstrip(), method_indent)
                src.insert(j, indented_addition + "\n\n")
                return src
    
    # If we get here, method extends to end of file
    indented_addition = textwrap.indent(addition.rstrip(), method_indent)
    src.append("\n" + indented_addition + "\n")
    return src

def apply_patch(file_path: Path, spec: dict[str, str], dry_run: bool = False) -> bool:
    """Apply all patches for a single file"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    text = file_path.read_text(encoding='utf-8').splitlines(keepends=True)
    original = text[:]
    
    # Apply imports
    if imp := spec.get("imports"):
        text = insert_imports(text, imp)
    
    # Apply init additions
    if ini := spec.get("init_additions"):
        text = insert_into_method(text, "__init__", ini)
    
    # Apply method-specific additions
    method_map = {
        "check_eigenvalues_additions": "check_eigenvalues",
        "propagate_additions": "propagate",
        "start_additions": "start",
        "check_health_additions": "_check_health",
        "handle_integrations_additions": "_handle_integrations",
        "broadcast_additions": "broadcast_loop"
    }
    
    for key, method_name in method_map.items():
        if addition := spec.get(key):
            text = insert_into_method(text, method_name, addition)
    
    # Check if anything changed
    if text != original:
        if dry_run:
            # Show diff
            diff = difflib.unified_diff(
                original, text,
                fromfile=str(file_path),
                tofile=str(file_path) + " (patched)",
                lineterm=""
            )
            print(f"\n{'='*60}")
            print(f"DIFF for {file_path}:")
            print(f"{'='*60}")
            for line in diff:
                print(line.rstrip())
        else:
            # Write changes
            file_path.write_text("".join(text), encoding='utf-8')
        return True
    
    return False

def write_status(root: Path, changed: list[str], skipped: list[str]) -> None:
    """Write status JSON file"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "patched_files": changed,
        "skipped_files": skipped,
        "patch_version": "0.3",
        "components": {
            "OriginSentry": "Dimensional emergence detection",
            "TemporalBraiding": "Multi-scale cognitive traces",
            "ObserverSynthesis": "Self-measurement operators",
            "CreativeFeedback": "Entropy injection control"
        }
    }
    (root / STATUS_FILE).write_text(json.dumps(data, indent=2))
    print(f"📝 Status written to {STATUS_FILE}")

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply Beyond Metacognition patches to TORI system"
    )
    p.add_argument("--dry", action="store_true", 
                   help="show diff but do not write")
    p.add_argument("--verify", action="store_true", 
                   help="run verify_beyond_integration after patching")
    p.add_argument("--status-only", action="store_true",
                   help="(re)generate status file without patching")
    p.add_argument("--single", type=str, metavar="FILE",
                   help="patch only a single file")
    args = p.parse_args()
    
    root = Path(__file__).parent
    
    # Status-only mode
    if args.status_only:
        write_status(root, [], list(PATCHES.keys()))
        return
    
    print("🚀 Applying Beyond-Metacognition patches")
    print("=" * 60)
    
    if args.dry:
        print("DRY RUN MODE - No files will be modified")
        print("=" * 60)
    
    changed = []
    skipped = []
    
    # Filter to single file if requested
    patches_to_apply = PATCHES
    if args.single:
        if args.single in PATCHES:
            patches_to_apply = {args.single: PATCHES[args.single]}
        else:
            print(f"❌ Unknown file: {args.single}")
            print(f"Available files: {', '.join(PATCHES.keys())}")
            return
    
    for rel_path, spec in patches_to_apply.items():
        target = root / rel_path
        
        if not target.exists():
            print(f"❌ {rel_path} not found")
            skipped.append(rel_path)
            continue
        
        if not args.dry:
            backup(target)
        
        patched = apply_patch(target, spec, dry_run=args.dry)
        
        if patched:
            changed.append(rel_path)
            print(f"✅ {'Would patch' if args.dry else 'Patched'} {rel_path}")
        else:
            skipped.append(rel_path)
            print(f"ℹ️  {rel_path} already up-to-date")
    
    print("\n" + "=" * 60)
    
    if args.dry:
        print("DRY RUN COMPLETE - Review diffs above")
        print("Run without --dry to apply changes")
    else:
        print(f"Patching complete!")
        print(f"  Files modified: {len(changed)}")
        print(f"  Files skipped: {len(skipped)}")
        
        # Write status
        write_status(root, changed, skipped)
        
        # Optional verification
        if args.verify:
            print("\n🔍 Running verification script...")
            verify_script = root / "verify_beyond_integration.py"
            if verify_script.exists():
                subprocess.call([sys.executable, str(verify_script)])
            else:
                print(f"❌ Verification script not found: {verify_script}")

if __name__ == "__main__":
    main()
