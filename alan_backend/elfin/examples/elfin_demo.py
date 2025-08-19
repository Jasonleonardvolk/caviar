"""
ELFIN DSL Demo

This script demonstrates the use of the ELFIN DSL to define a concept network
with ψ-based stability monitoring.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from alan_backend.elfin.parser import parse_elfin
from alan_backend.elfin.compiler import compile_elfin_to_lcn
from alan_backend.elfin.runtime import ElfinRuntime, StabilityMonitor

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example ELFIN DSL code
ELFIN_CODE = """
// Define a system with phase-coupled concepts and stability constraints

// Communication module concept with ψ-mode 3
concept CommunicationModule "Main communication interface" {
    psi_mode(3) {
        amplitude = 1.0,
        phase = 0.0,
        is_primary = true
    }
    
    stability LYAPUNOV {
        threshold = 0.01,
        expression = "lyapunov(CommunicationModule) < 0",
        is_global = false
    }
    
    protocol = "MQTT",
    max_connections = 100
}

// Data processing concept with ψ-mode 2
concept DataProcessor "Real-time data processing unit" {
    psi_mode(2) {
        amplitude = 0.8,
        phase = 0.5
    }
    
    stability ASYMPTOTIC {
        threshold = 0.05,
        affected_concepts = [CommunicationModule]
    }
    
    batch_size = 64,
    processing_rate = 1000
}

// Synchronization concept with ψ-mode 1
concept SyncController "Phase synchronization controller" {
    psi_mode(1) {
        amplitude = 1.2,
        phase = 0.0,
        is_primary = true,
        components = [
            { mode_index = 2, weight = 0.5, phase_offset = 0.0 },
            { mode_index = 3, weight = 0.3, phase_offset = 0.1 }
        ]
    }
    
    stability EXPONENTIAL {
        threshold = 0.001,
        expression = "lyapunov(SyncController) < -0.1",
        affected_concepts = [CommunicationModule, DataProcessor],
        is_global = true
    }
    
    control_rate = 100,
    feedback_gain = 0.8
}

// Phase coupling relationships between concepts
relation CommunicationModule synchronizes_with DataProcessor {
    phase_coupling = {
        coupling_strength = 0.5,
        coupling_function = "sin",
        phase_lag = 0.2,
        bidirectional = true
    }
}

relation SyncController stabilizes CommunicationModule {
    phase_coupling = {
        coupling_strength = 0.8,
        coupling_function = "sin",
        phase_lag = 0.0,
        bidirectional = false
    }
}

relation SyncController stabilizes DataProcessor {
    phase_coupling = {
        coupling_strength = 0.6,
        coupling_function = "sin",
        phase_lag = 0.1,
        bidirectional = false
    }
}

// Agent directive for stability monitoring
agent StabilityMonitor: "Monitor system stability" {
    targets = [CommunicationModule, DataProcessor, SyncController],
    trigger = psi(1, SyncController) > 0.9,
    monitoring_interval = 0.1
}

// System synchronization goal
goal SystemSync: "Maintain system synchronization" = 
    lyapunov(SyncController) < -0.05 and
    SyncController synchronizes CommunicationModule and
    SyncController synchronizes DataProcessor {
    
    targets = [CommunicationModule, DataProcessor, SyncController],
    priority = 0.9
}

// Assumption about system state
assume SystemStability: "System is initially stable" = 
    stable(CommunicationModule) and stable(DataProcessor) {
    
    confidence = 0.95,
    validated = false
}
"""


def stability_alert_handler(alert):
    """Handle stability alerts."""
    logger.warning(f"Stability alert: {alert['is_stable']}")
    for concept_id, details in alert.get('details', {}).items():
        logger.warning(f"  Concept {concept_id}: {details}")


def main():
    """Main function to demonstrate the ELFIN DSL."""
    logger.info("Parsing ELFIN code...")
    ast = parse_elfin(ELFIN_CODE)
    logger.info(f"Parsed AST with {len(ast.declarations)} declarations")
    
    logger.info("Compiling to LocalConceptNetwork...")
    lcn = compile_elfin_to_lcn(ast)
    logger.info(f"Compiled LCN with {len(lcn.concepts)} concepts, "
                f"{len(lcn.relations)} relations, "
                f"{len(lcn.agent_directives)} agent directives, "
                f"{len(lcn.goals)} goals, and "
                f"{len(lcn.assumptions)} assumptions")
    
    logger.info("Creating ELFIN runtime...")
    
    # In a real application, we would use the actual ALAN system components
    # For this demo, we'll use the ElfinRuntime with default settings
    runtime = ElfinRuntime(lcn=lcn)
    
    # Create a stability monitor
    stability_monitor = StabilityMonitor(runtime)
    stability_monitor.add_alert_listener(stability_alert_handler)
    
    logger.info("Starting stability monitoring...")
    stability_monitor.start_monitoring()
    
    logger.info("Starting simulation...")
    runtime.start()
    
    # Run a few simulation steps
    for i in range(10):
        logger.info(f"Simulation step {i+1}...")
        results = runtime.step(dt=0.1)
        
        # Log some results
        stability = results["stability"]["is_stable"]
        goals = results["goals"]
        logger.info(f"  System stable: {stability}")
        logger.info(f"  Goals satisfied: {sum(goals.values())}/{len(goals)}")
    
    logger.info("Stopping simulation...")
    runtime.stop()
    
    # Check final stability
    is_stable, details = stability_monitor.check_stability()
    logger.info(f"Final stability check: {is_stable}")
    
    logger.info("ELFIN demo completed")


if __name__ == "__main__":
    main()
