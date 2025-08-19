#!/usr/bin/env python3
"""
TORI Production System - Fully Integrated Edition
Complete production integration with all new components

This version includes:
- Kaizen introspection scheduler
- Braid Wormhole cross-instance sync
- Critic Consensus decision making
- Chaos burst recycling
- BdG stability monitoring
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import json
import os
from pathlib import Path

# Import all components
from python.core.unified_metacognitive_integration import (
    UnifiedMetacognitiveSystem, MetacognitiveState
)
from python.core.temporal_reasoning_integration import TemporalConceptMesh
from python.core.cognitive_dynamics_monitor import CognitiveStateManager
from python.core.eigensentry.core import EigenSentry2
from python.core.chaos_control_layer import ChaosControlLayer
from python.core.metacognitive_adapters import (
    MetacognitiveAdapterSystem, AdapterMode, AdapterConfig
)
from python.core.safety_calibration import SafetyCalibrationLoop, SafetyLevel

# Import new integrated components
from kaizen.introspection_scheduler import IntrospectionScheduler
from python.core.braid_wormhole import BraidWormhole
from python.core.critic_consensus import CriticConsensus, Decision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Configuration ==========

@dataclass
class TORIProductionConfig:
    """Production configuration for TORI"""
    # Chaos settings
    enable_chaos: bool = True
    default_adapter_mode: AdapterMode = AdapterMode.HYBRID
    chaos_energy_budget: int = 1000
    
    # Safety settings
    enable_safety_monitoring: bool = True
    safety_checkpoint_interval_minutes: int = 30
    emergency_rollback_enabled: bool = True
    
    # Performance settings
    max_concurrent_chaos_tasks: int = 5
    ccl_isolation: bool = True
    
    # Persistence settings
    state_persistence_path: Optional[Path] = Path("./tori_state")
    checkpoint_retention_days: int = 7
    
    # Feature flags
    enable_dark_solitons: bool = True
    enable_attractor_hopping: bool = True
    enable_phase_explosion: bool = True
    enable_concept_evolution: bool = True
    
    # New integration features
    enable_introspection: bool = True
    enable_braid_wormhole: bool = True
    enable_critic_consensus: bool = True
    braid_instance_id: str = "tori_primary"
    
    # Monitoring
    enable_prometheus_metrics: bool = False
    enable_tracing: bool = False
    enable_graphql_api: bool = False
    prometheus_port: int = 9090
    graphql_port: int = 8080
    otlp_endpoint: str = "http://localhost:4317"

# ========== Production System ==========

class TORIProductionSystem:
    """
    Main production system that orchestrates all components
    Provides unified interface for chaos-enhanced TORI
    """
    
    def __init__(self, config: Optional[TORIProductionConfig] = None):
        self.config = config or TORIProductionConfig()
        
        # Initialize core components
        logger.info("Initializing TORI Production System...")
        
        # State management
        self.state_manager = CognitiveStateManager(state_dim=100)
        
        # Concept mesh
        self.concept_mesh = TemporalConceptMesh()
        
        # Metacognitive system (original)
        self.metacognitive_system = UnifiedMetacognitiveSystem(
            self.concept_mesh,
            enable_all_systems=True
        )
        
        # Chaos components
        self.eigen_sentry = EigenSentry2(
            self.state_manager,
            enable_chaos=self.config.enable_chaos
        )
        
        self.ccl = ChaosControlLayer(
            self.eigen_sentry,
            self.state_manager
        )
        
        # Adapter system
        adapter_config = AdapterConfig(
            mode=self.config.default_adapter_mode,
            energy_allocation=self.config.chaos_energy_budget // 10
        )
        
        self.adapter_system = MetacognitiveAdapterSystem(
            self.metacognitive_system,
            self.eigen_sentry,
            self.ccl,
            adapter_config
        )
        
        # Safety system
        self.safety_system = SafetyCalibrationLoop(
            self.eigen_sentry,
            self.ccl,
            self.adapter_system
        )
        
        # Background tasks
        self.background_tasks = []
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'chaos_events': 0,
            'chaos_bursts': 0,
            'safety_interventions': 0,
            'energy_efficiency_gain': [],
            'critic_rejections': 0,
            'wormhole_syncs': 0
        }
        
        # New integrated components
        self.introspection_scheduler = None
        self.braid_wormhole = None
        self.critic_consensus = None
        
        if self.config.enable_introspection:
            self.introspection_scheduler = IntrospectionScheduler(self)
            logger.info("Initialized Kaizen introspection scheduler")
            
        if self.config.enable_braid_wormhole:
            self.braid_wormhole = BraidWormhole(
                instance_id=self.config.braid_instance_id
            )
            logger.info(f"Initialized Braid Wormhole as {self.config.braid_instance_id}")
            
        if self.config.enable_critic_consensus:
            self.critic_consensus = CriticConsensus()
            logger.info("Initialized Critic Consensus system")
        
        logger.info("TORI Production System initialized")
        
    async def start(self):
        """Start all production systems"""
        logger.info("Starting TORI production systems...")
        
        # Start CCL processing
        ccl_task = asyncio.create_task(self.ccl.process_tasks())
        self.background_tasks.append(ccl_task)
        
        # Start safety monitoring
        if self.config.enable_safety_monitoring:
            await self.safety_system.start_monitoring()
            
        # Start periodic checkpointing
        checkpoint_task = asyncio.create_task(self._periodic_checkpointing())
        self.background_tasks.append(checkpoint_task)
        
        # Start new integrated components
        if self.introspection_scheduler:
            introspection_task = asyncio.create_task(self.introspection_scheduler.start())
            self.background_tasks.append(introspection_task)
            logger.info("Started Kaizen introspection scheduler")
            
        if self.braid_wormhole:
            await self.braid_wormhole.start()
            # Add sync monitoring task
            sync_task = asyncio.create_task(self._monitor_wormhole_sync())
            self.background_tasks.append(sync_task)
            logger.info("Started Braid Wormhole synchronization")
        
        # Load persisted state if available
        await self._load_persisted_state()
        
        logger.info("TORI production systems started")
        
    async def stop(self):
        """Stop all production systems"""
        logger.info("Stopping TORI production systems...")
        
        # Stop new components first
        if self.introspection_scheduler:
            await self.introspection_scheduler.stop()
            
        if self.braid_wormhole:
            await self.braid_wormhole.stop()
        
        # Stop safety monitoring
        await self.safety_system.stop_monitoring()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save state
        await self._persist_state()
        
        # Shutdown CCL executor
        self.ccl.executor.shutdown(wait=True)
        
        logger.info("TORI production systems stopped")
        
    async def process_query(self, query: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main query processing interface with full integration
        """
        start_time = datetime.now(timezone.utc)
        
        # Update stats
        self.stats['queries_processed'] += 1
        
        # Check safety status
        safety_report = self.safety_system.get_safety_report()
        current_safety = SafetyLevel[safety_report['current_safety_level'].upper()]
        
        # Get critic consensus if enabled
        should_proceed = True
        critic_metadata = {}
        
        if self.critic_consensus and self.config.enable_critic_consensus:
            # Create decision for evaluation
            decision = Decision(
                decision_id=f"query_{self.stats['queries_processed']}",
                action_type="process_query",
                parameters={
                    'query': query,
                    'chaos_enabled': self.config.enable_chaos,
                    'expected_efficiency_gain': 1.0
                },
                context={
                    'current_eigenvalue': self.eigen_sentry.get_status().get('current_max_eigenvalue', 0),
                    'energy_level': self.eigen_sentry.get_status().get('energy_stats', {}).get('total_available', 100),
                    'safety_level': current_safety.value
                },
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            consensus_result = await self.critic_consensus.evaluate_decision(decision)
            should_proceed = consensus_result.approved
            
            critic_metadata = {
                'approved': consensus_result.approved,
                'weighted_score': consensus_result.weighted_score,
                'votes': consensus_result.vote_count
            }
            
            if not should_proceed:
                self.stats['critic_rejections'] += 1
                
                if consensus_result.rollback_to:
                    # Execute rollback if recommended
                    await self.rollback(consensus_result.rollback_to)
        
        if not should_proceed:
            return {
                'response': "Query processing was rejected by safety critics. Please try a simpler query or wait for system stabilization.",
                'metadata': {
                    'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds(),
                    'critic_consensus': critic_metadata,
                    'safety_level': current_safety.value
                },
                'reasoning_paths': []
            }
        
        # Determine if chaos should be enabled
        enable_chaos_for_query = (
            self.config.enable_chaos and
            current_safety in [SafetyLevel.OPTIMAL, SafetyLevel.NOMINAL] and
            self._should_use_chaos(query, context)
        )
        
        # Add chaos flag to context
        if context is None:
            context = {}
        context['enable_chaos'] = enable_chaos_for_query
        
        try:
            # Monitor system state with BdG integration
            jacobian = self._estimate_system_jacobian()
            
            # Include lattice state for BdG monitoring
            state_dict = {
                'lattice': getattr(self.state_manager, 'lattice', None),
                'g': 1.0,
                'dx': 1.0
            }
            
            eigen_result = await self.eigen_sentry.monitor_and_conduct(jacobian, state_dict)
            
            # Process through metacognitive system
            response = await self.metacognitive_system.process_query_metacognitively(
                query, context
            )
            
            # Track chaos events
            if eigen_result.get('state') != 'stable':
                self.stats['chaos_events'] += 1
                
            # Try chaos burst if appropriate
            if enable_chaos_for_query and hasattr(self.ccl, 'trigger_chaos_burst'):
                burst_result = await self.ccl.trigger_chaos_burst(
                    intensity=1.0,
                    duration=2.0,
                    mode='hybrid'
                )
                if burst_result.get('success'):
                    self.stats['chaos_bursts'] += 1
            
            # Calculate efficiency
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Prepare result
            result = {
                'response': response.text,
                'metadata': {
                    **response.metadata,
                    'processing_time': processing_time,
                    'chaos_enabled': enable_chaos_for_query,
                    'safety_level': current_safety.value,
                    'eigen_state': eigen_result.get('state', 'unknown'),
                    'critic_consensus': critic_metadata if critic_metadata else None
                },
                'reasoning_paths': [
                    {
                        'nodes': [n.name for n in path.chain],
                        'score': path.score,
                        'confidence': path.confidence
                    }
                    for path in response.reasoning_paths[:3]  # Top 3 paths
                ]
            }
            
            # Track efficiency if chaos was used
            if enable_chaos_for_query and 'efficiency_ratio' in self.ccl.get_status():
                efficiency = self.ccl.get_status()['efficiency_ratio']
                self.stats['energy_efficiency_gain'].append(efficiency)
                
            # Sync with wormhole if enabled
            if self.braid_wormhole and response.reasoning_paths:
                # Extract phase vector from reasoning paths
                phase_vector = np.array([path.score for path in response.reasoning_paths])
                await self.braid_wormhole.inject_state(phase_vector)
                
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Check if we need emergency response
            if current_safety == SafetyLevel.EMERGENCY:
                await self._handle_emergency()
                
            # Return error response
            return {
                'response': "I encountered an error processing your query. The system is being stabilized.",
                'metadata': {
                    'error': str(e),
                    'safety_level': current_safety.value,
                    'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds(),
                    'critic_consensus': critic_metadata if critic_metadata else None
                },
                'reasoning_paths': []
            }
            
    def _should_use_chaos(self, query: str, context: Optional[Dict[str, Any]]) -> bool:
        """Determine if chaos should be used for this query"""
        # Explicit context override
        if context and 'force_chaos' in context:
            return context['force_chaos']
            
        # Check query complexity indicators
        chaos_indicators = [
            'analyze', 'explore', 'discover', 'pattern',
            'creative', 'novel', 'innovative', 'complex'
        ]
        
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in chaos_indicators):
            return True
            
        # Check if query would benefit from specific chaos modes
        if 'search' in query_lower and self.config.enable_attractor_hopping:
            return True
            
        if 'remember' in query_lower and self.config.enable_dark_solitons:
            return True
            
        if 'brainstorm' in query_lower and self.config.enable_phase_explosion:
            return True
            
        return False
        
    def _estimate_system_jacobian(self) -> np.ndarray:
        """Estimate current system Jacobian for eigenvalue analysis"""
        dim = self.state_manager.state_dim
        
        # Get current state
        state = self.state_manager.get_state()
        
        # Simple estimation based on state magnitude
        # In production, would use actual system dynamics
        jacobian = np.eye(dim)
        
        # Add some structure based on state
        for i in range(min(10, dim)):
            jacobian[i, i] = 0.5 + 0.1 * np.tanh(state[i])
            if i < dim - 1:
                jacobian[i, i+1] = 0.1 * state[i]
                
        return jacobian
        
    async def _handle_emergency(self):
        """Handle emergency conditions"""
        logger.error("Emergency handler activated")
        
        # Create emergency checkpoint
        checkpoint_id = await self.safety_system.create_checkpoint("emergency")
        
        # Attempt stabilization
        if self.config.emergency_rollback_enabled and checkpoint_id:
            # Try to rollback to last stable checkpoint
            checkpoints = list(self.safety_system.checkpoints)
            for checkpoint in reversed(checkpoints[:-1]):  # Skip the emergency one
                if checkpoint.metrics.fidelity > 0.9:
                    success = await self.safety_system.rollback_to_checkpoint(
                        checkpoint.checkpoint_id
                    )
                    if success:
                        logger.info(f"Rolled back to stable checkpoint {checkpoint.checkpoint_id}")
                        break
                        
        self.stats['safety_interventions'] += 1
        
    async def _periodic_checkpointing(self):
        """Create periodic safety checkpoints"""
        interval = self.config.safety_checkpoint_interval_minutes * 60
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Only checkpoint if system is stable
                safety_report = self.safety_system.get_safety_report()
                if safety_report['current_safety_level'] in ['optimal', 'nominal']:
                    await self.safety_system.create_checkpoint("periodic")
                    
                # Cleanup old checkpoints
                self.safety_system.cleanup_old_checkpoints()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpointing error: {e}")
                
    async def _monitor_wormhole_sync(self):
        """Monitor braid wormhole synchronization"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.braid_wormhole:
                    # Check connected instances
                    connected = self.braid_wormhole.get_connected_instances()
                    if connected:
                        # Extract consensus state
                        consensus = await self.braid_wormhole.extract_consensus()
                        if consensus:
                            self.stats['wormhole_syncs'] += 1
                            logger.info(f"Wormhole sync with {len(connected)} instances")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Wormhole monitoring error: {e}")
                
    async def _load_persisted_state(self):
        """Load persisted system state"""
        if not self.config.state_persistence_path:
            return
            
        state_file = self.config.state_persistence_path / "tori_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                    
                # Restore statistics
                self.stats.update(saved_state.get('stats', {}))
                
                # Restore adapter mode
                mode_str = saved_state.get('adapter_mode', 'HYBRID')
                self.adapter_system.set_adapter_mode(AdapterMode[mode_str])
                
                logger.info("Loaded persisted state")
                
            except Exception as e:
                logger.error(f"Failed to load persisted state: {e}")
                
    async def _persist_state(self):
        """Persist system state"""
        if not self.config.state_persistence_path:
            return
            
        # Create directory if needed
        self.config.state_persistence_path.mkdir(parents=True, exist_ok=True)
        
        state_file = self.config.state_persistence_path / "tori_state.json"
        
        try:
            # Gather state to save
            state_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stats': self.stats,
                'adapter_mode': self.adapter_system.global_config.mode.value,
                'safety_level': self.safety_system.current_safety_level.value
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.info("Persisted system state")
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
            
    # ========== Public API ==========
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'operational': True,
            'chaos_enabled': self.config.enable_chaos,
            'adapter_mode': self.adapter_system.global_config.mode.value,
            'safety': self.safety_system.get_safety_report(),
            'eigensentry': self.eigen_sentry.get_status(),
            'ccl': self.ccl.get_status(),
            'statistics': self.stats,
            'integrations': {
                'introspection': self.introspection_scheduler.get_status() if self.introspection_scheduler else None,
                'braid_wormhole': self.braid_wormhole.get_status() if self.braid_wormhole else None,
                'critic_consensus': self.critic_consensus.get_status() if self.critic_consensus else None
            },
            'config': {
                'max_concurrent_chaos': self.config.max_concurrent_chaos_tasks,
                'safety_monitoring': self.config.enable_safety_monitoring,
                'features': {
                    'dark_solitons': self.config.enable_dark_solitons,
                    'attractor_hopping': self.config.enable_attractor_hopping,
                    'phase_explosion': self.config.enable_phase_explosion,
                    'concept_evolution': self.config.enable_concept_evolution
                },
                'integrations': {
                    'introspection': self.config.enable_introspection,
                    'braid_wormhole': self.config.enable_braid_wormhole,
                    'critic_consensus': self.config.enable_critic_consensus
                }
            }
        }
        
    def set_chaos_mode(self, mode: AdapterMode):
        """Set the chaos adaptation mode"""
        self.adapter_system.set_adapter_mode(mode)
        logger.info(f"Chaos mode set to: {mode.value}")
        
    async def create_checkpoint(self, label: str = "manual") -> str:
        """Create a safety checkpoint"""
        return await self.safety_system.create_checkpoint(label)
        
    async def rollback(self, checkpoint_id: str) -> bool:
        """Rollback to a checkpoint"""
        return await self.safety_system.rollback_to_checkpoint(checkpoint_id)
        
    def get_efficiency_report(self) -> Dict[str, float]:
        """Get chaos efficiency report"""
        if not self.stats['energy_efficiency_gain']:
            return {
                'average_gain': 1.0,
                'samples': 0
            }
            
        gains = self.stats['energy_efficiency_gain']
        return {
            'average_gain': np.mean(gains),
            'max_gain': np.max(gains),
            'min_gain': np.min(gains),
            'samples': len(gains)
        }
        
    async def trigger_chaos_burst(self, intensity: float = 1.0, duration: float = None) -> Dict[str, Any]:
        """Manually trigger a chaos burst"""
        if not self.config.enable_chaos:
            return {'success': False, 'reason': 'chaos_disabled'}
            
        return await self.ccl.trigger_chaos_burst(intensity, duration)
        
    def get_introspection_metrics(self, scale: str = 'micro', count: int = 10) -> List[Any]:
        """Get recent introspection metrics"""
        if not self.introspection_scheduler:
            return []
            
        return self.introspection_scheduler.get_recent_metrics(scale, count)
        
    async def sync_with_instance(self, instance_id: str) -> bool:
        """Manually sync with another TORI instance"""
        if not self.braid_wormhole:
            return False
            
        # Force a sync
        await self.braid_wormhole._perform_sync()
        return instance_id in self.braid_wormhole.get_connected_instances()

# ========== Main Entry Point ==========

async def main():
    """Main entry point for TORI production system"""
    
    # Load configuration
    config = TORIProductionConfig()
    
    # Override from environment if available
    if os.getenv('TORI_ENABLE_CHAOS', '').lower() == 'false':
        config.enable_chaos = False
        
    if os.getenv('TORI_SAFETY_MONITORING', '').lower() == 'false':
        config.enable_safety_monitoring = False
        
    if os.getenv('TORI_INSTANCE_ID'):
        config.braid_instance_id = os.getenv('TORI_INSTANCE_ID')
        
    # Create system
    tori = TORIProductionSystem(config)
    
    # Start system
    await tori.start()
    
    try:
        # Example queries
        print("\n🚀 TORI Fully Integrated Production System")
        print("=" * 60)
        
        # Test queries
        test_queries = [
            "How does consciousness emerge from physical processes?",
            "Search for patterns in quantum entanglement data",
            "Remember the key insights about emergent systems",
            "Brainstorm novel approaches to artificial consciousness"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            result = await tori.process_query(query)
            print(f"📊 Response: {result['response'][:200]}...")
            print(f"⚡ Chaos enabled: {result['metadata']['chaos_enabled']}")
            print(f"🛡️ Safety level: {result['metadata']['safety_level']}")
            
            if result['metadata'].get('critic_consensus'):
                print(f"🎭 Critic approval: {result['metadata']['critic_consensus']['approved']}")
                
        # Show status
        print("\n📈 System Status:")
        status = tori.get_status()
        print(f"  Queries processed: {status['statistics']['queries_processed']}")
        print(f"  Chaos events: {status['statistics']['chaos_events']}")
        print(f"  Chaos bursts: {status['statistics']['chaos_bursts']}")
        print(f"  Safety interventions: {status['statistics']['safety_interventions']}")
        print(f"  Critic rejections: {status['statistics']['critic_rejections']}")
        print(f"  Wormhole syncs: {status['statistics']['wormhole_syncs']}")
        
        # Show efficiency
        efficiency = tori.get_efficiency_report()
        if efficiency['samples'] > 0:
            print(f"  Average efficiency gain: {efficiency['average_gain']:.2f}x")
            
        # Show integrations
        print("\n🔗 Active Integrations:")
        for name, integration in status['integrations'].items():
            if integration:
                print(f"  {name}: Active")
            
    finally:
        # Stop system
        await tori.stop()

# ========== Demo Script ==========

async def interactive_demo():
    """Interactive demo of TORI production system"""
    
    print("🎮 TORI Interactive Demo - Fully Integrated Edition")
    print("=" * 60)
    print("Commands:")
    print("  query <text> - Process a query")
    print("  status - Show system status")
    print("  chaos <mode> - Set chaos mode (passthrough/hybrid/chaos_assisted/chaos_only)")
    print("  burst <intensity> - Trigger chaos burst")
    print("  checkpoint - Create safety checkpoint")
    print("  efficiency - Show efficiency report")
    print("  metrics <scale> - Show introspection metrics (micro/meso/macro)")
    print("  sync - Show wormhole sync status")
    print("  exit - Exit demo")
    print("=" * 60)
    
    # Initialize system
    config = TORIProductionConfig()
    tori = TORIProductionSystem(config)
    await tori.start()
    
    try:
        while True:
            command = input("\n> ").strip()
            
            if command.startswith("query "):
                query = command[6:]
                result = await tori.process_query(query)
                print(f"\n{result['response']}")
                print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
                
            elif command == "status":
                status = tori.get_status()
                print(json.dumps(status, indent=2))
                
            elif command.startswith("chaos "):
                mode_str = command[6:].upper()
                try:
                    mode = AdapterMode[mode_str]
                    tori.set_chaos_mode(mode)
                    print(f"Chaos mode set to: {mode.value}")
                except KeyError:
                    print("Invalid mode. Use: passthrough, hybrid, chaos_assisted, or chaos_only")
                    
            elif command.startswith("burst "):
                try:
                    intensity = float(command[6:])
                    result = await tori.trigger_chaos_burst(intensity)
                    print(f"Chaos burst result: {json.dumps(result, indent=2)}")
                except ValueError:
                    print("Invalid intensity. Use a number between 0.1 and 5.0")
                    
            elif command == "checkpoint":
                checkpoint_id = await tori.create_checkpoint("interactive")
                print(f"Created checkpoint: {checkpoint_id}")
                
            elif command == "efficiency":
                report = tori.get_efficiency_report()
                print(f"Efficiency Report: {json.dumps(report, indent=2)}")
                
            elif command.startswith("metrics "):
                scale = command[8:]
                if scale in ['micro', 'meso', 'macro']:
                    metrics = tori.get_introspection_metrics(scale, 5)
                    print(f"Recent {scale} metrics:")
                    for m in metrics:
                        print(f"  {m}")
                else:
                    print("Invalid scale. Use: micro, meso, or macro")
                    
            elif command == "sync":
                if tori.braid_wormhole:
                    status = tori.braid_wormhole.get_status()
                    print(f"Wormhole status: {json.dumps(status, indent=2)}")
                else:
                    print("Braid Wormhole not enabled")
                    
            elif command == "exit":
                break
                
            else:
                print("Unknown command. Type 'exit' to quit.")
                
    finally:
        await tori.stop()

if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())
    
    # Uncomment for interactive demo
    # asyncio.run(interactive_demo())
