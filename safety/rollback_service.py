#!/usr/bin/env python3
"""
Rollback Watchdog Service - Automatic state recovery for CCL
Monitors system health and triggers rollback when necessary
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import pickle
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class SystemCheckpoint:
    """System state checkpoint"""
    checkpoint_id: str
    timestamp: float
    state_snapshot: Dict[str, Any]
    metrics: Dict[str, float]
    is_stable: bool

class RollbackWatchdog:
    """
    Automatic rollback service for chaos system safety
    Monitors health metrics and triggers rollback on violations
    """
    
    def __init__(self, ccl, eigen_sentry, energy_broker):
        self.ccl = ccl
        self.eigen_sentry = eigen_sentry
        self.energy_broker = energy_broker
        
        # Watchdog configuration
        self.check_interval = 1.0  # seconds
        self.checkpoint_interval = 30.0  # seconds
        self.max_checkpoints = 100
        
        # Health thresholds
        self.max_lyapunov_threshold = 0.1  # Emergency threshold
        self.min_energy_threshold = 10  # Minimum system energy
        self.max_error_rate = 0.05  # 5% error rate
        self.min_coherence = 0.5  # Phase coherence threshold
        
        # State management
        self.checkpoints: Deque[SystemCheckpoint] = deque(maxlen=self.max_checkpoints)
        self.current_errors = 0
        self.total_checks = 0
        self.rollback_count = 0
        
        # Watchdog state
        self.running = False
        self.last_checkpoint_time = 0
        
    async def start(self):
        """Start watchdog service"""
        self.running = True
        logger.info("Rollback watchdog started")
        
        # Start monitoring loops
        asyncio.create_task(self._health_monitor_loop())
        asyncio.create_task(self._checkpoint_loop())
        
    async def stop(self):
        """Stop watchdog service"""
        self.running = False
        logger.info("Rollback watchdog stopped")
        
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        while self.running:
            try:
                # Check system health
                health = await self._check_system_health()
                self.total_checks += 1
                
                # Determine if rollback needed
                if self._should_rollback(health):
                    await self._execute_rollback(health)
                    
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.current_errors += 1
                await asyncio.sleep(self.check_interval)
                
    async def _checkpoint_loop(self):
        """Periodic checkpoint creation"""
        while self.running:
            try:
                current_time = asyncio.get_event_loop().time()
                
                if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
                    # Check if system is stable enough to checkpoint
                    health = await self._check_system_health()
                    
                    if health['is_stable']:
                        checkpoint = await self._create_checkpoint("periodic")
                        self.checkpoints.append(checkpoint)
                        self.last_checkpoint_time = current_time
                        logger.info(f"Created checkpoint {checkpoint.checkpoint_id}")
                        
                await asyncio.sleep(self.checkpoint_interval / 10)
                
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
                await asyncio.sleep(self.checkpoint_interval)
                
    async def _check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health = {
            'timestamp': datetime.now(timezone.utc).timestamp(),
            'is_stable': True,
            'violations': []
        }
        
        # Check Lyapunov exponents
        eigen_status = self.eigen_sentry.get_status()
        max_lyapunov = eigen_status.get('current_max_eigenvalue', 0)
        health['max_lyapunov'] = max_lyapunov
        
        if max_lyapunov > self.max_lyapunov_threshold:
            health['is_stable'] = False
            health['violations'].append(f"Lyapunov too high: {max_lyapunov:.3f}")
            
        # Check energy levels
        energy_status = self.energy_broker.get_status()
        total_energy = sum(energy_status.get('module_balances', {}).values())
        health['total_energy'] = total_energy
        
        if total_energy < self.min_energy_threshold:
            health['is_stable'] = False
            health['violations'].append(f"Low energy: {total_energy}")
            
        # Check error rate
        if self.total_checks > 0:
            error_rate = self.current_errors / self.total_checks
            health['error_rate'] = error_rate
            
            if error_rate > self.max_error_rate:
                health['is_stable'] = False
                health['violations'].append(f"High error rate: {error_rate:.1%}")
                
        # Check CCL status
        ccl_status = self.ccl.get_status()
        active_sessions = ccl_status.get('active_sessions', 0)
        health['active_chaos_sessions'] = active_sessions
        
        # Check phase coherence (from pump if available)
        if hasattr(self.ccl, 'lyap_pump'):
            pump_status = self.ccl.lyap_pump.get_status()
            recent_gains = pump_status.get('recent_lyapunov', [])
            if recent_gains:
                coherence = 1.0 / (1.0 + np.std(recent_gains))
                health['phase_coherence'] = coherence
                
                if coherence < self.min_coherence:
                    health['is_stable'] = False
                    health['violations'].append(f"Low coherence: {coherence:.3f}")
                    
        return health
        
    def _should_rollback(self, health: Dict[str, Any]) -> bool:
        """Determine if rollback is necessary"""
        if not health['is_stable']:
            # Critical violations trigger immediate rollback
            critical_violations = [
                v for v in health['violations']
                if 'Lyapunov too high' in v or 'Low energy' in v
            ]
            
            if critical_violations:
                logger.warning(f"Critical violations detected: {critical_violations}")
                return True
                
            # Multiple non-critical violations
            if len(health['violations']) >= 3:
                logger.warning(f"Multiple violations: {health['violations']}")
                return True
                
        return False
        
    async def _execute_rollback(self, health: Dict[str, Any]):
        """Execute system rollback to stable state"""
        logger.error(f"Executing rollback due to: {health['violations']}")
        self.rollback_count += 1
        
        # Find best checkpoint to rollback to
        best_checkpoint = self._find_best_checkpoint()
        
        if not best_checkpoint:
            logger.error("No valid checkpoint found for rollback!")
            # Emergency shutdown
            await self._emergency_shutdown()
            return
            
        try:
            # Restore system state
            await self._restore_checkpoint(best_checkpoint)
            logger.info(f"Successfully rolled back to {best_checkpoint.checkpoint_id}")
            
            # Reset error counter after successful rollback
            self.current_errors = 0
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            await self._emergency_shutdown()
            
    def _find_best_checkpoint(self) -> Optional[SystemCheckpoint]:
        """Find the best stable checkpoint to rollback to"""
        # Search backwards for most recent stable checkpoint
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.is_stable:
                # Verify checkpoint is not too old (max 5 minutes)
                age = asyncio.get_event_loop().time() - checkpoint.timestamp
                if age < 300:  # 5 minutes
                    return checkpoint
                    
        return None
        
    async def _create_checkpoint(self, label: str) -> SystemCheckpoint:
        """Create a system checkpoint"""
        checkpoint_id = f"ckpt_{label}_{int(time.time())}"
        
        # Capture system state
        state_snapshot = {
            'ccl_state': self._capture_ccl_state(),
            'energy_state': self._capture_energy_state(),
            'eigen_state': self._capture_eigen_state()
        }
        
        # Capture metrics
        health = await self._check_system_health()
        metrics = {
            'max_lyapunov': health.get('max_lyapunov', 0),
            'total_energy': health.get('total_energy', 0),
            'error_rate': health.get('error_rate', 0),
            'phase_coherence': health.get('phase_coherence', 1)
        }
        
        checkpoint = SystemCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=asyncio.get_event_loop().time(),
            state_snapshot=state_snapshot,
            metrics=metrics,
            is_stable=health['is_stable']
        )
        
        # Persist to disk
        self._save_checkpoint_to_disk(checkpoint)
        
        return checkpoint
        
    def _capture_ccl_state(self) -> Dict[str, Any]:
        """Capture CCL state"""
        return {
            'active_sessions': list(self.ccl.active_sessions.keys()),
            'total_chaos_generated': self.ccl.total_chaos_generated,
            'config': {
                'max_lyapunov': self.ccl.config.max_lyapunov,
                'target_lyapunov': self.ccl.config.target_lyapunov,
                'energy_threshold': self.ccl.config.energy_threshold
            }
        }
        
    def _capture_energy_state(self) -> Dict[str, Any]:
        """Capture energy broker state"""
        return {
            'module_balances': dict(self.energy_broker._credits),
            'total_spent': self.energy_broker._total_energy_spent
        }
        
    def _capture_eigen_state(self) -> Dict[str, Any]:
        """Capture EigenSentry state"""
        return self.eigen_sentry.get_status()
        
    async def _restore_checkpoint(self, checkpoint: SystemCheckpoint):
        """Restore system from checkpoint"""
        state = checkpoint.state_snapshot
        
        # Stop all active chaos sessions
        for session_id in list(self.ccl.active_sessions.keys()):
            await self.ccl.exit_chaos_session(session_id)
            
        # Restore CCL config
        ccl_state = state['ccl_state']
        self.ccl.config.max_lyapunov = ccl_state['config']['max_lyapunov']
        self.ccl.config.target_lyapunov = ccl_state['config']['target_lyapunov']
        self.ccl.config.energy_threshold = ccl_state['config']['energy_threshold']
        self.ccl.total_chaos_generated = ccl_state['total_chaos_generated']
        
        # Restore energy state
        energy_state = state['energy_state']
        self.energy_broker._credits = energy_state['module_balances'].copy()
        self.energy_broker._total_energy_spent = energy_state['total_spent']
        
        # Reset Lyapunov estimators
        if hasattr(self.ccl, 'lyap_estimator'):
            self.ccl.lyap_estimator.reset()
            
    def _save_checkpoint_to_disk(self, checkpoint: SystemCheckpoint):
        """Persist checkpoint to disk"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(checkpoint_dir, f"{checkpoint.checkpoint_id}.pkl")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    async def _emergency_shutdown(self):
        """Emergency system shutdown"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Close all topological gates
        if hasattr(self, 'topo_switch'):
            self.topo_switch.emergency_close_all()
            
        # Stop CCL
        await self.ccl.stop()
        
        # Drain all energy
        for module in self.energy_broker._credits:
            self.energy_broker._credits[module] = 0
            
        self.running = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get watchdog status"""
        return {
            'running': self.running,
            'total_checks': self.total_checks,
            'current_errors': self.current_errors,
            'error_rate': self.current_errors / max(1, self.total_checks),
            'rollback_count': self.rollback_count,
            'checkpoint_count': len(self.checkpoints),
            'latest_checkpoint': self.checkpoints[-1].checkpoint_id if self.checkpoints else None
        }

# Test the watchdog
async def test_rollback_watchdog():
    """Test the rollback watchdog"""
    print("🛡️ Testing Rollback Watchdog")
    print("=" * 50)
    
    from unittest.mock import Mock
    
    # Mock components
    ccl = Mock()
    ccl.get_status = Mock(return_value={'active_sessions': 0})
    ccl.active_sessions = {}
    ccl.total_chaos_generated = 0
    ccl.config = Mock(max_lyapunov=0.05, target_lyapunov=0.02, energy_threshold=100)
    
    eigen_sentry = Mock()
    eigen_sentry.get_status = Mock(return_value={'current_max_eigenvalue': 0.03})
    
    energy_broker = Mock()
    energy_broker.get_status = Mock(return_value={'module_balances': {'test': 100}})
    energy_broker._credits = {'test': 100}
    energy_broker._total_energy_spent = 0
    
    # Create watchdog
    watchdog = RollbackWatchdog(ccl, eigen_sentry, energy_broker)
    
    # Start watchdog
    await watchdog.start()
    
    # Run for a bit
    await asyncio.sleep(2)
    
    # Simulate instability
    eigen_sentry.get_status = Mock(return_value={'current_max_eigenvalue': 0.15})
    
    # Wait for rollback
    await asyncio.sleep(2)
    
    # Check status
    status = watchdog.get_status()
    print(f"\nWatchdog Status:")
    print(f"  Total checks: {status['total_checks']}")
    print(f"  Rollback count: {status['rollback_count']}")
    print(f"  Error rate: {status['error_rate']:.1%}")
    
    await watchdog.stop()

if __name__ == "__main__":
    asyncio.run(test_rollback_watchdog())
