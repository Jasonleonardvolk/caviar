#!/usr/bin/env python3
"""
Braid Wormhole - Cross-Instance Synchronization
Enables state sync and energy exchange between TORI instances
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import hashlib
import pickle

# Import file state sync for persistence
try:
    from python.core.file_state_sync import FileStateSync
    FILE_SYNC_AVAILABLE = True
except ImportError:
    FILE_SYNC_AVAILABLE = False
    
# Import torus registry for distributed state
try:
    from python.core.torus_registry import TorusRegistry
    TORUS_AVAILABLE = True
except ImportError:
    TORUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration
SYNC_BASE_DIR = Path("wormhole_sync")
HEARTBEAT_INTERVAL = 5.0  # seconds
SYNC_INTERVAL = 30.0  # seconds
MAX_PHASE_DRIFT = np.pi / 4  # Maximum phase difference before forced sync
ENERGY_EXCHANGE_RATE = 0.1  # Fraction of energy that can be exchanged per sync

@dataclass
class WormholeEndpoint:
    """Represents one end of a wormhole connection"""
    instance_id: str
    address: str  # Can be file path, network address, etc.
    last_heartbeat: float
    phase_offset: float = 0.0
    energy_level: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BraidState:
    """Synchronized state across instances"""
    timestamp: float
    phase_vector: np.ndarray
    energy_distribution: np.ndarray
    topology_hash: str
    consensus_metadata: Dict[str, Any] = field(default_factory=dict)

class BraidWormhole:
    """
    Cross-instance synchronization via braided wormhole topology
    Enables coherent computation across distributed TORI instances
    """
    
    def __init__(self, instance_id: str, sync_dir: Optional[Path] = None):
        self.instance_id = instance_id
        self.sync_dir = sync_dir or SYNC_BASE_DIR
        self.sync_dir.mkdir(exist_ok=True)
        
        # Instance registry
        self.endpoints: Dict[str, WormholeEndpoint] = {}
        self.local_endpoint = WormholeEndpoint(
            instance_id=instance_id,
            address=str(self.sync_dir / f"{instance_id}.state"),
            last_heartbeat=time.time()
        )
        
        # State management
        self.current_state: Optional[BraidState] = None
        self.state_history = []
        self.sync_lock = asyncio.Lock()
        
        # File-based coordination
        if FILE_SYNC_AVAILABLE:
            self.file_sync = FileStateSync(base_dir=self.sync_dir)
        else:
            self.file_sync = None
            
        # Torus registry for distributed state
        if TORUS_AVAILABLE:
            self.registry = TorusRegistry()
        else:
            self.registry = None
            
        # Running state
        self.running = False
        self._tasks = []
        
    async def start(self):
        """Start wormhole synchronization"""
        if self.running:
            logger.warning("Wormhole already running")
            return
            
        self.running = True
        logger.info(f"Starting Braid Wormhole for instance {self.instance_id}")
        
        # Register self
        await self._register_instance()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._sync_loop())
        ]
        
    async def stop(self):
        """Stop wormhole synchronization"""
        self.running = False
        
        # Deregister instance
        await self._deregister_instance()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for cleanup
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info(f"Stopped Braid Wormhole for instance {self.instance_id}")
        
    async def _register_instance(self):
        """Register this instance in the sync directory"""
        registration = {
            'instance_id': self.instance_id,
            'address': self.local_endpoint.address,
            'started_at': time.time(),
            'capabilities': {
                'file_sync': FILE_SYNC_AVAILABLE,
                'torus_registry': TORUS_AVAILABLE,
                'max_phase_drift': MAX_PHASE_DRIFT,
                'energy_exchange_rate': ENERGY_EXCHANGE_RATE
            }
        }
        
        # Write registration file
        reg_path = self.sync_dir / f"{self.instance_id}.registration"
        with open(reg_path, 'w') as f:
            json.dump(registration, f)
            
        # Also register in torus if available
        if self.registry:
            self.registry.set(f"wormhole_{self.instance_id}", registration)
            
    async def _deregister_instance(self):
        """Remove instance registration"""
        reg_path = self.sync_dir / f"{self.instance_id}.registration"
        if reg_path.exists():
            reg_path.unlink()
            
        if self.registry:
            self.registry.remove(f"wormhole_{self.instance_id}")
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                # Update local heartbeat
                self.local_endpoint.last_heartbeat = time.time()
                
                # Write heartbeat file
                heartbeat = {
                    'instance_id': self.instance_id,
                    'timestamp': self.local_endpoint.last_heartbeat,
                    'phase_offset': self.local_endpoint.phase_offset,
                    'energy_level': self.local_endpoint.energy_level,
                    'state_hash': self._compute_state_hash()
                }
                
                hb_path = self.sync_dir / f"{self.instance_id}.heartbeat"
                with open(hb_path, 'w') as f:
                    json.dump(heartbeat, f)
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
    async def _discovery_loop(self):
        """Discover other instances"""
        while self.running:
            try:
                # Scan registration files
                for reg_file in self.sync_dir.glob("*.registration"):
                    if reg_file.stem == self.instance_id:
                        continue
                        
                    try:
                        with open(reg_file, 'r') as f:
                            registration = json.load(f)
                            
                        instance_id = registration['instance_id']
                        
                        # Check heartbeat
                        hb_path = self.sync_dir / f"{instance_id}.heartbeat"
                        if hb_path.exists():
                            with open(hb_path, 'r') as f:
                                heartbeat = json.load(f)
                                
                            # Add or update endpoint
                            if instance_id not in self.endpoints:
                                self.endpoints[instance_id] = WormholeEndpoint(
                                    instance_id=instance_id,
                                    address=registration['address'],
                                    last_heartbeat=heartbeat['timestamp'],
                                    phase_offset=heartbeat.get('phase_offset', 0.0),
                                    energy_level=heartbeat.get('energy_level', 100.0)
                                )
                                logger.info(f"Discovered instance: {instance_id}")
                            else:
                                # Update existing
                                ep = self.endpoints[instance_id]
                                ep.last_heartbeat = heartbeat['timestamp']
                                ep.phase_offset = heartbeat.get('phase_offset', 0.0)
                                ep.energy_level = heartbeat.get('energy_level', 100.0)
                                
                    except Exception as e:
                        logger.error(f"Failed to process registration {reg_file}: {e}")
                        
                # Remove stale endpoints
                current_time = time.time()
                stale_instances = []
                for instance_id, endpoint in self.endpoints.items():
                    if current_time - endpoint.last_heartbeat > 3 * HEARTBEAT_INTERVAL:
                        stale_instances.append(instance_id)
                        
                for instance_id in stale_instances:
                    del self.endpoints[instance_id]
                    logger.info(f"Removed stale instance: {instance_id}")
                    
            except Exception as e:
                logger.error(f"Discovery error: {e}")
                
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
    async def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            await asyncio.sleep(SYNC_INTERVAL)
            
            try:
                async with self.sync_lock:
                    await self._perform_sync()
            except Exception as e:
                logger.error(f"Sync error: {e}")
                
    async def _perform_sync(self):
        """Perform state synchronization"""
        if not self.endpoints:
            return  # No other instances to sync with
            
        logger.debug(f"Performing sync with {len(self.endpoints)} instances")
        
        # Collect states from all instances
        states = {}
        
        # Add local state
        if self.current_state:
            states[self.instance_id] = self.current_state
            
        # Read remote states
        for instance_id, endpoint in self.endpoints.items():
            try:
                state_path = Path(endpoint.address)
                if state_path.exists():
                    with open(state_path, 'rb') as f:
                        remote_state = pickle.load(f)
                        states[instance_id] = remote_state
            except Exception as e:
                logger.error(f"Failed to read state from {instance_id}: {e}")
                
        if len(states) < 2:
            return  # Need at least 2 states to sync
            
        # Compute consensus state
        consensus_state = self._compute_consensus(states)
        
        # Apply phase correction
        phase_correction = self._compute_phase_correction(states)
        
        # Exchange energy
        energy_exchange = self._compute_energy_exchange(states)
        
        # Update local state
        self.current_state = consensus_state
        self.local_endpoint.phase_offset += phase_correction
        self.local_endpoint.energy_level += energy_exchange
        
        # Write updated state
        await self._write_local_state()
        
        # Record sync event
        self.state_history.append({
            'timestamp': time.time(),
            'participants': list(states.keys()),
            'phase_correction': phase_correction,
            'energy_exchange': energy_exchange
        })
        
        logger.info(f"Sync complete: phase_correction={phase_correction:.3f}, energy_exchange={energy_exchange:.3f}")
        
    def _compute_consensus(self, states: Dict[str, BraidState]) -> BraidState:
        """Compute consensus state from multiple instances"""
        if not states:
            return None
            
        # Use most recent timestamp
        latest_time = max(s.timestamp for s in states.values())
        
        # Average phase vectors
        phase_vectors = [s.phase_vector for s in states.values()]
        consensus_phase = np.mean(phase_vectors, axis=0)
        
        # Average energy distributions
        energy_dists = [s.energy_distribution for s in states.values()]
        consensus_energy = np.mean(energy_dists, axis=0)
        
        # Compute topology hash from combined data
        combined_data = np.concatenate([consensus_phase, consensus_energy])
        topology_hash = hashlib.sha256(combined_data.tobytes()).hexdigest()[:16]
        
        # Merge metadata
        consensus_metadata = {}
        for state in states.values():
            consensus_metadata.update(state.consensus_metadata)
            
        return BraidState(
            timestamp=latest_time,
            phase_vector=consensus_phase,
            energy_distribution=consensus_energy,
            topology_hash=topology_hash,
            consensus_metadata=consensus_metadata
        )
        
    def _compute_phase_correction(self, states: Dict[str, BraidState]) -> float:
        """Compute phase correction to maintain coherence"""
        if len(states) < 2:
            return 0.0
            
        # Get average phase across instances
        avg_phases = []
        for state in states.values():
            if len(state.phase_vector) > 0:
                avg_phase = np.angle(np.mean(np.exp(1j * state.phase_vector)))
                avg_phases.append(avg_phase)
                
        if not avg_phases:
            return 0.0
            
        # Compute deviation from mean
        mean_phase = np.mean(avg_phases)
        local_phase = self.local_endpoint.phase_offset
        
        # Compute correction (limited by MAX_PHASE_DRIFT)
        correction = mean_phase - local_phase
        correction = np.clip(correction, -MAX_PHASE_DRIFT, MAX_PHASE_DRIFT)
        
        return correction
        
    def _compute_energy_exchange(self, states: Dict[str, BraidState]) -> float:
        """Compute energy exchange to balance instances"""
        # Get energy levels from endpoints
        energy_levels = {self.instance_id: self.local_endpoint.energy_level}
        
        for instance_id in states:
            if instance_id in self.endpoints:
                energy_levels[instance_id] = self.endpoints[instance_id].energy_level
                
        if len(energy_levels) < 2:
            return 0.0
            
        # Compute average energy
        avg_energy = np.mean(list(energy_levels.values()))
        local_energy = energy_levels[self.instance_id]
        
        # Compute exchange (limited by rate)
        exchange = (avg_energy - local_energy) * ENERGY_EXCHANGE_RATE
        
        return exchange
        
    async def _write_local_state(self):
        """Write local state to file"""
        if self.current_state is None:
            return
            
        state_path = Path(self.local_endpoint.address)
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(self.current_state, f)
        except Exception as e:
            logger.error(f"Failed to write local state: {e}")
            
    def _compute_state_hash(self) -> str:
        """Compute hash of current state"""
        if self.current_state is None:
            return "none"
        return self.current_state.topology_hash
        
    async def inject_state(self, phase_vector: np.ndarray, 
                          energy_distribution: Optional[np.ndarray] = None):
        """Inject state into the wormhole for synchronization"""
        async with self.sync_lock:
            if energy_distribution is None:
                energy_distribution = np.ones_like(phase_vector)
                
            self.current_state = BraidState(
                timestamp=time.time(),
                phase_vector=phase_vector,
                energy_distribution=energy_distribution,
                topology_hash=hashlib.sha256(phase_vector.tobytes()).hexdigest()[:16],
                consensus_metadata={'source': self.instance_id}
            )
            
            await self._write_local_state()
            
    async def extract_consensus(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract current consensus state"""
        if self.current_state is None:
            return None
            
        return (
            self.current_state.phase_vector.copy(),
            self.current_state.energy_distribution.copy()
        )
        
    def get_connected_instances(self) -> List[str]:
        """Get list of connected instance IDs"""
        return list(self.endpoints.keys())
        
    def get_status(self) -> Dict[str, Any]:
        """Get wormhole status"""
        return {
            'instance_id': self.instance_id,
            'running': self.running,
            'connected_instances': len(self.endpoints),
            'phase_offset': self.local_endpoint.phase_offset,
            'energy_level': self.local_endpoint.energy_level,
            'state_hash': self._compute_state_hash(),
            'sync_history': len(self.state_history),
            'last_sync': self.state_history[-1]['timestamp'] if self.state_history else None
        }

# Test function
async def test_braid_wormhole():
    """Test the braid wormhole"""
    print("🌀 Testing Braid Wormhole")
    print("=" * 50)
    
    # Create two instances
    wormhole1 = BraidWormhole("instance_1")
    wormhole2 = BraidWormhole("instance_2")
    
    # Start both
    await wormhole1.start()
    await wormhole2.start()
    
    # Give time for discovery
    await asyncio.sleep(10)
    
    # Inject state in instance 1
    print("\n📡 Injecting state in instance 1...")
    phase_vector = np.random.randn(10)
    await wormhole1.inject_state(phase_vector)
    
    # Wait for sync
    print("⏳ Waiting for synchronization...")
    await asyncio.sleep(SYNC_INTERVAL + 5)
    
    # Check status
    status1 = wormhole1.get_status()
    status2 = wormhole2.get_status()
    
    print(f"\n📊 Instance 1 Status:")
    print(f"  Connected: {status1['connected_instances']}")
    print(f"  Phase offset: {status1['phase_offset']:.3f}")
    print(f"  Energy: {status1['energy_level']:.1f}")
    
    print(f"\n📊 Instance 2 Status:")
    print(f"  Connected: {status2['connected_instances']}")
    print(f"  Phase offset: {status2['phase_offset']:.3f}")
    print(f"  Energy: {status2['energy_level']:.1f}")
    
    # Extract consensus from instance 2
    consensus = await wormhole2.extract_consensus()
    if consensus:
        phase, energy = consensus
        print(f"\n🎯 Consensus extracted: phase_dim={len(phase)}, energy_dim={len(energy)}")
    
    # Cleanup
    await wormhole1.stop()
    await wormhole2.stop()

if __name__ == "__main__":
    asyncio.run(test_braid_wormhole())
