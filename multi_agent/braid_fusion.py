# /multi_agent/braid_fusion.py
"""
Multi-Agent Braid Fusion System
Enables multiple TORI instances to share and merge their self-knowledge in real-time.
Implements collective metacognition through distributed self-model synchronization.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import asyncio
import websockets
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from alan_backend.braid_aggregator import BraidAggregator
from python.core.soliton_memory_integration import SolitonMemory
from meta_genome.memory_bridge import MetacognitiveMemoryBridge
from audit.logger import log_event

class SyncMode(Enum):
    LOCAL = "local"      # Same machine, shared memory
    LAN = "lan"          # Local network, TCP/IP
    WAN = "wan"          # Internet, secure channels
    WORMHOLE = "wormhole"  # Direct concept mesh transfer

@dataclass
class BraidStrand:
    """Represents a unit of knowledge to be shared"""
    source_agent: str
    timestamp: datetime
    knowledge_type: str  # "insight", "error_pattern", "strategy", "relationship"
    content: Dict[str, Any]
    confidence: float
    signature: str  # Cryptographic hash for verification
    
    def to_soliton(self) -> Dict[str, Any]:
        """Convert to soliton wave packet for transmission"""
        return {
            "amplitude": self.confidence,
            "frequency": hash(self.knowledge_type) % 1000,
            "phase": self.timestamp.timestamp() % (2 * np.pi),
            "payload": self.content,
            "signature": self.signature
        }

class MultiBraidFusion:
    """
    Implements multi-agent knowledge fusion through braid synchronization.
    Each TORI instance can share partial self-models with others.
    """
    
    def __init__(self, agent_id: str, memory_bridge: MetacognitiveMemoryBridge,
                 sync_mode: SyncMode = SyncMode.LOCAL):
        self.agent_id = agent_id
        self.memory_bridge = memory_bridge
        self.sync_mode = sync_mode
        self.braid_aggregator = BraidAggregator()
        
        # Track known peers
        self.peer_agents: Set[str] = set()
        self.peer_reliabilities: Dict[str, float] = {}
        
        # Knowledge buffers
        self.outgoing_strands: List[BraidStrand] = []
        self.incoming_strands: List[BraidStrand] = []
        
        # Wormhole connections (direct concept mesh links)
        self.wormhole_links: Dict[str, 'ConceptMeshWormhole'] = {}
        
        # Conflict resolution policy
        self.conflict_policy = "confidence_weighted"  # or "newest", "consensus"
        
        # Start sync daemon if networked
        if sync_mode in [SyncMode.LAN, SyncMode.WAN]:
            asyncio.create_task(self._start_network_sync())
    
    def share_insight(self, insight_type: str, content: Dict[str, Any], 
                     confidence: float = 0.8):
        """Share a learned insight with peer agents"""
        strand = BraidStrand(
            source_agent=self.agent_id,
            timestamp=datetime.now(),
            knowledge_type=insight_type,
            content=content,
            confidence=confidence,
            signature=self._sign_content(content)
        )
        
        self.outgoing_strands.append(strand)
        
        # Log the sharing intent
        log_event("braid_share", {
            "agent_id": self.agent_id,
            "insight_type": insight_type,
            "confidence": confidence,
            "peer_count": len(self.peer_agents)
        })
        
        # Trigger immediate sync if high confidence
        if confidence > 0.9:
            asyncio.create_task(self._broadcast_strand(strand))
    
    def _sign_content(self, content: Dict[str, Any]) -> str:
        """Create cryptographic signature for content verification"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(f"{self.agent_id}:{content_str}".encode()).hexdigest()
    
    async def _broadcast_strand(self, strand: BraidStrand):
        """Broadcast a knowledge strand to all peers"""
        if self.sync_mode == SyncMode.LOCAL:
            # Direct memory sharing for local instances
            self._local_broadcast(strand)
        elif self.sync_mode in [SyncMode.LAN, SyncMode.WAN]:
            # Network broadcast
            await self._network_broadcast(strand)
        elif self.sync_mode == SyncMode.WORMHOLE:
            # Direct concept mesh injection
            self._wormhole_broadcast(strand)
    
    def _local_broadcast(self, strand: BraidStrand):
        """Share via shared memory (for local instances)"""
        # In production, this would use shared memory or IPC
        # For now, simulate with file-based exchange
        exchange_path = Path("multi_agent/local_exchange.jsonl")
        exchange_path.parent.mkdir(exist_ok=True)
        
        with open(exchange_path, "a") as f:
            f.write(json.dumps({
                "strand": strand.__dict__,
                "timestamp": datetime.now().isoformat()
            }) + "\n")
    
    async def _network_broadcast(self, strand: BraidStrand):
        """Broadcast over network to peer agents"""
        tasks = []
        for peer_id in self.peer_agents:
            if peer_id != self.agent_id:
                task = self._send_to_peer(peer_id, strand)
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_peer(self, peer_id: str, strand: BraidStrand):
        """Send strand to specific peer via websocket"""
        try:
            # In production, this would connect to peer's websocket endpoint
            uri = f"ws://tori-{peer_id}:8765"
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({
                    "type": "braid_strand",
                    "data": strand.to_soliton()
                }))
        except Exception as e:
            self._adjust_peer_reliability(peer_id, success=False)
            log_event("peer_send_error", {"peer": peer_id, "error": str(e)})
    
    def _wormhole_broadcast(self, strand: BraidStrand):
        """Inject knowledge directly into peer concept meshes via wormhole"""
        for peer_id, wormhole in self.wormhole_links.items():
            wormhole.inject_concept(strand)
    
    def receive_strand(self, strand: BraidStrand) -> bool:
        """
        Receive and integrate a knowledge strand from another agent.
        Returns True if successfully integrated.
        """
        # Verify signature
        expected_sig = self._sign_content(strand.content)
        if strand.signature != expected_sig:
            log_event("strand_verification_failed", {
                "source": strand.source_agent,
                "type": strand.knowledge_type
            })
            return False
        
        # Check if we already have this knowledge
        if self._is_duplicate(strand):
            return False
        
        # Add to incoming buffer
        self.incoming_strands.append(strand)
        
        # Process based on knowledge type
        if strand.knowledge_type == "error_pattern":
            self._integrate_error_pattern(strand)
        elif strand.knowledge_type == "strategy":
            self._integrate_strategy(strand)
        elif strand.knowledge_type == "insight":
            self._integrate_insight(strand)
        elif strand.knowledge_type == "relationship":
            self._integrate_relationship(strand)
        
        # Update peer reliability
        self._adjust_peer_reliability(strand.source_agent, success=True)
        
        return True
    
    def _is_duplicate(self, strand: BraidStrand) -> bool:
        """Check if we already have this knowledge"""
        # Simple duplicate detection based on content hash
        content_hash = hashlib.md5(
            json.dumps(strand.content, sort_keys=True).encode()
        ).hexdigest()
        
        # Check recent strands
        for existing in self.incoming_strands[-100:]:
            existing_hash = hashlib.md5(
                json.dumps(existing.content, sort_keys=True).encode()
            ).hexdigest()
            if content_hash == existing_hash:
                return True
        
        return False
    
    def _integrate_error_pattern(self, strand: BraidStrand):
        """Integrate error pattern from another agent"""
        error_type = strand.content.get("error_type", "unknown")
        context = strand.content.get("context", {})
        
        # Add to our error patterns with source attribution
        context["learned_from"] = strand.source_agent
        context["confidence"] = strand.confidence
        
        self.memory_bridge.record_error_pattern(error_type, context)
        
        # Reflect on learning from peer
        self.memory_bridge.add_self_reflection("peer_learning", {
            "thought": f"Learned about {error_type} errors from {strand.source_agent}",
            "integration_time": datetime.now().isoformat()
        })
    
    def _integrate_strategy(self, strand: BraidStrand):
        """Integrate strategy from another agent"""
        strategy_name = strand.content.get("name", "unnamed_strategy")
        performance = strand.content.get("performance_metrics", {})
        
        # Weight by peer reliability and strand confidence
        peer_reliability = self.peer_reliabilities.get(strand.source_agent, 0.5)
        integration_weight = peer_reliability * strand.confidence
        
        # Add to our strategy evolution with weighted metrics
        weighted_metrics = {
            k: v * integration_weight 
            for k, v in performance.items() 
            if isinstance(v, (int, float))
        }
        
        self.memory_bridge.evolve_strategy(
            f"{strategy_name}_from_{strand.source_agent}",
            weighted_metrics
        )
    
    def _integrate_insight(self, strand: BraidStrand):
        """Integrate general insight from another agent"""
        self.memory_bridge.add_self_reflection("peer_insight", {
            "thought": strand.content.get("thought", ""),
            "source_agent": strand.source_agent,
            "confidence": strand.confidence,
            "integrated_at": datetime.now().isoformat()
        })
    
    def _integrate_relationship(self, strand: BraidStrand):
        """Integrate relationship knowledge from another agent"""
        # Only integrate if high confidence (privacy consideration)
        if strand.confidence < 0.8:
            return
        
        entity_id = strand.content.get("entity_id")
        attributes = strand.content.get("attributes", {})
        
        if entity_id:
            # Add source attribution
            attributes["shared_by"] = strand.source_agent
            self.memory_bridge.remember_entity(f"shared_{entity_id}", attributes)
    
    def _adjust_peer_reliability(self, peer_id: str, success: bool):
        """Update reliability score for a peer using Bayesian update"""
        current = self.peer_reliabilities.get(peer_id, 0.5)
        
        # Simple Bayesian update
        if success:
            new_reliability = (current * 2 + 1) / 3  # Move toward 1
        else:
            new_reliability = (current * 2) / 3      # Move toward 0
        
        self.peer_reliabilities[peer_id] = new_reliability
    
    def establish_wormhole(self, peer_id: str, concept_mesh_endpoint: str):
        """Establish direct concept mesh wormhole with peer"""
        wormhole = ConceptMeshWormhole(
            self.agent_id, 
            peer_id,
            concept_mesh_endpoint
        )
        self.wormhole_links[peer_id] = wormhole
        
        log_event("wormhole_established", {
            "local_agent": self.agent_id,
            "remote_agent": peer_id,
            "endpoint": concept_mesh_endpoint
        })
    
    def get_collective_knowledge_summary(self) -> Dict[str, Any]:
        """Summarize knowledge gained from collective"""
        peer_contributions = {}
        
        for strand in self.incoming_strands:
            agent = strand.source_agent
            if agent not in peer_contributions:
                peer_contributions[agent] = {
                    "insights": 0, "errors": 0, 
                    "strategies": 0, "relationships": 0
                }
            
            if strand.knowledge_type == "insight":
                peer_contributions[agent]["insights"] += 1
            elif strand.knowledge_type == "error_pattern":
                peer_contributions[agent]["errors"] += 1
            elif strand.knowledge_type == "strategy":
                peer_contributions[agent]["strategies"] += 1
            elif strand.knowledge_type == "relationship":
                peer_contributions[agent]["relationships"] += 1
        
        return {
            "total_strands_received": len(self.incoming_strands),
            "total_strands_shared": len(self.outgoing_strands),
            "peer_count": len(self.peer_agents),
            "peer_reliabilities": self.peer_reliabilities,
            "peer_contributions": peer_contributions,
            "wormhole_connections": list(self.wormhole_links.keys()),
            "sync_mode": self.sync_mode.value
        }
    
    async def _start_network_sync(self):
        """Start network synchronization daemon"""
        # This would implement the full network protocol
        # For now, it's a placeholder for the architecture
        pass
    
    def initiate_collective_introspection(self) -> Dict[str, Any]:
        """
        Trigger collective introspection across all connected agents.
        Each agent introspects, then shares key insights.
        """
        # First, do our own introspection
        local_introspection = self.memory_bridge.add_self_reflection(
            "collective_introspection_participant",
            {
                "thought": "Participating in collective introspection",
                "agent_id": self.agent_id,
                "peer_count": len(self.peer_agents)
            }
        )
        
        # Share our introspection as high-confidence insight
        self.share_insight(
            "collective_introspection",
            {
                "agent_id": self.agent_id,
                "phase": "introspecting",
                "thought": "What have I learned from my peers?"
            },
            confidence=0.95
        )
        
        # Request introspections from peers
        for peer_id in self.peer_agents:
            # This would send introspection request
            pass
        
        return {
            "status": "collective_introspection_initiated",
            "participants": list(self.peer_agents) + [self.agent_id]
        }


class ConceptMeshWormhole:
    """
    Direct high-bandwidth link between concept meshes of different agents.
    Enables instantaneous concept transfer bypassing normal learning paths.
    """
    
    def __init__(self, local_agent: str, remote_agent: str, endpoint: str):
        self.local_agent = local_agent
        self.remote_agent = remote_agent
        self.endpoint = endpoint
        
        # Soliton memory for stable transfer
        self.soliton_memory = SolitonMemory()
        
        # Track transferred concepts
        self.transferred_concepts: List[Dict] = []
    
    def inject_concept(self, strand: BraidStrand):
        """
        Inject a concept directly into remote agent's concept mesh.
        Uses bright/dark solitons for stable transmission.
        """
        # Determine soliton type based on content
        if self._is_knowledge_gap(strand):
            soliton_type = "dark"  # Represents missing knowledge
        else:
            soliton_type = "bright"  # Represents new knowledge
        
        # Package as soliton wave
        soliton = self._create_soliton(strand, soliton_type)
        
        # Transmit through wormhole
        self._transmit_soliton(soliton)
        
        # Record transfer
        self.transferred_concepts.append({
            "timestamp": datetime.now().isoformat(),
            "strand": strand.knowledge_type,
            "soliton_type": soliton_type,
            "success": True
        })
    
    def _is_knowledge_gap(self, strand: BraidStrand) -> bool:
        """Determine if this represents a knowledge gap vs new knowledge"""
        # Simple heuristic: questions or unknowns are gaps
        content_str = json.dumps(strand.content).lower()
        gap_indicators = ["unknown", "question", "missing", "need", "gap"]
        return any(indicator in content_str for indicator in gap_indicators)
    
    def _create_soliton(self, strand: BraidStrand, soliton_type: str) -> Dict:
        """Create a soliton wave packet from knowledge strand"""
        base_soliton = strand.to_soliton()
        
        if soliton_type == "dark":
            # Dark soliton: stable dip representing knowledge gap
            base_soliton["amplitude"] *= -1  # Invert amplitude
            base_soliton["stability"] = 0.95  # Higher stability
        else:
            # Bright soliton: concentrated knowledge pulse
            base_soliton["amplitude"] *= 1.5  # Boost amplitude
            base_soliton["stability"] = 0.85
        
        base_soliton["type"] = soliton_type
        return base_soliton
    
    def _transmit_soliton(self, soliton: Dict):
        """Transmit soliton through wormhole connection"""
        # In production, this would use the actual endpoint
        # For now, we simulate with soliton memory recording
        self.soliton_memory.record_thought(
            f"Wormhole transmission: {soliton['type']} soliton",
            metadata={
                "soliton": soliton,
                "source": self.local_agent,
                "destination": self.remote_agent
            }
        )
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get statistics on concept transfers through this wormhole"""
        bright_count = sum(1 for t in self.transferred_concepts 
                          if t["soliton_type"] == "bright")
        dark_count = sum(1 for t in self.transferred_concepts 
                        if t["soliton_type"] == "dark")
        
        return {
            "total_transfers": len(self.transferred_concepts),
            "bright_solitons": bright_count,
            "dark_solitons": dark_count,
            "link_stability": 0.95,  # Would be calculated from success rate
            "bandwidth_utilization": len(self.transferred_concepts) / 100  # Normalized
        }
