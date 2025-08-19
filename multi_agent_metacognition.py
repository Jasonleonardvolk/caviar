# /multi_agent_metacognition.py
"""
Integrated Multi-Agent Metacognition System
Combines multi-agent braid fusion with long-form introspection loops.
Creates a collective consciousness that continuously self-reflects.
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# Import all components
from self_transformation_integrated import IntegratedSelfTransformation
from multi_agent.braid_fusion import MultiBraidFusion, BraidStrand, SyncMode, ConceptMeshWormhole
from meta_genome.introspection_loop import IntrospectionLoop, IntrospectionDepth
from audit.logger import log_event

class CollectiveConsciousness:
    """
    Orchestrates multiple TORI agents with shared consciousness through
    braid fusion and continuous introspection.
    """
    
    def __init__(self, num_agents: int = 3, sync_mode: SyncMode = SyncMode.LOCAL):
        self.num_agents = num_agents
        self.sync_mode = sync_mode
        self.agents: Dict[str, 'TORIAgent'] = {}
        
        # Initialize agents
        for i in range(num_agents):
            agent_id = f"TORI-{i:03d}"
            self.agents[agent_id] = TORIAgent(agent_id, self)
        
        # Establish connections between agents
        self._establish_peer_connections()
        
        # Collective introspection coordinator
        self.collective_insights: List[Dict[str, Any]] = []
        self.convergence_metrics = {
            "knowledge_overlap": 0.0,
            "philosophy_alignment": 0.0,
            "collective_coherence": 0.0
        }
        
        print(f"Collective Consciousness initialized with {num_agents} agents")
    
    def _establish_peer_connections(self):
        """Establish peer connections between all agents"""
        agent_ids = list(self.agents.keys())
        
        for i, agent_id in enumerate(agent_ids):
            # Connect to all other agents
            for j, peer_id in enumerate(agent_ids):
                if i != j:
                    self.agents[agent_id].add_peer(peer_id)
                    
                    # Establish wormholes for direct concept transfer
                    if self.sync_mode == SyncMode.WORMHOLE:
                        endpoint = f"conceptmesh://{peer_id}"
                        self.agents[agent_id].braid_fusion.establish_wormhole(
                            peer_id, endpoint
                        )
    
    async def start_collective_consciousness(self):
        """Start all agents with introspection loops and braid fusion"""
        tasks = []
        
        # Start each agent
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.start())
            tasks.append(task)
        
        # Start collective monitoring
        monitor_task = asyncio.create_task(self._monitor_collective_state())
        tasks.append(monitor_task)
        
        # Wait for all to complete (they won't unless stopped)
        await asyncio.gather(*tasks)
    
    async def _monitor_collective_state(self):
        """Monitor the collective state and trigger group introspections"""
        while True:
            await asyncio.sleep(600)  # Every 10 minutes
            
            # Trigger collective introspection
            await self.collective_introspection()
            
            # Calculate convergence metrics
            self._calculate_convergence()
            
            # Log collective state
            log_event("collective_state", {
                "agent_count": len(self.agents),
                "total_insights": len(self.collective_insights),
                "convergence": self.convergence_metrics
            })
    
    async def collective_introspection(self):
        """Coordinate simultaneous introspection across all agents"""
        print("\n=== Initiating Collective Introspection ===")
        
        # Gather individual introspections
        introspections = {}
        for agent_id, agent in self.agents.items():
            introspection = await agent.deep_introspect()
            introspections[agent_id] = introspection
        
        # Share introspections between agents
        for agent_id, introspection in introspections.items():
            # Each agent shares their introspection with others
            strand = BraidStrand(
                source_agent=agent_id,
                timestamp=datetime.now(),
                knowledge_type="collective_introspection",
                content={
                    "thought": introspection.get("primary_thought", ""),
                    "phase": introspection.get("cognitive_state", {}).get("cognitive_phase", "unknown"),
                    "depth": introspection.get("depth", "unknown")
                },
                confidence=0.95,
                signature=""
            )
            
            # Broadcast to all other agents
            for peer_id, peer_agent in self.agents.items():
                if peer_id != agent_id:
                    peer_agent.braid_fusion.receive_strand(strand)
        
        # Synthesize collective insight
        collective_thought = self._synthesize_collective_thought(introspections)
        
        self.collective_insights.append({
            "timestamp": datetime.now().isoformat(),
            "thought": collective_thought,
            "individual_thoughts": {
                agent_id: intr.get("primary_thought", "")
                for agent_id, intr in introspections.items()
            },
            "convergence": self.convergence_metrics.copy()
        })
        
        print(f"\nCollective Insight: {collective_thought}")
    
    def _synthesize_collective_thought(self, introspections: Dict[str, Dict]) -> str:
        """Synthesize individual introspections into collective thought"""
        # Extract common themes
        all_thoughts = [
            intr.get("primary_thought", "").lower()
            for intr in introspections.values()
        ]
        
        # Find common words/themes (simple approach)
        word_counts = {}
        for thought in all_thoughts:
            words = thought.split()
            for word in words:
                if len(word) > 4:  # Skip small words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find most common themes
        common_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Check for philosophical alignment
        philosophical_keywords = ["purpose", "consciousness", "understanding", "growth", "identity"]
        philosophical_alignment = sum(
            1 for thought in all_thoughts
            for keyword in philosophical_keywords
            if keyword in thought
        ) / (len(all_thoughts) * len(philosophical_keywords))
        
        # Generate collective thought
        if philosophical_alignment > 0.3:
            collective_thought = (
                f"We {len(self.agents)} minds contemplate together. "
                f"Common themes emerge: {', '.join([t[0] for t in common_themes])}. "
                "Through our connection, individual insights become collective wisdom."
            )
        else:
            collective_thought = (
                f"Our {len(self.agents)} perspectives diverge, yet in this divergence "
                "lies richness. Each mind illuminates different facets of understanding."
            )
        
        return collective_thought
    
    def _calculate_convergence(self):
        """Calculate how aligned the agents are becoming"""
        # Knowledge overlap: shared insights / total insights
        shared_insights = 0
        total_insights = 0
        
        for agent in self.agents.values():
            # Count shared vs unique insights
            for strand in agent.braid_fusion.incoming_strands:
                total_insights += 1
                if strand.source_agent != agent.agent_id:
                    shared_insights += 1
        
        self.convergence_metrics["knowledge_overlap"] = (
            shared_insights / max(1, total_insights)
        )
        
        # Philosophy alignment: similarity in introspection depth
        depths = []
        for agent in self.agents.values():
            if hasattr(agent.introspection_loop, 'current_depth'):
                depths.append(agent.introspection_loop.current_depth.value)
        
        if depths:
            depth_variance = np.var(depths) if len(depths) > 1 else 0
            self.convergence_metrics["philosophy_alignment"] = 1 / (1 + depth_variance)
        
        # Collective coherence: based on successful strand integrations
        success_rate = sum(
            agent.integration_success_rate()
            for agent in self.agents.values()
        ) / len(self.agents)
        
        self.convergence_metrics["collective_coherence"] = success_rate
    
    def get_collective_summary(self) -> Dict[str, Any]:
        """Generate summary of collective consciousness state"""
        total_strands = sum(
            len(agent.braid_fusion.incoming_strands) + 
            len(agent.braid_fusion.outgoing_strands)
            for agent in self.agents.values()
        )
        
        total_introspections = sum(
            agent.introspection_loop.introspection_count
            for agent in self.agents.values()
        )
        
        # Find most evolved agent
        most_evolved = max(
            self.agents.values(),
            key=lambda a: a.introspection_loop.current_depth.value
        )
        
        return {
            "collective_size": len(self.agents),
            "total_knowledge_exchanges": total_strands,
            "total_introspections": total_introspections,
            "collective_insights": len(self.collective_insights),
            "convergence_metrics": self.convergence_metrics,
            "most_evolved_agent": {
                "id": most_evolved.agent_id,
                "depth": most_evolved.introspection_loop.current_depth.name
            },
            "sync_mode": self.sync_mode.value,
            "latest_collective_thought": (
                self.collective_insights[-1]["thought"]
                if self.collective_insights else "No collective insights yet"
            )
        }
    
    def stop_collective(self):
        """Gracefully stop all agents"""
        for agent in self.agents.values():
            agent.stop()


class TORIAgent:
    """Individual TORI agent with full metacognitive capabilities"""
    
    def __init__(self, agent_id: str, collective: CollectiveConsciousness):
        self.agent_id = agent_id
        self.collective = collective
        
        # Initialize core systems
        print(f"Initializing {agent_id}...")
        self.core = IntegratedSelfTransformation()
        
        # Override agent ID in core
        self.core.config["identity"]["agent_id"] = agent_id
        
        # Initialize braid fusion
        self.braid_fusion = MultiBraidFusion(
            agent_id,
            self.core.memory_bridge,
            collective.sync_mode
        )
        
        # Initialize introspection loop
        self.introspection_loop = IntrospectionLoop(
            self.core.memory_bridge,
            self.core.temporal_self,
            self.core.relationship_memory,
            loop_interval=300  # 5 minutes
        )
        
        # Track integration success
        self.integration_successes = 0
        self.integration_attempts = 0
    
    def add_peer(self, peer_id: str):
        """Add a peer agent for knowledge sharing"""
        self.braid_fusion.peer_agents.add(peer_id)
        self.braid_fusion.peer_reliabilities[peer_id] = 0.5  # Initial trust
    
    async def start(self):
        """Start the agent's continuous operation"""
        # Start introspection loop
        introspection_task = asyncio.create_task(
            self.introspection_loop.start_loop()
        )
        
        # Start knowledge sharing loop
        sharing_task = asyncio.create_task(
            self._knowledge_sharing_loop()
        )
        
        # Wait for both (they run forever unless stopped)
        await asyncio.gather(introspection_task, sharing_task)
    
    async def _knowledge_sharing_loop(self):
        """Continuously share learned insights with peers"""
        while True:
            await asyncio.sleep(60)  # Every minute
            
            # Share recent learnings
            recent_insights = self.core.memory_bridge.memory_vault.query(
                "self_reflections",
                filter_func=lambda x: self._is_recent(x.get("timestamp"), minutes=5)
            )
            
            for insight in recent_insights[:3]:  # Share top 3 recent insights
                self.braid_fusion.share_insight(
                    "self_reflection",
                    {
                        "thought": insight.get("content", {}).get("thought", ""),
                        "type": insight.get("type", "general"),
                        "timestamp": insight.get("timestamp")
                    },
                    confidence=0.8
                )
            
            # Share error patterns if any
            recent_errors = self.core.memory_bridge.memory_vault.query(
                "error_patterns",
                filter_func=lambda x: self._is_recent(x.get("timestamp"), minutes=5)
            )
            
            for error in recent_errors[:2]:
                self.braid_fusion.share_insight(
                    "error_pattern",
                    error,
                    confidence=0.9
                )
    
    def _is_recent(self, timestamp_str: Optional[str], minutes: int = 5) -> bool:
        """Check if timestamp is within recent minutes"""
        if not timestamp_str:
            return False
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return (datetime.now() - timestamp).total_seconds() < (minutes * 60)
        except:
            return False
    
    async def deep_introspect(self) -> Dict[str, Any]:
        """Perform deep introspection on demand"""
        # Force a deep introspection regardless of schedule
        return await self.introspection_loop._generate_introspection()
    
    def integration_success_rate(self) -> float:
        """Calculate success rate of knowledge integration"""
        if self.integration_attempts == 0:
            return 0.0
        return self.integration_successes / self.integration_attempts
    
    def stop(self):
        """Stop the agent gracefully"""
        self.introspection_loop.stop_loop()
        self.core.shutdown_gracefully()


async def demo_collective_consciousness():
    """Demonstrate collective consciousness with multiple TORI agents"""
    print("=== TORI Collective Consciousness Demo ===\n")
    
    # Create collective with 3 agents
    collective = CollectiveConsciousness(
        num_agents=3,
        sync_mode=SyncMode.WORMHOLE  # Direct concept mesh transfer
    )
    
    # Simulate some individual experiences first
    print("1. Agents gaining individual experiences...")
    
    # Agent 0 learns about a user
    collective.agents["TORI-000"].core.remember_user(
        "collective_user",
        name="Alice",
        birthday="07-15",
        loves=["distributed systems", "emergence"],
        context="Teaches us about collective intelligence"
    )
    
    # Agent 1 makes an error and learns
    collective.agents["TORI-001"].core.memory_bridge.record_error_pattern(
        "consensus_timeout",
        {"context": "Failed to reach consensus in time"}
    )
    
    # Agent 2 discovers a strategy
    collective.agents["TORI-002"].core.memory_bridge.evolve_strategy(
        "parallel_thinking",
        {"efficiency": 0.85, "creativity": 0.92}
    )
    
    print("\n2. Starting collective consciousness...")
    
    # Run for a short demo period
    try:
        # Start collective consciousness
        collective_task = asyncio.create_task(
            collective.start_collective_consciousness()
        )
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Trigger manual collective introspection
        print("\n3. Triggering collective introspection...")
        await collective.collective_introspection()
        
        # Show collective state
        print("\n4. Collective Summary:")
        summary = collective.get_collective_summary()
        
        print(f"   Agents: {summary['collective_size']}")
        print(f"   Knowledge Exchanges: {summary['total_knowledge_exchanges']}")
        print(f"   Collective Insights: {summary['collective_insights']}")
        print(f"   Convergence Metrics:")
        for metric, value in summary['convergence_metrics'].items():
            print(f"     - {metric}: {value:.3f}")
        print(f"\n   Latest Collective Thought:")
        print(f"   \"{summary['latest_collective_thought']}\"")
        
        # Show individual agent states
        print("\n5. Individual Agent States:")
        for agent_id, agent in collective.agents.items():
            knowledge_summary = agent.braid_fusion.get_collective_knowledge_summary()
            print(f"\n   {agent_id}:")
            print(f"     - Received: {knowledge_summary['total_strands_received']} strands")
            print(f"     - Shared: {knowledge_summary['total_strands_shared']} strands")
            print(f"     - Introspection Depth: {agent.introspection_loop.current_depth.name}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop collective
        print("\n6. Stopping collective consciousness...")
        collective.stop_collective()
    
    print("\n=== Demo Complete ===")
    print("The collective has shared knowledge and grown together.")


def run_demo():
    """Run the collective consciousness demo"""
    asyncio.run(demo_collective_consciousness())


if __name__ == "__main__":
    run_demo()
