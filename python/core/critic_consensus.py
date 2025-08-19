#!/usr/bin/env python3
"""
Critic Consensus - Weighted Voting and Rollback Mechanism
Implements multi-critic consensus for safe decision making
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)

# Consensus parameters
MIN_CRITICS = 3
CONSENSUS_THRESHOLD = 0.67  # 2/3 majority
ROLLBACK_WINDOW = 100  # Keep last 100 decisions for rollback
CRITIC_TIMEOUT = 5.0  # seconds

class VoteType(Enum):
    """Types of votes critics can cast"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    ROLLBACK = "rollback"

@dataclass
class CriticVote:
    """Individual critic's vote"""
    critic_id: str
    vote_type: VoteType
    confidence: float  # 0-1
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    """A decision to be evaluated by critics"""
    decision_id: str
    action_type: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: float
    requester: Optional[str] = None

@dataclass
class ConsensusResult:
    """Result of critic consensus"""
    decision_id: str
    approved: bool
    vote_count: Dict[str, int]
    weighted_score: float
    critics_participated: List[str]
    rollback_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Critic:
    """Base class for system critics"""
    
    def __init__(self, critic_id: str, weight: float = 1.0):
        self.critic_id = critic_id
        self.weight = weight
        self.vote_history = deque(maxlen=1000)
        
    async def evaluate(self, decision: Decision) -> CriticVote:
        """Evaluate a decision and return vote"""
        raise NotImplementedError("Subclasses must implement evaluate()")
        
    def get_performance_score(self) -> float:
        """Get critic's historical performance score"""
        if not self.vote_history:
            return 0.5  # Neutral starting score
            
        # Calculate accuracy based on outcomes
        correct_votes = sum(1 for v in self.vote_history if v.get('correct', False))
        return correct_votes / len(self.vote_history)

class SafetyCritic(Critic):
    """Critic focused on system safety"""
    
    def __init__(self):
        super().__init__("safety_critic", weight=2.0)  # Higher weight for safety
        self.safety_thresholds = {
            'max_eigenvalue': 0.95,
            'min_energy': 10.0,
            'max_phase_spread': np.pi
        }
        
    async def evaluate(self, decision: Decision) -> CriticVote:
        """Evaluate decision from safety perspective"""
        # Check if decision involves risky parameters
        risky_actions = ['chaos_burst', 'eigenvalue_push', 'emergency_override']
        
        if decision.action_type in risky_actions:
            # Check safety conditions
            context = decision.context
            
            violations = []
            if context.get('current_eigenvalue', 0) > self.safety_thresholds['max_eigenvalue']:
                violations.append("eigenvalue_too_high")
                
            if context.get('energy_level', 100) < self.safety_thresholds['min_energy']:
                violations.append("insufficient_energy")
                
            if violations:
                return CriticVote(
                    critic_id=self.critic_id,
                    vote_type=VoteType.REJECT,
                    confidence=0.9,
                    reasoning=f"Safety violations: {', '.join(violations)}",
                    metadata={'violations': violations}
                )
                
        # Default: approve with caution
        return CriticVote(
            critic_id=self.critic_id,
            vote_type=VoteType.APPROVE,
            confidence=0.7,
            reasoning="No safety concerns detected"
        )

class EfficiencyCritic(Critic):
    """Critic focused on computational efficiency"""
    
    def __init__(self):
        super().__init__("efficiency_critic", weight=1.0)
        self.efficiency_history = deque(maxlen=100)
        
    async def evaluate(self, decision: Decision) -> CriticVote:
        """Evaluate decision from efficiency perspective"""
        # Check expected efficiency
        expected_gain = decision.parameters.get('expected_efficiency_gain', 1.0)
        
        if expected_gain < 0.5:
            return CriticVote(
                critic_id=self.critic_id,
                vote_type=VoteType.REJECT,
                confidence=0.8,
                reasoning=f"Expected efficiency gain too low: {expected_gain}",
                metadata={'expected_gain': expected_gain}
            )
            
        # Check resource usage
        resource_cost = decision.parameters.get('resource_cost', 0)
        if resource_cost > 1000:  # Arbitrary threshold
            return CriticVote(
                critic_id=self.critic_id,
                vote_type=VoteType.ABSTAIN,
                confidence=0.6,
                reasoning="High resource cost, deferring to other critics",
                metadata={'resource_cost': resource_cost}
            )
            
        return CriticVote(
            critic_id=self.critic_id,
            vote_type=VoteType.APPROVE,
            confidence=0.8,
            reasoning=f"Efficiency gain {expected_gain}x justifies action"
        )

class StabilityCritic(Critic):
    """Critic focused on system stability"""
    
    def __init__(self):
        super().__init__("stability_critic", weight=1.5)
        self.stability_window = deque(maxlen=50)
        
    async def evaluate(self, decision: Decision) -> CriticVote:
        """Evaluate decision from stability perspective"""
        # Check recent stability metrics
        recent_instabilities = decision.context.get('recent_instabilities', 0)
        
        if recent_instabilities > 5:
            # System has been unstable, recommend rollback
            last_stable = decision.context.get('last_stable_checkpoint')
            if last_stable:
                return CriticVote(
                    critic_id=self.critic_id,
                    vote_type=VoteType.ROLLBACK,
                    confidence=0.85,
                    reasoning=f"System unstable ({recent_instabilities} events), recommend rollback",
                    metadata={'rollback_to': last_stable}
                )
            else:
                return CriticVote(
                    critic_id=self.critic_id,
                    vote_type=VoteType.REJECT,
                    confidence=0.9,
                    reasoning="System unstable, no safe rollback point"
                )
                
        # Check if action might destabilize
        destabilizing_actions = ['phase_explosion', 'coupling_increase', 'damping_disable']
        if decision.action_type in destabilizing_actions:
            return CriticVote(
                critic_id=self.critic_id,
                vote_type=VoteType.ABSTAIN,
                confidence=0.5,
                reasoning="Potentially destabilizing action, monitoring required"
            )
            
        return CriticVote(
            critic_id=self.critic_id,
            vote_type=VoteType.APPROVE,
            confidence=0.75,
            reasoning="No stability concerns"
        )

class CriticConsensus:
    """
    Manages multiple critics and consensus mechanism
    """
    
    def __init__(self):
        self.critics: Dict[str, Critic] = {}
        self.decision_history = deque(maxlen=ROLLBACK_WINDOW)
        self.pending_decisions: Dict[str, Decision] = {}
        self.consensus_callbacks: List[Callable] = []
        
        # Initialize default critics
        self._init_default_critics()
        
    def _init_default_critics(self):
        """Initialize the default critic ensemble"""
        self.add_critic(SafetyCritic())
        self.add_critic(EfficiencyCritic())
        self.add_critic(StabilityCritic())
        
    def add_critic(self, critic: Critic):
        """Add a critic to the ensemble"""
        self.critics[critic.critic_id] = critic
        logger.info(f"Added critic: {critic.critic_id} (weight: {critic.weight})")
        
    def remove_critic(self, critic_id: str):
        """Remove a critic from the ensemble"""
        if critic_id in self.critics:
            del self.critics[critic_id]
            logger.info(f"Removed critic: {critic_id}")
            
    def register_callback(self, callback: Callable):
        """Register callback for consensus results"""
        self.consensus_callbacks.append(callback)
        
    async def evaluate_decision(self, decision: Decision) -> ConsensusResult:
        """
        Evaluate a decision through all critics
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            ConsensusResult with voting outcome
        """
        if len(self.critics) < MIN_CRITICS:
            logger.warning(f"Insufficient critics ({len(self.critics)} < {MIN_CRITICS})")
            return ConsensusResult(
                decision_id=decision.decision_id,
                approved=False,
                vote_count={'insufficient_critics': 1},
                weighted_score=0.0,
                critics_participated=[],
                metadata={'error': 'insufficient_critics'}
            )
            
        # Store pending decision
        self.pending_decisions[decision.decision_id] = decision
        
        # Collect votes from all critics
        votes = await self._collect_votes(decision)
        
        # Compute consensus
        result = self._compute_consensus(decision, votes)
        
        # Store in history
        self.decision_history.append({
            'decision': decision,
            'result': result,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Remove from pending
        del self.pending_decisions[decision.decision_id]
        
        # Notify callbacks
        for callback in self.consensus_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Consensus callback error: {e}")
                
        return result
        
    async def _collect_votes(self, decision: Decision) -> List[CriticVote]:
        """Collect votes from all critics"""
        votes = []
        
        # Create tasks for parallel evaluation
        tasks = []
        for critic_id, critic in self.critics.items():
            task = asyncio.create_task(
                self._get_critic_vote(critic, decision)
            )
            tasks.append((critic_id, task))
            
        # Wait for all votes with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
                timeout=CRITIC_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning("Some critics timed out")
            
        # Collect results
        for critic_id, task in tasks:
            try:
                if task.done():
                    vote = task.result()
                    if isinstance(vote, CriticVote):
                        votes.append(vote)
                    else:
                        logger.error(f"Invalid vote from {critic_id}: {vote}")
            except Exception as e:
                logger.error(f"Failed to get vote from {critic_id}: {e}")
                
        return votes
        
    async def _get_critic_vote(self, critic: Critic, decision: Decision) -> CriticVote:
        """Get vote from a single critic"""
        try:
            vote = await critic.evaluate(decision)
            # Update critic's history
            critic.vote_history.append({
                'decision_id': decision.decision_id,
                'vote': vote.vote_type.value,
                'confidence': vote.confidence
            })
            return vote
        except Exception as e:
            logger.error(f"Critic {critic.critic_id} evaluation failed: {e}")
            # Return abstain on error
            return CriticVote(
                critic_id=critic.critic_id,
                vote_type=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning=f"Evaluation error: {e}"
            )
            
    def _compute_consensus(self, decision: Decision, votes: List[CriticVote]) -> ConsensusResult:
        """Compute weighted consensus from votes"""
        # Count votes by type
        vote_count = {
            VoteType.APPROVE.value: 0,
            VoteType.REJECT.value: 0,
            VoteType.ABSTAIN.value: 0,
            VoteType.ROLLBACK.value: 0
        }
        
        # Weighted scores
        weighted_approve = 0.0
        weighted_reject = 0.0
        weighted_rollback = 0.0
        total_weight = 0.0
        
        # Rollback targets
        rollback_targets = {}
        
        # Process each vote
        critics_participated = []
        for vote in votes:
            critic = self.critics.get(vote.critic_id)
            if not critic:
                continue
                
            critics_participated.append(vote.critic_id)
            vote_count[vote.vote_type.value] += 1
            
            # Apply weight and confidence
            vote_weight = critic.weight * vote.confidence
            total_weight += critic.weight
            
            if vote.vote_type == VoteType.APPROVE:
                weighted_approve += vote_weight
            elif vote.vote_type == VoteType.REJECT:
                weighted_reject += vote_weight
            elif vote.vote_type == VoteType.ROLLBACK:
                weighted_rollback += vote_weight
                # Track rollback target
                target = vote.metadata.get('rollback_to')
                if target:
                    rollback_targets[target] = rollback_targets.get(target, 0) + vote_weight
                    
        # Normalize scores
        if total_weight > 0:
            weighted_approve /= total_weight
            weighted_reject /= total_weight
            weighted_rollback /= total_weight
        
        # Determine outcome
        approved = False
        rollback_to = None
        
        if weighted_rollback > CONSENSUS_THRESHOLD:
            # Rollback takes precedence
            approved = False
            # Find most voted rollback target
            if rollback_targets:
                rollback_to = max(rollback_targets.items(), key=lambda x: x[1])[0]
        elif weighted_approve > weighted_reject and weighted_approve > CONSENSUS_THRESHOLD:
            approved = True
        
        # Calculate overall weighted score
        weighted_score = weighted_approve - weighted_reject - weighted_rollback
        
        return ConsensusResult(
            decision_id=decision.decision_id,
            approved=approved,
            vote_count=vote_count,
            weighted_score=weighted_score,
            critics_participated=critics_participated,
            rollback_to=rollback_to,
            metadata={
                'weighted_approve': weighted_approve,
                'weighted_reject': weighted_reject,
                'weighted_rollback': weighted_rollback,
                'total_weight': total_weight
            }
        )
        
    async def execute_rollback(self, checkpoint_id: str) -> bool:
        """
        Execute rollback to a previous checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            Success flag
        """
        logger.info(f"Executing rollback to {checkpoint_id}")
        
        # Find the checkpoint in history
        checkpoint = None
        for entry in self.decision_history:
            if entry['decision'].decision_id == checkpoint_id:
                checkpoint = entry
                break
                
        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
            
        # Create rollback decision
        rollback_decision = Decision(
            decision_id=f"rollback_{checkpoint_id}",
            action_type="system_rollback",
            parameters={'target_checkpoint': checkpoint_id},
            context={'reason': 'critic_consensus_rollback'},
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        
        # Evaluate through critics (they might have opinions on the rollback itself)
        result = await self.evaluate_decision(rollback_decision)
        
        return result.approved
        
    def get_critic_performance(self) -> Dict[str, float]:
        """Get performance scores for all critics"""
        scores = {}
        for critic_id, critic in self.critics.items():
            scores[critic_id] = critic.get_performance_score()
        return scores
        
    def adjust_critic_weights(self):
        """Adjust critic weights based on performance"""
        scores = self.get_critic_performance()
        
        for critic_id, score in scores.items():
            if critic_id in self.critics:
                # Adjust weight based on performance
                # Good performance (>0.7) increases weight
                # Poor performance (<0.3) decreases weight
                adjustment = (score - 0.5) * 0.2  # ±0.1 adjustment
                new_weight = self.critics[critic_id].weight * (1 + adjustment)
                new_weight = np.clip(new_weight, 0.5, 3.0)  # Keep weights reasonable
                
                self.critics[critic_id].weight = new_weight
                logger.info(f"Adjusted {critic_id} weight to {new_weight:.2f} (score: {score:.2f})")
                
    def get_status(self) -> Dict[str, Any]:
        """Get consensus system status"""
        return {
            'critics': list(self.critics.keys()),
            'critic_weights': {cid: c.weight for cid, c in self.critics.items()},
            'pending_decisions': len(self.pending_decisions),
            'history_size': len(self.decision_history),
            'performance_scores': self.get_critic_performance()
        }

# Test function
async def test_critic_consensus():
    """Test the critic consensus system"""
    print("🎭 Testing Critic Consensus")
    print("=" * 50)
    
    # Create consensus system
    consensus = CriticConsensus()
    
    # Test 1: Safe decision
    print("\n1️⃣ Testing safe decision...")
    safe_decision = Decision(
        decision_id="test_safe_001",
        action_type="parameter_update",
        parameters={'learning_rate': 0.01},
        context={'current_eigenvalue': 0.5, 'energy_level': 80},
        timestamp=datetime.now(timezone.utc).timestamp()
    )
    
    result = await consensus.evaluate_decision(safe_decision)
    print(f"Result: {'APPROVED' if result.approved else 'REJECTED'}")
    print(f"Weighted score: {result.weighted_score:.3f}")
    print(f"Votes: {result.vote_count}")
    
    # Test 2: Risky decision
    print("\n2️⃣ Testing risky decision...")
    risky_decision = Decision(
        decision_id="test_risky_002",
        action_type="chaos_burst",
        parameters={'intensity': 5.0, 'expected_efficiency_gain': 0.3},
        context={'current_eigenvalue': 0.98, 'energy_level': 15},
        timestamp=datetime.now(timezone.utc).timestamp()
    )
    
    result = await consensus.evaluate_decision(risky_decision)
    print(f"Result: {'APPROVED' if result.approved else 'REJECTED'}")
    print(f"Weighted score: {result.weighted_score:.3f}")
    print(f"Votes: {result.vote_count}")
    
    # Test 3: Unstable system decision
    print("\n3️⃣ Testing decision in unstable system...")
    unstable_decision = Decision(
        decision_id="test_unstable_003",
        action_type="phase_explosion",
        parameters={},
        context={
            'recent_instabilities': 8,
            'last_stable_checkpoint': 'checkpoint_001'
        },
        timestamp=datetime.now(timezone.utc).timestamp()
    )
    
    result = await consensus.evaluate_decision(unstable_decision)
    print(f"Result: {'APPROVED' if result.approved else 'REJECTED'}")
    print(f"Rollback to: {result.rollback_to}")
    print(f"Votes: {result.vote_count}")
    
    # Test 4: Status check
    print("\n4️⃣ Consensus status:")
    status = consensus.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_critic_consensus())
