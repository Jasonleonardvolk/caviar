#!/usr/bin/env python3
"""
Synthetic Data Generator
=========================
Builds fine-tune data from mesh contexts, intent gaps, and psi archive.
Generates Q&A pairs for continuous adapter improvement.
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.concept_mesh_v5 import MeshManager
from python.core.conversation_manager import IntentGap

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

INTENT_VAULT_DIR = Path("data/intent_vault")
TRAINING_DATA_DIR = Path("data/training")
VALIDATION_DATA_DIR = Path("data/validation")
PSI_ARCHIVE_DIR = Path("data/psi_archive")

MIN_CONFIDENCE_THRESHOLD = 0.7
MAX_SAMPLES_PER_USER = 1000
VALIDATION_SPLIT = 0.1

# Template patterns for Q&A generation
QA_TEMPLATES = [
    {
        "pattern": "What is {concept}?",
        "response": "{concept} is {description}. It relates to {related_concepts}."
    },
    {
        "pattern": "Explain {concept} in the context of {domain}",
        "response": "In {domain}, {concept} refers to {description}. This is important because {importance}."
    },
    {
        "pattern": "How does {concept1} relate to {concept2}?",
        "response": "{concept1} and {concept2} are connected through {relationship}. Specifically, {details}."
    },
    {
        "pattern": "Can you describe the {attribute} of {concept}?",
        "response": "The {attribute} of {concept} is characterized by {characteristics}. This means {explanation}."
    },
    {
        "pattern": "What are the applications of {concept}?",
        "response": "{concept} has several applications including {applications}. The most notable is {primary_application}."
    }
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SyntheticSample:
    """Represents a synthetic training sample."""
    user_id: str
    prompt: str
    response: str
    source: str  # "mesh", "intent", "psi", "template"
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": f"User: {self.prompt}\nAssistant: {self.response}",
            "prompt": self.prompt,
            "response": self.response,
            "user_id": self.user_id,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class GenerationStats:
    """Statistics for data generation."""
    total_samples: int = 0
    mesh_samples: int = 0
    intent_samples: int = 0
    psi_samples: int = 0
    template_samples: int = 0
    users_processed: int = 0
    average_confidence: float = 0.0

# ============================================================================
# DATA SOURCES
# ============================================================================

class DataSourceManager:
    """Manages various data sources for synthesis."""
    
    def __init__(self):
        self.mesh_manager = MeshManager()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all data directories exist."""
        INTENT_VAULT_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PSI_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_intent_gaps(self, user_id: Optional[str] = None) -> List[IntentGap]:
        """Load intent gaps from vault."""
        gaps = []
        
        # Read from intent gap logs
        for gap_file in INTENT_VAULT_DIR.glob("gaps_*.jsonl"):
            try:
                with open(gap_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            gap_data = json.loads(line)
                            if not user_id or gap_data.get("user_id") == user_id:
                                # Reconstruct IntentGap (simplified)
                                gaps.append(gap_data)
            except Exception as e:
                logger.error(f"Failed to load gaps from {gap_file}: {e}")
        
        return gaps
    
    def load_conversation_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """Load conversation history from vault."""
        conversations = []
        
        for conv_file in INTENT_VAULT_DIR.glob("conversations_*.jsonl"):
            try:
                with open(conv_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            conv = json.loads(line)
                            if not user_id or conv.get("user_id") == user_id:
                                conversations.append(conv)
            except Exception as e:
                logger.error(f"Failed to load conversations from {conv_file}: {e}")
        
        return conversations
    
    def load_psi_archive(self, user_id: Optional[str] = None) -> List[Dict]:
        """Load psi-morphon synthesis archive."""
        psi_data = []
        
        synthesis_file = PSI_ARCHIVE_DIR / "synthesis_log.jsonl"
        if synthesis_file.exists():
            try:
                with open(synthesis_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            psi_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load psi archive: {e}")
        
        return psi_data

# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticDataGenerator:
    """Generates synthetic training data from multiple sources."""
    
    def __init__(self):
        self.source_manager = DataSourceManager()
        self.mesh_manager = MeshManager()
        self.stats = GenerationStats()
    
    def generate_for_user(self, 
                         user_id: str,
                         max_samples: int = MAX_SAMPLES_PER_USER) -> List[SyntheticSample]:
        """
        Generate synthetic data for a specific user.
        
        Args:
            user_id: User identifier
            max_samples: Maximum samples to generate
            
        Returns:
            List of synthetic samples
        """
        samples = []
        
        # 1. Generate from mesh context
        mesh_samples = self._generate_from_mesh(user_id)
        samples.extend(mesh_samples[:max_samples // 3])
        
        # 2. Generate from intent gaps
        intent_samples = self._generate_from_intents(user_id)
        samples.extend(intent_samples[:max_samples // 3])
        
        # 3. Generate from conversations
        conv_samples = self._generate_from_conversations(user_id)
        samples.extend(conv_samples[:max_samples // 3])
        
        # 4. Generate from templates
        template_samples = self._generate_from_templates(user_id, max_samples // 4)
        samples.extend(template_samples)
        
        # 5. Generate from psi archive if available
        psi_samples = self._generate_from_psi(user_id)
        samples.extend(psi_samples[:max_samples // 6])
        
        # Filter by confidence
        samples = [s for s in samples if s.confidence >= MIN_CONFIDENCE_THRESHOLD]
        
        # Limit to max samples
        samples = samples[:max_samples]
        
        # Update statistics
        self._update_stats(samples)
        
        logger.info(f"Generated {len(samples)} samples for user {user_id}")
        
        return samples
    
    def _generate_from_mesh(self, user_id: str) -> List[SyntheticSample]:
        """Generate samples from mesh context."""
        samples = []
        
        mesh = self.mesh_manager.load_mesh(user_id)
        if not mesh:
            return samples
        
        nodes = mesh.get("nodes", [])
        edges = mesh.get("edges", [])
        
        # Generate Q&A from nodes
        for node in nodes:
            if node.get("confidence", 0) < MIN_CONFIDENCE_THRESHOLD:
                continue
            
            # Generate "What is X?" questions
            prompt = f"What is {node.get('label', 'this concept')}?"
            response = f"{node.get('label', 'This concept')} is {node.get('description', 'a key concept in this domain')}."
            
            sample = SyntheticSample(
                user_id=user_id,
                prompt=prompt,
                response=response,
                source="mesh",
                confidence=node.get("confidence", 0.8),
                metadata={"node_id": node.get("id")},
                timestamp=datetime.now()
            )
            samples.append(sample)
        
        # Generate relationship questions from edges
        for edge in edges[:10]:  # Limit edge samples
            source_node = next((n for n in nodes if n.get("id") == edge.get("source")), None)
            target_node = next((n for n in nodes if n.get("id") == edge.get("target")), None)
            
            if source_node and target_node:
                prompt = f"How does {source_node.get('label')} relate to {target_node.get('label')}?"
                response = f"{source_node.get('label')} {edge.get('relationship', 'is connected to')} {target_node.get('label')}."
                
                sample = SyntheticSample(
                    user_id=user_id,
                    prompt=prompt,
                    response=response,
                    source="mesh",
                    confidence=0.85,
                    metadata={"edge": edge},
                    timestamp=datetime.now()
                )
                samples.append(sample)
        
        self.stats.mesh_samples += len(samples)
        return samples
    
    def _generate_from_intents(self, user_id: str) -> List[SyntheticSample]:
        """Generate samples from intent gaps."""
        samples = []
        
        gaps = self.source_manager.load_intent_gaps(user_id)
        
        for gap in gaps[:50]:  # Limit to recent gaps
            gap_type = gap.get("gap_type", "knowledge")
            user_input = gap.get("user_input", "")
            
            if not user_input:
                continue
            
            # Generate improved response for the gap
            if gap_type == "knowledge":
                response = f"Based on available information, {user_input} involves [detailed explanation would go here]."
            elif gap_type == "capability":
                response = f"To address your request about {user_input}, [capability explanation would go here]."
            else:  # context
                response = f"To provide better context for {user_input}, [contextual information would go here]."
            
            sample = SyntheticSample(
                user_id=user_id,
                prompt=user_input,
                response=response,
                source="intent",
                confidence=0.75,  # Lower confidence for gap-based samples
                metadata={"gap_type": gap_type},
                timestamp=datetime.now()
            )
            samples.append(sample)
        
        self.stats.intent_samples += len(samples)
        return samples
    
    def _generate_from_conversations(self, user_id: str) -> List[SyntheticSample]:
        """Generate samples from conversation history."""
        samples = []
        
        conversations = self.source_manager.load_conversation_history(user_id)
        
        # Extract successful Q&A pairs
        for conv in conversations[:100]:  # Limit to recent conversations
            user_input = conv.get("user_input", "")
            assistant_response = conv.get("assistant_response", "")
            
            if user_input and assistant_response and len(assistant_response) > 20:
                # Use successful conversations as training data
                sample = SyntheticSample(
                    user_id=user_id,
                    prompt=user_input,
                    response=assistant_response,
                    source="conversation",
                    confidence=0.9,  # High confidence for actual conversations
                    metadata={"conversation_id": conv.get("conversation_id")},
                    timestamp=datetime.now()
                )
                samples.append(sample)
        
        return samples
    
    def _generate_from_templates(self, user_id: str, count: int) -> List[SyntheticSample]:
        """Generate samples from templates."""
        samples = []
        
        # Load user's mesh for concepts
        mesh = self.mesh_manager.load_mesh(user_id)
        if not mesh:
            return samples
        
        nodes = mesh.get("nodes", [])
        if not nodes:
            return samples
        
        # Generate template-based samples
        for _ in range(min(count, len(nodes) * 2)):
            template = random.choice(QA_TEMPLATES)
            
            # Select random concepts
            node = random.choice(nodes)
            concept = node.get("label", "concept")
            
            # Fill template
            prompt = template["pattern"].format(
                concept=concept,
                concept1=concept,
                concept2=random.choice(nodes).get("label", "another concept"),
                domain=node.get("domain", "general"),
                attribute="properties"
            )
            
            response = template["response"].format(
                concept=concept,
                concept1=concept,
                concept2=random.choice(nodes).get("label", "another concept"),
                description=node.get("description", "a fundamental concept"),
                related_concepts=", ".join([n.get("label", "") for n in nodes[:3]]),
                domain=node.get("domain", "general"),
                importance="it provides key insights",
                relationship="conceptual connection",
                details="they share common properties",
                attribute="properties",
                characteristics="unique features",
                explanation="it defines core behavior",
                applications="research, development, analysis",
                primary_application="advanced research"
            )
            
            sample = SyntheticSample(
                user_id=user_id,
                prompt=prompt,
                response=response,
                source="template",
                confidence=0.8,
                metadata={"template": template["pattern"]},
                timestamp=datetime.now()
            )
            samples.append(sample)
        
        self.stats.template_samples += len(samples)
        return samples
    
    def _generate_from_psi(self, user_id: str) -> List[SyntheticSample]:
        """Generate samples from psi-morphon synthesis."""
        samples = []
        
        psi_data = self.source_manager.load_psi_archive(user_id)
        
        for synthesis in psi_data[:20]:  # Limit psi samples
            concept_id = synthesis.get("id", "")
            source_concepts = synthesis.get("source_concepts", [])
            
            if concept_id and source_concepts:
                # Generate synthesis explanation
                prompt = f"What emerged from combining {' and '.join(source_concepts)}?"
                response = f"The synthesis of {' and '.join(source_concepts)} produces {concept_id}, which represents an emergent concept with unique properties."
                
                sample = SyntheticSample(
                    user_id=user_id,
                    prompt=prompt,
                    response=response,
                    source="psi",
                    confidence=synthesis.get("confidence", 0.8),
                    metadata={"synthesis_id": concept_id},
                    timestamp=datetime.now()
                )
                samples.append(sample)
        
        self.stats.psi_samples += len(samples)
        return samples
    
    def _update_stats(self, samples: List[SyntheticSample]):
        """Update generation statistics."""
        self.stats.total_samples += len(samples)
        self.stats.users_processed += 1
        
        if samples:
            confidences = [s.confidence for s in samples]
            self.stats.average_confidence = np.mean(confidences)
    
    def save_samples(self, 
                    samples: List[SyntheticSample],
                    user_id: str,
                    split_validation: bool = True):
        """
        Save samples to training/validation files.
        
        Args:
            samples: List of samples to save
            user_id: User identifier
            split_validation: Whether to split validation set
        """
        if not samples:
            return
        
        # Shuffle samples
        random.shuffle(samples)
        
        if split_validation:
            # Split into train/validation
            split_idx = int(len(samples) * (1 - VALIDATION_SPLIT))
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]
        else:
            train_samples = samples
            val_samples = []
        
        # Save training data
        train_file = TRAINING_DATA_DIR / f"user_{user_id}_data.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        logger.info(f"Saved {len(train_samples)} training samples to {train_file}")
        
        # Save validation data
        if val_samples:
            val_file = VALIDATION_DATA_DIR / f"user_{user_id}_validation.jsonl"
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
            
            logger.info(f"Saved {len(val_samples)} validation samples to {val_file}")
    
    def generate_batch(self, user_ids: List[str]) -> Dict[str, List[SyntheticSample]]:
        """
        Generate data for multiple users.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            Dictionary mapping user_id to samples
        """
        all_samples = {}
        
        for user_id in user_ids:
            logger.info(f"Generating data for user: {user_id}")
            samples = self.generate_for_user(user_id)
            all_samples[user_id] = samples
            
            # Save immediately
            self.save_samples(samples, user_id)
        
        return all_samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_samples": self.stats.total_samples,
            "mesh_samples": self.stats.mesh_samples,
            "intent_samples": self.stats.intent_samples,
            "psi_samples": self.stats.psi_samples,
            "template_samples": self.stats.template_samples,
            "users_processed": self.stats.users_processed,
            "average_confidence": self.stats.average_confidence
        }

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_synthetic_data(user_id: str,
                           max_samples: int = MAX_SAMPLES_PER_USER,
                           save: bool = True) -> List[SyntheticSample]:
    """
    Main function to generate synthetic data.
    
    Args:
        user_id: User identifier
        max_samples: Maximum samples to generate
        save: Whether to save to files
        
    Returns:
        List of generated samples
    """
    generator = SyntheticDataGenerator()
    samples = generator.generate_for_user(user_id, max_samples)
    
    if save:
        generator.save_samples(samples, user_id)
    
    return samples

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to generate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save to files")
    parser.add_argument("--stats", action="store_true",
                       help="Show generation statistics")
    parser.add_argument("--batch", nargs="+",
                       help="Generate for multiple users")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = SyntheticDataGenerator()
    
    if args.batch:
        # Batch generation
        print(f"Generating data for {len(args.batch)} users...")
        all_samples = generator.generate_batch(args.batch)
        
        for user_id, samples in all_samples.items():
            print(f"  {user_id}: {len(samples)} samples")
    else:
        # Single user generation
        print(f"Generating synthetic data for user: {args.user_id}")
        samples = generator.generate_for_user(args.user_id, args.max_samples)
        
        if not args.no_save:
            generator.save_samples(samples, args.user_id)
        
        print(f"\nGenerated {len(samples)} samples:")
        
        # Show sample distribution
        by_source = {}
        for sample in samples:
            by_source[sample.source] = by_source.get(sample.source, 0) + 1
        
        for source, count in by_source.items():
            print(f"  {source}: {count}")
        
        # Show examples
        print("\nSample examples:")
        for sample in samples[:3]:
            print(f"\n  Prompt: {sample.prompt[:100]}...")
            print(f"  Response: {sample.response[:100]}...")
            print(f"  Source: {sample.source}, Confidence: {sample.confidence:.2f}")
    
    if args.stats:
        stats = generator.get_statistics()
        print("\nGeneration Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
