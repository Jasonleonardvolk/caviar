#!/usr/bin/env python3
"""
Temporal Knowledge Management System
Tracks concept evolution and knowledge drift over time
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from python.core.temporal_reasoning_integration import (
    TemporalConceptMesh,
    TemporalReasoningAnalyzer,
    TrainingDataExporter
)
from python.core.reasoning_traversal import EdgeType

class TemporalKnowledgeManager:
    """Manage knowledge evolution over time"""
    
    def __init__(self, mesh: TemporalConceptMesh):
        self.mesh = mesh
        self.analyzer = TemporalReasoningAnalyzer()
        self.exporter = TrainingDataExporter()
        self.knowledge_history = []
    
    def add_knowledge_version(self, concept_id: str, name: str, 
                            description: str, sources: List[str],
                            timestamp: Optional[str] = None):
        """Add a new version of a concept"""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        
        # Create versioned ID
        version_id = f"{concept_id}_v{ts[:10]}"
        
        # Add node with timestamp in sources
        sources_with_time = sources + [ts]
        node = type('Node', (), {
            'id': version_id,
            'name': name,
            'description': description,
            'sources': sources_with_time,
            'edges': []
        })()
        
        self.mesh.add_node(node)
        
        # Link to previous version if exists
        prev_versions = [nid for nid in self.mesh.nodes 
                        if nid.startswith(f"{concept_id}_v") and nid != version_id]
        
        if prev_versions:
            latest_prev = sorted(prev_versions)[-1]
            self.mesh.add_temporal_edge(
                latest_prev, version_id, EdgeType.RELATED_TO,
                justification="evolved into",
                timestamp=ts
            )
        
        # Track in history
        self.knowledge_history.append({
            'concept_id': concept_id,
            'version_id': version_id,
            'timestamp': ts,
            'action': 'add_version'
        })
        
        return version_id
    
    def deprecate_concept(self, concept_id: str, reason: str,
                         replacement_id: Optional[str] = None):
        """Mark a concept as deprecated"""
        ts = datetime.now(timezone.utc).isoformat()
        
        if concept_id in self.mesh.nodes:
            node = self.mesh.nodes[concept_id]
            
            # Add deprecation metadata
            if not hasattr(node, 'metadata'):
                node.metadata = {}
            
            node.metadata['deprecated'] = {
                'timestamp': ts,
                'reason': reason,
                'replacement': replacement_id
            }
            
            # Add edge to replacement if provided
            if replacement_id and replacement_id in self.mesh.nodes:
                self.mesh.add_temporal_edge(
                    concept_id, replacement_id, EdgeType.RELATED_TO,
                    justification=f"deprecated: {reason}",
                    timestamp=ts
                )
            
            self.knowledge_history.append({
                'concept_id': concept_id,
                'timestamp': ts,
                'action': 'deprecate',
                'reason': reason,
                'replacement': replacement_id
            })
    
    def get_concept_timeline(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get evolution timeline for a concept"""
        timeline = []
        
        # Find all versions
        versions = [(nid, node) for nid, node in self.mesh.nodes.items()
                   if nid.startswith(f"{concept_id}_v")]
        
        for version_id, node in sorted(versions):
            # Extract timestamp from sources
            timestamps = [s for s in node.sources if "T" in s]
            if timestamps:
                timeline.append({
                    'version_id': version_id,
                    'timestamp': timestamps[0],
                    'description': node.description,
                    'sources': [s for s in node.sources if "T" not in s]
                })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def analyze_drift_patterns(self, window_days: int = 30) -> Dict[str, Any]:
        """Analyze knowledge drift patterns"""
        analysis = self.analyzer.analyze_knowledge_drift(self.mesh, window_days)
        
        # Enhanced analysis
        now = datetime.now(timezone.utc)
        
        # Categorize by age
        age_buckets = {
            'fresh': 0,      # < 30 days
            'recent': 0,     # 30-90 days
            'stable': 0,     # 90-365 days
            'legacy': 0,     # > 365 days
        }
        
        for node in self.mesh.nodes.values():
            # Get newest timestamp
            timestamps = []
            for source in node.sources:
                if "T" in source:
                    try:
                        timestamps.append(datetime.fromisoformat(source))
                    except:
                        pass
            
            if timestamps:
                newest = max(timestamps)
                age_days = (now - newest).days
                
                if age_days < 30:
                    age_buckets['fresh'] += 1
                elif age_days < 90:
                    age_buckets['recent'] += 1
                elif age_days < 365:
                    age_buckets['stable'] += 1
                else:
                    age_buckets['legacy'] += 1
        
        analysis['age_distribution'] = age_buckets
        
        # Find update hotspots
        update_frequency = {}
        for event in self.knowledge_history:
            concept = event.get('concept_id', '')
            update_frequency[concept] = update_frequency.get(concept, 0) + 1
        
        analysis['update_hotspots'] = sorted(
            update_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return analysis
    
    def generate_drift_report(self, output_path: str = "drift_report.json"):
        """Generate comprehensive drift report"""
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_concepts': len(self.mesh.nodes),
            'drift_analysis': self.analyze_drift_patterns(),
            'deprecated_concepts': [],
            'version_counts': {},
            'temporal_inconsistencies': self.analyzer.find_temporal_inconsistencies(self.mesh)
        }
        
        # Find deprecated concepts
        for node_id, node in self.mesh.nodes.items():
            if hasattr(node, 'metadata') and 'deprecated' in node.metadata:
                report['deprecated_concepts'].append({
                    'id': node_id,
                    'deprecated_at': node.metadata['deprecated']['timestamp'],
                    'reason': node.metadata['deprecated']['reason'],
                    'replacement': node.metadata['deprecated'].get('replacement')
                })
        
        # Count versions per concept
        concept_versions = {}
        for node_id in self.mesh.nodes:
            if "_v" in node_id:
                base_id = node_id.split("_v")[0]
                concept_versions[base_id] = concept_versions.get(base_id, 0) + 1
        
        report['version_counts'] = concept_versions
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def visualize_knowledge_evolution(self, concept_id: str, 
                                    output_path: str = "knowledge_evolution.png"):
        """Visualize how a concept evolved over time"""
        timeline = self.get_concept_timeline(concept_id)
        
        if not timeline:
            print(f"No timeline data for concept: {concept_id}")
            return
        
        # Extract dates and version numbers
        dates = []
        versions = []
        
        for i, entry in enumerate(timeline):
            try:
                dt = datetime.fromisoformat(entry['timestamp'])
                dates.append(dt)
                versions.append(i + 1)
            except:
                pass
        
        if not dates:
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates, versions, 'bo-', markersize=10)
        
        # Add annotations
        for i, (date, version) in enumerate(zip(dates, versions)):
            plt.annotate(f"v{version}", 
                        (date, version), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.xlabel('Date')
        plt.ylabel('Version')
        plt.title(f'Evolution of Concept: {concept_id}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()
        
        print(f"Evolution visualization saved to: {output_path}")

def demonstrate_temporal_knowledge_management():
    """Demonstrate temporal knowledge management features"""
    
    # Create mesh and manager
    mesh = TemporalConceptMesh()
    manager = TemporalKnowledgeManager(mesh)
    
    print("🕐 Temporal Knowledge Management Demo")
    print("=" * 60)
    
    # Simulate knowledge evolution
    # Day 1: Initial concepts
    base_time = datetime.now(timezone.utc) - timedelta(days=365)
    
    manager.add_knowledge_version(
        "quantum", "Quantum Computing",
        "Computing using quantum mechanical phenomena",
        ["arxiv_2023_001"],
        timestamp=base_time.isoformat()
    )
    
    # Day 90: Updated understanding
    update1_time = base_time + timedelta(days=90)
    manager.add_knowledge_version(
        "quantum", "Quantum Computing",
        "Computing paradigm leveraging superposition and entanglement",
        ["arxiv_2023_100", "nature_2023_05"],
        timestamp=update1_time.isoformat()
    )
    
    # Day 180: Major revision
    update2_time = base_time + timedelta(days=180)
    v3_id = manager.add_knowledge_version(
        "quantum", "Quantum Computing",
        "Computational model using qubits for exponential speedup in specific problems",
        ["science_2024_02", "ieee_2024_03"],
        timestamp=update2_time.isoformat()
    )
    
    # Day 270: Deprecate old version
    manager.deprecate_concept(
        "quantum_v" + base_time.isoformat()[:10],
        "Outdated definition, see latest version",
        replacement_id=v3_id
    )
    
    # Add related concepts
    manager.add_knowledge_version(
        "qubit", "Quantum Bit",
        "Basic unit of quantum information",
        ["textbook_2024"],
        timestamp=update2_time.isoformat()
    )
    
    mesh.add_temporal_edge(
        v3_id, f"qubit_v{update2_time.isoformat()[:10]}",
        EdgeType.PART_OF,
        justification="qubits are fundamental to quantum computing",
        timestamp=update2_time.isoformat()
    )
    
    # Show concept timeline
    print("\n📅 Concept Timeline for 'quantum':")
    timeline = manager.get_concept_timeline("quantum")
    for entry in timeline:
        print(f"  - {entry['timestamp'][:10]}: {entry['description'][:50]}...")
    
    # Analyze drift
    print("\n📊 Knowledge Drift Analysis:")
    drift = manager.analyze_drift_patterns(window_days=90)
    print(f"  Age distribution: {drift['age_distribution']}")
    print(f"  Update hotspots: {drift['update_hotspots'][:3]}")
    
    # Generate report
    report = manager.generate_drift_report()
    print(f"\n📋 Drift report generated with {len(report['deprecated_concepts'])} deprecated concepts")
    
    # Export training data for recent paths
    print("\n🎯 Exporting training data for temporal reasoning...")
    
    recent_paths = mesh.traverse_temporal("quantum_v" + update2_time.isoformat()[:10])
    training_samples = []
    
    for path in recent_paths[:5]:
        sample = manager.exporter.export_path(
            path,
            "Explain the evolution of quantum computing concepts",
            {"temporal_context": "knowledge_evolution"}
        )
        training_samples.append(sample)
    
    if training_samples:
        output_file = manager.exporter.save_training_batch(
            training_samples, 
            "temporal_reasoning_samples.jsonl"
        )
        print(f"  Exported {len(training_samples)} samples to {output_file}")
    
    # Visualize evolution
    manager.visualize_knowledge_evolution("quantum")
    
    print("\n✅ Temporal knowledge management demo complete!")

if __name__ == "__main__":
    demonstrate_temporal_knowledge_management()
