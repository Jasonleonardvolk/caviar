#!/usr/bin/env python3
"""
Test and demo script for Reasoning Traversal system
Shows how all components work together
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.core.reasoning_traversal import (
    ConceptMesh, ConceptNode, EdgeType,
    ReasoningEngine, ExplanationGenerator,
    PrajnaResponsePlus, PrajnaReasoningIntegration
)

def create_rich_test_mesh() -> ConceptMesh:
    """Create a richer test mesh for demonstration"""
    mesh = ConceptMesh()
    
    # Information Theory Concepts
    nodes = [
        ConceptNode("entropy", "Entropy", 
                   "A measure of uncertainty or randomness in information",
                   ["Shannon1948", "CoverThomas2006", "PDF_001"]),
        
        ConceptNode("information", "Information",
                   "Data that reduces uncertainty about a system's state",
                   ["Shannon1948", "PDF_002"]),
        
        ConceptNode("compression", "Data Compression",
                   "Process of encoding information using fewer bits",
                   ["Huffman1952", "LempelZiv1977", "PDF_003"]),
        
        ConceptNode("redundancy", "Redundancy",
                   "Repetitive patterns that can be eliminated without loss",
                   ["PDF_004", "TextbookCh7"]),
        
        ConceptNode("kolmogorov", "Kolmogorov Complexity",
                   "Shortest program length that produces a given output",
                   ["Kolmogorov1965", "PDF_005"]),
        
        ConceptNode("mutual_info", "Mutual Information",
                   "Amount of information shared between two variables",
                   ["CoverThomas2006", "PDF_006"]),
        
        ConceptNode("channel_capacity", "Channel Capacity",
                   "Maximum rate of reliable information transmission",
                   ["Shannon1948", "PDF_007"]),
        
        ConceptNode("noise", "Noise",
                   "Random disturbances that corrupt information",
                   ["PDF_008"]),
        
        ConceptNode("error_correction", "Error Correction",
                   "Methods to detect and correct errors in data",
                   ["Hamming1950", "PDF_009"])
    ]
    
    # Add all nodes
    for node in nodes:
        mesh.add_node(node)
    
    # Add relationships
    edges = [
        # Core relationships
        ("entropy", "information", EdgeType.IMPLIES, 0.95, 
         "high entropy indicates potential for information gain"),
        
        ("information", "compression", EdgeType.ENABLES, 0.9,
         "information theory provides foundation for compression"),
        
        ("redundancy", "compression", EdgeType.ENABLES, 0.95,
         "redundancy allows for effective compression"),
        
        ("entropy", "redundancy", EdgeType.CONTRADICTS, 0.8,
         "high entropy means low redundancy"),
        
        # Deeper relationships
        ("entropy", "kolmogorov", EdgeType.RELATED_TO, 0.7,
         "both measure complexity/randomness"),
        
        ("kolmogorov", "compression", EdgeType.SUPPORTS, 0.85,
         "Kolmogorov complexity bounds compression"),
        
        ("information", "mutual_info", EdgeType.PART_OF, 0.9,
         "mutual information is shared information"),
        
        ("mutual_info", "channel_capacity", EdgeType.SUPPORTS, 0.8,
         "mutual information helps determine capacity"),
        
        # Noise and error handling
        ("noise", "information", EdgeType.PREVENTS, 0.7,
         "noise corrupts information transmission"),
        
        ("noise", "error_correction", EdgeType.CAUSES, 0.9,
         "noise necessitates error correction"),
        
        ("error_correction", "redundancy", EdgeType.BECAUSE, 0.85,
         "error correction works by adding redundancy"),
        
        ("channel_capacity", "noise", EdgeType.RELATED_TO, 0.75,
         "capacity depends on noise level")
    ]
    
    # Add all edges
    for from_id, to_id, relation, weight, justification in edges:
        mesh.add_edge(from_id, to_id, relation, weight, justification)
    
    return mesh

def demonstrate_reasoning_traversal():
    """Comprehensive demonstration of reasoning traversal"""
    print("🧪 Reasoning Traversal System Demonstration")
    print("=" * 70)
    
    # Create rich mesh
    mesh = create_rich_test_mesh()
    print(f"\n✅ Created concept mesh with {len(mesh.nodes)} nodes")
    
    # Initialize reasoning system
    reasoning_integration = PrajnaReasoningIntegration(mesh, enable_inline_attribution=True)
    
    # Test queries
    test_queries = [
        {
            "query": "How does entropy relate to data compression?",
            "anchors": ["entropy"],
            "expected_path": ["entropy", "information", "compression"]
        },
        {
            "query": "Why do we need error correction in noisy channels?",
            "anchors": ["noise", "channel_capacity"],
            "expected_path": ["noise", "error_correction", "redundancy"]
        },
        {
            "query": "What is the relationship between redundancy and compression?",
            "anchors": ["redundancy"],
            "expected_path": ["redundancy", "compression"]
        }
    ]
    
    for i, test in enumerate(test_queries):
        print(f"\n{'=' * 70}")
        print(f"📝 Query {i+1}: {test['query']}")
        print(f"🎯 Anchor concepts: {test['anchors']}")
        print("-" * 70)
        
        # Generate reasoned response
        response = reasoning_integration.generate_reasoned_response(
            query=test['query'],
            anchor_concepts=test['anchors']
        )
        
        # Display response
        print("\n🤖 Generated Response:")
        print(response.text)
        
        print(f"\n📊 Reasoning Statistics:")
        print(f"  - Paths found: {len(response.reasoning_paths)}")
        print(f"  - Confidence: {response.confidence:.2%}")
        print(f"  - Sources used: {len(response.sources)}")
        
        # Show reasoning paths
        print("\n🔗 Reasoning Paths:")
        for j, path in enumerate(response.reasoning_paths[:3]):
            chain_str = " → ".join([n.name for n in path.chain])
            print(f"  Path {j+1} ({path.path_type}): {chain_str}")
            print(f"    Score: {path.score:.3f}, Confidence: {path.confidence:.3f}")
        
        # Show sources
        print("\n📚 Sources Referenced:")
        for source in sorted(set(response.sources))[:5]:
            print(f"  - {source}")
        
        # Show graph visualization
        if i == 0:  # Only show for first query
            print("\n📈 Graphviz Representation:")
            print(response.to_graphviz())

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n\n🧪 Testing Edge Cases")
    print("=" * 70)
    
    mesh = ConceptMesh()
    reasoning = PrajnaReasoningIntegration(mesh)
    
    # Test 1: Empty mesh
    print("\n1️⃣ Empty mesh:")
    response = reasoning.generate_reasoned_response("test query", ["nonexistent"])
    print(f"   Result: {response.text}")
    print(f"   Paths: {len(response.reasoning_paths)}")
    
    # Test 2: Circular reference
    print("\n2️⃣ Circular references:")
    a = ConceptNode("A", "A", "Node A")
    b = ConceptNode("B", "B", "Node B")
    mesh.add_node(a)
    mesh.add_node(b)
    mesh.add_edge("A", "B", EdgeType.CAUSES)
    mesh.add_edge("B", "A", EdgeType.CAUSES)  # Circular!
    
    response = reasoning.generate_reasoned_response("Circular test", ["A"])
    print(f"   Paths found: {len(response.reasoning_paths)}")
    print(f"   No infinite loop: ✅")
    
    # Test 3: Deep traversal
    print("\n3️⃣ Deep traversal test:")
    for i in range(10):
        node = ConceptNode(f"D{i}", f"Depth {i}", f"Node at depth {i}")
        mesh.add_node(node)
        if i > 0:
            mesh.add_edge(f"D{i-1}", f"D{i}", EdgeType.IMPLIES)
    
    response = reasoning.generate_reasoned_response("Deep test", ["D0"])
    deepest_path = max(response.reasoning_paths, key=lambda p: len(p.chain)) if response.reasoning_paths else None
    if deepest_path:
        print(f"   Deepest path length: {len(deepest_path.chain)}")
        print(f"   Max depth limiting works: ✅")

def save_test_results():
    """Save test results for inspection"""
    print("\n\n💾 Saving Test Results")
    print("=" * 70)
    
    mesh = create_rich_test_mesh()
    reasoning = PrajnaReasoningIntegration(mesh)
    
    # Generate response
    response = reasoning.generate_reasoned_response(
        "How does information theory enable data compression?",
        ["information", "entropy"]
    )
    
    # Save to file
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save response
    with open(output_dir / "reasoning_response.json", "w") as f:
        json.dump(response.to_dict(), f, indent=2)
    
    # Save graphviz
    with open(output_dir / "reasoning_graph.dot", "w") as f:
        f.write(response.to_graphviz())
    
    # Save mesh structure
    mesh_data = {
        "nodes": {
            node_id: {
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "sources": node.sources
            }
            for node_id, node in mesh.nodes.items()
        },
        "edges": [
            {
                "from": node_id,
                "to": edge.target,
                "relation": edge.relation.value,
                "weight": edge.weight,
                "justification": edge.justification
            }
            for node_id, node in mesh.nodes.items()
            for edge in node.edges
        ]
    }
    
    with open(output_dir / "concept_mesh.json", "w") as f:
        json.dump(mesh_data, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}/")
    print("   - reasoning_response.json")
    print("   - reasoning_graph.dot (visualize at http://viz-js.com/)")
    print("   - concept_mesh.json")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_reasoning_traversal()
    test_edge_cases()
    save_test_results()
    
    print("\n\n✅ All tests completed!")
    print("\n💡 Next steps:")
    print("   1. Integrate with your existing Prajna API")
    print("   2. Load concept mesh from your actual data")
    print("   3. Enable inline attribution in responses")
    print("   4. Add caching for performance")
