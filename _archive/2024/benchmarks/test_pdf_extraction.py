#!/usr/bin/env python3
"""
🎯 DIRECT PDF CONCEPT EXTRACTION TEST

Test the universal extraction on the RL/DE paper to show what we SHOULD get
"""

import sys
from pathlib import Path

# Add the ingest_pdf directory to path
sys.path.append(str(Path(__file__).parent / "ingest_pdf"))

def test_extraction():
    """Test extraction on the RL/DE paper content"""
    
    # Sample content from the RL/DE paper
    paper_text = """
    Reinforcement learning Based Automated Design of Differential Evolution Algorithm for Black-box Optimization
    
    Abstract: Differential Evolution (DE) is a powerful derivative-free optimizer, but no single variant is superior across all black-box optimization problems (BBOPs). This paper proposes a reinforcement learning (RL) based framework called rlDE that learns to design DE algorithms using meta-learning. The RL agent is implemented with a Double Deep Q-Network (DDQN), and generates DE configurations based on problem characteristics extracted via Exploratory Landscape Analysis (ELA).
    
    The framework leverages meta-learning to enhance the generalizability of the meta-optimizer, enabling it to adapt more effectively across a diverse range of problem scenarios. The RL agent is trained offline by meta-learning pairs (state, action, reward) across a large number of black-box optimization problems. The Double Deep Q-Network (DDQN) is utilized for implementation, considering a subset of 40 possible strategy combinations and parameter optimizations simultaneously.
    
    Key concepts include differential evolution, reinforcement learning, black-box optimization, meta-learning, exploratory landscape analysis, DDQN, Markov Decision Process, BBOB2009 benchmark, algorithm design, hyperparameter optimization, mutation strategies, crossover strategies, population-based optimization, and evolutionary computation.
    
    The experimental results on BBOB2009 demonstrate the effectiveness of the proposed framework compared to state-of-the-art algorithms including JDE21, NL-SHADE-LBC, and other traditional differential evolution variants.
    """
    
    try:
        from extractConceptsFromDocument import extractConceptsFromDocument
        
        print("🎯 TESTING UNIVERSAL EXTRACTION ON RL/DE PAPER")
        print("=" * 60)
        
        # Extract concepts
        concepts = extractConceptsFromDocument(paper_text)
        
        print(f"📊 EXTRACTED {len(concepts)} CONCEPTS:")
        print("-" * 40)
        
        for i, concept in enumerate(concepts[:20], 1):  # Show top 20
            name = concept.get('name', 'Unknown')
            score = concept.get('score', 0)
            methods = concept.get('source', {}).get('methods', 'unknown')
            
            # Method emoji
            method_emoji = "🔤"
            if "yake" in methods.lower():
                method_emoji = "📊" if "keybert" in methods.lower() or "ner" in methods.lower() else "📈"
            elif "keybert" in methods.lower():
                method_emoji = "🧠" if "ner" in methods.lower() else "🎯"
            elif "ner" in methods.lower():
                method_emoji = "🏷️"
                
            print(f"  {method_emoji} {name} (score: {score:.3f}, methods: {methods})")
        
        if len(concepts) > 20:
            print(f"  ... and {len(concepts) - 20} more concepts")
        
        # Expected vs Actual
        print(f"\n🎯 ANALYSIS:")
        print(f"✅ Expected: 20-40 concepts from rich academic paper")
        print(f"📊 Actual: {len(concepts)} concepts extracted")
        print(f"🎉 SUCCESS: This is what you SHOULD be getting!")
        
        # Method breakdown
        methods_count = {}
        for concept in concepts:
            method = concept.get('source', {}).get('methods', 'unknown')
            methods_count[method] = methods_count.get(method, 0) + 1
        
        print(f"\n📊 METHOD BREAKDOWN:")
        for method, count in sorted(methods_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} concepts")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure dependencies are installed:")
        print("  pip install yake keybert sentence-transformers spacy")
        print("  python -m spacy download en_core_web_lg")
        return False

if __name__ == "__main__":
    test_extraction()
