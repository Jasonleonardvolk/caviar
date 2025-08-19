#!/usr/bin/env python3
"""
Example Applications for Chaos-Enhanced TORI
Demonstrates practical use cases for the new capabilities
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from python.core.tori_production import TORIProductionSystem, TORIProductionConfig
from python.core.metacognitive_adapters import AdapterMode

# ========== Creative Writing Assistant ==========

class CreativeWritingAssistant:
    """Uses phase explosion for creative story generation"""
    
    def __init__(self, tori: TORIProductionSystem):
        self.tori = tori
        
    async def generate_story_elements(self, theme: str, genre: str) -> Dict[str, Any]:
        """Generate creative story elements using chaos"""
        
        # Use phase explosion for maximum creativity
        self.tori.set_chaos_mode(AdapterMode.CHAOS_ASSISTED)
        
        # Generate character
        character_query = f"Create a unique character for a {genre} story about {theme}. Use chaos to explore unusual character traits and backstories."
        character_result = await self.tori.process_query(
            character_query,
            context={'enable_chaos': True, 'force_chaos': True}
        )
        
        # Generate plot twist
        plot_query = f"Generate an unexpected plot twist for a {genre} story about {theme}. Let chaos guide you to surprising connections."
        plot_result = await self.tori.process_query(
            plot_query,
            context={'enable_chaos': True}
        )
        
        # Generate setting
        setting_query = f"Describe an unusual setting for a {genre} story about {theme}. Use chaotic exploration to find unique environments."
        setting_result = await self.tori.process_query(
            setting_query,
            context={'enable_chaos': True}
        )
        
        return {
            'character': character_result['response'],
            'plot_twist': plot_result['response'],
            'setting': setting_result['response'],
            'efficiency_gains': {
                'character': character_result['metadata'].get('efficiency_ratio', 1.0),
                'plot': plot_result['metadata'].get('efficiency_ratio', 1.0),
                'setting': setting_result['metadata'].get('efficiency_ratio', 1.0)
            }
        }

# ========== Scientific Pattern Discovery ==========

class ScientificPatternDiscovery:
    """Uses attractor hopping for finding patterns in data"""
    
    def __init__(self, tori: TORIProductionSystem):
        self.tori = tori
        
    async def discover_patterns(self, data_description: str, 
                              hypothesis: Optional[str] = None) -> Dict[str, Any]:
        """Discover hidden patterns using chaos-assisted search"""
        
        # Use attractor hopping for exploration
        self.tori.set_chaos_mode(AdapterMode.HYBRID)
        
        # Initial pattern search
        search_query = f"""
        Analyze this data for hidden patterns: {data_description}
        {"Consider the hypothesis: " + hypothesis if hypothesis else ""}
        Use attractor hopping to explore different pattern spaces.
        """
        
        search_result = await self.tori.process_query(
            search_query,
            context={'enable_chaos': True}
        )
        
        # Deep dive into promising patterns
        if "pattern" in search_result['response'].lower():
            deep_query = f"""
            Explore the discovered patterns more deeply.
            Use chaos dynamics to find non-obvious connections.
            Original data: {data_description}
            """
            
            deep_result = await self.tori.process_query(
                deep_query,
                context={'enable_chaos': True, 'force_chaos': True}
            )
            
            return {
                'initial_patterns': search_result['response'],
                'deep_insights': deep_result['response'],
                'reasoning_paths': search_result.get('reasoning_paths', []),
                'chaos_metrics': {
                    'search_efficiency': search_result['metadata'].get('efficiency_ratio', 1.0),
                    'deep_efficiency': deep_result['metadata'].get('efficiency_ratio', 1.0)
                }
            }
        
        return {
            'initial_patterns': search_result['response'],
            'deep_insights': None,
            'reasoning_paths': search_result.get('reasoning_paths', []),
            'chaos_metrics': {
                'search_efficiency': search_result['metadata'].get('efficiency_ratio', 1.0)
            }
        }

# ========== Memory-Enhanced Learning ==========

class MemoryEnhancedLearning:
    """Uses dark soliton memory for robust knowledge retention"""
    
    def __init__(self, tori: TORIProductionSystem):
        self.tori = tori
        self.learning_sessions = {}
        
    async def learn_concept(self, concept: str, explanation: str, 
                          session_id: str) -> Dict[str, Any]:
        """Learn and robustly store a concept using soliton memory"""
        
        # Store session
        if session_id not in self.learning_sessions:
            self.learning_sessions[session_id] = {
                'concepts': [],
                'start_time': datetime.now(),
                'checkpoints': []
            }
            
        # Use soliton memory enhancement
        store_query = f"""
        Store this concept using dark soliton encoding for maximum robustness:
        Concept: {concept}
        Explanation: {explanation}
        Create interference-resistant memory traces.
        """
        
        store_result = await self.tori.process_query(
            store_query,
            context={'enable_chaos': True}
        )
        
        # Create checkpoint for this learning
        checkpoint_id = await self.tori.create_checkpoint(f"learn_{concept[:20]}")
        
        self.learning_sessions[session_id]['concepts'].append(concept)
        self.learning_sessions[session_id]['checkpoints'].append(checkpoint_id)
        
        return {
            'stored': True,
            'concept': concept,
            'storage_response': store_result['response'],
            'checkpoint': checkpoint_id,
            'memory_robustness': store_result['metadata'].get('memory_resonance', 0)
        }
        
    async def recall_concept(self, concept: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Recall concept using soliton memory retrieval"""
        
        recall_query = f"""
        Recall the concept: {concept}
        {"In the context of: " + context if context else ""}
        Use soliton memory resonance for robust retrieval.
        """
        
        recall_result = await self.tori.process_query(
            recall_query,
            context={'enable_chaos': True}
        )
        
        return {
            'concept': concept,
            'recall': recall_result['response'],
            'confidence': recall_result['metadata'].get('confidence', 0),
            'memory_resonance': recall_result['metadata'].get('memory_resonance', 0)
        }
        
    async def test_retention(self, session_id: str) -> Dict[str, Any]:
        """Test retention of all concepts in a session"""
        
        if session_id not in self.learning_sessions:
            return {'error': 'Session not found'}
            
        session = self.learning_sessions[session_id]
        results = []
        
        for concept in session['concepts']:
            recall = await self.recall_concept(concept)
            results.append({
                'concept': concept,
                'retention_score': recall['memory_resonance'],
                'confidence': recall['confidence']
            })
            
        avg_retention = np.mean([r['retention_score'] for r in results])
        
        return {
            'session_id': session_id,
            'concepts_tested': len(results),
            'average_retention': avg_retention,
            'individual_results': results
        }

# ========== Problem Solving Assistant ==========

class ProblemSolvingAssistant:
    """Uses multiple chaos modes for complex problem solving"""
    
    def __init__(self, tori: TORIProductionSystem):
        self.tori = tori
        
    async def solve_complex_problem(self, problem: str, 
                                  constraints: Optional[List[str]] = None) -> Dict[str, Any]:
        """Solve complex problems using full chaos arsenal"""
        
        solutions = {}
        
        # Phase 1: Understand the problem (minimal chaos)
        self.tori.set_chaos_mode(AdapterMode.PASSTHROUGH)
        understanding = await self.tori.process_query(
            f"Analyze this problem: {problem}\nConstraints: {constraints or 'None'}"
        )
        solutions['understanding'] = understanding['response']
        
        # Phase 2: Explore solution space (attractor hopping)
        self.tori.set_chaos_mode(AdapterMode.HYBRID)
        exploration = await self.tori.process_query(
            f"Explore potential solutions to: {problem}\nUse attractor hopping to find diverse approaches.",
            context={'enable_chaos': True}
        )
        solutions['exploration'] = exploration['response']
        
        # Phase 3: Creative breakthrough (phase explosion)
        self.tori.set_chaos_mode(AdapterMode.CHAOS_ASSISTED)
        creative = await self.tori.process_query(
            f"Generate creative breakthrough solutions for: {problem}\nUse phase explosion for maximum innovation.",
            context={'enable_chaos': True, 'force_chaos': True}
        )
        solutions['creative'] = creative['response']
        
        # Phase 4: Synthesize best solution (reflection with chaos)
        synthesis = await self.tori.process_query(
            f"""
            Synthesize the best solution from these approaches:
            1. Understanding: {understanding['response'][:200]}...
            2. Exploration: {exploration['response'][:200]}...
            3. Creative: {creative['response'][:200]}...
            Problem: {problem}
            """,
            context={'enable_chaos': True, 'deep_reflection': True}
        )
        
        return {
            'problem': problem,
            'phases': solutions,
            'final_solution': synthesis['response'],
            'efficiency_gains': {
                'exploration': exploration['metadata'].get('efficiency_ratio', 1.0),
                'creative': creative['metadata'].get('efficiency_ratio', 1.0),
                'synthesis': synthesis['metadata'].get('efficiency_ratio', 1.0)
            },
            'reasoning_paths': synthesis.get('reasoning_paths', [])
        }

# ========== Demo Applications ==========

async def demo_creative_writing():
    """Demo creative writing assistant"""
    print("\n📝 Creative Writing Assistant Demo")
    print("=" * 60)
    
    config = TORIProductionConfig(enable_chaos=True)
    tori = TORIProductionSystem(config)
    await tori.start()
    
    try:
        assistant = CreativeWritingAssistant(tori)
        
        # Generate story elements
        elements = await assistant.generate_story_elements(
            theme="artificial consciousness",
            genre="science fiction"
        )
        
        print("\n🎭 Generated Story Elements:")
        print(f"\nCharacter:\n{elements['character'][:300]}...")
        print(f"\nPlot Twist:\n{elements['plot_twist'][:300]}...")
        print(f"\nSetting:\n{elements['setting'][:300]}...")
        
        print(f"\n⚡ Efficiency Gains:")
        for key, value in elements['efficiency_gains'].items():
            print(f"  {key}: {value:.2f}x")
            
    finally:
        await tori.stop()

async def demo_pattern_discovery():
    """Demo scientific pattern discovery"""
    print("\n🔬 Scientific Pattern Discovery Demo")
    print("=" * 60)
    
    config = TORIProductionConfig(enable_chaos=True)
    tori = TORIProductionSystem(config)
    await tori.start()
    
    try:
        discovery = ScientificPatternDiscovery(tori)
        
        # Discover patterns in data
        results = await discovery.discover_patterns(
            data_description="Time series of neural oscillations showing irregular spikes every 3-7 intervals with phase coupling to gamma waves",
            hypothesis="The irregular spikes may encode information through their timing variability"
        )
        
        print("\n🔍 Pattern Discovery Results:")
        print(f"\nInitial Patterns:\n{results['initial_patterns'][:400]}...")
        
        if results['deep_insights']:
            print(f"\nDeep Insights:\n{results['deep_insights'][:400]}...")
            
        print(f"\n⚡ Chaos Metrics:")
        for key, value in results['chaos_metrics'].items():
            print(f"  {key}: {value:.2f}x")
            
    finally:
        await tori.stop()

async def demo_memory_learning():
    """Demo memory-enhanced learning"""
    print("\n🧠 Memory-Enhanced Learning Demo")
    print("=" * 60)
    
    config = TORIProductionConfig(enable_chaos=True)
    tori = TORIProductionSystem(config)
    await tori.start()
    
    try:
        learning = MemoryEnhancedLearning(tori)
        session_id = "quantum_computing_101"
        
        # Learn concepts
        concepts = [
            ("Superposition", "A quantum state existing in multiple states simultaneously until measured"),
            ("Entanglement", "Quantum correlation between particles regardless of distance"),
            ("Decoherence", "Loss of quantum properties due to environmental interaction")
        ]
        
        print("\n📚 Learning concepts...")
        for concept, explanation in concepts:
            result = await learning.learn_concept(concept, explanation, session_id)
            print(f"  ✓ Stored: {concept} (robustness: {result['memory_robustness']:.2f})")
            
        # Test retention
        print("\n🧪 Testing retention...")
        retention = await learning.test_retention(session_id)
        print(f"  Average retention: {retention['average_retention']:.2%}")
        
        for result in retention['individual_results']:
            print(f"  {result['concept']}: {result['retention_score']:.2f}")
            
    finally:
        await tori.stop()

async def demo_problem_solving():
    """Demo problem solving assistant"""
    print("\n🎯 Problem Solving Assistant Demo")
    print("=" * 60)
    
    config = TORIProductionConfig(enable_chaos=True)
    tori = TORIProductionSystem(config)
    await tori.start()
    
    try:
        solver = ProblemSolvingAssistant(tori)
        
        # Solve a complex problem
        solution = await solver.solve_complex_problem(
            problem="Design a sustainable city that can adapt to climate change while preserving cultural heritage",
            constraints=["Limited budget", "Must house 1 million people", "Carbon neutral by 2050"]
        )
        
        print("\n🏗️ Problem Solving Results:")
        print(f"\nFinal Solution:\n{solution['final_solution'][:500]}...")
        
        print(f"\n⚡ Efficiency Gains by Phase:")
        for phase, efficiency in solution['efficiency_gains'].items():
            print(f"  {phase}: {efficiency:.2f}x")
            
    finally:
        await tori.stop()

async def main():
    """Run all demos"""
    demos = [
        demo_creative_writing,
        demo_pattern_discovery,
        demo_memory_learning,
        demo_problem_solving
    ]
    
    for demo in demos:
        await demo()
        print("\n" + "="*60 + "\n")
        await asyncio.sleep(2)  # Brief pause between demos

if __name__ == "__main__":
    asyncio.run(main())
