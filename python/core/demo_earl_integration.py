#!/usr/bin/env python3
"""
Example integration of EARL intent reasoning with pattern matching.
Shows how to use the hybrid pipeline in production.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from intent_driven_reasoning import ReasoningIntentParser, IntentDrivenReasoning
from earl_intent_reasoner import EARLIntentReasoner, ActionEvent, HybridIntentEngine

def demo_hybrid_intent_engine():
    """Demonstrate the hybrid intent engine in action."""
    
    print("\n" + "="*60)
    print("🚀 HYBRID INTENT ENGINE DEMO")
    print("="*60)
    
    # Initialize components
    pattern_parser = ReasoningIntentParser()
    earl_reasoner = EARLIntentReasoner(
        max_hypotheses=5,
        confidence_threshold=0.7
    )
    
    # Create hybrid engine
    hybrid = HybridIntentEngine(
        pattern_parser=pattern_parser,
        earl_reasoner=earl_reasoner,
        confidence_threshold=0.6,  # When to use EARL
        always_update_earl=True  # Always learn
    )
    
    # Simulate user action sequence
    events = [
        ActionEvent("click", "new_document_button"),
        ActionEvent("type", "meeting_notes"),
        ActionEvent("click", "format_menu"),
        ActionEvent("select", "bullet_list"),
        ActionEvent("type", "agenda_item_1"),
    ]
    
    print("\n📋 Processing action sequence:")
    print("-" * 40)
    
    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}: {event}")
        
        # Process through hybrid engine
        intent, strategy, confidence, method = hybrid.process_event(event)
        
        print(f"  → Intent: {intent}")
        print(f"  → Strategy: {strategy}")
        print(f"  → Confidence: {confidence:.1%}")
        print(f"  → Method: {method}")
        
        # Show EARL's reasoning state
        if i % 2 == 0 or i == len(events):
            print("\n" + hybrid.earl.explain_reasoning())
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    stats = hybrid.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_earl_solo():
    """Demonstrate EARL reasoning alone."""
    
    print("\n" + "="*60)
    print("🧠 EARL INTENT REASONER DEMO")
    print("="*60)
    
    earl = EARLIntentReasoner(
        max_hypotheses=5,
        confidence_threshold=0.7
    )
    
    # Simulate troubleshooting sequence
    events = [
        ActionEvent("click", "error_message"),
        ActionEvent("search", "error_code_404"),
        ActionEvent("navigate", "help_docs"),
        ActionEvent("click", "troubleshooting_guide"),
        ActionEvent("copy", "solution_steps"),
    ]
    
    print("\n🔍 Troubleshooting sequence:")
    print("-" * 40)
    
    for event in events:
        print(f"\nProcessing: {event}")
        earl.update(event)
        
        # Get current best hypothesis
        intent, strategy, confidence = earl.get_top_intent()
        print(f"  Top intent: {intent} ({confidence:.1%})")
        
        # Show all hypotheses
        all_hyps = earl.get_all_hypotheses()
        if len(all_hyps) > 1:
            print("  Other hypotheses:")
            for hyp in all_hyps[1:3]:
                print(f"    - {hyp.intent}: {hyp.confidence:.1%}")
    
    print("\n" + earl.explain_reasoning())


def demo_pattern_confidence():
    """Demonstrate pattern matching with confidence scores."""
    
    print("\n" + "="*60)
    print("🎯 PATTERN MATCHING WITH CONFIDENCE")
    print("="*60)
    
    parser = ReasoningIntentParser()
    
    test_queries = [
        "why did the system crash",  # Clear intent
        "compare these options",  # Clear intent
        "hmm interesting",  # Ambiguous
        "fix problem analyze data",  # Multiple intents
        "the quick brown fox",  # No clear intent
    ]
    
    print("\n📝 Testing queries:")
    print("-" * 40)
    
    for query in test_queries:
        intent, strategy, confidence = parser.parse_intent(query)
        print(f"\nQuery: \"{query}\"")
        print(f"  Intent: {intent.value}")
        print(f"  Strategy: {strategy.value}")
        print(f"  Confidence: {confidence:.1%}")
        
        if confidence < 0.5:
            print("  ⚠️ Low confidence - would trigger EARL")


if __name__ == "__main__":
    # Run all demos
    demo_pattern_confidence()
    demo_earl_solo()
    demo_hybrid_intent_engine()
    
    print("\n✅ Demo complete!")
