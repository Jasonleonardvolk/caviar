#!/usr/bin/env python3
"""
🧠 TORI Cognitive System Examples
Demonstrating how to use the FastAPI bridge for cognitive processing
"""

import asyncio
import httpx
import json
from datetime import datetime

FASTAPI_URL = "http://localhost:8000"

async def example_simple_chat():
    """Example 1: Simple chat processing"""
    print("🗣️ Example 1: Simple Chat Processing")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        data = {
            "message": "Explain the concept of emergence in complex systems",
            "glyphs": ["anchor", "concept-synthesizer", "meta-echo:reflect", "return"]
        }
        
        response = await client.post(f"{FASTAPI_URL}/api/chat", json=data)
        result = response.json()
        
        print(f"Question: {data['message']}")
        print(f"Answer: {result['answer']}")
        print(f"Processing Time: {result['trace']['processingTime']}ms")
        print(f"Coherence Final: {result['trace']['coherenceTrace'][-1]:.3f}")
        print()

async def example_smart_ask():
    """Example 2: Smart ask with automatic glyph generation"""
    print("🎯 Example 2: Smart Ask (Auto Glyphs)")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        data = {
            "message": "Help me understand the relationship between quantum entanglement and information theory",
            "complexity": "research"
        }
        
        response = await client.post(f"{FASTAPI_URL}/api/smart/ask", json=data)
        result = response.json()
        
        print(f"Question: {data['message']}")
        print(f"Auto-Generated Glyphs: {result['trace']['glyphPath']}")
        print(f"Answer: {result['answer']}")
        print(f"Smart Processing: {result['trace']['metadata']['smartProcessing']}")
        print()

async def example_research_mode():
    """Example 3: Deep research mode"""
    print("🔬 Example 3: Deep Research Mode")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        data = {
            "query": "Analyze the implications of artificial general intelligence on human consciousness and societal structures",
            "depth": "deep"
        }
        
        response = await client.post(f"{FASTAPI_URL}/api/smart/research", json=data)
        result = response.json()
        
        print(f"Research Query: {data['query']}")
        print(f"Depth: {data['depth']}")
        print(f"Glyphs Used: {len(result['trace']['glyphPath'])}")
        print(f"Research Answer: {result['answer']}")
        print(f"Contradictions Resolved: {len(result['trace']['metadata']['contradictionPeaks'])}")
        print()

async def example_batch_processing():
    """Example 4: Batch cognitive processing"""
    print("📦 Example 4: Batch Processing")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        requests = [
            {
                "message": "What is machine learning?",
                "glyphs": ["anchor", "concept-synthesizer", "return"]
            },
            {
                "message": "What is deep learning?", 
                "glyphs": ["anchor", "concept-synthesizer", "return"]
            },
            {
                "message": "How do ML and DL relate?",
                "glyphs": ["anchor", "paradox-analyzer", "meta-echo:reflect", "return"]
            }
        ]
        
        data = {"requests": requests}
        
        response = await client.post(f"{FASTAPI_URL}/api/cognitive/batch", json=data)
        result = response.json()
        
        print(f"Batch Requests: {len(requests)}")
        print(f"Processed: {result['processed']}")
        print(f"Successful: {result['successful']}")
        
        for i, batch_result in enumerate(result['results']):
            if batch_result['success']:
                print(f"  {i+1}. {batch_result['answer'][:80]}...")
        print()

async def example_glyph_suggestions():
    """Example 5: Get glyph suggestions"""
    print("💡 Example 5: Glyph Suggestions")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        params = {
            "message": "Help me debug this complex software architecture problem",
            "complexity": "complex"
        }
        
        response = await client.get(f"{FASTAPI_URL}/api/glyph-suggestions", params=params)
        result = response.json()
        
        print(f"Message: {params['message']}")
        print(f"Complexity: {params['complexity']}")
        print(f"Suggested Glyphs: {result['suggestedGlyphs']}")
        print(f"Description: {result['description']}")
        print()

async def example_system_status():
    """Example 6: Check system status"""
    print("📊 Example 6: System Status")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{FASTAPI_URL}/api/status")
        result = response.json()
        
        print(f"Bridge Status: {result['bridge']['status']}")
        print(f"Cognitive Engine: {result['cognitive']['cognitive']['engine']['engineReady']}")
        print(f"Active Loops: {result['cognitive']['cognitive']['engine']['activeLoops']}")
        print(f"Total Processed: {result['cognitive']['cognitive']['engine']['totalProcessed']}")
        print(f"Current Coherence: {result['cognitive']['cognitive']['engine']['currentCoherence']:.3f}")
        print()

async def run_all_examples():
    """Run all examples"""
    print("🧠 TORI Cognitive System - Usage Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_simple_chat,
        example_smart_ask,
        example_research_mode,
        example_batch_processing,
        example_glyph_suggestions,
        example_system_status
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
            print()
    
    print("✅ All examples completed!")
    print()
    print("🎯 Key Takeaways:")
    print("• Use /api/chat for direct cognitive processing")
    print("• Use /api/smart/ask for auto-glyph generation")
    print("• Use /api/smart/research for deep analysis")
    print("• Use /api/cognitive/batch for multiple requests")
    print("• Check /api/status for system health")
    print()
    print("📚 Full API documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    asyncio.run(run_all_examples())
