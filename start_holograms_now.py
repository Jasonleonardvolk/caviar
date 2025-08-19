#!/usr/bin/env python3
"""
Hologram Auto-Seeder - Populates holograms immediately after TORI startup
Runs automatically to ensure holograms appear instead of just metrics
"""

import asyncio
import websockets
import json
import time

async def seed_concept_holograms():
    """Seed the concept mesh with default holograms"""
    print("🌟 Seeding concept holograms...")

    default_concepts = [
        {
            "id": "active_consciousness",
            "name": "Active Consciousness", 
            "description": "Real-time awareness processing",
            "position": {"x": 0, "y": 0, "z": 0},
            "color": {"r": 1.0, "g": 0.8, "b": 0.2},
            "size": 1.5,
            "connections": ["active_cognition", "active_perception"]
        },
        {
            "id": "active_cognition",
            "name": "Active Cognition",
            "description": "Dynamic thinking and reasoning",
            "position": {"x": 2, "y": 1, "z": 0}, 
            "color": {"r": 0.2, "g": 0.8, "b": 1.0},
            "size": 1.2,
            "connections": ["active_consciousness", "active_intelligence"]
        },
        {
            "id": "active_perception",
            "name": "Active Perception",
            "description": "Real-time sensory processing",
            "position": {"x": -1, "y": 2, "z": 1},
            "color": {"r": 0.8, "g": 0.2, "b": 1.0},
            "size": 1.0,
            "connections": ["active_consciousness"]
        },
        {
            "id": "active_intelligence",
            "name": "Active Intelligence", 
            "description": "Adaptive problem solving",
            "position": {"x": 1, "y": -1, "z": -1},
            "color": {"r": 1.0, "g": 0.5, "b": 0.0},
            "size": 1.3,
            "connections": ["active_cognition"]
        },
        {
            "id": "active_memory",
            "name": "Active Memory",
            "description": "Dynamic information storage",
            "position": {"x": -2, "y": 0, "z": 2},
            "color": {"r": 0.0, "g": 1.0, "b": 0.5},
            "size": 1.1,
            "connections": ["active_intelligence"]
        }
    ]

    try:
        uri = "ws://localhost:8766/concepts"
        print(f"Connecting to concept bridge: {uri}")

        async with websockets.connect(uri) as websocket:
            print("✅ Connected to concept bridge")

            # Add each concept
            for concept in default_concepts:
                message = {
                    "type": "add_concept",
                    "concept": concept
                }

                await websocket.send(json.dumps(message))
                print(f"  ➕ Added concept: {concept['name']}")
                await asyncio.sleep(0.1)

            # Request immediate hologram rendering
            render_message = {
                "type": "render_holograms",
                "enable_particles": True,
                "enable_connections": True,
                "enable_animations": True,
                "render_quality": "high"
            }

            await websocket.send(json.dumps(render_message))
            print("🎨 Requested immediate hologram rendering")

            # Wait for confirmation
            await asyncio.sleep(1)

        print("✅ Concept hologram seeding complete!")

    except Exception as e:
        print(f"❌ Failed to seed concept holograms: {e}")

async def seed_audio_patterns():
    """Seed the audio bridge with patterns to drive visualizations"""
    print("🎵 Seeding audio patterns...")

    try:
        uri = "ws://localhost:8765/audio_stream"
        print(f"Connecting to audio bridge: {uri}")

        async with websockets.connect(uri) as websocket:
            print("✅ Connected to audio bridge")

            # Send audio patterns that create hologram effects
            patterns = [
                {"frequency": 220, "amplitude": 0.7, "waveform": "sine"},
                {"frequency": 330, "amplitude": 0.6, "waveform": "triangle"}, 
                {"frequency": 440, "amplitude": 0.8, "waveform": "sine"},
                {"frequency": 660, "amplitude": 0.5, "waveform": "square"},
                {"frequency": 880, "amplitude": 0.6, "waveform": "sine"}
            ]

            for pattern in patterns:
                audio_data = {
                    "type": "audio_features",
                    "amplitude": pattern["amplitude"],
                    "frequency": pattern["frequency"],
                    "waveform": pattern["waveform"],
                    "timestamp": time.time(),
                    "hologram_enable": True
                }

                await websocket.send(json.dumps(audio_data))
                print(f"  🎵 Sent audio pattern: {pattern['frequency']}Hz")
                await asyncio.sleep(0.3)
                
        print("✅ Audio pattern seeding complete!")

    except Exception as e:
        print(f"❌ Failed to seed audio patterns: {e}")

async def main():
    """Main seeding function"""
    print("🚀 TORI Hologram Auto-Seeder Starting...")
    print("This will populate your hologram system with initial content")
    print()

    # Wait for TORI to be fully ready
    print("⏳ Waiting 3 seconds for TORI to initialize...")
    await asyncio.sleep(3)

    # Seed both systems
    await seed_concept_holograms()
    print()
    await seed_audio_patterns()

    print()
    print("🎉 Hologram seeding complete!")
    print("✨ You should now see holograms in your TORI interface!")
    print("🔄 If not visible, refresh your browser or restart TORI")

if __name__ == "__main__":
    asyncio.run(main())
